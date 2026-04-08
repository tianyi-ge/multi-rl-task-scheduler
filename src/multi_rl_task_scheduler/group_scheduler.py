from __future__ import annotations

import time
from collections import defaultdict
from typing import DefaultDict, Dict, List, Sequence, Tuple

from .algorithms import (
    assess_range,
    build_idle_workers_per_machine,
    dont_starve,
    feed_more,
    find_best_placement_global,
)
from .interfaces import GroupSchedulerProtocol, InferScheduler
from .models import ManagedTask, SchedulerTuning, TaskAllocation, TaskConfig, TaskStateReport, WorkerInfo


class GroupScheduler(GroupSchedulerProtocol):
    """Minimal in-memory scheduler skeleton aligned with the current spec.

    This is intentionally not the full production scheduler. It provides the
    public interface, stale-state protection, resource bookkeeping, and the
    issue #2 placement contract so future implementation work has a concrete
    code anchor.
    """

    def __init__(
        self,
        workers: Sequence[WorkerInfo],
        tuning: SchedulerTuning | None = None,
    ) -> None:
        self.worker_index: Dict[str, WorkerInfo] = {w.worker_id: w for w in workers}
        self.total_workers = len(self.worker_index)
        self.free_workers: Dict[str, WorkerInfo] = dict(self.worker_index)
        self.tasks: Dict[str, ManagedTask] = {}
        self.task_schedulers: Dict[str, InferScheduler] = {}
        self.last_state_version: Dict[str, int] = {}
        self.pending_reports: DefaultDict[str, List[TaskStateReport]] = defaultdict(list)
        self.tuning = tuning or SchedulerTuning()
        self.consecutive_reclaim_count = 0

    @property
    def free_gpus(self) -> List[WorkerInfo]:
        return list(self.free_workers.values())

    @property
    def idle_workers_per_machine(self) -> Dict[int, List[str]]:
        return build_idle_workers_per_machine(self.free_workers.keys(), self.worker_index)

    def register_task(self, config: TaskConfig, scheduler: InferScheduler) -> bool:
        if config.task_id in self.tasks:
            raise ValueError(f"task already registered: {config.task_id}")

        requested_baseline = config.base_instances * config.workers_per_instance
        reserved_baseline = sum(
            task.base_instances * task.workers_per_instance for task in self.tasks.values()
        )
        if reserved_baseline + requested_baseline > self.total_workers:
            return False

        self.tasks[config.task_id] = ManagedTask(config=config)
        self.task_schedulers[config.task_id] = scheduler
        self.last_state_version[config.task_id] = -1
        return True

    def report_state(self, report: TaskStateReport) -> None:
        self.pending_reports[report.task_id].append(report)
        self.process_pending_state_reports()

    def unregister_task(self, task_id: str) -> None:
        task = self.tasks.pop(task_id, None)
        self.task_schedulers.pop(task_id, None)
        self.last_state_version.pop(task_id, None)
        self.pending_reports.pop(task_id, None)
        if task is None:
            return
        for worker_id, worker in task.assigned_workers.items():
            self.free_workers[worker_id] = worker

    def process_pending_state_reports(self) -> None:
        for task_id, reports in list(self.pending_reports.items()):
            if not reports:
                continue
            reports.sort(key=lambda report: report.state_version)
            for report in reports:
                self.apply_task_state_report(report)
            self.pending_reports[task_id].clear()

    def managed_tasks(self) -> List[ManagedTask]:
        return list(self.tasks.values())

    def apply_task_state_report(self, report: TaskStateReport) -> bool:
        task = self.tasks.get(report.task_id)
        if task is None:
            return False
        if report.state_version <= self.last_state_version.get(report.task_id, -1):
            return False

        new_assigned = {worker.worker_id: worker for worker in report.assigned_workers}
        old_worker_ids = set(task.assigned_workers)
        new_worker_ids = set(new_assigned)

        released = old_worker_ids - new_worker_ids
        acquired = new_worker_ids - old_worker_ids

        for worker_id in released:
            self.free_workers[worker_id] = task.assigned_workers[worker_id]
        for worker_id in acquired:
            self.free_workers.pop(worker_id, None)
        for worker_id in old_worker_ids & new_worker_ids:
            self.free_workers.pop(worker_id, None)

        task.assigned_workers = new_assigned
        task.state = report
        self.last_state_version[report.task_id] = report.state_version
        return True

    def assign_from_plan(
        self,
        allocation_requests: Sequence[Tuple[str, int]],
    ) -> List[TaskStateReport]:
        workers_per_task = {
            task_id: self.tasks[task_id].workers_per_instance for task_id, _ in allocation_requests
        }
        placements = find_best_placement_global(
            allocation_requests,
            workers_per_task=workers_per_task,
            idle_workers=list(self.free_workers.keys()),
            idle_workers_per_machine=self.idle_workers_per_machine,
        )

        reports: List[TaskStateReport] = []
        for task_id, task_placements in placements:
            if not task_placements:
                continue
            scheduler = self.task_schedulers[task_id]
            report = scheduler.assign(task_placements)
            if self.apply_task_state_report(report):
                reports.append(report)
        return reports

    def reclaim_from_plan(
        self,
        reclaim_requests: Sequence[Tuple[str, int]],
    ) -> List[TaskStateReport]:
        reports: List[TaskStateReport] = []
        for task_id, num_instances in reclaim_requests:
            scheduler = self.task_schedulers[task_id]
            report = scheduler.reclaim(num_instances)
            if self.apply_task_state_report(report):
                reports.append(report)
        return reports

    def assess_range(self) -> Dict[str, Tuple[int, int]]:
        return assess_range(
            self.managed_tasks(),
            catch_up_ratio=self.tuning.catch_up_ratio,
            acceleration_limit_ratio=self.tuning.acceleration_limit_ratio,
        )

    def dont_starve(
        self,
        delta_card_ranges: Dict[str, Tuple[int, int]],
    ) -> Tuple[Dict[str, int], int]:
        return dont_starve(
            self.managed_tasks(),
            delta_card_ranges,
            free_card_count=len(self.free_workers),
        )

    def feed_more(
        self,
        delta_card_ranges: Dict[str, Tuple[int, int]],
        plan: Dict[str, int],
        excess_cards: int,
    ) -> Dict[str, int]:
        return feed_more(
            self.managed_tasks(),
            delta_card_ranges,
            plan,
            excess_cards=excess_cards,
        )

    def build_task_allocations(self, plan: Dict[str, int]) -> List[TaskAllocation]:
        allocations: List[TaskAllocation] = []
        for task_id, card_delta in plan.items():
            task = self.tasks[task_id]
            instance_delta = card_delta // task.workers_per_instance
            allocations.append(TaskAllocation(task_id=task_id, instance_delta=instance_delta))
        return allocations

    def run_scheduling_cycle(self) -> List[TaskAllocation]:
        delta_card_ranges = self.assess_range()
        plan, excess_cards = self.dont_starve(delta_card_ranges)
        if excess_cards:
            plan = self.feed_more(delta_card_ranges, plan, excess_cards)
        return self.execute_plan(plan)

    def execute_plan(self, plan: Dict[str, int]) -> List[TaskAllocation]:
        executed_cards = {task_id: 0 for task_id in self.tasks}
        has_reclaim = any(delta < 0 for delta in plan.values())
        has_assign = any(delta > 0 for delta in plan.values())

        if has_reclaim:
            reclaim_requests: List[Tuple[str, int]] = []
            for task_id, delta_cards in plan.items():
                if delta_cards >= 0:
                    continue
                task = self.tasks[task_id]
                if task.idle_instances > 0 or task.busy_instances > task.base_instances:
                    instance_count = (-delta_cards) // task.workers_per_instance
                    reclaim_requests.append(
                        (task_id, instance_count)
                    )
                    executed_cards[task_id] -= instance_count * task.workers_per_instance

            if reclaim_requests:
                self.reclaim_from_plan(reclaim_requests)
                self.process_pending_state_reports()
                self.consecutive_reclaim_count += 1

                free_worker_ratio = len(self.free_workers) / max(1, self.total_workers)
                force_assign = (
                    self.consecutive_reclaim_count >= self.tuning.max_consecutive_reclaims
                    or free_worker_ratio >= self.tuning.max_free_worker_ratio
                )

                if not force_assign:
                    return self.build_task_allocations(executed_cards)

                self.consecutive_reclaim_count = 0
                delta_card_ranges = self.assess_range()
                plan, excess_cards = self.dont_starve(delta_card_ranges)
                if excess_cards:
                    plan = self.feed_more(delta_card_ranges, plan, excess_cards)
                has_assign = any(delta > 0 for delta in plan.values())

        if has_assign:
            assign_requests: List[Tuple[str, int]] = []
            for task_id, delta_cards in plan.items():
                if delta_cards <= 0:
                    continue
                task = self.tasks[task_id]
                if task.in_rollout_phase and task.remaining_samples > 0:
                    instance_count = delta_cards // task.workers_per_instance
                    assign_requests.append(
                        (task_id, instance_count)
                    )
                    executed_cards[task_id] += instance_count * task.workers_per_instance
            if assign_requests:
                self.assign_from_plan(assign_requests)
            self.consecutive_reclaim_count = 0

        return self.build_task_allocations(executed_cards)

    def snapshot_decision_timestamp(self) -> float:
        return time.time()
