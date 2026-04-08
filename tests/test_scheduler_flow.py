from __future__ import annotations

import unittest

from src.multi_rl_task_scheduler.algorithms import (
    compute_allocation_score,
    find_best_placement_global,
)
from src.multi_rl_task_scheduler.group_scheduler import GroupScheduler
from src.multi_rl_task_scheduler.interfaces import InferScheduler
from src.multi_rl_task_scheduler.models import TaskConfig, TaskStateReport, WorkerInfo


class FakeInferScheduler(InferScheduler):
    def __init__(self, task_id: str, workers: list[WorkerInfo], initial_worker_ids: list[str]) -> None:
        self.task_id = task_id
        self.worker_index = {worker.worker_id: worker for worker in workers}
        self.assigned_worker_ids = list(initial_worker_ids)
        self.state_version = 0
        self.in_rollout_phase = True
        self.remaining_samples = 64

    def report(self) -> TaskStateReport:
        current_instances = len(self.assigned_worker_ids) // 2
        busy_instances = current_instances
        idle_instances = 0
        if self.remaining_samples <= 0:
            busy_instances = 0
            idle_instances = current_instances
        elif current_instances > 1:
            busy_instances = current_instances - 1
            idle_instances = 1

        return TaskStateReport(
            task_id=self.task_id,
            state_version=self.state_version,
            done_samples=0,
            done_rounds=0,
            elapsed_time_sec=0.0,
            remaining_samples=self.remaining_samples,
            current_instances=current_instances,
            idle_instances=idle_instances,
            busy_instances=busy_instances,
            in_rollout_phase=self.in_rollout_phase,
            assigned_workers=[self.worker_index[wid] for wid in self.assigned_worker_ids],
            idle_worker_ids=self.assigned_worker_ids[-2:] if idle_instances else [],
        )

    def reclaim(self, num_instances: int) -> TaskStateReport:
        workers_to_remove = num_instances * 2
        if workers_to_remove > len(self.assigned_worker_ids):
            workers_to_remove = len(self.assigned_worker_ids)
        self.assigned_worker_ids = self.assigned_worker_ids[:-workers_to_remove]
        self.state_version += 1
        return self.report()

    def assign(self, placements: list[list[str]]) -> TaskStateReport:
        for placement in placements:
            self.assigned_worker_ids.extend(placement)
        self.state_version += 1
        return self.report()


class SchedulerFlowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.workers = [
            WorkerInfo(worker_id=f"m0g{i}", machine_id=0, gpu_id=i) for i in range(4)
        ] + [
            WorkerInfo(worker_id=f"m1g{i}", machine_id=1, gpu_id=i) for i in range(4)
        ]

    def test_stale_report_is_ignored(self) -> None:
        scheduler = GroupScheduler(self.workers)
        fake = FakeInferScheduler("task-a", self.workers, ["m0g0", "m0g1"])
        config = TaskConfig("task-a", base_instances=1, tp=1, pp=2, samples_per_round=64, total_samples=128)
        self.assertTrue(scheduler.register_task(config, fake))

        report_v0 = fake.report()
        scheduler.report_state(report_v0)
        self.assertEqual(len(scheduler.free_workers), 6)

        fake.assigned_worker_ids = ["m0g0", "m0g1", "m0g2", "m0g3"]
        fake.state_version = 1
        scheduler.report_state(fake.report())
        self.assertEqual(len(scheduler.free_workers), 4)

        scheduler.report_state(report_v0)
        self.assertEqual(len(scheduler.free_workers), 4)

    def test_run_scheduling_cycle_assigns_free_workers(self) -> None:
        scheduler = GroupScheduler(self.workers)
        task_a = FakeInferScheduler("task-a", self.workers, ["m0g0", "m0g1"])
        task_b = FakeInferScheduler("task-b", self.workers, ["m1g0", "m1g1"])
        task_a.remaining_samples = 128
        task_b.remaining_samples = 128

        config_a = TaskConfig("task-a", base_instances=2, tp=1, pp=2, samples_per_round=64, total_samples=128)
        config_b = TaskConfig("task-b", base_instances=1, tp=1, pp=2, samples_per_round=64, total_samples=128)
        self.assertTrue(scheduler.register_task(config_a, task_a))
        self.assertTrue(scheduler.register_task(config_b, task_b))
        scheduler.report_state(task_a.report())
        scheduler.report_state(task_b.report())

        allocations = scheduler.run_scheduling_cycle()
        allocation_map = {item.task_id: item.instance_delta for item in allocations}
        self.assertGreaterEqual(allocation_map["task-a"], 1)
        self.assertGreaterEqual(len(task_a.assigned_worker_ids), 4)

    def test_reclaim_and_force_assign_recomputes_plan(self) -> None:
        scheduler = GroupScheduler(self.workers[:6])
        donor = FakeInferScheduler("donor", self.workers[:6], ["m0g0", "m0g1", "m0g2", "m0g3"])
        needy = FakeInferScheduler("needy", self.workers[:6], ["m1g0", "m1g1"])
        donor.remaining_samples = 0
        needy.remaining_samples = 64

        config_donor = TaskConfig("donor", base_instances=1, tp=1, pp=2, samples_per_round=64, total_samples=128)
        config_needy = TaskConfig("needy", base_instances=2, tp=1, pp=2, samples_per_round=64, total_samples=128)
        self.assertTrue(scheduler.register_task(config_donor, donor))
        self.assertTrue(scheduler.register_task(config_needy, needy))
        scheduler.report_state(donor.report())
        scheduler.report_state(needy.report())

        allocations = scheduler.run_scheduling_cycle()
        allocation_map = {item.task_id: item.instance_delta for item in allocations}
        self.assertLessEqual(allocation_map["donor"], -1)
        self.assertGreaterEqual(allocation_map["needy"], 1)
        self.assertGreaterEqual(len(needy.assigned_worker_ids), 4)

    def test_compute_allocation_score_prefers_deficit_task(self) -> None:
        scheduler = GroupScheduler(self.workers)
        task_low = FakeInferScheduler("low", self.workers, ["m0g0", "m0g1"])
        task_high = FakeInferScheduler("high", self.workers, ["m1g0", "m1g1", "m1g2", "m1g3"])
        scheduler.register_task(TaskConfig("low", 2, 1, 2, 64, 128), task_low)
        scheduler.register_task(TaskConfig("high", 1, 1, 2, 64, 128), task_high)
        scheduler.report_state(task_low.report())
        scheduler.report_state(task_high.report())

        low_score = compute_allocation_score(scheduler.tasks["low"], 0)
        high_score = compute_allocation_score(scheduler.tasks["high"], 0)
        self.assertGreater(low_score, high_score)

    def test_find_best_placement_global_prefers_single_machine(self) -> None:
        placements = find_best_placement_global(
            [("task-a", 2)],
            workers_per_task={"task-a": 2},
            idle_workers=["m0g0", "m0g1", "m0g2", "m1g0", "m1g1"],
            idle_workers_per_machine={0: ["m0g0", "m0g1", "m0g2"], 1: ["m1g0", "m1g1"]},
        )
        self.assertEqual(placements[0][0], "task-a")
        self.assertEqual(placements[0][1][0], ["m0g0", "m0g1"])


if __name__ == "__main__":
    unittest.main()
