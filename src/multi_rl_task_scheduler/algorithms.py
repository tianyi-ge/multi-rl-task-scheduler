from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

from .models import ManagedTask


def compute_allocation_score(task: ManagedTask, plan_num_instance: int) -> float:
    """Issue #2 scoring rule.

    The score combines an exponential instance-balance term and an exponential
    sample-sufficiency term. It assumes the caller passes the number of extra
    instances already planned for this task in the current scheduling round.
    """
    if not task.has_state:
        return 0.0
    if not task.in_rollout_phase:
        return 0.0
    if task.remaining_samples <= 0:
        return 0.0
    if task.base_instances <= 0:
        return 0.0
    if task.total_samples <= 0:
        return 0.0

    weight_instances = 40.0
    weight_samples = 60.0

    assumed_num_instance = plan_num_instance + task.current_instances
    deficit = task.base_instances - assumed_num_instance
    score = weight_instances * math.exp(deficit / task.base_instances)

    if task.busy_instances > 0:
        remaining_ratio = task.remaining_samples / task.busy_instances
    else:
        remaining_ratio = task.remaining_samples * 5

    total_ratio = task.total_samples / task.base_instances
    sample_sufficiency = remaining_ratio / total_ratio
    score += weight_samples * math.exp(sample_sufficiency - 1.0)
    return score


def build_idle_workers_per_machine(idle_workers: Iterable[str], worker_index: Dict[str, object]) -> Dict[int, List[str]]:
    """Group idle worker ids by machine id.

    `worker_index` only needs to expose a `machine_id` attribute for each worker.
    """
    free_by_machine: Dict[int, List[str]] = defaultdict(list)
    for worker_id in idle_workers:
        worker = worker_index[worker_id]
        free_by_machine[worker.machine_id].append(worker_id)
    return dict(free_by_machine)


def assess_range(
    tasks: Sequence[ManagedTask],
    *,
    catch_up_ratio: float,
    acceleration_limit_ratio: float,
) -> Dict[str, Tuple[int, int]]:
    """Return per-task card delta range as `(min_cards, max_cards)`."""
    delta_card_ranges: Dict[str, Tuple[int, int]] = {}

    for task in tasks:
        cards_per_instance = task.workers_per_instance
        current_cards = task.current_instances * cards_per_instance
        max_total_cards = int(
            acceleration_limit_ratio * task.base_instances * cards_per_instance
        )

        if not task.in_rollout_phase:
            min_cards = -task.idle_instances * cards_per_instance
            max_cards = 0
        elif task.busy_instances < task.base_instances and task.remaining_samples > 0:
            must_instances = max(
                0,
                math.ceil(catch_up_ratio * task.base_instances - task.busy_instances),
            )
            min_cards = must_instances * cards_per_instance
            max_cards = max(0, max_total_cards - current_cards)
        elif task.remaining_samples == 0:
            must_reclaim_instances = task.idle_instances
            if task.busy_instances > task.base_instances:
                must_reclaim_instances += task.busy_instances - task.base_instances
            min_cards = -must_reclaim_instances * cards_per_instance
            max_cards = 0
        elif task.busy_instances > task.base_instances:
            min_cards = -(task.busy_instances - task.base_instances) * cards_per_instance
            max_cards = max(0, max_total_cards - current_cards)
        else:
            min_cards = 0
            max_cards = max(0, max_total_cards - current_cards)

        delta_card_ranges[task.task_id] = (min_cards, max_cards)

    return delta_card_ranges


def dont_starve(
    tasks: Sequence[ManagedTask],
    delta_card_ranges: Dict[str, Tuple[int, int]],
    *,
    free_card_count: int,
) -> Tuple[Dict[str, int], int]:
    """Satisfy required allocations first and return card-level plan."""
    task_map = {task.task_id: task for task in tasks}
    plan = {task.task_id: 0 for task in tasks}
    needy_tasks: List[Tuple[str, int]] = []
    reclaimable_tasks: List[Tuple[int, int, str]] = []

    for task in tasks:
        min_cards, _ = delta_card_ranges[task.task_id]
        if min_cards > 0:
            needy_tasks.append((task.task_id, min_cards))
        elif min_cards < 0:
            priority = 0 if task.idle_instances > 0 else 1
            reclaimable_tasks.append((priority, -min_cards, task.task_id))

    reclaimable_tasks.sort(key=lambda item: (-item[0], -item[1]))

    for task_id, needed_cards in needy_tasks:
        task = task_map[task_id]
        cards_per_instance = task.workers_per_instance
        while needed_cards > 0:
            if free_card_count >= cards_per_instance:
                plan[task_id] += cards_per_instance
                free_card_count -= cards_per_instance
                needed_cards -= cards_per_instance
                continue

            if not reclaimable_tasks:
                break

            priority, reclaimable_cards, donor_id = reclaimable_tasks.pop()
            donor = task_map[donor_id]
            reclaim_amount = min(reclaimable_cards, donor.workers_per_instance)
            plan[donor_id] -= reclaim_amount
            free_card_count += reclaim_amount

            remaining = reclaimable_cards - reclaim_amount
            if remaining > 0:
                reclaimable_tasks.append((priority, remaining, donor_id))
                reclaimable_tasks.sort(key=lambda item: (-item[0], -item[1]))

    return plan, free_card_count


def feed_more(
    tasks: Sequence[ManagedTask],
    delta_card_ranges: Dict[str, Tuple[int, int]],
    plan: Dict[str, int],
    *,
    excess_cards: int,
) -> Dict[str, int]:
    """Allocate excess cards using issue #2's score."""
    task_map = {task.task_id: task for task in tasks}
    task_scores: List[Tuple[float, str]] = []

    for task in tasks:
        _, max_cards = delta_card_ranges[task.task_id]
        if max_cards <= 0:
            continue
        plan_num_instance = plan[task.task_id] // task.workers_per_instance
        score = compute_allocation_score(task, plan_num_instance)
        if score > 0:
            task_scores.append((-score, task.task_id))

    task_scores.sort()

    while excess_cards > 0 and task_scores:
        _, task_id = task_scores.pop(0)
        task = task_map[task_id]
        cards_per_instance = task.workers_per_instance
        _, max_cards = delta_card_ranges[task_id]
        current_cards = task.current_instances * cards_per_instance + plan[task_id]
        max_allowed = task.current_instances * cards_per_instance + max_cards
        if current_cards >= max_allowed:
            continue

        if excess_cards >= cards_per_instance:
            plan[task_id] += cards_per_instance
            excess_cards -= cards_per_instance
            new_plan_num_instance = plan[task_id] // cards_per_instance
            new_score = compute_allocation_score(task, new_plan_num_instance)
            new_current = current_cards + cards_per_instance
            if new_score > 0 and new_current < max_allowed:
                task_scores.append((-new_score, task_id))
                task_scores.sort()

    return plan


def find_best_placement_global(
    allocation_requests: Sequence[Tuple[str, int]],
    *,
    workers_per_task: Dict[str, int],
    idle_workers: Sequence[str],
    idle_workers_per_machine: Dict[int, List[str]],
) -> List[Tuple[str, List[List[str]]]]:
    """Issue #2 placement rule.

    First try to place each full instance on a single machine. Any remaining
    requests are filled by slicing the remaining idle worker ids.
    """
    placements: List[Tuple[str, List[List[str]]]] = []
    used_workers = set()
    left_allocation_requests: List[Tuple[str, int]] = []

    for task_id, num_instances in allocation_requests:
        worker_per_instance = workers_per_task[task_id]
        instance_placements: List[List[str]] = []

        for _ in range(num_instances):
            for machine_id, machine_workers in idle_workers_per_machine.items():
                available = [wid for wid in machine_workers if wid not in used_workers]
                if len(available) >= worker_per_instance:
                    selected = available[:worker_per_instance]
                    instance_placements.append(selected)
                    used_workers.update(selected)
                    break
            else:
                left_allocation_requests.append(
                    (task_id, num_instances - len(instance_placements))
                )
                break

        placements.append((task_id, instance_placements))

    start = 0
    left_idle_workers = [wid for wid in idle_workers if wid not in used_workers]
    for task_id, num_instances in left_allocation_requests:
        worker_per_instance = workers_per_task[task_id]
        instance_placements: List[List[str]] = []
        for _ in range(num_instances):
            if start + worker_per_instance > len(left_idle_workers):
                break
            selected = left_idle_workers[start : start + worker_per_instance]
            start += worker_per_instance
            instance_placements.append(selected)
        placements.append((task_id, instance_placements))

    return placements
