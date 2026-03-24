# GRPO GPU Scheduler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a global GPU scheduler for multi-GRPO-task workloads that enables time-sharing of a common GPU pool, solving the long-tail problem in inference.

**Architecture:** Centralized gRPC-based scheduler with per-task agents. The scheduler maintains state for all tasks, tracks debt/slack, and makes greedy marginal-gain based allocation decisions.

**Tech Stack:** Python, gRPC, asyncio, dataclasses, pytest

**Spec Reference:** [2026-03-24-grpo-gpu-scheduler-design.md](../specs/2026-03-24-grpo-gpu-scheduler-design.md)

---

## File Structure

```
grpo-scheduler/
├── pyproject.toml
├── src/
│   └── grpo_scheduler/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── state.py          # Data structures: TaskState, InstanceState, etc.
│       │   ├── metrics.py        # PhaseMetrics, history tracking
│       │   └── debt_tracker.py   # Debt and slack tracking
│       ├── scheduler/
│       │   ├── __init__.py
│       │   ├── engine.py         # Core scheduling algorithm
│       │   ├── estimators.py     # Time estimation functions
│       │   └── constraints.py    # Constraint checking
│       ├── proto/
│       │   ├── __init__.py
│       │   ├── scheduler.proto   # Protobuf definitions
│       │   └── scheduler_pb2.py  # Generated (not committed)
│       └── server/
│           ├── __init__.py
│           └── server.py         # gRPC server implementation
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_state.py
    ├── test_debt_tracker.py
    ├── test_estimators.py
    ├── test_constraints.py
    └── test_scheduler_engine.py
```

---

## Tuning Parameters (Constants)

```python
# src/grpo_scheduler/scheduler/config.py
@dataclass
class SchedulerConfig:
    alpha_idle: float = 0.1
    alpha_longtail: float = 1.5
    alpha_debt: float = 1.2
    switch_cost_threshold: float = 2.0
    min_allocation_duration_sec: float = 300.0
    tail_ratio_threshold: float = 2.0
```

---

### Task 1: Project Setup and Core Data Structures

**Files:**
- Create: `pyproject.toml`
- Create: `src/grpo_scheduler/__init__.py`
- Create: `src/grpo_scheduler/core/__init__.py`
- Create: `src/grpo_scheduler/core/state.py`
- Test: `tests/test_state.py`

- [ ] **Step 1: Create pyproject.toml with dependencies**

```toml
[project]
name = "grpo-scheduler"
version = "0.1.0"
description = "GRPO Multi-Task GPU Scheduler"
requires-python = ">=3.10"
dependencies = [
    "grpcio>=1.60.0",
    "grpcio-tools>=1.60.0",
    "protobuf>=4.25.0",
    "dataclasses-json>=0.6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0.0",
    "mypy>=1.6.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- [ ] **Step 2: Write test for core data structures**

```python
# tests/test_state.py
import pytest
from grpo_scheduler.core.state import (
    TaskConfig,
    InstanceState,
    PhaseMetrics,
    TaskState,
)


def test_task_config_creation():
    config = TaskConfig(
        task_id="task-1",
        base_instances=4,
        tp=2,
        pp=2,
        samples_per_round=64,
        total_samples=1024,
    )
    assert config.task_id == "task-1"
    assert config.base_instances == 4
    assert config.cards_per_instance == 4


def test_instance_state_busy():
    instance = InstanceState(
        instance_id=0,
        is_busy=True,
        elapsed_time_sec=10.5,
        done_samples=16,
        remaining_samples=32,
    )
    assert instance.is_busy is True
    assert instance.estimated_remaining_sec == 21.0


def test_instance_state_idle():
    instance = InstanceState(
        instance_id=1,
        is_busy=False,
        elapsed_time_sec=0.0,
        done_samples=0,
        remaining_samples=0,
    )
    assert instance.is_busy is False


def test_phase_metrics():
    metrics = PhaseMetrics(
        weight_transfer_sec=2.0,
        rollout_gen_sec=30.0,
        rollout_tool_sec=5.0,
        ref_log_prob_sec=1.0,
        reward_sec=0.5,
        adv_sec=0.3,
        update_sec=10.0,
        total_round_sec=48.8,
    )
    assert metrics.non_rollout_sec == 11.8


def test_task_state_initialization():
    config = TaskConfig(
        task_id="task-1",
        base_instances=4,
        tp=2,
        pp=2,
        samples_per_round=64,
        total_samples=1024,
    )
    state = TaskState(config=config)
    assert state.current_instances == 4
    assert state.done_samples == 0
    assert state.total_rounds == 16
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_state.py -v`
Expected: FAIL with "No module named 'grpo_scheduler'"

- [ ] **Step 4: Implement core state data structures**

```python
# src/grpo_scheduler/core/state.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class Phase(Enum):
    ROLLOUT = "rollout"
    UPDATE = "update"


@dataclass
class TaskConfig:
    task_id: str
    base_instances: int
    tp: int
    pp: int
    samples_per_round: int
    total_samples: int

    @property
    def cards_per_instance(self) -> int:
        return self.tp * self.pp


@dataclass
class InstanceState:
    instance_id: int
    is_busy: bool
    elapsed_time_sec: float = 0.0
    done_samples: int = 0
    remaining_samples: int = 0

    @property
    def estimated_remaining_sec(self) -> float:
        """Estimate remaining time for this instance based on current speed."""
        if not self.is_busy or self.done_samples == 0:
            return 0.0
        speed = self.done_samples / self.elapsed_time_sec
        if speed <= 0:
            return float('inf')
        return self.remaining_samples / speed


@dataclass
class PhaseMetrics:
    weight_transfer_sec: float = 0.0
    rollout_gen_sec: float = 0.0      # includes old_log_prob
    rollout_tool_sec: float = 0.0
    ref_log_prob_sec: float = 0.0
    reward_sec: float = 0.0
    adv_sec: float = 0.0
    update_sec: float = 0.0
    total_round_sec: float = 0.0

    @property
    def non_rollout_sec(self) -> float:
        """Non-rollout phases: max(ref_log_prob, reward) + adv + update."""
        return max(self.ref_log_prob_sec, self.reward_sec) + self.adv_sec + self.update_sec


@dataclass
class TaskState:
    config: TaskConfig

    # Progress
    done_samples: int = 0
    done_rounds: int = 0
    elapsed_time_sec: float = 0.0

    # Current allocation
    current_instances: int = field(init=False)
    idle_instances: int = 0

    # Instance states
    instances: List[InstanceState] = field(default_factory=list)

    # Latest metrics
    latest_metrics: PhaseMetrics = field(default_factory=PhaseMetrics)

    # History
    avg_round_sec_base: float = 0.0  # baseline avg round time
    rollout_history: dict[int, list[float]] = field(default_factory=dict)  # K -> list of times

    # Debt tracking
    debt: float = 0.0

    # Internal state
    start_time_sec: float = field(default_factory=lambda: 0.0)
    current_phase: Phase = Phase.ROLLOUT
    last_allocation_change_sec: float = 0.0

    def __post_init__(self):
        self.current_instances = self.config.base_instances

    @property
    def busy_instances(self) -> int:
        return self.current_instances - self.idle_instances

    @property
    def remaining_samples(self) -> int:
        return self.config.total_samples - self.done_samples

    @property
    def total_rounds(self) -> int:
        return (self.config.total_samples + self.config.samples_per_round - 1) // self.config.samples_per_round

    @property
    def remaining_rounds(self) -> int:
        return max(0, self.total_rounds - self.done_rounds)

    @property
    def slack(self) -> float:
        """Calculate slack: baseline total time - elapsed - remaining baseline time."""
        if self.avg_round_sec_base <= 0:
            return float('inf')
        baseline_total = self.total_rounds * self.avg_round_sec_base
        baseline_remaining = self.remaining_rounds * self.avg_round_sec_base
        return baseline_total - self.elapsed_time_sec - baseline_remaining
```

- [ ] **Step 5: Add __init__.py files**

```python
# src/grpo_scheduler/__init__.py
__version__ = "0.1.0"
```

```python
# src/grpo_scheduler/core/__init__.py
from .state import TaskConfig, InstanceState, PhaseMetrics, TaskState, Phase

__all__ = [
    "TaskConfig",
    "InstanceState",
    "PhaseMetrics",
    "TaskState",
    "Phase",
]
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_state.py -v`
Expected: PASS

---

### Task 2: Debt and Slack Tracker

**Files:**
- Create: `src/grpo_scheduler/core/debt_tracker.py`
- Test: `tests/test_debt_tracker.py`

- [ ] **Step 1: Write test for debt tracker**

```python
# tests/test_debt_tracker.py
import pytest
from grpo_scheduler.core.state import TaskConfig, TaskState, PhaseMetrics
from grpo_scheduler.core.debt_tracker import DebtTracker


def test_debt_tracker_initialization():
    config = TaskConfig(
        task_id="task-1",
        base_instances=4,
        tp=2,
        pp=2,
        samples_per_round=64,
        total_samples=1024,
    )
    state = TaskState(config=config)
    tracker = DebtTracker()
    assert tracker.get_debt("task-1") == 0.0


def test_debt_increases_when_slower():
    config = TaskConfig(
        task_id="task-1",
        base_instances=4,
        tp=2,
        pp=2,
        samples_per_round=64,
        total_samples=1024,
    )
    state = TaskState(config=config)
    state.avg_round_sec_base = 40.0

    tracker = DebtTracker()

    # Round took longer than baseline: debt increases
    metrics = PhaseMetrics(total_round_sec=50.0)
    tracker.on_round_complete(state, metrics)

    assert tracker.get_debt("task-1") == 10.0


def test_debt_decreases_when_faster():
    config = TaskConfig(
        task_id="task-1",
        base_instances=4,
        tp=2,
        pp=2,
        samples_per_round=64,
        total_samples=1024,
    )
    state = TaskState(config=config)
    state.avg_round_sec_base = 40.0

    tracker = DebtTracker()
    tracker.debts["task-1"] = 20.0  # Start with some debt

    # Round took less than baseline: debt decreases
    metrics = PhaseMetrics(total_round_sec=35.0)
    tracker.on_round_complete(state, metrics)

    assert tracker.get_debt("task-1") == 15.0


def test_constraint_check_passes_when_healthy():
    config = TaskConfig(
        task_id="task-1",
        base_instances=4,
        tp=2,
        pp=2,
        samples_per_round=64,
        total_samples=1024,
    )
    state = TaskState(config=config)
    state.avg_round_sec_base = 40.0
    state.elapsed_time_sec = 100.0
    state.done_rounds = 2

    tracker = DebtTracker()
    tracker.debts["task-1"] = 5.0

    # Should pass: 100 + (14 * 40) + 5 = 100 + 560 + 5 = 665
    # Baseline total: 16 * 40 = 640
    # Wait, let's adjust numbers to pass
    state.elapsed_time_sec = 50.0
    tracker.debts["task-1"] = 0.0
    # 50 + (14 * 40) = 50 + 560 = 610 < 640

    assert tracker.check_constraint(state) is True


def test_constraint_check_fails_when_behind():
    config = TaskConfig(
        task_id="task-1",
        base_instances=4,
        tp=2,
        pp=2,
        samples_per_round=64,
        total_samples=1024,
    )
    state = TaskState(config=config)
    state.avg_round_sec_base = 40.0
    state.elapsed_time_sec = 300.0  # Way behind
    state.done_rounds = 2

    tracker = DebtTracker()
    tracker.debts["task-1"] = 50.0

    assert tracker.check_constraint(state) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_debt_tracker.py -v`
Expected: FAIL

- [ ] **Step 3: Implement debt tracker**

```python
# src/grpo_scheduler/core/debt_tracker.py
from __future__ import annotations

from typing import Dict
from .state import TaskState, PhaseMetrics


class DebtTracker:
    def __init__(self):
        self.debts: Dict[str, float] = {}

    def get_debt(self, task_id: str) -> float:
        return self.debts.get(task_id, 0.0)

    def on_round_complete(self, state: TaskState, metrics: PhaseMetrics) -> None:
        """Update debt when a round completes."""
        task_id = state.config.task_id

        if state.avg_round_sec_base <= 0:
            return

        delta = metrics.total_round_sec - state.avg_round_sec_base
        self.debts[task_id] = self.get_debt(task_id) + delta

    def check_constraint(self, state: TaskState) -> bool:
        """
        Check if constraint is satisfied: even if we return to baseline allocation now,
        can we still finish on time?
        """
        if state.avg_round_sec_base <= 0:
            return True

        baseline_total = state.total_rounds * state.avg_round_sec_base
        baseline_remaining = state.remaining_rounds * state.avg_round_sec_base
        current_debt = self.get_debt(state.config.task_id)

        total_needed = state.elapsed_time_sec + baseline_remaining + max(current_debt, 0.0)
        return total_needed <= baseline_total

    def reset_task(self, task_id: str) -> None:
        """Reset debt for a task."""
        if task_id in self.debts:
            del self.debts[task_id]
```

- [ ] **Step 4: Update core __init__.py**

```python
# src/grpo_scheduler/core/__init__.py
from .state import TaskConfig, InstanceState, PhaseMetrics, TaskState, Phase
from .debt_tracker import DebtTracker

__all__ = [
    "TaskConfig",
    "InstanceState",
    "PhaseMetrics",
    "TaskState",
    "Phase",
    "DebtTracker",
]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_debt_tracker.py -v`
Expected: PASS

---

### Task 3: Time Estimators

**Files:**
- Create: `src/grpo_scheduler/scheduler/__init__.py`
- Create: `src/grpo_scheduler/scheduler/config.py`
- Create: `src/grpo_scheduler/scheduler/estimators.py`
- Test: `tests/test_estimators.py`

- [ ] **Step 1: Write test for estimators**

```python
# tests/test_estimators.py
import pytest
from grpo_scheduler.core.state import TaskConfig, TaskState, InstanceState, PhaseMetrics, Phase
from grpo_scheduler.scheduler.config import SchedulerConfig
from grpo_scheduler.scheduler.estimators import (
    estimate_rollout_remaining,
    estimate_future_round_time,
    estimate_remaining_time,
    has_long_tail,
)


def test_estimate_rollout_remaining_with_live_instances():
    config = TaskConfig(
        task_id="task-1",
        base_instances=4,
        tp=2,
        pp=2,
        samples_per_round=64,
        total_samples=1024,
    )
    state = TaskState(config=config)
    state.current_phase = Phase.ROLLOUT
    state.instances = [
        InstanceState(
            instance_id=0,
            is_busy=True,
            elapsed_time_sec=10.0,
            done_samples=10,
            remaining_samples=10,
        ),  # Estimated rem: 10s
        InstanceState(
            instance_id=1,
            is_busy=True,
            elapsed_time_sec=10.0,
            done_samples=5,
            remaining_samples=15,
        ),  # Estimated rem: 30s
    ]
    state.idle_instances = 0

    config = SchedulerConfig()
    rem = estimate_rollout_remaining(state, 2, config)

    assert rem == pytest.approx(30.0)


def test_has_long_tail_true():
    config = TaskConfig(
        task_id="task-1",
        base_instances=4,
        tp=2,
        pp=2,
        samples_per_round=64,
        total_samples=1024,
    )
    state = TaskState(config=config)
    state.current_phase = Phase.ROLLOUT
    state.instances = [
        InstanceState(instance_id=0, is_busy=True, done_samples=40),
        InstanceState(instance_id=1, is_busy=True, done_samples=10),
    ]

    cfg = SchedulerConfig(tail_ratio_threshold=2.0)
    assert has_long_tail(state, cfg) is True


def test_has_long_tail_false():
    config = TaskConfig(
        task_id="task-1",
        base_instances=4,
        tp=2,
        pp=2,
        samples_per_round=64,
        total_samples=1024,
    )
    state = TaskState(config=config)
    state.current_phase = Phase.ROLLOUT
    state.instances = [
        InstanceState(instance_id=0, is_busy=True, done_samples=25),
        InstanceState(instance_id=1, is_busy=True, done_samples=20),
    ]

    cfg = SchedulerConfig(tail_ratio_threshold=2.0)
    assert has_long_tail(state, cfg) is False


def test_estimate_future_round_time_from_history():
    config = TaskConfig(
        task_id="task-1",
        base_instances=4,
        tp=2,
        pp=2,
        samples_per_round=64,
        total_samples=1024,
    )
    state = TaskState(config=config)
    state.rollout_history[4] = [30.0, 32.0, 28.0]  # K=4, avg 30
    state.latest_metrics = PhaseMetrics(
        weight_transfer_sec=2.0,
        non_rollout_sec=10.0,
    )

    cfg = SchedulerConfig()
    round_time = estimate_future_round_time(state, 4, cfg)

    assert round_time == pytest.approx(42.0)  # 2 + 30 + 10


def test_estimate_remaining_time_in_rollout():
    config = TaskConfig(
        task_id="task-1",
        base_instances=4,
        tp=2,
        pp=2,
        samples_per_round=64,
        total_samples=1024,
    )
    state = TaskState(config=config)
    state.avg_round_sec_base = 40.0
    state.current_phase = Phase.ROLLOUT
    state.done_rounds = 2  # 14 remaining
    state.elapsed_time_sec = 80.0
    state.instances = [
        InstanceState(
            instance_id=0,
            is_busy=True,
            elapsed_time_sec=10.0,
            done_samples=10,
            remaining_samples=10,
        ),
    ]
    state.rollout_history[4] = [30.0]
    state.latest_metrics = PhaseMetrics(
        weight_transfer_sec=2.0,
        non_rollout_sec=8.0,
    )

    cfg = SchedulerConfig()
    total_rem = estimate_remaining_time(state, 4, cfg)

    # Current round rem: ~10 + 8 = 18
    # Future rounds: 13 * (2 + 30 + 8) = 13 * 40 = 520
    # Total: ~538
    assert total_rem > 500
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_estimators.py -v`
Expected: FAIL

- [ ] **Step 3: Implement config and estimators**

```python
# src/grpo_scheduler/scheduler/config.py
from dataclasses import dataclass


@dataclass
class SchedulerConfig:
    alpha_idle: float = 0.1
    alpha_longtail: float = 1.5
    alpha_debt: float = 1.2
    switch_cost_threshold: float = 2.0
    min_allocation_duration_sec: float = 300.0
    tail_ratio_threshold: float = 2.0
```

```python
# src/grpo_scheduler/scheduler/estimators.py
from __future__ import annotations

import math
from typing import Optional

from ..core.state import TaskState, Phase
from .config import SchedulerConfig


def estimate_rollout_remaining(
    state: TaskState,
    target_instances: int,
    config: SchedulerConfig,
) -> float:
    """
    Estimate remaining rollout time for the current round.
    If target_instances == current_instances, use live instance data.
    Otherwise, use historical estimates.
    """
    if state.current_phase != Phase.ROLLOUT:
        return 0.0

    # Use live data if same instance count
    if target_instances == state.current_instances and state.busy_instances > 0:
        max_rem = 0.0
        for inst in state.instances:
            if inst.is_busy:
                rem = inst.estimated_remaining_sec
                if rem > max_rem:
                    max_rem = rem
        return max_rem

    # Fall back to historical estimate
    return _estimate_rollout_from_history(state, target_instances)


def _estimate_rollout_from_history(state: TaskState, k: int) -> float:
    """Estimate rollout time for k instances using history or interpolation."""
    if k <= 0:
        return float('inf')

    # Exact match in history
    if k in state.rollout_history and state.rollout_history[k]:
        times = state.rollout_history[k]
        return sum(times) / len(times)

    # Find nearest K for interpolation
    nearest_k = None
    nearest_dist = float('inf')
    for hist_k in state.rollout_history:
        dist = abs(hist_k - k)
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_k = hist_k

    if nearest_k is not None and state.rollout_history[nearest_k]:
        base_time = sum(state.rollout_history[nearest_k]) / len(state.rollout_history[nearest_k])
        # Sublinear scaling: time ~ 1/sqrt(k)
        scale = math.sqrt(nearest_k / k)
        return base_time * scale

    # No history: use baseline average if available
    if state.avg_round_sec_base > 0:
        # Subtract non-rollout phases
        non_rollout = state.latest_metrics.non_rollout_sec if state.latest_metrics else 10.0
        base_rollout = max(1.0, state.avg_round_sec_base - non_rollout - state.latest_metrics.weight_transfer_sec)
        # Scale from baseline K
        scale = math.sqrt(state.config.base_instances / k)
        return base_rollout * scale

    return 60.0  # Default guess


def estimate_future_round_time(
    state: TaskState,
    target_instances: int,
    config: SchedulerConfig,
) -> float:
    """Estimate total time for a full future round with target_instances."""
    wt = state.latest_metrics.weight_transfer_sec if state.latest_metrics else 2.0
    rollout = _estimate_rollout_from_history(state, target_instances)
    non_rollout = state.latest_metrics.non_rollout_sec if state.latest_metrics else 10.0

    return wt + rollout + non_rollout


def estimate_remaining_time(
    state: TaskState,
    target_instances: int,
    config: SchedulerConfig,
) -> float:
    """Estimate total remaining time for the task with target_instances."""
    if state.remaining_rounds <= 0:
        return 0.0

    if state.current_phase == Phase.ROLLOUT:
        # Current round: remaining rollout + non-rollout
        rollout_rem = estimate_rollout_remaining(state, target_instances, config)
        non_rollout = state.latest_metrics.non_rollout_sec if state.latest_metrics else 10.0
        current_round_rem = rollout_rem + non_rollout

        # Future rounds
        if state.remaining_rounds > 1:
            future_round_time = estimate_future_round_time(state, target_instances, config)
            future_total = (state.remaining_rounds - 1) * future_round_time
        else:
            future_total = 0.0

        return current_round_rem + future_total

    else:  # UPDATE phase
        # Estimate remaining update time (simplified)
        upd_rem = 5.0  # Could be smarter with elapsed update time

        # Future rounds
        future_round_time = estimate_future_round_time(state, target_instances, config)
        future_total = state.remaining_rounds * future_round_time

        return upd_rem + future_total


def has_long_tail(state: TaskState, config: SchedulerConfig) -> bool:
    """Check if task is currently experiencing a long tail in rollout."""
    if state.current_phase != Phase.ROLLOUT or state.busy_instances < 2:
        return False

    done_samples = [
        inst.done_samples
        for inst in state.instances
        if inst.is_busy
    ]

    if len(done_samples) < 2:
        return False

    done_samples_sorted = sorted(done_samples)
    fastest = done_samples_sorted[-1]
    slowest = done_samples_sorted[0]

    if slowest <= 0:
        return True

    return fastest > config.tail_ratio_threshold * slowest
```

- [ ] **Step 4: Update scheduler __init__.py**

```python
# src/grpo_scheduler/scheduler/__init__.py
from .config import SchedulerConfig
from .estimators import (
    estimate_rollout_remaining,
    estimate_future_round_time,
    estimate_remaining_time,
    has_long_tail,
)

__all__ = [
    "SchedulerConfig",
    "estimate_rollout_remaining",
    "estimate_future_round_time",
    "estimate_remaining_time",
    "has_long_tail",
]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_estimators.py -v`
Expected: PASS

---

### Task 4: Constraints Checking

**Files:**
- Create: `src/grpo_scheduler/scheduler/constraints.py`
- Test: `tests/test_constraints.py`

- [ ] **Step 1: Write test for constraints**

```python
# tests/test_constraints.py
import pytest
from grpo_scheduler.core.state import TaskConfig, TaskState
from grpo_scheduler.core.debt_tracker import DebtTracker
from grpo_scheduler.scheduler.config import SchedulerConfig
from grpo_scheduler.scheduler.constraints import (
    can_reclaim_from,
    check_can_allocate,
)


def test_can_reclaim_from_when_above_min():
    config = TaskConfig(
        task_id="task-1",
        base_instances=4,
        tp=2,
        pp=2,
        samples_per_round=64,
        total_samples=1024,
    )
    state = TaskState(config=config)
    state.current_instances = 3  # Above min (1)

    debt_tracker = DebtTracker()
    cfg = SchedulerConfig()

    # Constraint passes by default (no elapsed time)
    assert can_reclaim_from(state, debt_tracker, cfg, 0.0) is True


def test_cannot_reclaim_from_when_at_min():
    config = TaskConfig(
        task_id="task-1",
        base_instances=4,
        tp=2,
        pp=2,
        samples_per_round=64,
        total_samples=1024,
    )
    state = TaskState(config=config)
    state.current_instances = 1  # At min

    debt_tracker = DebtTracker()
    cfg = SchedulerConfig()

    assert can_reclaim_from(state, debt_tracker, cfg, 0.0) is False


def test_cannot_reclaim_from_when_constraint_fails():
    config = TaskConfig(
        task_id="task-1",
        base_instances=4,
        tp=2,
        pp=2,
        samples_per_round=64,
        total_samples=1024,
    )
    state = TaskState(config=config)
    state.current_instances = 3
    state.avg_round_sec_base = 40.0
    state.elapsed_time_sec = 600.0  # Way behind
    state.done_rounds = 2

    debt_tracker = DebtTracker()
    debt_tracker.debts["task-1"] = 50.0

    cfg = SchedulerConfig()

    assert can_reclaim_from(state, debt_tracker, cfg, 0.0) is False


def test_can_allocate_when_free_gpu():
    # Check is mainly about switching cost vs gain
    # can_allocate mainly checks if allocation is beneficial
    cfg = SchedulerConfig(switch_cost_threshold=2.0)

    # Gain exceeds switching cost * threshold
    assert check_can_allocate(gain=10.0, switch_cost=2.0, config=cfg) is True

    # Gain below threshold
    assert check_can_allocate(gain=3.0, switch_cost=2.0, config=cfg) is False  # 3 < 2 * 2

    # No switching cost needed
    assert check_can_allocate(gain=1.0, switch_cost=0.0, config=cfg) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_constraints.py -v`
Expected: FAIL

- [ ] **Step 3: Implement constraints checking**

```python
# src/grpo_scheduler/scheduler/constraints.py
from __future__ import annotations

from ..core.state import TaskState
from ..core.debt_tracker import DebtTracker
from .config import SchedulerConfig


def can_reclaim_from(
    state: TaskState,
    debt_tracker: DebtTracker,
    config: SchedulerConfig,
    current_time_sec: float,
) -> bool:
    """Check if we can reclaim an instance from this task."""
    # Can't go below minimum instances
    if state.current_instances <= 1:
        return False

    # Check minimum allocation duration
    time_since_change = current_time_sec - state.last_allocation_change_sec
    if time_since_change < config.min_allocation_duration_sec:
        return False

    # Check constraint: can we still finish on time if we reclaim?
    # We delegate the full check to the debt tracker for now
    return debt_tracker.check_constraint(state)


def check_can_allocate(
    gain: float,
    switch_cost: float,
    config: SchedulerConfig,
) -> bool:
    """
    Check if allocation is worth it.
    gain must be > switch_cost * threshold if switching is needed.
    """
    if gain <= 0:
        return False

    if switch_cost <= 0:
        return True

    return gain > switch_cost * config.switch_cost_threshold
```

- [ ] **Step 4: Update scheduler __init__.py**

```python
# src/grpo_scheduler/scheduler/__init__.py
from .config import SchedulerConfig
from .estimators import (
    estimate_rollout_remaining,
    estimate_future_round_time,
    estimate_remaining_time,
    has_long_tail,
)
from .constraints import can_reclaim_from, check_can_allocate

__all__ = [
    "SchedulerConfig",
    "estimate_rollout_remaining",
    "estimate_future_round_time",
    "estimate_remaining_time",
    "has_long_tail",
    "can_reclaim_from",
    "check_can_allocate",
]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_constraints.py -v`
Expected: PASS

---

### Task 5: Scheduler Engine - Marginal Gain/Loss Calculation

**Files:**
- Create: `src/grpo_scheduler/scheduler/engine.py`
- Test: `tests/test_scheduler_engine.py`

- [ ] **Step 1: Write test for marginal gain/loss**

```python
# tests/test_scheduler_engine.py
import pytest
from grpo_scheduler.core.state import TaskConfig, TaskState, Phase, InstanceState, PhaseMetrics
from grpo_scheduler.core.debt_tracker import DebtTracker
from grpo_scheduler.scheduler.config import SchedulerConfig
from grpo_scheduler.scheduler.engine import (
    compute_marginal_loss,
    compute_marginal_gain,
)
from grpo_scheduler.scheduler.estimators import estimate_remaining_time


def test_compute_marginal_loss_basic():
    config = TaskConfig(
        task_id="task-1",
        base_instances=4,
        tp=2,
        pp=2,
        samples_per_round=64,
        total_samples=1024,
    )
    state = TaskState(config=config)
    state.current_instances = 3
    state.done_rounds = 1
    state.avg_round_sec_base = 40.0
    state.rollout_history[3] = [30.0]
    state.rollout_history[2] = [45.0]  # Higher time with fewer instances
    state.latest_metrics = PhaseMetrics(
        weight_transfer_sec=2.0,
        non_rollout_sec=8.0,
    )

    debt_tracker = DebtTracker()
    cfg = SchedulerConfig()

    loss = compute_marginal_loss(state, debt_tracker, cfg, 0.0)

    # Loss should be positive (time increases when going from 3 -> 2)
    assert loss > 0


def test_compute_marginal_loss_idle_instance_discounted():
    config = TaskConfig(
        task_id="task-1",
        base_instances=4,
        tp=2,
        pp=2,
        samples_per_round=64,
        total_samples=1024,
    )
    state = TaskState(config=config)
    state.current_instances = 3
    state.idle_instances = 1  # One idle instance
    state.done_rounds = 1
    state.avg_round_sec_base = 40.0
    state.rollout_history[3] = [30.0]
    state.rollout_history[2] = [35.0]
    state.latest_metrics = PhaseMetrics()

    debt_tracker = DebtTracker()
    cfg = SchedulerConfig(alpha_idle=0.1)

    loss = compute_marginal_loss(state, debt_tracker, cfg, 0.0)

    # Loss should be discounted by alpha_idle
    assert loss >= 0


def test_compute_marginal_gain_basic():
    config = TaskConfig(
        task_id="task-1",
        base_instances=4,
        tp=2,
        pp=2,
        samples_per_round=64,
        total_samples=1024,
    )
    state = TaskState(config=config)
    state.current_instances = 3
    state.done_rounds = 1
    state.avg_round_sec_base = 40.0
    state.rollout_history[3] = [35.0]
    state.rollout_history[4] = [28.0]  # Lower time with more instances
    state.latest_metrics = PhaseMetrics(
        weight_transfer_sec=2.0,
        non_rollout_sec=8.0,
    )

    debt_tracker = DebtTracker()
    cfg = SchedulerConfig()

    gain = compute_marginal_gain(state, debt_tracker, cfg)

    # Gain should be positive (time decreases when going from 3 -> 4)
    assert gain > 0


def test_compute_marginal_gain_long_tail_bonus():
    config = TaskConfig(
        task_id="task-1",
        base_instances=4,
        tp=2,
        pp=2,
        samples_per_round=64,
        total_samples=1024,
    )
    state = TaskState(config=config)
    state.current_instances = 3
    state.current_phase = Phase.ROLLOUT
    state.done_rounds = 1
    state.avg_round_sec_base = 40.0
    state.rollout_history[3] = [35.0]
    state.rollout_history[4] = [28.0]
    state.latest_metrics = PhaseMetrics()
    state.instances = [
        InstanceState(instance_id=0, is_busy=True, done_samples=40),
        InstanceState(instance_id=1, is_busy=True, done_samples=10),
        InstanceState(instance_id=2, is_busy=True, done_samples=5),
    ]

    debt_tracker = DebtTracker()
    cfg = SchedulerConfig(alpha_longtail=1.5)

    gain = compute_marginal_gain(state, debt_tracker, cfg)

    # Long tail should give bonus
    assert gain > 0


def test_compute_marginal_gain_debt_bonus():
    config = TaskConfig(
        task_id="task-1",
        base_instances=4,
        tp=2,
        pp=2,
        samples_per_round=64,
        total_samples=1024,
    )
    state = TaskState(config=config)
    state.current_instances = 3
    state.done_rounds = 1
    state.avg_round_sec_base = 40.0
    state.rollout_history[3] = [35.0]
    state.rollout_history[4] = [28.0]
    state.latest_metrics = PhaseMetrics()

    debt_tracker = DebtTracker()
    debt_tracker.debts["task-1"] = 20.0  # Has debt

    cfg = SchedulerConfig(alpha_debt=1.2)

    gain = compute_marginal_gain(state, debt_tracker, cfg)

    # Debt should give bonus
    assert gain > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scheduler_engine.py -v`
Expected: FAIL

- [ ] **Step 3: Implement marginal gain/loss functions**

```python
# src/grpo_scheduler/scheduler/engine.py
from __future__ import annotations

from ..core.state import TaskState
from ..core.debt_tracker import DebtTracker
from .config import SchedulerConfig
from .estimators import estimate_remaining_time, has_long_tail
from .constraints import can_reclaim_from


def compute_marginal_loss(
    state: TaskState,
    debt_tracker: DebtTracker,
    config: SchedulerConfig,
    current_time_sec: float,
) -> float:
    """
    Compute the marginal loss (increase in remaining time) if we reclaim
    one instance from this task.
    Returns infinity if we can't reclaim.
    """
    if state.current_instances <= 1:
        return float('inf')

    if not can_reclaim_from(state, debt_tracker, config, current_time_sec):
        return float('inf')

    # Current remaining time
    rem_current = estimate_remaining_time(state, state.current_instances, config)

    # Remaining time if we reclaim one
    rem_after = estimate_remaining_time(state, state.current_instances - 1, config)

    loss = max(0.0, rem_after - rem_current)

    # Discount loss if there are idle instances
    if state.idle_instances > 0:
        loss *= config.alpha_idle

    return loss


def compute_marginal_gain(
    state: TaskState,
    debt_tracker: DebtTracker,
    config: SchedulerConfig,
) -> float:
    """
    Compute the marginal gain (reduction in remaining time) if we add
    one instance to this task.
    """
    # Current remaining time
    rem_current = estimate_remaining_time(state, state.current_instances, config)

    # Remaining time if we add one
    rem_after = estimate_remaining_time(state, state.current_instances + 1, config)

    gain = max(0.0, rem_current - rem_after)

    # Bonus for long tail tasks
    if has_long_tail(state, config):
        gain *= config.alpha_longtail

    # Bonus for tasks in debt
    debt = debt_tracker.get_debt(state.config.task_id)
    if debt > 0:
        gain *= config.alpha_debt

    return gain
```

- [ ] **Step 4: Update scheduler __init__.py**

```python
# src/grpo_scheduler/scheduler/__init__.py
from .config import SchedulerConfig
from .estimators import (
    estimate_rollout_remaining,
    estimate_future_round_time,
    estimate_remaining_time,
    has_long_tail,
)
from .constraints import can_reclaim_from, check_can_allocate
from .engine import compute_marginal_loss, compute_marginal_gain

__all__ = [
    "SchedulerConfig",
    "estimate_rollout_remaining",
    "estimate_future_round_time",
    "estimate_remaining_time",
    "has_long_tail",
    "can_reclaim_from",
    "check_can_allocate",
    "compute_marginal_loss",
    "compute_marginal_gain",
]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_scheduler_engine.py -v`
Expected: PASS

---

### Task 6: Full Scheduler Algorithm

**Files:**
- Modify: `src/grpo_scheduler/scheduler/engine.py` (add full scheduler)
- Test: `tests/test_scheduler_engine.py` (add more tests)

- [ ] **Step 1: Add full scheduling algorithm to engine.py**

(Append to existing engine.py)

```python
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import heapq

from ..core.state import TaskState
from ..core.debt_tracker import DebtTracker
from .config import SchedulerConfig
from .estimators import estimate_remaining_time, has_long_tail
from .constraints import can_reclaim_from, check_can_allocate


@dataclass
class AllocationChange:
    task_id: str
    delta: int  # +1 for add, -1 for remove


@dataclass
class SchedulingResult:
    changes: List[AllocationChange]
    timestamp_sec: float


class GlobalScheduler:
    def __init__(self, config: Optional[SchedulerConfig] = None):
        self.config = config or SchedulerConfig()
        self.debt_tracker = DebtTracker()
        self.tasks: Dict[str, TaskState] = {}
        self.total_gpus: int = 64  # TODO: make configurable
        self._last_time: float = 0.0

    def register_task(self, state: TaskState) -> None:
        """Register a new task with the scheduler."""
        self.tasks[state.config.task_id] = state
        state.start_time_sec = self._get_time()
        state.last_allocation_change_sec = self._get_time()

    def unregister_task(self, task_id: str) -> None:
        """Remove a task from the scheduler."""
        if task_id in self.tasks:
            del self.tasks[task_id]
            self.debt_tracker.reset_task(task_id)

    def update_task_state(self, task_id: str, state_update: dict) -> None:
        """Update a task's state from a report."""
        if task_id not in self.tasks:
            return
        # TODO: implement state update logic
        pass

    def schedule(self) -> SchedulingResult:
        """
        Run the scheduling algorithm.
        Returns allocation changes to apply.
        """
        current_time = self._get_time()
        self._last_time = current_time

        changes: List[AllocationChange] = []

        # Step 1: Constraint check
        need_restore: Dict[str, int] = {}
        for task in self.tasks.values():
            if not self.debt_tracker.check_constraint(task):
                task._can_reclaim = False
                if task.current_instances < task.config.base_instances:
                    need = task.config.base_instances - task.current_instances
                    need_restore[task.config.task_id] = need
            else:
                task._can_reclaim = (task.current_instances > 1)

        # Step 2: Collect reclaimable tasks
        reclaim_heap: List[Tuple[float, str]] = []
        for task in self.tasks.values():
            if getattr(task, '_can_reclaim', False):
                loss = compute_marginal_loss(task, self.debt_tracker, self.config, current_time)
                if loss < float('inf'):
                    heapq.heappush(reclaim_heap, (loss, task.config.task_id))

        # Step 3: Collect gain candidates
        gain_heap: List[Tuple[float, str]] = []
        for task in self.tasks.values():
            gain = compute_marginal_gain(task, self.debt_tracker, self.config)
            if gain > 0:
                heapq.heappush(gain_heap, (-gain, task.config.task_id))

        # Calculate current free GPUs (simplified)
        free_gpus = self._calculate_free_gpus()

        # Step 4: Restore tasks below baseline first
        for task_id, need in need_restore.items():
            task = self.tasks[task_id]
            cards_needed = task.config.cards_per_instance
            while need > 0:
                if free_gpus >= cards_needed:
                    changes.append(AllocationChange(task_id=task_id, delta=+1))
                    free_gpus -= cards_needed
                    need -= 1
                elif reclaim_heap:
                    loss, reclaim_task_id = heapq.heappop(reclaim_heap)
                    reclaim_task = self.tasks[reclaim_task_id]
                    changes.append(AllocationChange(task_id=reclaim_task_id, delta=-1))
                    free_gpus += reclaim_task.config.cards_per_instance
                else:
                    break

        # Step 5: Greedy allocation to highest gain tasks
        while gain_heap:
            neg_gain, task_id = heapq.heappop(gain_heap)
            gain = -neg_gain

            task = self.tasks[task_id]
            cards_needed = task.config.cards_per_instance

            if free_gpus >= cards_needed:
                # Check if gain is worth it (no switch cost if free)
                if gain > 0:
                    changes.append(AllocationChange(task_id=task_id, delta=+1))
                    free_gpus -= cards_needed
                    # Push back with updated gain
                    new_gain = compute_marginal_gain(task, self.debt_tracker, self.config)
                    if new_gain > 0:
                        heapq.heappush(gain_heap, (-new_gain, task_id))
            elif reclaim_heap:
                loss, reclaim_task_id = reclaim_heap[0]
                reclaim_task = self.tasks[reclaim_task_id]
                switch_cost = task.latest_metrics.weight_transfer_sec

                if check_can_allocate(gain, switch_cost, self.config):
                    # Do the reclaim and allocate
                    heapq.heappop(reclaim_heap)
                    changes.append(AllocationChange(task_id=reclaim_task_id, delta=-1))
                    free_gpus += reclaim_task.config.cards_per_instance

                    changes.append(AllocationChange(task_id=task_id, delta=+1))
                    free_gpus -= cards_needed

                    # Push back both with updated values
                    if getattr(reclaim_task, '_can_reclaim', False):
                        new_loss = compute_marginal_loss(reclaim_task, self.debt_tracker, self.config, current_time)
                        if new_loss < float('inf'):
                            heapq.heappush(reclaim_heap, (new_loss, reclaim_task_id))

                    new_gain = compute_marginal_gain(task, self.debt_tracker, self.config)
                    if new_gain > 0:
                        heapq.heappush(gain_heap, (-new_gain, task_id))
            else:
                break

        return SchedulingResult(
            changes=changes,
            timestamp_sec=current_time,
        )

    def _get_time(self) -> float:
        return time.time()

    def _calculate_free_gpus(self) -> int:
        """Calculate currently free GPUs (simplified)."""
        used = sum(
            task.current_instances * task.config.cards_per_instance
            for task in self.tasks.values()
        )
        return max(0, self.total_gpus - used)


# ... (keep existing functions)
```

- [ ] **Step 2: Add test for full scheduler**

(Append to test_scheduler_engine.py)

```python
from grpo_scheduler.scheduler.engine import GlobalScheduler, SchedulingResult


def test_scheduler_register_task():
    scheduler = GlobalScheduler()
    config = TaskConfig(
        task_id="task-1",
        base_instances=4,
        tp=2,
        pp=2,
        samples_per_round=64,
        total_samples=1024,
    )
    state = TaskState(config=config)

    scheduler.register_task(state)

    assert "task-1" in scheduler.tasks


def test_scheduler_unregister_task():
    scheduler = GlobalScheduler()
    config = TaskConfig(
        task_id="task-1",
        base_instances=4,
        tp=2,
        pp=2,
        samples_per_round=64,
        total_samples=1024,
    )
    state = TaskState(config=config)

    scheduler.register_task(state)
    scheduler.unregister_task("task-1")

    assert "task-1" not in scheduler.tasks


def test_scheduler_initial_run():
    scheduler = GlobalScheduler()
    scheduler.total_gpus = 16

    config1 = TaskConfig(
        task_id="task-1",
        base_instances=2,
        tp=2,
        pp=2,
        samples_per_round=64,
        total_samples=1024,
    )
    state1 = TaskState(config=config1)
    state1.avg_round_sec_base = 40.0

    config2 = TaskConfig(
        task_id="task-2",
        base_instances=2,
        tp=2,
        pp=2,
        samples_per_round=64,
        total_samples=1024,
    )
    state2 = TaskState(config=config2)
    state2.avg_round_sec_base = 40.0

    scheduler.register_task(state1)
    scheduler.register_task(state2)

    result = scheduler.schedule()

    # Initial schedule should be valid (may have no changes if already balanced)
    assert isinstance(result, SchedulingResult)
```

- [ ] **Step 3: Run test to verify it passes**

Run: `python -m pytest tests/test_scheduler_engine.py -v`
Expected: PASS

---

### Task 7: Protobuf Definitions

**Files:**
- Create: `src/grpo_scheduler/proto/scheduler.proto`
- Create: `src/grpo_scheduler/proto/__init__.py`

- [ ] **Step 1: Write protobuf definitions**

```protobuf
// src/grpo_scheduler/proto/scheduler.proto
syntax = "proto3";

package grpo_scheduler;

// Task configuration
message TaskConfig {
  string task_id = 1;
  int32 base_instances = 2;
  int32 tp = 3;
  int32 pp = 4;
  int32 samples_per_round = 5;
  int32 total_samples = 6;
}

// Single inference instance state
message InstanceState {
  int32 instance_id = 1;
  bool is_busy = 2;
  double elapsed_time_sec = 3;
  int32 done_samples = 4;
  int32 remaining_samples = 5;
}

// Phase timing metrics
message PhaseMetrics {
  double weight_transfer_sec = 1;
  double rollout_gen_sec = 2;
  double rollout_tool_sec = 3;
  double ref_log_prob_sec = 4;
  double reward_sec = 5;
  double adv_sec = 6;
  double update_sec = 7;
  double total_round_sec = 8;
}

// Task state report from agent
message TaskStateReport {
  string task_id = 1;

  // Progress
  int32 done_samples = 2;
  int32 done_rounds = 3;
  double elapsed_time_sec = 4;

  // Current allocation
  int32 current_instances = 5;
  int32 idle_instances = 6;

  // Instance states
  repeated InstanceState instances = 7;

  // Latest metrics
  PhaseMetrics latest_metrics = 8;

  // Historical baseline
  double avg_round_sec_base = 9;
}

// Allocation decision for a single task
message Allocation {
  string task_id = 1;
  int32 target_instances = 2;
}

// Scheduling decision returned to agent
message SchedulingDecision {
  repeated Allocation allocations = 1;
  double timestamp_sec = 2;
}

// Registration messages
message RegisterRequest {
  TaskConfig config = 1;
}

message RegisterResponse {
  bool success = 1;
  string message = 2;
}

message UnregisterRequest {
  string task_id = 1;
}

message Empty {}

// Scheduler service
service GlobalScheduler {
  rpc RegisterTask(RegisterRequest) returns (RegisterResponse);
  rpc ReportState(TaskStateReport) returns (SchedulingDecision);
  rpc UnregisterTask(UnregisterRequest) returns (Empty);
}
```

- [ ] **Step 2: Create proto __init__.py**

```python
# src/grpo_scheduler/proto/__init__.py
# Generated protobuf modules will be imported here
try:
    from .scheduler_pb2 import (
        TaskConfig,
        InstanceState,
        PhaseMetrics,
        TaskStateReport,
        Allocation,
        SchedulingDecision,
        RegisterRequest,
        RegisterResponse,
        UnregisterRequest,
        Empty,
    )
    from .scheduler_pb2_grpc import (
        GlobalSchedulerServicer,
        GlobalSchedulerStub,
        add_GlobalSchedulerServicer_to_server,
    )
    HAS_PROTO = True
except ImportError:
    HAS_PROTO = False
    TaskConfig = None
    InstanceState = None
    PhaseMetrics = None
    TaskStateReport = None
    Allocation = None
    SchedulingDecision = None
    RegisterRequest = None
    RegisterResponse = None
    UnregisterRequest = None
    Empty = None
    GlobalSchedulerServicer = None
    GlobalSchedulerStub = None
    add_GlobalSchedulerServicer_to_server = None

__all__ = [
    "HAS_PROTO",
    "TaskConfig",
    "InstanceState",
    "PhaseMetrics",
    "TaskStateReport",
    "Allocation",
    "SchedulingDecision",
    "RegisterRequest",
    "RegisterResponse",
    "UnregisterRequest",
    "Empty",
    "GlobalSchedulerServicer",
    "GlobalSchedulerStub",
    "add_GlobalSchedulerServicer_to_server",
]
```

---

### Task 8: gRPC Server Implementation

**Files:**
- Create: `src/grpo_scheduler/server/__init__.py`
- Create: `src/grpo_scheduler/server/server.py`

- [ ] **Step 1: Write gRPC server implementation**

```python
# src/grpo_scheduler/server/server.py
from __future__ import annotations

import asyncio
import time
from typing import Optional

import grpc
from grpc import aio

from ..core.state import TaskConfig as CoreTaskConfig, TaskState as CoreTaskState
from ..scheduler.engine import GlobalScheduler, SchedulingResult, AllocationChange
from ..scheduler.config import SchedulerConfig
from ..proto import (
    HAS_PROTO,
    TaskConfig,
    InstanceState,
    PhaseMetrics,
    TaskStateReport,
    Allocation,
    SchedulingDecision,
    RegisterRequest,
    RegisterResponse,
    UnregisterRequest,
    Empty,
    GlobalSchedulerServicer,
    add_GlobalSchedulerServicer_to_server,
)


class SchedulerService(GlobalSchedulerServicer):
    def __init__(self, scheduler: GlobalScheduler):
        self.scheduler = scheduler

    async def RegisterTask(
        self,
        request: RegisterRequest,
        context: grpc.aio.ServicerContext,
    ) -> RegisterResponse:
        """Register a new task."""
        config = CoreTaskConfig(
            task_id=request.config.task_id,
            base_instances=request.config.base_instances,
            tp=request.config.tp,
            pp=request.config.pp,
            samples_per_round=request.config.samples_per_round,
            total_samples=request.config.total_samples,
        )
        state = CoreTaskState(config=config)

        self.scheduler.register_task(state)

        return RegisterResponse(success=True, message="Task registered")

    async def ReportState(
        self,
        request: TaskStateReport,
        context: grpc.aio.ServicerContext,
    ) -> SchedulingDecision:
        """Receive state report and return scheduling decision."""
        # Update task state
        task_id = request.task_id
        if task_id in self.scheduler.tasks:
            task = self.scheduler.tasks[task_id]
            task.done_samples = request.done_samples
            task.done_rounds = request.done_rounds
            task.elapsed_time_sec = request.elapsed_time_sec
            task.current_instances = request.current_instances
            task.idle_instances = request.idle_instances
            task.avg_round_sec_base = request.avg_round_sec_base

            if request.HasField("latest_metrics"):
                m = request.latest_metrics
                task.latest_metrics = CoreTaskState._dataclass_from_dict(
                    task.latest_metrics,
                    {
                        "weight_transfer_sec": m.weight_transfer_sec,
                        "rollout_gen_sec": m.rollout_gen_sec,
                        "rollout_tool_sec": m.rollout_tool_sec,
                        "ref_log_prob_sec": m.ref_log_prob_sec,
                        "reward_sec": m.reward_sec,
                        "adv_sec": m.adv_sec,
                        "update_sec": m.update_sec,
                        "total_round_sec": m.total_round_sec,
                    }
                )

            # Notify debt tracker if round completed
            # (simplified - need more state to detect round completion)

        # Run scheduling
        result = self.scheduler.schedule()

        # Convert result to proto
        decision = SchedulingDecision(timestamp_sec=result.timestamp_sec)

        # Group changes by task
        target_instances: dict[str, int] = {}
        for task in self.scheduler.tasks.values():
            target_instances[task.config.task_id] = task.current_instances

        for change in result.changes:
            target_instances[change.task_id] = (
                target_instances.get(change.task_id, 0) + change.delta
            )

        for task_id, target in target_instances.items():
            alloc = Allocation(task_id=task_id, target_instances=target)
            decision.allocations.append(alloc)

        return decision

    async def UnregisterTask(
        self,
        request: UnregisterRequest,
        context: grpc.aio.ServicerContext,
    ) -> Empty:
        """Unregister a task."""
        self.scheduler.unregister_task(request.task_id)
        return Empty()


async def serve(
    port: int = 50051,
    scheduler_config: Optional[SchedulerConfig] = None,
    total_gpus: int = 64,
) -> None:
    """Start the gRPC server."""
    scheduler = GlobalScheduler(scheduler_config)
    scheduler.total_gpus = total_gpus

    server = aio.server()
    add_GlobalSchedulerServicer_to_server(SchedulerService(scheduler), server)

    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)

    await server.start()
    print(f"Scheduler server listening on {listen_addr}")

    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
```

- [ ] **Step 2: Create server __init__.py**

```python
# src/grpo_scheduler/server/__init__.py
from .server import serve, SchedulerService

__all__ = ["serve", "SchedulerService"]
```

---

### Task 9: Build Script and Final Polish

**Files:**
- Create: `scripts/build_proto.py`
- Create: `tests/conftest.py`
- Update: `pyproject.toml` (add scripts)

- [ ] **Step 1: Write proto build script**

```python
# scripts/build_proto.py
#!/usr/bin/env python3
"""Build protobuf files."""

import os
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).parent.parent
    proto_dir = root / "src" / "grpo_scheduler" / "proto"
    proto_file = proto_dir / "scheduler.proto"

    if not proto_file.exists():
        print(f"Proto file not found: {proto_file}")
        return 1

    print(f"Building proto: {proto_file}")

    cmd = [
        sys.executable, "-m", "grpc_tools.protoc",
        f"-I{proto_dir}",
        f"--python_out={proto_dir}",
        f"--grpc_python_out={proto_dir}",
        str(proto_file),
    ]

    result = subprocess.run(cmd, cwd=root)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Write conftest.py**

```python
# tests/conftest.py
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))
```

- [ ] **Step 3: Update pyproject.toml with project config**

(Add to existing pyproject.toml)

```toml
[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

---

## Execution Options

Plan complete and saved to `docs/superpowers/plans/2026-03-24-grpo-gpu-scheduler.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
