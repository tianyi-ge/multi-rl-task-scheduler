from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class WorkerInfo:
    worker_id: str
    gpu_id: int
    machine_id: int


@dataclass
class TaskConfig:
    task_id: str
    base_instances: int
    tp: int
    pp: int
    samples_per_round: int
    total_samples: int

    @property
    def workers_per_instance(self) -> int:
        return self.tp * self.pp


@dataclass
class TaskStateReport:
    task_id: str
    state_version: int
    done_samples: int
    done_rounds: int
    elapsed_time_sec: float
    remaining_samples: int
    current_instances: int
    idle_instances: int
    busy_instances: int
    in_rollout_phase: bool
    assigned_workers: List[WorkerInfo] = field(default_factory=list)
    idle_worker_ids: List[str] = field(default_factory=list)

    @property
    def has_state(self) -> bool:
        return True


@dataclass
class TaskAllocation:
    task_id: str
    instance_delta: int
    recommended_workers: List[WorkerInfo] = field(default_factory=list)


@dataclass
class SchedulingDecision:
    allocations: List[TaskAllocation]
    timestamp_sec: float


@dataclass
class SchedulerTuning:
    catch_up_ratio: float = 1.2
    acceleration_limit_ratio: float = 1.5
    max_consecutive_reclaims: int = 3
    max_free_worker_ratio: float = 0.3


@dataclass
class ManagedTask:
    config: TaskConfig
    state: Optional[TaskStateReport] = None
    assigned_workers: Dict[str, WorkerInfo] = field(default_factory=dict)

    @property
    def task_id(self) -> str:
        return self.config.task_id

    @property
    def has_state(self) -> bool:
        return self.state is not None

    @property
    def base_instances(self) -> int:
        return self.config.base_instances

    @property
    def tp(self) -> int:
        return self.config.tp

    @property
    def pp(self) -> int:
        return self.config.pp

    @property
    def workers_per_instance(self) -> int:
        return self.config.workers_per_instance

    @property
    def total_samples(self) -> int:
        return self.config.total_samples

    @property
    def current_instances(self) -> int:
        return 0 if self.state is None else self.state.current_instances

    @property
    def busy_instances(self) -> int:
        return 0 if self.state is None else self.state.busy_instances

    @property
    def idle_instances(self) -> int:
        return 0 if self.state is None else self.state.idle_instances

    @property
    def remaining_samples(self) -> int:
        return 0 if self.state is None else self.state.remaining_samples

    @property
    def in_rollout_phase(self) -> bool:
        return False if self.state is None else self.state.in_rollout_phase
