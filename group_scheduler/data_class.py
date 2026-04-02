from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from worker import WorkerInfo
else:
    from worker import WorkerInfo
@dataclass
class ReclaimConfirm:
    task_id: str
    reclaimed_instances: int
    reclaimed_workers: list[WorkerInfo]

@dataclass
class TaskAllocation:
    task_id: str
    instance_delta: int  # 正数=增加，负数=回收，0=不变
    recommended_gpus: list[WorkerInfo]  # 只有分配时会填，回收时不会填

@dataclass
class SchedulingDecision:
    allocations: list[TaskAllocation]
    timestamp_sec: float

@dataclass
class TaskConfig:
    task_id: str
    base_instances: int
    tp: int
    pp: int
    samples_per_round: int
    total_samples: int

@dataclass
class TaskStateReport:
    task_id: str

    # 进度
    done_samples: int
    done_rounds: int
    elapsed_time_sec: float
    remaining_samples: int

    # 当前分配
    current_instances: int
    idle_instances: int
    busy_instances: int

    # 阶段
    in_rollout_phase: bool

    voluntary_reclaim: Optional[ReclaimConfirm] = None