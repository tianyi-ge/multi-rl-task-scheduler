"""Cluster模块 - 集群模型和机器模型"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
import sys

# 尝试导入真实GS的WorkerTable
try:
    sys.path.insert(0, '/home/hhc/rl_wp/multi-rl-task-scheduler/gs-code')
    from worker import WorkerInfo, WorkerTable
    REAL_GS_AVAILABLE = True
except Exception:
    REAL_GS_AVAILABLE = False
    WorkerTable = None


@dataclass
class Machine:
    """单个机器模型"""
    machine_id: int
    gpu_count: int
    gpu_states: List[int] = field(default_factory=list)  # 0=空闲, 1=已分配

    def __post_init__(self):
        if not self.gpu_states:
            self.gpu_states = [0] * self.gpu_count


class ClusterModel:
    """集群模型，维护GPU资源池"""
    def __init__(self, machines: List[Machine]):
        self.machines = machines
        self.free_gpus: List[GPUPlacement] = []
        self._init_free_gpus()

        # 真实GS的WorkerTable（真实调度器模式）
        self._worker_table: Optional[WorkerTable] = None

    @classmethod
    def from_config(cls, machine_count: int, gpus_per_machine: int, enable_real_scheduler: bool = False):
        """从配置创建集群"""
        machines = []
        for i in range(machine_count):
            machines.append(Machine(
                machine_id=i,
                gpu_count=gpus_per_machine
            ))
        cluster = cls(machines=machines)

        # 如果启用真实调度器，创建WorkerTable并初始化Worker
        if enable_real_scheduler and REAL_GS_AVAILABLE:
            cluster._init_worker_table(machine_count, gpus_per_machine)

        return cluster

    def _init_worker_table(self, machine_count: int, gpus_per_machine: int) -> None:
        """初始化真实GS的WorkerTable"""
        if not REAL_GS_AVAILABLE:
            return

        self._worker_table = WorkerTable()

        num_workers = machine_count * gpus_per_machine
        for worker_id in range(num_workers):
            local_rank = worker_id % gpus_per_machine
            node_id = worker_id // gpus_per_machine
            node_type = "shared"

            worker = WorkerInfo(node_type, node_id, local_rank)
            worker.set_id(str(worker_id))
            self._worker_table.register(worker)

    def get_worker_table(self) -> Optional[WorkerTable]:
        """获取真实GS的WorkerTable"""
        return self._worker_table

    def _init_free_gpus(self) -> None:
        """初始化空闲GPU列表"""
        for machine in self.machines:
            for gpu_id in range(machine.gpu_count):
                from .instance import GPUPlacement
                self.free_gpus.append(GPUPlacement(machine.machine_id, gpu_id))

    def allocate_instance(self, tp: int, pp: int) -> Optional[List]:
        """
        分配一个实例的GPU（tp*pp张卡）

        优先级：
        1. 同机分配（拓扑感知）
        2. 跨机分配

        返回：GPU列表（成功）或None（失败）
        """
        cards_needed = tp * pp
        result = self._try_same_machine(cards_needed)
        if result is not None:
            return result
        return self._try_cross_machine(cards_needed)

    def _try_same_machine(self, cards_needed: int) -> Optional[List]:
        """尝试同机分配"""
        from .instance import GPUPlacement
        for machine in self.machines:
            available_gpus = []
            for gpu_id in range(machine.gpu_count):
                if machine.gpu_states[gpu_id] == 0:
                    available_gpus.append(GPUPlacement(machine.machine_id, gpu_id))
            if len(available_gpus) >= cards_needed:
                selected = available_gpus[:cards_needed]
                for gpu in selected:
                    machine.gpu_states[gpu.gpu_id] = 1
                    self.free_gpus.remove(gpu)
                return selected
        return None

    def _try_cross_machine(self, cards_needed: int) -> Optional[List]:
        """尝试跨机分配"""
        if len(self.free_gpus) < cards_needed:
            return None
        selected = self.free_gpus[:cards_needed]
        for gpu in selected:
            self.machines[gpu.machine_id].gpu_states[gpu.gpu_id] = 1
        self.free_gpus = self.free_gpus[cards_needed:]
        return selected

    def reclaim_gpus(self, gpus: List) -> None:
        """回收GPU"""
        for gpu in gpus:
            self.machines[gpu.machine_id].gpu_states[gpu.gpu_id] = 0
            self.free_gpus.append(gpu)

    def get_utilization(self) -> float:
        """计算GPU利用率"""
        total = sum(m.gpu_count for m in self.machines)
        used = sum(sum(m.gpu_states) for m in self.machines)
        return used / total if total > 0 else 0.0

    def total_gpus(self) -> int:
        """获取总GPU数"""
        return sum(m.gpu_count for m in self.machines)

    def get_available_gpus(self) -> int:
        """获取可用GPU数量"""
        return len(self.free_gpus)

    def get_machine_count(self) -> int:
        """获取机器数量"""
        return len(self.machines)
