from dataclasses import dataclass
from typing import Any, Optional
import math
from data_class import ReclaimConfirm, TaskConfig, TaskStateReport
class TaskInfo:
    """作业信息类，用于存储作业的基本信息"""
    def __init__(self, config: TaskConfig):
        # 基本属性
        self.task_id: str = config.task_id
        self.base_instances: int =  config.base_instances
        self.tp: int = config.tp
        self.pp: int = config.pp
        self.samples_per_round: int = config.samples_per_round
        self.total_samples: int = config.total_samples
        self._used_workers: list[Any] = []

        # state
        self._has_state: bool = False

        # 进度
        self.done_samples: Optional[int] = None
        self.done_rounds: Optional[int] = None
        self.elapsed_time_sec: Optional[float] = None
        self.remaining_samples: Optional[int] = None

        # 当前分配
        self.current_instances: Optional[int] = 0
        self.idle_instances: Optional[int] = 0
        self.busy_instances: Optional[int] = 0

        # 阶段
        self.in_rollout_phase: Optional[bool] = True

    def update_from_report(self, state: TaskStateReport):
        self._has_state = True
        self.done_samples = state.done_samples
        self.done_rounds = state.done_rounds
        self.elapsed_time_sec = state.elapsed_time_sec
        self.remaining_samples = state.remaining_samples
        self.current_instances = state.current_instances
        self.idle_instances = state.idle_instances
        self.busy_instances = state.busy_instances
        self.in_rollout_phase = state.in_rollout_phase
    
    @property
    def has_state(self) -> bool:
        return self._has_state

    @property
    def used_workers(self) -> list:
        """返回当前已分配的worker列表"""
        return self._used_workers

    @property
    def num_used_worker(self) -> int:
        """返回当前已分配的worker数量"""
        return len(self._used_workers)

class TaskTable:
    """作业表类，用于管理所有作业的信息"""
    def __init__(self):
        self._task_table = {}  # 存储所有作业信息
    
    def check_task_exist(self, task_id):
        return task_id in self._task_table
    

    def register(self, config: TaskConfig) -> bool:
        if self.check_task_exist(config.task_id):
            return False
        self._task_table[config.task_id] = TaskInfo(config)
        return True
    
    def get_task(self, task_id):
        return self._task_table.get(task_id, None)
    
    def get_all_tasks(self) -> list[TaskInfo]:
        return list(self._task_table.values())

    def update_task_info(self, info: TaskStateReport) -> bool:
        if not self.check_task_exist(info.task_id):
            return False
        self._task_table[info.task_id].update_from_report(info)
        return True
    
    def del_workers_from_used(self, task_id, workers):
        """从任务的已使用worker列表中删除指定workers
        Args:
            task_id: 任务ID
            workers: 要删除的worker列表
        """
        task = self.get_task(task_id)
        if task is None:
            # warning
            return False

        workers_set = set(workers)
        # 使用列表推导式过滤保留未被删除的workers
        task._used_workers = [w for w in task._used_workers if w not in workers_set]
        return True

    def add_workers_to_used(self, task_id, workers):
        """将workers添加到任务的已使用worker列表
        Args:
            task_id: 任务ID
            workers: 要添加的worker列表
        Returns:
            bool: 是否成功
        """
        task = self.get_task(task_id)
        if task is None:
            return False
        task._used_workers.extend(workers)
        return True

    def assess_range(self):
        pass

    def compute_allocation_score(self, task_id, plan_num_instance):
        task = self.get_task(task_id)
        if not task.has_state:
            return 0

        # 在训练阶段，给实例没用
        if not task.in_rollout_phase:
            return 0

        # 没有剩余样本了
        if task.remaining_samples <= 0:
            return 0

        # 基准实例数为0时直接返回0（避免除0错误）
        if task.base_instances <= 0:
            return 0

        weight_instances = 40
        weight_smaples = 60

        score = 0
        # 假设当前实例数（包含计划实例）
        assumed_num_instance = plan_num_instance + task.current_instances
        # 信号1&2合并：指数实例平衡信号
        deficit = task.base_instances - assumed_num_instance
        score += weight_instances * math.exp(deficit / task.base_instances)

        # 信号3：改进的样本充足度
        if task.busy_instances > 0:
            # 计算剩余样本/忙实例比率
            remaining_ratio = task.remaining_samples / task.busy_instances
        else:
            remaining_ratio = task.remaining_samples * 5

        # 计算总样本/基准实例比率
        total_ratio = task.total_samples / task.base_instances

        # 样本充足度基于比率比较（若total_ratio=0 -> total_samples=0 -> remaining_samples=0, 则会提前return）
        sample_sufficiency = remaining_ratio / total_ratio
        score += weight_smaples * math.exp(sample_sufficiency - 1.0)

        return score