"""TestCase模块 - 测试用例"""

from dataclasses import dataclass
from typing import List

from .cluster_config import ClusterConfig
from .task_config import TaskConfig


@dataclass
class TestCase:
    """测试用例"""
    name: str
    description: str
    cluster: ClusterConfig
    tasks: List[TaskConfig]

    def validate_initial_constraints(self) -> bool:
        """验证初始约束：
        1. 所有任务base_instances不超过集群容量
        2. 不同任务有不同的tp/pp组合
        """
        total_cluster_gpus = self.cluster.total_gpus()
        total_required = sum(t.total_cards() for t in self.tasks)

        if total_required > total_cluster_gpus:
            raise ValueError(
                f"初始分配超过集群容量："
                f"集群{total_cluster_gpus}张卡，任务需要{total_required}张卡"
            )

        # 检查任务ID唯一性
        task_ids = [t.task_id for t in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("任务ID必须唯一")

        return True
