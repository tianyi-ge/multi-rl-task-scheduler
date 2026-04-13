"""ClusterConfig模块 - 集群配置"""

from dataclasses import dataclass


@dataclass
class ClusterConfig:
    """集群配置"""
    machine_count: int
    gpus_per_machine: int

    def total_gpus(self) -> int:
        """获取总GPU数"""
        return self.machine_count * self.gpus_per_machine
