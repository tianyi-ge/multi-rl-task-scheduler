"""TaskConfig模块 - 任务配置"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any


@dataclass
class TaskConfig:
    """任务配置"""
    task_id: str
    tp: int
    pp: int
    base_instances: int = 0
    samples_per_round: int = 16
    num_rounds: int = 4  # 新增：迭代轮数

    # total_samples 自动计算（init=False）
    total_samples: int = field(init=False)

    # 时间分布参数（用于预计算推理时间）
    time_distribution: str = "longtail_normal"  # longtail_normal, lognormal, exponential
    distribution_params: Dict[str, Any] = field(default_factory=dict)

    # 兼容旧参数
    long_tail_ratio: Optional[float] = None
    slow_factor_range: Optional[Tuple[float, float]] = None

    def __post_init__(self):
        """自动计算总样本数"""
        self.total_samples = self.samples_per_round * self.num_rounds

    def cards_per_instance(self) -> int:
        """获取每个实例的GPU卡数"""
        return self.tp * self.pp

    def total_cards(self) -> int:
        """获取任务总GPU卡数"""
        return self.base_instances * self.cards_per_instance()
