"""Instance模块 - 任务实例（DP切片）"""

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Set
from enum import Enum


class InstanceState(Enum):
    """实例状态"""
    INIT = 0
    COLD_STARTING = 1  # 冷启动中（分配后~60s）
    IDLE = 2  # 空闲（没有样本可处理）
    BUSY = 3  # 忙（正在推理）


class GPUPlacement:
    """GPU位置信息"""
    def __init__(self, machine_id: int, gpu_id: int):
        self.machine_id = machine_id
        self.gpu_id = gpu_id

    def __eq__(self, other):
        if not isinstance(other, GPUPlacement):
            return False
        return self.machine_id == other.machine_id and self.gpu_id == other.gpu_id

    def __hash__(self):
        return hash((self.machine_id, self.gpu_id))

    def __repr__(self):
        return f"GPUPlacement(machine_id={self.machine_id}, gpu_id={self.gpu_id})"


@dataclass
class Instance:
    """任务实例（DP切片）"""
    instance_id: int
    gpus: List[GPUPlacement]  # 分配的GPU
    state: InstanceState = InstanceState.INIT
    speed_factor: float = 1.0  # 速度因子（基于固定种子）
    current_sample_start_time: Optional[float] = None
    samples_processed: int = 0
    cold_start_end_time: Optional[float] = None

    # 预计算的推理时间表
    cards_per_instance: int = field(init=False, default=0)  # tp * pp（固定）
    inference_time_table: List[float] = field(default_factory=list)
    current_global_index: int = 0  # 当前处理的全局样本索引

    # 跨节点通信建模
    placement_nodes: Set[int] = field(default_factory=set)  # 实例跨越的节点ID集合
    communication_factor: float = 1.0  # 通信因子：1.0=同节点，>1.0=跨节点有通信开销

    # 冷启动标记：用于跟踪是否已应用冷启动时间
    has_applied_cold_start: bool = False

    # 用于生成推理时间的参数（动态生成时需要）
    _time_distribution: str = field(default="longtail_normal")
    _distribution_params: Dict[str, Any] = field(default_factory=dict)
    _random_seed: int = field(default=42)
    _base_time: float = field(default=100.0)

    def precompute_inference_times(
        self,
        total_samples: int,
        tp: int,
        pp: int,
        time_distribution: str = "longtail_normal",
        distribution_params: Dict[str, Any] = None,
        random_seed: int = 42
    ) -> None:
        """
        预计算所有样本的推理时间表

        【公平性设计】推理时间只由全局样本索引决定：
        - seed = random_seed + global_sample_index
        - 同一个 sample 在任何实例上都有相同的推理时间
        - 确保无GS和有GS方案公平比较

        长尾效应体现在样本层面：
        - 不同 sample 有不同的推理时间（由分布参数决定）
        - 实例完成时间取决于它处理了哪些样本
        - 例如：某实例处理了多个慢样本，会比其他实例慢完成

        Args:
            total_samples: 总样本数
            tp: tensor parallelism
            pp: pipeline parallelism
            time_distribution: 时间分布类型
            distribution_params: 分布参数
            random_seed: 随机种子（任务的原始种子）
        """
        if distribution_params is None:
            distribution_params = {}

        # 保存参数用于动态生成
        self._time_distribution = time_distribution
        self._distribution_params = distribution_params
        self._random_seed = random_seed

        # 设置 cards_per_instance
        self.cards_per_instance = tp * pp

        # 基础时间与卡数相关：8卡为基准100秒
        base_time = 100.0 * (8.0 / (tp * pp)) * self.communication_factor
        self._base_time = base_time

        # 获取默认参数
        params = {
            "slow_ratio": 0.5,
            "slow_min": 0.8,
            "slow_max": 5.0,
        }
        params.update(distribution_params)

        # 预计算推理时间表（只依赖全局样本索引）
        self.inference_time_table.clear()

        for global_index in range(total_samples):
            # 种子只依赖任务种子和样本索引，不依赖实例ID
            # 这样同一个sample在任何实例上都有相同的推理时间
            seed = random_seed + global_index
            speed_factor = self._generate_speed_factor(time_distribution, params, seed)
            self.inference_time_table.append(base_time * speed_factor)

    def get_inference_time_for_sample(self, global_index: int) -> float:
        """
        获取指定全局样本索引的推理时间

        【公平性】同一个 sample 的推理时间在任何实例上都相同

        Args:
            global_index: 全局样本索引（0 ~ total_samples-1）

        Returns:
            推理时间（秒）
        """
        # 如果在预计算表中，直接返回
        if global_index < len(self.inference_time_table):
            return self.inference_time_table[global_index]

        # 如果超出预计算表，动态生成（用于动态扩容）
        # 使用相同的种子公式确保公平性
        seed = self._random_seed + global_index
        speed_factor = self._generate_speed_factor(
            self._time_distribution, self._distribution_params, seed
        )
        return self._base_time * speed_factor

    def _generate_speed_factor(
        self,
        dist_type: str,
        params: Dict[str, Any],
        seed: int
    ) -> float:
        """
        根据分布类型生成速度因子，体现长尾特性

        Args:
            dist_type: 分布类型
            params: 分布参数
            seed: 随机种子

        Returns:
            速度因子（时间倍数）
        """
        rng = random.Random(seed)

        if dist_type == "longtail_normal":
            # 长尾正态分布：大部分正常，少部分很慢
            if rng.random() < params.get("slow_ratio", 0.5):
                return rng.uniform(params.get("slow_min", 0.8), params.get("slow_max", 5.0))
            return 1.0

        elif dist_type == "lognormal":
            # 对数正态分布：自然产生长尾
            sigma = params.get("sigma", 0.8)
            return rng.lognormvariate(0, sigma)

        elif dist_type == "exponential":
            # 指数分布
            lam = params.get("lambda", 2.0)
            return min(rng.expovariate(lam), 2.0)

        return 1.0

    def get_current_inference_time(self) -> Optional[float]:
        """获取当前处理样本的推理时间（向后兼容）"""
        return self.get_inference_time_for_sample(self.current_global_index)
