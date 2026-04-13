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
    current_sample_index: int = 0
    sample_start_index: int = 0  # 该实例开始处理的样本索引（用于动态分配实例时从当前进度继续）

    # 跨节点通信建模
    placement_nodes: Set[int] = field(default_factory=set)  # 实例跨越的节点ID集合
    communication_factor: float = 1.0  # 通信因子：1.0=同节点，>1.0=跨节点有通信开销

    # 冷启动标记：用于跟踪是否已应用冷启动时间
    has_applied_cold_start: bool = False

    def precompute_inference_times(
        self,
        total_samples: int,
        tp: int,
        pp: int,
        time_distribution: str = "longtail_normal",
        distribution_params: Dict[str, Any] = None,
        random_seed: int = 42,
        sample_start_index: int = 0
    ) -> None:
        """
        预计算所有样本的推理时间

        推理时间 = 基础时间(与TP*PP相关) * 速度因子

        Args:
            total_samples: 总样本数
            tp: tensor parallelism
            pp: pipeline parallelism
            time_distribution: 时间分布类型 (longtail_normal, lognormal, exponential)
            distribution_params: 分布参数
            random_seed: 随机种子（任务的原始种子，不包含instance_id偏移）
            sample_start_index: 该实例开始处理的样本索引（用于动态分配实例）
        """
        if distribution_params is None:
            distribution_params = {}

        # 设置 cards_per_instance
        self.cards_per_instance = tp * pp
        self.sample_start_index = sample_start_index

        # 基础时间与卡数相关：8卡为基准100秒
        # 调整为几十秒到几百秒范围
        cards_per_instance = tp * pp
        base_time = 100.0 * (8.0 / cards_per_instance) * self.communication_factor

        # 获取默认参数（长尾效应更严重）
        params = {
            "slow_ratio": 0.5,  # 50%的样本是长尾
            "slow_min": 0.8,   # 最慢0.8倍
            "slow_max": 5.0,   # 最慢5倍
        }
        params.update(distribution_params)

        # 预计算每个样本的推理时间
        # 使用 sample_start_index 偏移，确保不同实例处理不同样本时使用正确的种子
        self.inference_time_table.clear()
        for i in range(total_samples):
            # 种子 = 任务种子 + 样本全局索引
            global_sample_index = sample_start_index + i
            speed_factor = self._generate_speed_factor(
                time_distribution, params, random_seed + global_sample_index
            )
            self.inference_time_table.append(base_time * speed_factor)

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
            速度因子
        """
        rng = random.Random(seed)

        if dist_type == "longtail_normal":
            # 长尾正态分布：大部分正常，少部分很慢
            if rng.random() < params.get("slow_ratio", 0.3):
                return rng.uniform(params.get("slow_min", 0.2), params.get("slow_max", 0.5))
            return 1.0

        elif dist_type == "lognormal":
            # 对数正态分布：自然产生长尾
            sigma = params.get("sigma", 0.8)
            return rng.lognormvariate(0, sigma)

        elif dist_type == "exponential":
            # 指数分布
            lam = params.get("lambda", 2.0)
            return min(rng.expovariate(lam), 2.0)  # 上限2倍

        return 1.0

    def get_current_inference_time(self) -> Optional[float]:
        """获取当前样本的预计算推理时间（不推进索引）"""
        if self.current_sample_index < len(self.inference_time_table):
            return self.inference_time_table[self.current_sample_index]
        return None

    def advance_to_next_sample(self) -> None:
        """推进到下一个样本（当前样本完成后调用）"""
        self.current_sample_index += 1

    def get_next_inference_time(self) -> Optional[float]:
        """获取下一个样本的预计算推理时间（向后兼容接口）"""
        if self.current_sample_index < len(self.inference_time_table):
            time = self.inference_time_table[self.current_sample_index]
            self.current_sample_index += 1
            return time
        return None
