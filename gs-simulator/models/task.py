"""Task模块 - 任务模型和样本队列"""

import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum

from .instance import Instance, InstanceState, GPUPlacement


class TaskPhase(Enum):
    """任务阶段"""
    REGISTERED = 0
    ROLLOUT = 1  # 推理阶段
    TRAIN = 2  # 训练阶段
    DONE = 3  # 完成


@dataclass
class SampleQueue:
    """样本队列 - 使用全局索引跟踪进度"""
    total_samples: int  # 总样本数
    samples_per_round: int  # 每轮样本数（用于进度显示）
    round_id: int = 0
    available_samples: int = field(init=False, default=0)  # 本轮剩余可用样本
    locked_samples: int = 0  # 已被实例取走但未完成的样本数
    next_global_index: int = 0  # 下一个要分配的全局样本索引
    returned_indices: List[int] = field(default_factory=list)  # 【新增】放回队列的样本索引列表

    def __post_init__(self):
        # 第一轮的可用样本数
        self.available_samples = min(self.samples_per_round, self.total_samples)

    def try_lock_samples(self, count: int) -> Optional[int]:
        """
        尝试锁定count个样本（实例取样本时调用）

        Returns:
            全局样本索引（成功时），None（失败时）
        """
        # 检查本轮是否还有可用样本（包括放回队列的样本）
        if self.available_samples >= count:
            self.available_samples -= count
            self.locked_samples += count

            # 【优先】检查是否有放回队列的样本索引
            if self.returned_indices:
                # 使用放回的索引（确保正确的推理时间）
                idx = self.returned_indices.pop(0)
                return idx

            # 检查是否还有未分配的全局样本索引
            if self.next_global_index < self.total_samples:
                idx = self.next_global_index
                self.next_global_index += count
                return idx

        return None

    def unlock_samples(self, count: int) -> None:
        """完成count个样本（推理完成后调用）"""
        self.locked_samples -= count

    def return_sample(self, original_index: int) -> None:
        """
        把样本放回队列（BUSY实例被强制回收时调用）

        Args:
            original_index: 原始样本索引（必须保留，确保正确的推理时间）

        当GS需要回收超过base_instances的BUSY实例时，样本正在处理中
        但还没完成，需要放回队列让其他实例处理。

        注意：
        - 保留原始索引，确保推理时间正确（预计算表使用索引）
        - 这与unlock_samples不同：unlock是样本完成了，return是样本被中止
        """
        if self.locked_samples > 0:
            self.locked_samples -= 1
            self.available_samples += 1
            # 【关键】记录放回的索引，后续分配时使用
            self.returned_indices.append(original_index)

    def check_round_complete(self, done_samples: int) -> bool:
        """
        检查当前轮是否完成

        Returns:
            True 如果本轮所有样本都已完成，False 否则
        """
        # 本轮完成条件：所有取走的样本都已完成，且本轮可用样本也已耗尽
        return (self.locked_samples == 0 and self.available_samples == 0)

    def next_round(self) -> bool:
        """
        进入下一轮

        Returns:
            True 如果还有更多轮次，False 如果所有样本已分配完毕
        """
        self.round_id += 1
        # 计算下一轮的可用样本数
        remaining_total = self.total_samples - self.next_global_index
        self.available_samples = min(self.samples_per_round, remaining_total)
        self.locked_samples = 0
        return remaining_total > 0


@dataclass
class TaskStateReport:
    """任务状态上报"""
    task_id: str
    done_samples: int
    done_rounds: int
    elapsed_time_sec: float
    remaining_samples: int
    current_instances: int
    idle_instances: int
    busy_instances: int
    in_rollout_phase: bool


class ClusterModel:
    """集群模型（占位，避免循环导入）"""
    pass


@dataclass
class TaskModel:
    """任务模型，模拟RL推理任务"""
    task_id: str
    tp: int
    pp: int
    base_instances: int
    samples_per_round: int
    num_rounds: int  # 新增：迭代轮数
    random_seed: int

    # total_samples 自动计算
    total_samples: int = field(init=False)

    # 时间分布参数
    time_distribution: str = "longtail_normal"
    distribution_params: Dict[str, Any] = field(default_factory=dict)

    # 状态
    phase: TaskPhase = TaskPhase.REGISTERED
    instances: List[Instance] = field(default_factory=list)
    sample_queue: SampleQueue = field(init=False, default=None)
    done_samples: int = 0
    done_rounds: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    workers_released: bool = False  # 标记worker是否已释放
    reclaimed_workers_cache: List[GPUPlacement] = field(default_factory=list)  # 缓存已回收的worker信息

    def __post_init__(self):
        """自动计算总样本数并初始化样本队列"""
        self.total_samples = self.samples_per_round * self.num_rounds
        self.sample_queue = SampleQueue(
            total_samples=self.total_samples,
            samples_per_round=self.samples_per_round
        )

    def release_idle_instances(self, cluster: ClusterModel) -> List[GPUPlacement]:
        """
        动态释放空闲实例

        【关键约束】保护base_instances：
        1. 任务DONE时：释放所有实例
        2. 长尾场景（有BUSY实例处理慢样本，有IDLE实例完成快样本）：
           - 可以释放超过base_instances的IDLE实例给其他任务
           - 实现资源动态分配和负载均衡
        3. 轮次切换（所有实例都IDLE且无样本）：
           - 保留实例等待下一轮

        Returns:
            被释放的GPU placement列表
        """
        self.reclaimed_workers_cache.clear()

        # 所有轮次完成 → 释放所有实例
        if self.phase == TaskPhase.DONE:
            for inst in self.instances:
                cluster.reclaim_gpus(inst.gpus)
                self.reclaimed_workers_cache.extend(inst.gpus)
            self.instances.clear()
            return self.reclaimed_workers_cache

        # 统计各状态实例数
        idle_instances = [inst for inst in self.instances if inst.state == InstanceState.IDLE]
        busy_instances = [inst for inst in self.instances if inst.state == InstanceState.BUSY]
        cold_starting_instances = [inst for inst in self.instances if inst.state == InstanceState.COLD_STARTING]

        # 如果没有空闲实例，不需要释放
        if not idle_instances:
            return []

        total_instances = len(self.instances)

        # 【长尾场景关键】当有BUSY实例时，可以释放IDLE实例
        if busy_instances or cold_starting_instances:
            # 有实例在忙（处理长尾样本）
            # 可以释放超过base_instances的IDLE实例给其他任务
            excess_instances = total_instances - self.base_instances

            if excess_instances > 0:
                # 只释放超额的IDLE实例（最多excess个）
                for inst in idle_instances[:excess_instances]:
                    cluster.reclaim_gpus(inst.gpus)
                    self.reclaimed_workers_cache.extend(inst.gpus)
                    self.instances.remove(inst)
                return self.reclaimed_workers_cache

            # 没有超额实例，不释放（保护base_instances）
            return []

        # 所有实例都空闲：轮次切换场景
        if not busy_instances and not cold_starting_instances:
            # 检查是否还有下一轮样本
            remaining_samples = self.total_samples - self.done_samples
            if remaining_samples > 0:
                # 还有下一轮 → 保留实例等待下一轮
                return []
            else:
                # 所有样本完成 → 释放所有实例
                for inst in idle_instances:
                    cluster.reclaim_gpus(inst.gpus)
                    self.reclaimed_workers_cache.extend(inst.gpus)
                    self.instances.remove(inst)
                return self.reclaimed_workers_cache

        # 本轮还有样本 → 空闲实例继续处理，不释放
        return []

    def init_instances(self, base_count: int, cluster: ClusterModel) -> None:
        """初始化实例，分配GPU并预计算推理时间

        关键改进：所有实例使用相同的随机种子和sample_start_index=0
        这样无论有多少实例，它们处理的样本时间是一致的（只是并行处理）
        """
        for i in range(base_count):
            gpus = cluster.allocate_instance(self.tp, self.pp)
            if gpus:
                inst = Instance(
                    instance_id=i,
                    gpus=gpus,
                    state=InstanceState.COLD_STARTING
                )

                # 预计算所有样本的推理时间表
                # 【公平性设计】推理时间只由全局样本索引决定
                # 同一个 sample 在任何实例上都有相同的推理时间
                # 确保无GS和有GS方案公平比较
                inst.precompute_inference_times(
                    total_samples=self.total_samples,
                    tp=self.tp,
                    pp=self.pp,
                    time_distribution=self.time_distribution,
                    distribution_params=self.distribution_params,
                    random_seed=self.random_seed
                )

                self.instances.append(inst)

    def step(self, current_time: float) -> None:
        """推进时间，模拟推理执行"""
        for inst in self.instances:
            self._step_instance(inst, current_time)

        # 检查rollout阶段是否完成
        if self.phase == TaskPhase.ROLLOUT:
            if self.done_samples >= self.total_samples:
                self.phase = TaskPhase.DONE
                self.end_time = current_time

    def _step_instance(self, inst: Instance, current_time: float) -> None:
        """推进单个实例的状态 - 使用预计算的推理时间

        关键改进：
        1. 冷启动完成后直接转为BUSY（如果有样本）
        2. 使用全局样本索引确保不同实例处理不同样本
        3. 每个实例的推理时间由全局样本索引决定（体现长尾效应）
        """
        # 冷启动检查：冷启动完成后直接转为 BUSY（如果有样本）
        if inst.state == InstanceState.COLD_STARTING:
            # 冷启动后立即尝试取样本，如果有样本则直接进入BUSY状态
            global_idx = self.sample_queue.try_lock_samples(1)
            if global_idx is not None:
                inst.state = InstanceState.BUSY
                inst.current_sample_start_time = current_time
                inst.current_global_index = global_idx  # 记录当前处理的全局样本索引
            else:
                # 只有真正没有样本时才变成IDLE
                inst.state = InstanceState.IDLE
            return

        # 空闲实例尝试取样本
        if inst.state == InstanceState.IDLE:
            global_idx = self.sample_queue.try_lock_samples(1)
            if global_idx is not None:
                inst.state = InstanceState.BUSY
                inst.current_sample_start_time = current_time
                inst.current_global_index = global_idx
            return

        # 忙实例检查是否完成
        if inst.state == InstanceState.BUSY:
            # 检查是否已处理完所有样本
            if inst.current_global_index >= self.total_samples:
                # 任务已完成，转为IDLE
                inst.state = InstanceState.IDLE
                return

            # 获取当前样本的推理时间（基于全局索引）
            inference_time = inst.get_inference_time_for_sample(inst.current_global_index)

            # 如果是第一次推理（未应用过冷启动时间），加上冷启动时间
            actual_inference_time = inference_time
            if not inst.has_applied_cold_start:
                actual_inference_time += 5.0  # 固定5秒冷启动时间
                inst.has_applied_cold_start = True  # 标记已应用冷启动

            elapsed = current_time - inst.current_sample_start_time

            if elapsed >= actual_inference_time:
                # 样本完成
                self.sample_queue.unlock_samples(1)
                self.done_samples += 1
                inst.samples_processed += 1
                inst.current_sample_start_time = 0.0

                # 调试输出：每完成10个样本输出一次
                if self.done_samples % 10 == 0 or self.done_samples == self.total_samples:
                    print(f"    {self.task_id}: 完成 {self.done_samples}/{self.total_samples} 样本")

                # 检查任务是否完成（所有样本都处理完）→ 所有轮次完成
                if self.done_samples >= self.total_samples:
                    self.phase = TaskPhase.DONE
                    self.end_time = current_time
                    print(f"    {self.task_id}: 所有轮次完成！总轮数: {self.done_rounds + 1}, 总时间: {current_time:.1f}s")
                    inst.state = InstanceState.IDLE
                    return

                # 检查本轮是否完成（本轮样本处理完，但还有下一轮）
                if self.sample_queue.check_round_complete(self.done_samples):
                    # 本轮完成，进入下一轮
                    self.done_rounds += 1
                    print(f"    {self.task_id}: 第 {self.done_rounds} 轮完成")

                    # 进入下一轮（不释放实例，保留分配状态）
                    if self.sample_queue.next_round():
                        print(f"    {self.task_id}: 进入第 {self.done_rounds + 1} 轮，实例保持IDLE等待新样本")
                        # 还有更多轮次，实例保持IDLE状态等待新样本
                        # 不释放实例，保留上一轮的分配状态（可能失衡）
                    # else: 所有样本已分配完毕，等待现有样本完成

                # 尝试取下一个样本（如果有样本则继续BUSY，否则变IDLE）
                if self.phase == TaskPhase.ROLLOUT:
                    global_idx = self.sample_queue.try_lock_samples(1)
                    if global_idx is not None:
                        inst.state = InstanceState.BUSY
                        inst.current_sample_start_time = current_time
                        inst.current_global_index = global_idx
                    else:
                        inst.state = InstanceState.IDLE

    def get_state_report(self) -> TaskStateReport:
        """生成状态上报"""
        idle = sum(1 for i in self.instances if i.state == InstanceState.IDLE)
        busy = sum(1 for i in self.instances if i.state == InstanceState.BUSY)
        return TaskStateReport(
            task_id=self.task_id,
            done_samples=self.done_samples,
            done_rounds=self.done_rounds,
            elapsed_time_sec=self._elapsed_time(),
            remaining_samples=self.total_samples - self.done_samples,
            current_instances=len(self.instances),
            idle_instances=idle,
            busy_instances=busy,
            in_rollout_phase=(self.phase == TaskPhase.ROLLOUT)
        )

    def get_state_report_for_gs(self):
        """
        生成给GroupScheduler的状态上报

        关键改进：
        1. 使用 GS 的 TaskStateReport（动态导入）
        2. 构建 voluntary_reclaim（检测空闲实例）
        3. 冷启动中的实例归类为busy（已分配给task，不占用GS可用资源）
        """
        # 【修复】将冷启动中的实例归类为busy
        # 原因：冷启动中的实例已经分配给task，不应该被GS当作"可用"资源
        cold_starting = sum(1 for i in self.instances if i.state == InstanceState.COLD_STARTING)
        idle = sum(1 for i in self.instances if i.state == InstanceState.IDLE)
        busy = sum(1 for i in self.instances if i.state == InstanceState.BUSY)
        busy += cold_starting  # 冷启动中的实例也算作busy（已分配，不占用GS可用资源）

        # 使用 reclaimed_workers_cache 构建 voluntary_reclaim
        # 注意：reclaimed_workers_cache 由 release_idle_instances() 填充
        voluntary_reclaim = None
        if self.reclaimed_workers_cache:
            try:
                from data_class import ReclaimConfirm, WorkerInfo

                # 创建 WorkerInfo 列表
                worker_infos = []
                for gpu in self.reclaimed_workers_cache:
                    worker_id = gpu.machine_id * 8 + gpu.gpu_id
                    wi = WorkerInfo("shared", gpu.machine_id, gpu.gpu_id)
                    wi.set_id(str(worker_id))
                    worker_infos.append(wi)

                # 计算 reclaimed_instances（每个实例需要 tp*pp 张卡）
                cards_per_instance = self.tp * self.pp
                reclaimed_instances = len(self.reclaimed_workers_cache) // cards_per_instance

                voluntary_reclaim = ReclaimConfirm(
                    task_id=self.task_id,
                    reclaimed_instances=reclaimed_instances,
                    reclaimed_workers=worker_infos
                )
            except ImportError:
                # GS 不可用时，不构建 voluntary_reclaim
                pass

        # 动态导入 GS 的 TaskStateReport
        try:
            from data_class import TaskStateReport as GSTaskStateReport

            # 【关键修复】remaining_samples 只报本轮可取的样本数
            # 这样GS能正确判断是否需要分配实例：
            # - available_samples=0 → remaining=0 → GS不分配
            # - 进入下一轮，available变为samples_per_round → GS分配
            current_round_available = self.sample_queue.available_samples

            return GSTaskStateReport(
                task_id=self.task_id,
                done_samples=self.done_samples,
                done_rounds=self.done_rounds,
                elapsed_time_sec=self._elapsed_time(),
                remaining_samples=current_round_available,  # 只报本轮可取样本
                current_instances=len(self.instances),
                idle_instances=idle,
                busy_instances=busy,
                in_rollout_phase=(self.phase == TaskPhase.ROLLOUT),
                voluntary_reclaim=voluntary_reclaim  # 新增
            )
        except ImportError:
            # 如果 GS 不可用，返回普通字典
            return {
                'task_id': self.task_id,
                'done_samples': self.done_samples,
                'done_rounds': self.done_rounds,
                'elapsed_time_sec': self._elapsed_time(),
                'remaining_samples': self.total_samples - self.done_samples,
                'current_instances': len(self.instances),
                'idle_instances': idle,
                'busy_instances': busy,
                'in_rollout_phase': (self.phase == TaskPhase.ROLLOUT),
                'voluntary_reclaim': voluntary_reclaim
            }

    def _elapsed_time(self) -> float:
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time
