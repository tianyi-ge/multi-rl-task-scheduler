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
    """每轮迭代的样本队列"""
    samples_per_round: int  # 本轮总样本数
    round_id: int = 0
    available_samples: int = field(init=False, default=0)
    locked_samples: int = 0  # 已被实例取走的样本数

    def __post_init__(self):
        self.available_samples = self.samples_per_round

    def try_lock_samples(self, count: int) -> bool:
        """尝试锁定count个样本（实例取样本时调用）"""
        if self.available_samples >= count:
            self.available_samples -= count
            self.locked_samples += count
            return True
        return False

    def unlock_samples(self, count: int) -> None:
        """完成count个样本（推理完成后调用）"""
        self.locked_samples -= count

    def next_round(self) -> None:
        """进入下一轮"""
        self.round_id += 1
        self.available_samples = self.samples_per_round
        self.locked_samples = 0


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
    total_samples: int
    random_seed: int

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
        self.sample_queue = SampleQueue(samples_per_round=self.samples_per_round)

    def release_idle_instances(self, cluster: ClusterModel) -> List[GPUPlacement]:
        """
        主动释放空闲实例

        逻辑：
        1. 如果任务已经完成（所有sample推理完），释放所有实例
        2. 如果所有实例都处于空闲状态（刚刚推理完），释放所有实例
        3. 不能释放 BUSY 或 COLD_STARTING 状态的实例

        Returns:
            被释放的GPU placement列表
        """
        self.reclaimed_workers_cache.clear()

        # 如果任务已经完成，释放所有实例
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

        # 如果所有实例都处于空闲状态（刚刚推理完），释放所有实例
        if not busy_instances and not cold_starting_instances:
            for inst in idle_instances:
                cluster.reclaim_gpus(inst.gpus)
                self.reclaimed_workers_cache.extend(inst.gpus)
                self.instances.remove(inst)
            return self.reclaimed_workers_cache

        # 否则不释放（有实例还在工作）
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

                # 预计算所有样本的推理时间
                # 所有实例使用相同的种子，sample_start_index=0
                # 这样无GS和有GS模式下，初始实例的时间表完全一致
                inst.precompute_inference_times(
                    total_samples=self.total_samples,
                    tp=self.tp,
                    pp=self.pp,
                    time_distribution=self.time_distribution,
                    distribution_params=self.distribution_params,
                    random_seed=self.random_seed,
                    sample_start_index=0  # 初始实例从头开始
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

        关键改进：冷启动完成后直接转为BUSY（如果有样本），不经过IDLE
        只有真正没有样本可取时才变成IDLE
        """
        # 冷启动检查：冷启动完成后直接转为 BUSY（如果有样本）
        if inst.state == InstanceState.COLD_STARTING:
            # 冷启动后立即尝试取样本，如果有样本则直接进入BUSY状态
            if self.sample_queue.try_lock_samples(1):
                inst.state = InstanceState.BUSY
                inst.current_sample_start_time = current_time
            else:
                # 只有真正没有样本时才变成IDLE
                inst.state = InstanceState.IDLE
            return

        # 空闲实例尝试取样本
        if inst.state == InstanceState.IDLE:
            if self.sample_queue.try_lock_samples(1):
                inst.state = InstanceState.BUSY
                inst.current_sample_start_time = current_time
            return

        # 忙实例检查是否完成
        if inst.state == InstanceState.BUSY:
            # 检查是否已处理完该实例的所有样本
            if inst.current_sample_index >= len(inst.inference_time_table):
                # 该实例的样本已全部完成，尝试取下一个
                if self.sample_queue.try_lock_samples(1):
                    inst.state = InstanceState.BUSY
                    inst.current_sample_start_time = current_time
                else:
                    inst.state = InstanceState.IDLE
                return

            inference_time = inst.inference_time_table[inst.current_sample_index]

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
                inst.current_sample_index += 1
                inst.current_sample_start_time = 0.0

                # 调试输出：每完成一个样本输出一次
                if self.done_samples % 10 == 0 or self.done_samples == self.total_samples:
                    print(f"    {self.task_id}: 完成 {self.done_samples}/{self.total_samples} 样本")

                # 检查本轮是否完成
                if (self.sample_queue.available_samples == 0 and
                    self.sample_queue.locked_samples == 0):
                    # 本轮完成
                    self.done_rounds += 1
                    if self.done_rounds * self.samples_per_round >= self.total_samples:
                        self.phase = TaskPhase.DONE
                        self.end_time = current_time
                        print(f"    {self.task_id}: 完成！总时间: {current_time:.1f}s")
                    else:
                        self.sample_queue.next_round()

                # 尝试取下一个样本（如果有样本则继续BUSY，否则变IDLE）
                if (self.phase == TaskPhase.ROLLOUT and
                    self.sample_queue.try_lock_samples(1)):
                    inst.state = InstanceState.BUSY
                    inst.current_sample_start_time = current_time
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
            return GSTaskStateReport(
                task_id=self.task_id,
                done_samples=self.done_samples,
                done_rounds=self.done_rounds,
                elapsed_time_sec=self._elapsed_time(),
                remaining_samples=self.total_samples - self.done_samples,
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
