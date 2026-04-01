"""Simulator模块 - 仿真器主类"""

import random
from typing import Dict, Optional, List, Callable, Any, Set
import os
import sys

# 导入模型
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.cluster import ClusterModel
from models.task import TaskModel, TaskPhase
from models.test_case import TestCase
from models.instance import InstanceState, GPUPlacement

# 导入适配器和日志器
from .gs_adapter import GSAdapter, NPUS_PER_NODE
from .scheduler_logger import SchedulerLogger


class SimulationClock:
    """仿真时钟 - 固定步长推进"""
    def __init__(self, time_step: float = 0.5):
        self.time_step = time_step
        self.current_time: float = 0.0

    def advance_one_step(self) -> float:
        """推进一个时间步"""
        self.current_time += self.time_step
        return self.current_time

    def peek_time(self) -> float:
        """获取当前时间"""
        return self.current_time


class Simulator:
    """仿真器主类"""

    def __init__(
        self,
        test_case: TestCase,
        enable_gs: bool = True,
        log_dir: str = "results"
    ):
        """
        Args:
            test_case: 测试用例
            enable_gs: 是否启用GroupScheduler调度
            log_dir: 调度日志目录
        """
        self.test_case = test_case
        self.enable_gs = enable_gs
        self.log_dir = log_dir

        # 创建集群
        self.cluster = ClusterModel.from_config(
            machine_count=test_case.cluster.machine_count,
            gpus_per_machine=test_case.cluster.gpus_per_machine
        )

        # 创建时钟（仿真步长从0.5改为1秒）
        self.clock = SimulationClock(time_step=1.0)

        # 任务字典
        self.tasks: Dict[str, TaskModel] = {}

        # GS适配器和日志器
        self.gs_adapter: Optional[GSAdapter] = None
        self.logger: Optional[SchedulerLogger] = None

        # 创建日志器（无论是否启用GS，都记录轮次进度）
        if log_dir:
            self.logger = SchedulerLogger(log_dir=log_dir)

        if enable_gs:
            # 创建 GS适配器（使用 ClusterConfig 对齐）
            self.gs_adapter = GSAdapter(
                machine_count=test_case.cluster.machine_count,
                total_workers=test_case.cluster.total_gpus(),
                logger=self.logger
            )

            # 设置回调函数
            self.gs_adapter.set_callbacks(
                expand_callback=self._on_gs_expand,
                reclaim_callback=self._on_gs_reclaim,
                get_idle_workers_callback=self._get_idle_workers_for_task,
                get_state_callback=self._get_task_state_for_gs,
                get_num_rounds_map_callback=self._get_num_rounds_map
            )

        # 初始化
        self._init_from_test_case(test_case)

    def _init_from_test_case(self, test_case: TestCase) -> None:
        """从测试用例初始化"""
        # 验证初始约束
        test_case.validate_initial_constraints()

        # 创建任务
        for task_config in test_case.tasks:
            task = TaskModel(
                task_id=task_config.task_id,
                tp=task_config.tp,
                pp=task_config.pp,
                base_instances=task_config.base_instances,
                samples_per_round=task_config.samples_per_round,
                num_rounds=task_config.num_rounds,
                random_seed=42 + hash(task_config.task_id),
                time_distribution=task_config.time_distribution,
                distribution_params=task_config.distribution_params
            )
            self.tasks[task_config.task_id] = task

            # 先设置为ROLLOUT阶段（必须在注册前设置）
            task.phase = TaskPhase.ROLLOUT

            # 注册到GS（GS会自动触发调度并分配实例）
            if self.gs_adapter:
                # 【正确流程】直接注册任务，让GS调度空闲实例并分配
                # 不预先分配实例，等待GS通过回调分配
                success = self.gs_adapter.register_task(task_config)
                if success:
                    print(f"任务 {task_config.task_id} 注册成功，等待GS分配实例")
                    # 等待GS完成初始调度和分配（通过_on_gs_expand回调）
                    self.gs_adapter._wait_for_gs_scheduling()
                else:
                    print(f"任务 {task_config.task_id} 注册失败")
            else:
                # 非GS模式：预先分配base_instances的实例
                task.init_instances(task_config.base_instances, self.cluster)

    def run(self) -> Any:
        """运行仿真"""
        print(f"开始仿真，启用GS调度: {self.enable_gs}")
        print(f"集群: {self.cluster.total_gpus()}张卡")
        print(f"任务: {len(self.tasks)}个")

        step = 0
        max_steps = 50000  # 增大上限以支持更多样本的测试场景

        # 记录间隔
        record_interval = max(1, int(5.0 / self.clock.time_step))

        while self._has_active_tasks() and step < max_steps:
            step += 1
            current_time = self.clock.advance_one_step()

            # 所有任务执行推理
            for task in self.tasks.values():
                task.step(current_time)

            # 上报状态到GS（触发调度）
            if self.enable_gs and step > 0:
                self._report_all_task_states(current_time, step)
            else:
                # 无GS模式：记录轮次进度（不触发调度）
                if self.logger and step > 0:
                    for task in self.tasks.values():
                        current_round_remaining = task.sample_queue.available_samples + task.sample_queue.locked_samples
                        self.logger.log_round_progress(
                            task_id=task.task_id,
                            done_rounds=task.done_rounds,
                            num_rounds=task.num_rounds,
                            done_samples=task.done_samples,
                            remaining_samples=task.total_samples - task.done_samples,
                            current_round=task.done_rounds + 1,
                            samples_in_current_round=current_round_remaining,
                            simulation_step=step,
                            current_time=current_time
                        )

            # 定期记录
            if step % record_interval == 0:
                self._print_progress(step, current_time)

            # 记录仿真步（每个步都记录，便于分析）
            if self.logger:
                self.logger.log_simulation_step(current_time, step)

        current_time = self.clock.peek_time()
        print(f"\n仿真完成，总步数: {step}, 总时间: {current_time:.2f}s")

        # 清理GS资源（停止后台线程）
        if self.gs_adapter:
            self.gs_adapter.cleanup()

        # 关闭日志器
        if self.logger:
            self.logger.close()

        return self._collect_results()

    def _report_all_task_states(self, current_time: float, step: int) -> None:
        """
        上报所有任务状态到GS，触发调度

        GS会根据状态自动决定是否需要调度
        """
        if not self.gs_adapter:
            return

        # 1. 只在所有轮次完成时释放实例（多轮迭代仿真）
        # 关键修改：不在每轮完成时释放，保留分配状态让GS进行"失衡→平衡"调度
        for task in self.tasks.values():
            # 检查任务是否所有轮次完成（所有样本都处理完）
            if task.done_samples >= task.total_samples:
                if task.phase != TaskPhase.DONE:
                    # 标记任务为完成
                    task.phase = TaskPhase.DONE
                    task.end_time = current_time
                    print(f"  {task.task_id}: 所有轮次完成，标记为DONE并释放资源")

            # 只在DONE时释放实例（不是每轮完成）
            if task.phase == TaskPhase.DONE and not task.workers_released:
                # 释放GPU资源
                for inst in task.instances:
                    self.cluster.reclaim_gpus(inst.gpus)

                # 清空实例列表
                task.instances.clear()

                # 释放GS worker
                worker_ids = self.gs_adapter.get_task_workers(task.task_id)
                if worker_ids:
                    # 将worker加回GS的idle列表
                    self.gs_adapter.gs.workers.add_workers_to_idle(worker_ids)
                    # 清除任务worker记录
                    if task.task_id in self.gs_adapter.allocated_workers:
                        del self.gs_adapter.allocated_workers[task.task_id]

                # 标记已释放
                task.workers_released = True
                print(f"  {task.task_id}: 释放了所有实例和资源")

        # 2. 轮次进行中的动态回收（通过 voluntary_reclaim 机制）
        # 注意：release_idle_instances 在轮次切换时不应触发（见 task.py 的修改）
        for task in self.tasks.values():
            if task.phase == TaskPhase.ROLLOUT:
                reclaimed_gpus = task.release_idle_instances(self.cluster)
                if reclaimed_gpus:
                    # 将回收的worker加回GS的idle列表
                    worker_ids = [str(gpu.machine_id * NPUS_PER_NODE + gpu.gpu_id) for gpu in reclaimed_gpus]
                    self.gs_adapter.gs.workers.add_workers_to_idle(worker_ids)
                    # 更新allocated_workers
                    if task.task_id in self.gs_adapter.allocated_workers:
                        worker_id_set = set(worker_ids)
                        self.gs_adapter.allocated_workers[task.task_id] = [
                            w for w in self.gs_adapter.allocated_workers[task.task_id] if w not in worker_id_set
                        ]
                    print(f"  [动态回收] {task.task_id} 释放了 {len(reclaimed_gpus)} 张卡 ({len(worker_ids)} workers)")

        # 3. 记录轮次进度（每个任务）
        for task in self.tasks.values():
            if self.logger:
                # 计算当前轮次剩余样本
                current_round_remaining = task.sample_queue.available_samples + task.sample_queue.locked_samples

                self.logger.log_round_progress(
                    task_id=task.task_id,
                    done_rounds=task.done_rounds,
                    num_rounds=task.num_rounds,
                    done_samples=task.done_samples,
                    remaining_samples=task.total_samples - task.done_samples,
                    current_round=task.done_rounds + 1,
                    samples_in_current_round=current_round_remaining,
                    simulation_step=step,
                    current_time=current_time
                )

        # 4. 上报所有任务状态到GS（触发调度）
        # GS会根据 current_instances vs base_instances 进行"失衡→平衡"调度
        states_reported = 0
        for task in self.tasks.values():
            state = task.get_state_report_for_gs()
            if state:
                # 记录状态上报（含轮次信息）
                if self.logger:
                    self.logger.log_report_state(
                        task.task_id, state, need_schedule=True,
                        simulation_step=step, current_time=current_time,
                        num_rounds=task.num_rounds
                    )
                self.gs_adapter.report_state(state, need_schedule=True)
                states_reported += 1
        if step % 10 == 0:
            print(f"  Step {step}, Reported {states_reported} task states to GS")

    def _has_idle_instances(self) -> bool:
        """检查是否有空闲实例"""
        for task in self.tasks.values():
            if task.phase == TaskPhase.ROLLOUT:
                for inst in task.instances:
                    if inst.state == InstanceState.IDLE:
                        return True
        return False

    def _on_gs_expand(self, task_id: str, worker_ids: List[str]) -> None:
        """GS回调：为任务分配worker（扩容）"""
        task = self.tasks.get(task_id)
        if not task:
            return

        # 跳过空的worker_ids（可能是bug导致的）
        if not worker_ids:
            print(f"  警告: _on_gs_expand收到空的worker_ids，跳过")
            return

        # 检查集群是否有足够的空闲GPU
        required_gpus = len(worker_ids)
        available_gpus = len(self.cluster.free_gpus)
        if available_gpus < required_gpus:
            print(f"  [警告] 集群空闲GPU不足: 需要{required_gpus}，可用{available_gpus}，跳过分配")
            return

        # 根据worker_ids确定GPU placement
        from models.instance import GPUPlacement
        gpus = []
        for worker_id_str in worker_ids:
            worker_id = int(worker_id_str)
            machine_id = worker_id // NPUS_PER_NODE
            machine_id = machine_id % 1000  # 处理machine_id格式如"shared_0"的情况
            gpu_id = worker_id % NPUS_PER_NODE
            gpus.append(GPUPlacement(machine_id, gpu_id))

        # 更新cluster的状态
        for gpu in gpus:
            self.cluster.machines[gpu.machine_id].gpu_states[gpu.gpu_id] = 1
            if gpu in self.cluster.free_gpus:
                self.cluster.free_gpus.remove(gpu)

        # 创建新实例
        new_instance_id = len(task.instances)
        from models.instance import Instance
        inst = Instance(
            instance_id=new_instance_id,
            gpus=gpus,
            state=InstanceState.COLD_STARTING
        )

        # 计算是否跨节点
        nodes = set(gpu.machine_id for gpu in gpus)
        inst.placement_nodes = nodes
        if len(nodes) > 1:
            inst.communication_factor = 1.2  # 跨节点通信开销
        else:
            inst.communication_factor = 1.0

        # 预计算推理时间表
        # 【公平性设计】使用相同的任务种子，确保同一个 sample 在任何实例上
        # 都有相同的推理时间，保证无GS和有GS方案公平比较
        inst.precompute_inference_times(
            total_samples=task.total_samples,
            tp=task.tp,
            pp=task.pp,
            time_distribution=task.time_distribution,
            distribution_params=task.distribution_params,
            random_seed=task.random_seed
        )

        task.instances.append(inst)
        inst.cold_start_end_time = self.clock.peek_time() + 5.0  # 冷启动时间固定为5秒

        print(f"  扩容: {task_id} -> {len(task.instances)}实例 (nodes: {nodes})")

    def _on_gs_reclaim(self, task_id: str, worker_ids: List[str]) -> None:
        """GS回调：回收worker（缩容）"""
        task = self.tasks.get(task_id)
        if not task or len(task.instances) <= task.base_instances:
            return

        # 根据worker_ids找到对应的实例
        worker_id_set = set(int(w) for w in worker_ids)
        instances_to_remove = []

        # 【关键修复】只移除超过base_instances的部分
        max_to_remove = len(task.instances) - task.base_instances

        for inst in task.instances:
            if len(instances_to_remove) >= max_to_remove:
                # 已经找到足够的实例，不再继续
                break
            inst_worker_ids = [gpu.machine_id * NPUS_PER_NODE + gpu.gpu_id for gpu in inst.gpus]
            # 如果实例的所有worker都在回收列表中，则移除该实例
            if all(w in worker_id_set for w in inst_worker_ids):
                instances_to_remove.append(inst)

        # 移除实例并回收GPU
        reclaimed_workers = []
        samples_returned = 0
        for inst in instances_to_remove:
            # 【关键修复】BUSY实例需要把样本放回队列
            if inst.state == InstanceState.BUSY:
                # 样本正在处理中，需要放回队列让其他实例处理
                # 【关键】传递原始索引，确保推理时间正确
                task.sample_queue.return_sample(inst.current_global_index)
                samples_returned += 1
                print(f"  [BUSY回收] {task_id} 样本{inst.current_global_index}放回队列")

            # 回收GPU到集群
            self.cluster.reclaim_gpus(inst.gpus)
            # 记录被回收的worker IDs
            for gpu in inst.gpus:
                worker_id = gpu.machine_id * NPUS_PER_NODE + gpu.gpu_id
                reclaimed_workers.append(str(worker_id))
            # 移除实例
            task.instances.remove(inst)

        # 【关键修复】将回收的workers加回GS的idle列表
        # 这样后续的分配才能找到空闲workers
        if reclaimed_workers and self.gs_adapter:
            self.gs_adapter.gs.workers.add_workers_to_idle(reclaimed_workers)
            # 更新allocated_workers记录
            if task_id in self.gs_adapter.allocated_workers:
                worker_id_set = set(reclaimed_workers)
                self.gs_adapter.allocated_workers[task_id] = [
                    w for w in self.gs_adapter.allocated_workers[task_id] if w not in worker_id_set
                ]

        if instances_to_remove:
            msg = f"  缩容: {task_id} -> {len(task.instances)}实例, 释放{len(reclaimed_workers)}workers到GS"
            if samples_returned > 0:
                msg += f", {samples_returned}样本放回队列"
            print(msg)

    def _get_idle_workers_for_task(self, task_id: str) -> List[str]:
        """
        获取任务可回收的worker IDs

        【关键约束】必须保护base_instances：
        1. 任务DONE时：所有实例可回收
        2. 长尾场景（有BUSY实例处理慢样本，有IDLE实例完成快样本）：
           - 可以回收IDLE实例给其他任务使用（资源动态分配）
           - 只保护base_instances数量
        3. 轮次切换（所有实例都IDLE且无样本）：
           - 不回收，保留实例等待下一轮

        Returns:
            可回收的worker ID列表
        """
        task = self.tasks.get(task_id)
        if not task:
            return []

        reclaimable_workers = []

        # 1. 任务DONE：所有实例可回收
        if task.phase == TaskPhase.DONE:
            for inst in task.instances:
                for gpu in inst.gpus:
                    worker_id = gpu.machine_id * NPUS_PER_NODE + gpu.gpu_id
                    reclaimable_workers.append(str(worker_id))
            return reclaimable_workers

        # 2. 检查是否有样本待处理
        remaining_total = task.total_samples - task.done_samples
        if remaining_total == 0:
            # 任务完成所有样本，即将标记为DONE
            return []

        # 3. 统计实例状态
        idle_instances = [inst for inst in task.instances if inst.state == InstanceState.IDLE]
        busy_instances = [inst for inst in task.instances if inst.state == InstanceState.BUSY]
        cold_starting = [inst for inst in task.instances if inst.state == InstanceState.COLD_STARTING]

        total_instances = len(task.instances)

        # 4. 【长尾场景关键】当有BUSY实例时，可以回收IDLE实例
        # 原因：BUSY实例处理长尾样本需要时间，IDLE实例已经完成快样本
        # 可以把IDLE实例的资源给其他任务，实现负载均衡
        if busy_instances or cold_starting:
            # 有实例在忙：可以回收IDLE实例（最多回收到保留base_instances）
            excess_instances = total_instances - task.base_instances

            if excess_instances > 0 and idle_instances:
                # 回收IDLE实例（最多excess个）
                for inst in idle_instances[:excess_instances]:
                    for gpu in inst.gpus:
                        worker_id = gpu.machine_id * NPUS_PER_NODE + gpu.gpu_id
                        reclaimable_workers.append(str(worker_id))

            # 如果还有超额BUSY实例，也可以回收
            remaining_excess = excess_instances - len(idle_instances)
            if remaining_excess > 0:
                for inst in busy_instances[:remaining_excess]:
                    for gpu in inst.gpus:
                        worker_id = gpu.machine_id * NPUS_PER_NODE + gpu.gpu_id
                        reclaimable_workers.append(str(worker_id))

            return reclaimable_workers

        # 5. 所有实例都IDLE且无BUSY：轮次切换场景
        # 检查是否还有下一轮样本
        if idle_instances and not busy_instances:
            # 所有实例都空闲
            if task.sample_queue.available_samples > 0:
                # 有样本待处理，实例会继续工作
                return []

            # 无样本：等待下一轮，保留base_instances
            # 但可以回收超过base的实例
            excess_instances = total_instances - task.base_instances
            if excess_instances > 0:
                for inst in idle_instances[:excess_instances]:
                    for gpu in inst.gpus:
                        worker_id = gpu.machine_id * NPUS_PER_NODE + gpu.gpu_id
                        reclaimable_workers.append(str(worker_id))

        return reclaimable_workers

    def _get_task_state_for_gs(self, task_id: str) -> Any:
        """获取任务的当前状态（用于GS回调时同步状态）"""
        task = self.tasks.get(task_id)
        if not task:
            return None
        return task.get_state_report_for_gs()

    def _get_num_rounds_map(self) -> Dict[str, int]:
        """获取所有任务的 num_rounds 映射（用于日志记录）"""
        return {task_id: task.num_rounds for task_id, task in self.tasks.items()}

    def _has_active_tasks(self) -> bool:
        """检查是否有活动任务"""
        return any(t.phase != TaskPhase.DONE for t in self.tasks.values())

    def _print_progress(self, step: int, current_time: float) -> None:
        """打印进度"""
        utilization = self.cluster.get_utilization() * 100
        done_count = sum(1 for t in self.tasks.values() if t.phase == TaskPhase.DONE)

        print(f"  Step {step}, Time: {current_time:.1f}s, "
              f"GPU利用率: {utilization:.1f}%, 完成: {done_count}/{len(self.tasks)}")

    def _collect_results(self) -> Any:
        """收集仿真结果"""
        completion_times = {
            task_id: task.end_time or 0
            for task_id, task in self.tasks.items()
        }

        # 从日志器获取有效调度决策数
        scheduling_count = 0
        if self.logger:
            scheduling_count = self.logger.effective_scheduling_count

        from .result import SimulationResult
        return SimulationResult(
            test_case_name=self.test_case.name,
            total_simulation_time=self.clock.peek_time(),
            task_completion_times=completion_times,
            scheduling_traces=[],  # 详细轨迹在JSON日志中
            scheduling_trace_count=scheduling_count,  # 有效调度决策数
            gpu_utilization_curve=[],  # 简化版暂不记录曲线
            task_states_history={},  # 简化版暂不记录
            event_history=[]
        )
