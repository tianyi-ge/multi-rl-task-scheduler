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

        if enable_gs:
            # 创建日志器（仅当log_dir不为None时）
            if log_dir:
                self.logger = SchedulerLogger(log_dir=log_dir)
            else:
                self.logger = None

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
                get_state_callback=self._get_task_state_for_gs
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
                total_samples=task_config.total_samples,
                random_seed=42 + hash(task_config.task_id),
                time_distribution=task_config.time_distribution,
                distribution_params=task_config.distribution_params
            )
            self.tasks[task_config.task_id] = task

            # 先设置为ROLLOUT阶段（必须在注册前设置）
            task.phase = TaskPhase.ROLLOUT

            # 注册到GS（GS会自动触发调度并分配实例）
            if self.gs_adapter:
                # 【修复】GS模式下也预先分配实例，确保和无GS模式相同的启动时机
                # 原因：如果等待GS调度分配，会有约6秒延迟（GS调度周期+状态上报延迟）
                # 修复：先分配实例，再向GS注册并上报状态，让GS知道这些worker已被占用
                task.init_instances(task_config.base_instances, self.cluster)

                # 收集已分配实例的worker IDs
                worker_ids = []
                for inst in task.instances:
                    for gpu in inst.gpus:
                        worker_id = gpu.machine_id * NPUS_PER_NODE + gpu.gpu_id
                        worker_ids.append(str(worker_id))

                # 注册任务到GS，带上已分配的worker IDs
                success = self.gs_adapter.register_task(task_config, worker_ids=worker_ids)
                if success:
                    print(f"任务 {task_config.task_id} 注册成功，base_instances={task_config.base_instances}, worker_ids={len(worker_ids)}")
                    # 等待GS调度完成（GS会根据初始状态触发调度，但此时worker已被占用）
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

        # 1. 检查并释放已完成任务的worker和GPU
        # 修复：当任务完成所有样本时，即使phase还未被设为DONE，也应该标记为DONE并释放资源
        for task in self.tasks.values():
            # 检查任务是否真正完成（所有样本都处理完毕）
            if task.done_samples >= task.total_samples:
                if task.phase != TaskPhase.DONE:
                    # 标记任务为完成
                    task.phase = TaskPhase.DONE
                    task.end_time = current_time
                    print(f"  {task.task_id}: 检测到完成，标记为DONE并释放资源")

            if task.phase == TaskPhase.DONE and not task.workers_released:
                # 释放GPU资源
                for inst in task.instances:
                    self.cluster.reclaim_gpus(inst.gpus)

                # 清空实例列表（修复：否则上报状态时仍显示有实例）
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

        # 2. 主动释放空闲实例并回收GPU资源
        # 逻辑：释放空闲实例，但保留至少 base_instances 个实例
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
                    print(f"  [主动回收] {task.task_id} 释放了 {len(reclaimed_gpus)} 张卡 ({len(worker_ids)} workers)")

        # 3. 上报所有任务状态到GS
        states_reported = 0
        for task in self.tasks.values():
            state = task.get_state_report_for_gs()
            if state:
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

        # 预计算推理时间 - 从当前进度开始
        # 新实例从done_samples处开始处理剩余样本
        # 使用sample_start_index确保种子一致性
        remaining_samples = task.total_samples - task.done_samples
        if remaining_samples > 0:
            inst.precompute_inference_times(
                total_samples=remaining_samples,  # 只需要计算剩余样本的时间
                tp=task.tp,
                pp=task.pp,
                time_distribution=task.time_distribution,
                distribution_params=task.distribution_params,
                random_seed=task.random_seed,  # 使用任务原始种子，不添加偏移
                sample_start_index=task.done_samples  # 从当前进度开始
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

        for inst in task.instances:
            inst_worker_ids = [gpu.machine_id * NPUS_PER_NODE + gpu.gpu_id for gpu in inst.gpus]
            # 如果实例的所有worker都在回收列表中，则移除该实例
            if all(w in worker_id_set for w in inst_worker_ids):
                instances_to_remove.append(inst)

        # 移除实例并回收GPU
        for inst in instances_to_remove:
            # 回收GPU
            self.cluster.reclaim_gpus(inst.gpus)
            # 移除实例
            task.instances.remove(inst)

        if instances_to_remove:
            print(f"  缩容: {task_id} -> {len(task.instances)}实例")

    def _get_idle_workers_for_task(self, task_id: str) -> List[str]:
        """获取任务的空闲worker IDs"""
        task = self.tasks.get(task_id)
        if not task:
            return []

        idle_workers = []
        for inst in task.instances:
            if inst.state == InstanceState.IDLE:
                for gpu in inst.gpus:
                    worker_id = gpu.machine_id * NPUS_PER_NODE + gpu.gpu_id
                    idle_workers.append(str(worker_id))

        return idle_workers

    def _get_task_state_for_gs(self, task_id: str) -> Any:
        """获取任务的当前状态（用于GS回调时同步状态）"""
        task = self.tasks.get(task_id)
        if not task:
            return None
        return task.get_state_report_for_gs()

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
