"""
GroupScheduler适配层 - 连接仿真器与真实GroupScheduler

使用真实GS的事件驱动机制，并通过 monkey patch 修复/增强其行为：
- 修复 concurrent_reclaim/concurrent_assign 的 task_id 调用bug
- 对齐 create() 使用正确的 total_workers
- 添加完整的调度决策日志（JSON + 控制台）
"""

import sys
import os
import time
import pickle
import copy
from types import ModuleType, MethodType
from typing import Dict, List, Tuple, Callable, Any, Optional, Set
from dataclasses import dataclass

# Global adapter registry for deepcopy support
_ADAPTER_REGISTRY: Dict[int, Any] = {}

# 相对导入真实GroupScheduler
_gs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../group_scheduler'))
if _gs_dir not in sys.path:
    sys.path.insert(0, _gs_dir)

# Setup yr mock first - MUST be done before importing group_scheduler
# Use group_scheduler's yr_mock
from yr_mock import mock_yr as yr

# Import yr into global scope for group_scheduler
sys.modules['yr'] = yr

# Setup mindspeed_llm mock BEFORE importing task.py
# This MUST be done before importing config, task, worker, data_class
if 'mindspeed_llm' not in sys.modules:
    # Create mock mindspeed_llm module structure with __path__ to make them packages
    mindspeed_llm = ModuleType('mindspeed_llm')
    mindspeed_llm.__path__ = []

    tasks = ModuleType('mindspeed_llm.tasks')
    tasks.__path__ = []

    posttrain = ModuleType('mindspeed_llm.tasks.posttrain')
    posttrain.__path__ = []

    rlxf = ModuleType('mindspeed_llm.tasks.posttrain.rlxf')
    rlxf.__path__ = []

    group_scheduler_gs = ModuleType('mindspeed_llm.tasks.posttrain.rlxf.group_scheduler')
    group_scheduler_gs.__path__ = []  # Make it a package

    # Register modules BEFORE importing (include all levels)
    sys.modules['mindspeed_llm'] = mindspeed_llm
    sys.modules['mindspeed_llm.tasks'] = tasks
    sys.modules['mindspeed_llm.tasks.posttrain'] = posttrain
    sys.modules['mindspeed_llm.tasks.posttrain.rlxf'] = rlxf
    sys.modules['mindspeed_llm.tasks.posttrain.rlxf.group_scheduler'] = group_scheduler_gs

    # Import actual modules (they now have mindspeed_llm available)
    import config as _config
    import data_class as _data_class  # Import data_class FIRST since task.py depends on it
    import task as _task
    import worker as _worker

    # Register sub-modules for direct imports
    sys.modules['mindspeed_llm.tasks.posttrain.rlxf.group_scheduler.config'] = _config
    sys.modules['mindspeed_llm.tasks.posttrain.rlxf.group_scheduler.task'] = _task
    sys.modules['mindspeed_llm.tasks.posttrain.rlxf.group_scheduler.worker'] = _worker
    sys.modules['mindspeed_llm.tasks.posttrain.rlxf.group_scheduler.data_class'] = _data_class

    # Set up mock structure with actual modules
    group_scheduler_gs.acceleration_limit_ratio = _config.acceleration_limit_ratio
    group_scheduler_gs.catch_up_ratio = _config.catch_up_ratio
    group_scheduler_gs.max_consecutive_reclaims = _config.max_consecutive_reclaims
    group_scheduler_gs.max_free_gpu_ratio = _config.max_free_gpu_ratio
    group_scheduler_gs.TaskTable = _task.TaskTable
    group_scheduler_gs.TaskConfig = _task.TaskConfig
    group_scheduler_gs.WorkerTable = _worker.WorkerTable
    group_scheduler_gs.WorkerInfo = _worker.WorkerInfo
    group_scheduler_gs.task = _task
    group_scheduler_gs.worker = _worker
    group_scheduler_gs.data_class = _data_class

    # Build module tree
    rlxf.group_scheduler = group_scheduler_gs
    posttrain.rlxf = rlxf
    tasks.posttrain = posttrain

    tasks_utils = ModuleType('mindspeed_llm.tasks_utils')
    tasks_utils.__path__ = []
    global_vars = ModuleType('mindspeed_llm.tasks_utils.global_vars')
    global_vars.NPUS_PER_NODE = 8
    tasks_utils.global_vars = global_vars

    mindspeed_llm.tasks = tasks
    mindspeed_llm.tasks_utils = tasks_utils

    # Register additional modules
    sys.modules['mindspeed_llm.tasks_utils'] = tasks_utils
    sys.modules['mindspeed_llm.tasks_utils.global_vars'] = global_vars

from group_scheduler import GroupScheduler, NPUS_PER_NODE
from data_class import TaskConfig as GSTaskConfig, TaskStateReport, ReclaimConfirm
from worker import WorkerInfo

# 导入日志器
from core.scheduler_logger import SchedulerLogger


@dataclass
class MockRevokeCallable:
    """Mock for TaskInfo.revoke.invoke() callback"""
    adapter: 'GSAdapter'
    task_id: str

    def __call__(self, num_instances: int):
        """直接调用此方法（兼容真实yr的revoke callable）"""
        return self.invoke(num_instances)

    def invoke(self, num_instances: int):
        """
        GS调用此方法来回收worker
        返回MockFuture，由yr.get()等待
        """
        from yr_mock import MockFuture

        future = MockFuture()
        # 立即执行并设置结果
        result = self.get_adapter()._handle_revoke_invoke(self.task_id, num_instances)
        future.set_result(result)
        return future

    def __deepcopy__(self, memo):
        """Support deepcopy by returning a new instance with same adapter and task_id"""
        # Deepcopy the adapter reference (it's the same object)
        return MockRevokeCallable(adapter=self.get_adapter(), task_id=self.task_id)

    def __getstate__(self):
        """Customize pickling - adapter reference is restored dynamically"""
        return {'task_id': self.task_id}

    def __setstate__(self, state):
        """Restore state"""
        self.task_id = state['task_id']
        # Adapter will be restored via a global registry

    def get_adapter(self):
        """Get adapter reference (for deepcopy support)"""
        return self.adapter


@dataclass
class MockAssignCallable:
    """Mock for TaskInfo.assign() callback"""
    adapter: 'GSAdapter'
    task_id: str

    def __call__(self, instance_placements: List[List[str]]):
        """直接调用（兼容真实yr的assign callable）"""
        return self.invoke(instance_placements)

    def invoke(self, instance_placements: List[List[str]]):
        """
        GS调用此方法来分配worker
        instance_placements: 每个实例的worker ID列表
        返回MockFuture，由yr.get()等待
        """
        from yr_mock import MockFuture

        future = MockFuture()
        # 立即执行并设置结果
        result = self.get_adapter()._handle_assign_invoke(self.task_id, instance_placements)
        future.set_result(result)
        return future

    def __deepcopy__(self, memo):
        """Support deepcopy by returning a new instance with same adapter and task_id"""
        # Deepcopy adapter reference (it's the same object)
        return MockAssignCallable(adapter=self.get_adapter(), task_id=self.task_id)

    def __getstate__(self):
        """Customize pickling - adapter reference is restored dynamically"""
        return {'task_id': self.task_id}

    def __setstate__(self, state):
        """Restore state"""
        self.task_id = state['task_id']
        # Adapter will be restored via a global registry

    def get_adapter(self):
        """Get adapter reference (for deepcopy support)"""
        return self.adapter


class GSAdapter:
    """GroupScheduler适配层，处理仿真器与真实GS的交互"""

    def __init__(
        self,
        machine_count: int,
        total_workers: int,
        logger: Optional[SchedulerLogger] = None
    ):
        """
        初始化GS适配器

        Args:
            machine_count: 机器数量
            total_workers: 总worker数（与仿真器对齐）
            logger: 调度日志器（可选）
        """
        self.machine_count = machine_count
        self.total_workers = total_workers
        self.allocated_workers = {}  # task_id -> [worker_id, ...]
        self.logger = logger

        # 应用所有 monkey patch（在初始化 GS 之前）
        self._apply_all_patches()

        # 初始化 GS（num_node 只用于内部计算）
        self.gs = GroupScheduler(machine_count)
        # 覆盖 num_workers 以对齐
        self.gs.num_workers = total_workers

        # 注入 logger 和 adapter 到 GS（用于 patch 内部访问）
        self.gs.logger = logger
        self.gs._adapter = self  # 引用适配器，用于同步

        # 调用 create（已被 patch 使用正确的 total_workers）
        self.gs.create()

        print(f"成功初始化GroupScheduler, 机器数: {machine_count}, 总worker数: {total_workers}")

        # 回调函数集合

        self.expand_callback: Optional[Callable[[str, List[str]], None]] = None
        self.reclaim_callback: Optional[Callable[[str, List[str]], None]] = None
        self.get_idle_workers_callback: Optional[Callable[[str], List[str]]] = None

        # 调度完成标志（用于同步）- 使用事件
        import threading
        self._scheduling_complete = threading.Event()
        self._scheduling_complete.set()  # 初始为已完成

    def _patch_loop(self):
        """
        Patch GroupScheduler.loop() 来修复同步问题

        原问题：schedule_tag 在 loop 开始时被清除，导致等待条件无法正确判断
        修复：在 execute 完成后才清除调度进行标志
        """
        from group_scheduler import GroupScheduler as GS
        original_class = GS._cls if hasattr(GS, '_cls') else GS
        original_loop = original_class.loop

        def patched_loop(self_gs):
            """修复版本的 loop - 使用超时等待避免退出死锁"""
            while self_gs.running_loop:
                with self_gs.loop_cv:
                    # 使用超时等待（500ms），定期检查 running_loop 状态
                    # 这样当 running_loop=False 时能及时退出
                    self_gs.loop_cv.wait_for(
                        lambda: self_gs.get_schedule_tag() or not self_gs.running_loop,
                        timeout=0.5
                    )
                    # 如果 running_loop 为 False，退出循环
                    if not self_gs.running_loop:
                        break
                    # 【关键】离开条件锁之前，不清除schedule_tag，保持调度进行状态

                # NOT clearing schedule_tag here - let it be cleared after work completes

                # 设置调度进行中标志（清除完成状态）
                adapter = getattr(self_gs, '_adapter', None)
                if adapter:
                    adapter._scheduling_complete.clear()

                try:
                    tasks_snapshot, workers_snapshot, plan = self_gs.compute_card_allocation()
                    self_gs.execute(tasks_snapshot, workers_snapshot, plan)
                    # Now clear schedule_tag after work completes
                    self_gs.set_schedule_tag(False)
                except Exception as e:
                    print(f"GS loop error: {e}")
                    import traceback
                    traceback.print_exc()
                    # Clear schedule_tag on error too
                    self_gs.set_schedule_tag(False)
                    # 错误时也要设置完成状态
                    if adapter:
                        adapter._scheduling_complete.set()
                    raise

                # 设置调度完成状态
                if adapter:
                    adapter._scheduling_complete.set()

        original_class.loop = patched_loop

    def _apply_all_patches(self):
        """应用所有 monkey patch 来修复和增强 GroupScheduler"""
        self._patch_create()
        self._patch_loop()  # 必须在其他 patch 之前应用
        self._patch_concurrent_reclaim()
        self._patch_concurrent_assign()
        self._patch_find_best_placement_global()
        self._patch_execute()
        self._patch_compute_card_allocation()
        self._patch_assess_range()
        self._patch_dont_starve()
        self._patch_feed_more()
        self._patch_do_assign()
        self._patch_trigger_schedule()
        self._patch_report_state()  # 添加voluntary_reclaim日志

    def _patch_trigger_schedule(self):
        """添加调试日志到 trigger_schedule"""
        from group_scheduler import GroupScheduler as GS
        original_class = GS._cls if hasattr(GS, '_cls') else GS
        original_trigger = original_class.trigger_schedule

        def wrapped_trigger(self_gs):
            return original_trigger(self_gs)

        original_class.trigger_schedule = wrapped_trigger

    def _patch_report_state(self):
        """
        Patch GroupScheduler.report_state() 添加voluntary_reclaim日志
        """
        from group_scheduler import GroupScheduler as GS
        original_class = GS._cls if hasattr(GS, '_cls') else GS
        original_report_state = original_class.report_state

        def wrapped_report_state(self_gs, state, need_schedule=True):
            """包装版本：记录voluntary_reclaim日志"""
            # 记录voluntary_reclaim日志
            if state.voluntary_reclaim and hasattr(self_gs, 'logger') and self_gs.logger:
                reclaimed_workers = [w.id for w in state.voluntary_reclaim.reclaimed_workers]
                task_info = self_gs.tasks.get_task(state.task_id)

                # 计算gpus_per_instance
                cards_per_instance = task_info.tp * task_info.pp if task_info else NPUS_PER_NODE

                self_gs.logger.log_instance_reclaim(
                    task_id=state.task_id,
                    worker_ids=reclaimed_workers,
                    total_instances=state.current_instances - state.voluntary_reclaim.reclaimed_instances,
                    gpus_per_instance=cards_per_instance
                )

                print(f"  [VOLUNTARY-RECLAIM] task={state.task_id}, "
                      f"reclaimed_instances={state.voluntary_reclaim.reclaimed_instances}, "
                      f"workers={reclaimed_workers}")

            # 调用原始实现
            return original_report_state(self_gs, state, need_schedule)

        original_class.report_state = wrapped_report_state

    def _patch_create(self):
        """
        Patch GroupScheduler.create() 使用正确的 total_workers

        原始问题：
            - create() 使用 self.config.npus_per_node * self.config.num_node
            - 与仿真器的 total_workers 可能不一致
        """
        from group_scheduler import GroupScheduler as GS
        # GS 被 @yr.instance 装饰器包装，需要从 _cls 获取原始类
        original_class = GS._cls if hasattr(GS, '_cls') else GS
        original_create = original_class.create

        def patched_create(self_gs):
            """使用 GSAdapter 注入的总worker数"""
            total_workers = getattr(self_gs, '_adapter_total_workers',
                                self_gs.config.npus_per_node * self_gs.config.num_node)

            # 记录日志
            if hasattr(self_gs, 'logger') and self_gs.logger:
                self_gs.logger.log_phase_start('create')

            # 创建 worker
            for id in range(total_workers):
                # 指定到shared节点
                local_rank = id % self_gs.config.npus_per_node
                node_type: str = "shared"
                node_id: int = id // self_gs.config.npus_per_node
                worker_info = WorkerInfo(node_type, node_id, local_rank)
                worker_info.set_id(str(id))
                self_gs.workers.register(worker_info)
                yr.kv_set(worker_info.id, pickle.dumps(worker_info))

            if hasattr(self_gs, 'logger') and self_gs.logger:
                duration = self_gs.logger.log_phase_end('create')
                print(f"  GS.create() 创建了 {total_workers} workers, 耗时 {duration:.3f}s")

        GS.create = patched_create

    def _patch_concurrent_reclaim(self):
        """
        Patch GroupScheduler.concurrent_reclaim() 修复 task_id.revoke bug

       原问题（第451行）：
            reclaim_jobs = [task_id.revoke.invoke(num_instances) for ...]
            AttributeError: 'str' object has no attribute 'revoke'

        修复：从 tasks.get_task(task_id) 获取真实的 TaskInfo 对象
        """
        from group_scheduler import GroupScheduler as GS
        from data_class import TaskStateReport

        # 获取原始类
        original_class = GS._cls if hasattr(GS, '_cls') else GS
        original_concurrent_reclaim = original_class.concurrent_reclaim

        def fixed_concurrent_reclaim(self_gs, reclaim_tasks):
            """
            修复版本：从 tasks.get_task() 获取 TaskInfo
            """
            # 记录日志
            if hasattr(self_gs, 'logger') and self_gs.logger:
                self_gs.logger.log_phase_start('concurrent_reclaim')

            # 修复：获取真实的 TaskInfo 对象
            reclaim_jobs = [
                self_gs.tasks.get_task(task_id).revoke.invoke(num_instances)
                for task_id, num_instances in reclaim_tasks
            ]

            reclaim_results = yr.get(reclaim_jobs)
            for reclaim_result in reclaim_results:
                # 维护本地的tasks状态信息
                self_gs.report_state(reclaim_result, need_schedule=False)

            if hasattr(self_gs, 'logger') and self_gs.logger:
                duration = self_gs.logger.log_phase_end('concurrent_reclaim')
                print(f"  GS.concurrent_reclaim() 回收了 {len(reclaim_tasks)} 个任务, 耗时 {duration:.3f}s")

        original_class.concurrent_reclaim = fixed_concurrent_reclaim

    def _patch_concurrent_assign(self):
        """
        Patch GroupScheduler.concurrent_assign() 修复 task_id.assign bug

        原始问题（第458行）：
            assign_jobs = [task_id.assign.invoke(instance_placements) for ...]
            AttributeError: 'str' object has no attribute 'assign'

        修复：从 tasks.get_task(task_id) 获取真实的 TaskInfo 对象
        """
        from group_scheduler import GroupScheduler as GS
        from data_class import TaskStateReport

        # 获取原始类
        original_class = GS._cls if hasattr(GS, '_cls') else GS
        original_concurrent_assign = original_class.concurrent_assign

        def fixed_concurrent_assign(self_gs, placements):
            """
            修复版本：从 tasks.get_task() 获取 TaskInfo
            """
            # 记录日志
            if hasattr(self_gs, 'logger') and self_gs.logger:
                self_gs.logger.log_phase_start('concurrent_assign')

            # 修复：获取真实的 TaskInfo 对象
            assign_jobs = [
                self_gs.tasks.get_task(task_id).assign.invoke(instance_placements)
                for task_id, instance_placements in placements
            ]

            assign_results = yr.get(assign_jobs)
            for assign_result in assign_results:
                # 维护本地的tasks状态信息
                self_gs.report_state(assign_result, need_schedule=False)

            if hasattr(self_gs, 'logger') and self_gs.logger:
                duration = self_gs.logger.log_phase_end('concurrent_assign')
                print(f"  GS.concurrent_assign() 分配了 {len(placements)} 个任务, 耗时 {duration:.3f}s")

        original_class.concurrent_assign = fixed_concurrent_assign

    def _patch_find_best_placement_global(self):
        """
        Patch GroupScheduler.find_best_placement_global() 添加日志

        包装原始实现，记录输入和输出（控制台 + JSONL文件）
        """
        from group_scheduler import GroupScheduler as GS
        original_class = GS._cls if hasattr(GS, '_cls') else GS
        original_find = original_class.find_best_placement_global

        def wrapped_find(self_gs, tasks_snapshot, workers_snapshot, allocation_requests):
            """包装版本：添加日志后调用原始实现"""
            # 记录日志
            if hasattr(self_gs, 'logger') and self_gs.logger:
                self_gs.logger.log_phase_start('find_best_placement_global')
                # 格式化分配请求
                alloc_str = ", ".join([f"{tid}: +{num}实例" for tid, num in allocation_requests])
                print(f"  GS.find_best_placement_global() 分配请求: [{alloc_str}]")

            # 调用原始实现（复用所有真实逻辑）
            # 修复接口不匹配：添加 get_idle_worker_per_machine 方法
            if not hasattr(workers_snapshot, 'get_idle_worker_per_machine'):
                workers_snapshot.get_idle_worker_per_machine = workers_snapshot.idle_workers_per_machine

            original_placements = original_find(self_gs, tasks_snapshot, workers_snapshot, allocation_requests)

            # 过滤空placement：保持原始格式 [(task_id, [[w1, w2], [w3, w4]]), ...]
            filtered_placements = []
            for task_id, instance_placements in original_placements:
                # 过滤掉空的placement，但保持实例分离的格式
                valid_instance_placements = []
                for placement in instance_placements:
                    if placement:  # 跳过空placement
                        valid_instance_placements.append(placement)
                if valid_instance_placements:
                    filtered_placements.append((task_id, valid_instance_placements))
                # 只有当请求了实例但全部失败时才打印警告
                elif instance_placements:  # 有placement请求但都为空
                    print(f"  警告: 任务{task_id}请求了实例但GPU不足，无法分配")

            placements = filtered_placements
            # 打印详细的 placements 信息（显示具体worker ID），只在有结果时打印
            if filtered_placements:
                print(f"  GS.find_best_placement_global() 分配结果:")
                for task_id, instance_placements in filtered_placements:
                    print(f"    {task_id}: {len(instance_placements)} 个实例")
                    for i, placement in enumerate(instance_placements):
                        # placement是worker_id列表，直接显示
                        workers_str = ", ".join(placement[:8]) + ("..." if len(placement) > 8 else "")
                        print(f"      实例{i}: workers=[{workers_str}] ({len(placement)}张卡)")
            else:
                print(f"  GS.find_best_placement_global() 分配结果: 无有效分配（GPU不足）")

            # 记录日志
            if hasattr(self_gs, 'logger') and self_gs.logger:
                duration = self_gs.logger.log_phase_end('find_best_placement_global')
                print(f"  GS.find_best_placement_global() 结果: {[(tid, len(pls)) for tid, pls in placements]}, 耗时 {duration:.3f}s")
                # 记录到JSONL文件
                cycle_id = getattr(self_gs, '_current_cycle_id', self_gs.logger.cycle_count)
                self_gs.logger.log_find_best_placement_global(allocation_requests, placements, cycle_id=cycle_id)

            return placements

        original_class.find_best_placement_global = wrapped_find

    def _patch_execute(self):
        """
        Patch GroupScheduler.execute() 添加日志并处理同步

        包装原始实现，记录输入和输出，并确保调度完成通知
        """
        from group_scheduler import GroupScheduler as GS
        original_class = GS._cls if hasattr(GS, '_cls') else GS
        original_execute = original_class.execute

        def wrapped_execute(self_gs, tasks_snapshot, workers_snapshot, plan):
            """包装版本：添加日志后调用原始实现"""
            # 记录日志
            if hasattr(self_gs, 'logger') and self_gs.logger:
                self_gs.logger.log_phase_start('execute')
                print(f"  GS.execute() plan: {plan}")

            # 调用原始实现（复用所有真实逻辑）
            result = original_execute(self_gs, tasks_snapshot, workers_snapshot, plan)

            # 记录日志
            if hasattr(self_gs, 'logger') and self_gs.logger:
                duration = self_gs.logger.log_phase_end('execute')
                print(f"  GS.execute() 完成, 耗时 {duration:.3f}s")

            return result

        original_class.execute = wrapped_execute

    def _patch_compute_card_allocation(self):
        """
        Patch GroupScheduler.compute_card_allocation() 添加日志并修复deepcopy问题

        关键改进：
        1. 使用手动deepcopy避免循环引用（TaskInfo的revoke/assign回调包含adapter引用）
        2. 记录完整的四阶段算法输出
        3. 记录task_table_snapshot状态
        """
        from group_scheduler import GroupScheduler as GS
        original_class = GS._cls if hasattr(GS, '_cls') else GS

        def safe_deepcopy_tasks_and_workers(original_tasks, original_workers):
            """
            手动deepcopy，跳过mock callable对象（revoke/assign）

            TaskInfo中的revoke和assign是MockRevokeCallable/MockAssignCallable，
            包含adapter引用，导致循环引用。手动复制时不复制这些回调。
            """
            # Copy workers
            new_workers = type(original_workers)()
            new_workers._worker_table = copy.deepcopy(original_workers._worker_table)
            new_workers._idle_workers = original_workers._idle_workers.copy()

            # Copy tasks
            new_tasks = type(original_tasks)()
            for task_id, task_info in original_tasks._task_table.items():
                # Create new TaskInfo without mock callables
                import task as task_module
                from data_class import TaskConfig

                # Create a minimal config
                config = TaskConfig(
                    task_id=task_info.task_id,
                    base_instances=task_info.base_instances,
                    tp=task_info.tp,
                    pp=task_info.pp,
                    samples_per_round=task_info.samples_per_round,
                    total_samples=task_info.total_samples
                )

                new_task_info = task_module.TaskInfo(config)

                # Copy state attributes (但不复制 revoke/assign 回调)
                new_task_info._has_state = task_info._has_state
                new_task_info.done_samples = task_info.done_samples
                new_task_info.done_rounds = task_info.done_rounds
                new_task_info.elapsed_time_sec = task_info.elapsed_time_sec
                new_task_info.remaining_samples = task_info.remaining_samples
                new_task_info.current_instances = task_info.current_instances
                new_task_info.idle_instances = task_info.idle_instances
                new_task_info.busy_instances = task_info.busy_instances
                new_task_info.in_rollout_phase = task_info.in_rollout_phase
                new_task_info._used_workers = copy.deepcopy(task_info._used_workers)

                # 保留原始的 assign 和 revoke 回调（直接引用）
                new_task_info.assign = getattr(task_info, 'assign', None)
                new_task_info.revoke = getattr(task_info, 'revoke', None)

                new_tasks._task_table[task_id] = new_task_info

            return new_tasks, new_workers

        def wrapped_compute(self_gs):
            """包装版本：使用安全deepcopy并记录完整的调度决策过程"""
            # 读取触发来源（如果有）并清除标记
            trigger_source = getattr(self_gs, '_trigger_source', None)
            if hasattr(self_gs, '_trigger_source'):
                delattr(self_gs, '_trigger_source')

            # 开始新的调度周期
            if hasattr(self_gs, 'logger') and self_gs.logger:
                cycle_id = self_gs.logger.start_cycle()
                self_gs._current_cycle_id = cycle_id
                self_gs.logger.log_phase_start('compute_card_allocation')

            # 使用安全deepcopy替代原来的 copy.deepcopy(self.tasks) 和 copy.deepcopy(self.workers)
            with self_gs.update_lock:
                tasks_snapshot, workers_snapshot = safe_deepcopy_tasks_and_workers(
                    self_gs.tasks, self_gs.workers
                )

            # 记录 task_table_snapshot（关键日志：用于分析调度决策依据）
            if hasattr(self_gs, 'logger') and self_gs.logger:
                # 获取 num_rounds_map
                num_rounds_map = None
                if hasattr(self_gs, '_adapter') and self_gs._adapter.get_num_rounds_map_callback:
                    num_rounds_map = self_gs._adapter.get_num_rounds_map_callback()
                self_gs.logger.log_task_table_snapshot(
                    tasks_snapshot,
                    cycle_id=cycle_id,
                    trigger_source=trigger_source,
                    num_rounds_map=num_rounds_map
                )

            # 调用四阶段算法（各自包含日志）
            delta_card_ranges = self_gs.assess_range(tasks_snapshot)
            plan, excess_cards = self_gs.dont_starve(tasks_snapshot, delta_card_ranges, workers_snapshot.num_idle_worker())
            if excess_cards:
                plan = self_gs.feed_more(tasks_snapshot, delta_card_ranges, plan, excess_cards)

            # 记录最终结果
            if hasattr(self_gs, 'logger') and self_gs.logger:
                self_gs.logger.log_compute_allocation_result(tasks_snapshot, workers_snapshot, plan, cycle_id=cycle_id)
                self_gs.logger.log_phase_end('compute_card_allocation')

            return tasks_snapshot, workers_snapshot, plan

        original_class.compute_card_allocation = wrapped_compute

    def _patch_assess_range(self):
        """
        Patch GroupScheduler.assess_range() 添加日志并修复TaskTable.values()问题

        记录每个任务的 min_cards 和 max_cards 范围
        """
        from group_scheduler import GroupScheduler as GS
        from mindspeed_llm.tasks.posttrain.rlxf.group_scheduler.config import (
            acceleration_limit_ratio,
            catch_up_ratio,
        )

        original_class = GS._cls if hasattr(GS, '_cls') else GS

        def patched_assess(self_gs, tasks_snapshot):
            """修复版本：使用get_all_tasks()替代values()"""
            # 记录日志
            if hasattr(self_gs, 'logger') and self_gs.logger:
                self_gs.logger.log_phase_start('assess_range')

            delta_card_ranges = []

            # 使用get_all_tasks()替代values()
            for task in tasks_snapshot.get_all_tasks():
                cards_per_instance = task.tp * task.pp
                # 处理None值（刚注册的任务还没有状态报告）
                current_instances = task.current_instances or 0
                idle_instances = task.idle_instances or 0
                busy_instances = task.busy_instances or 0
                remaining_samples = task.remaining_samples or 0
                in_rollout_phase = task.in_rollout_phase or True

                current_cards = current_instances * cards_per_instance
                max_total_cards = acceleration_limit_ratio * task.base_instances * cards_per_instance


                # 情况0: 不在rollout阶段
                if not in_rollout_phase:
                    min_cards = -idle_instances * cards_per_instance
                    max_cards = 0
                    delta_card_ranges.append((min_cards, max_cards))

                # 情况1: 有剩余样本，但忙实例数没到基线 → 必须增加
                elif busy_instances < task.base_instances and remaining_samples > 0:
                    min_cards = (catch_up_ratio * task.base_instances - busy_instances) * cards_per_instance
                    max_cards = max_total_cards - current_cards
                    delta_card_ranges.append((min_cards, max_cards))

                # 情况2: 有空闲实例，且没有剩余样本 → 必须回收
                elif remaining_samples == 0:
                    must_reclaim_instances = idle_instances
                    if busy_instances > task.base_instances:
                        must_reclaim_instances += (busy_instances - task.base_instances)
                    min_cards = -must_reclaim_instances * cards_per_instance
                    max_cards = 0
                    delta_card_ranges.append((min_cards, max_cards))

                # 情况3: 忙实例数超过基线 → 可以回收超额部分
                elif busy_instances > task.base_instances:
                    min_cards = -(busy_instances - task.base_instances) * cards_per_instance
                    max_cards = max_total_cards - current_cards
                    delta_card_ranges.append((min_cards, max_cards))

                # 情况4: 其他 → 不强制调整
                else:
                    min_cards = 0
                    max_cards = max_total_cards - current_cards
                    delta_card_ranges.append((min_cards, max_cards))

            # 记录日志
            if hasattr(self_gs, 'logger') and self_gs.logger:
                self_gs.logger.log_assess_range_result(delta_card_ranges, cycle_id=getattr(self_gs, '_current_cycle_id', self_gs.logger.cycle_count))
                duration = self_gs.logger.log_phase_end('assess_range')
                print(f"  GS.assess_range() 完成: {delta_card_ranges}, 耗时 {duration:.3f}s")

            return delta_card_ranges

        original_class.assess_range = patched_assess

    def _patch_dont_starve(self):
        """
        Patch GroupScheduler.dont_starve() 添加日志

        计算满足 min > 0 的任务，生成初步 plan
        """
        from group_scheduler import GroupScheduler as GS
        original_class = GS._cls if hasattr(GS, '_cls') else GS
        original_dont_starve = original_class.dont_starve

        def wrapped_dont_starve(self_gs, tasks_snapshot, delta_card_ranges, free_card_count):
            """包装版本：添加日志后调用原始实现"""
            # 记录日志
            if hasattr(self_gs, 'logger') and self_gs.logger:
                self_gs.logger.log_phase_start('dont_starve')
                print(f"  GS.dont_starve() 开始: free_card_count={free_card_count}")

            # 调用原始实现
            plan, excess_cards = original_dont_starve(self_gs, tasks_snapshot, delta_card_ranges, free_card_count)

            # 记录日志
            if hasattr(self_gs, 'logger') and self_gs.logger:
                cycle_id = getattr(self_gs, '_current_cycle_id', self_gs.logger.cycle_count)
                self_gs.logger.log_dont_starve_result(plan, excess_cards, cycle_id=cycle_id)
                duration = self_gs.logger.log_phase_end('dont_starve')
                print(f"  GS.dont_starve() 完成: plan={plan}, excess_cards={excess_cards}, 耗时 {duration:.3f}s")

            return plan, excess_cards

        original_class.dont_starve = wrapped_dont_starve

    def _patch_feed_more(self):
        """
        Patch GroupScheduler.feed_more() 添加日志

        把富余卡分配给收益最大的任务
        """
        from group_scheduler import GroupScheduler as GS
        original_class = GS._cls if hasattr(GS, '_cls') else GS
        original_feed_more = original_class.feed_more

        def wrapped_feed_more(self_gs, tasks_snapshot, delta_card_ranges, plan, excess_cards):
            """包装版本：添加日志后调用原始实现"""
            # 记录日志
            if hasattr(self_gs, 'logger') and self_gs.logger:
                self_gs.logger.log_phase_start('feed_more')
                print(f"  GS.feed_more() 开始: excess_cards={excess_cards}")

            # 调用原始实现
            result_plan = original_feed_more(self_gs, tasks_snapshot, delta_card_ranges, plan, excess_cards)

            # 记录日志
            if hasattr(self_gs, 'logger') and self_gs.logger:
                cycle_id = getattr(self_gs, '_current_cycle_id', self_gs.logger.cycle_count)
                self_gs.logger.log_feed_more_result(result_plan, cycle_id=cycle_id)
                duration = self_gs.logger.log_phase_end('feed_more')
                print(f"  GS.feed_more() 完成: plan={result_plan}, 耗时 {duration:.3f}s")

            return result_plan

        original_class.feed_more = wrapped_feed_more

    def _patch_do_assign(self):
        """
        Patch GroupScheduler.do_assign() 添加日志

        执行分配（被情况A的强制分配和情况B复用）
        """
        from group_scheduler import GroupScheduler as GS
        original_class = GS._cls if hasattr(GS, '_cls') else GS
        original_do_assign = original_class.do_assign

        def wrapped_do_assign(self_gs, tasks_snapshot, workers_snapshot, plan):
            """包装版本：添加日志后调用原始实现"""
            # 记录日志
            if hasattr(self_gs, 'logger') and self_gs.logger:
                self_gs.logger.log_phase_start('do_assign')
                print(f"  GS.do_assign() 开始: plan={plan}")

            # 调用原始实现
            original_do_assign(self_gs, tasks_snapshot, workers_snapshot, plan)

            # 记录日志
            if hasattr(self_gs, 'logger') and self_gs.logger:
                duration = self_gs.logger.log_phase_end('do_assign')
                print(f"  GS.do_assign() 完成, 耗时 {duration:.3f}s")

        original_class.do_assign = wrapped_do_assign

    def set_callbacks(
        self,
        expand_callback: Callable[[str, List[str]], None],
        reclaim_callback: Callable[[str, List[str]], None],
        get_idle_workers_callback: Callable[[str], List[str]],
        get_state_callback: Optional[Callable[[str], Any]] = None,
        get_num_rounds_map_callback: Optional[Callable[[], Dict[str, int]]] = None
    ) -> None:
        """设置仿真器的回调函数"""
        self.expand_callback = expand_callback
        self.reclaim_callback = reclaim_callback
        self.get_idle_workers_callback = get_idle_workers_callback
        self.get_state_callback = get_state_callback
        self.get_num_rounds_map_callback = get_num_rounds_map_callback

    def register_task(self, task_config: Any) -> bool:
        """
        注册任务到GroupScheduler

        【正确流程】直接调用gs.register_task()，GS会自动：
        1. 注册任务到TaskTable
        2. 设置mock callbacks
        3. 触发调度（trigger_schedule）
        4. 调度完成后通过assign回调分配实例

        Args:
            task_config: TaskConfig对象（来自models/task_config.py）

        Returns:
            是否注册成功
        """
        gs_config = GSTaskConfig(
            task_id=task_config.task_id,
            base_instances=task_config.base_instances,
            tp=task_config.tp,
            pp=task_config.pp,
            samples_per_round=task_config.samples_per_round,
            total_samples=task_config.total_samples
        )

        # 【修改】使用 gs.register_task() 自动触发调度
        # gs.register_task() 内部会调用 trigger_schedule()
        success = self.gs.register_task(gs_config)

        # Patch TaskInfo with mock callbacks (在注册之后立即设置）
        if success:
            task_info = self.gs.tasks.get_task(task_config.task_id)
            task_info.revoke = MockRevokeCallable(self, task_config.task_id)
            task_info.assign = MockAssignCallable(self, task_config.task_id)

        return success

    def report_state(self, state: TaskStateReport, need_schedule: bool = True) -> bool:
        """
        上报任务状态到GS

        如果 need_schedule=True，GS会自动触发调度（通过事件驱动loop线程）
        调度完成后，需要等待loop线程完成所有操作

        Args:
            state: TaskStateReport
            need_schedule: 是否触发调度

        Returns:
            是否成功
        """
        # 注意：日志记录已在 simulator.py 中完成，此处不再重复记录
        # 避免 total_rounds 显示为 "unknown" 的问题

        # 设置触发来源标记（用于日志关联）
        if need_schedule:
            self.gs._trigger_source = f"report_state:{state.task_id}"

        # 调用GS的report_state，让GS自动触发调度
        result = self.gs.report_state(state, need_schedule=need_schedule)

        # 【不等待调度完成，让它异步执行】
        # 原因：等待调度可能导致复杂的线程同步问题
        # 仿真器可以通过后续的状态上报和分配回调来获取结果

        return result

    def _wait_for_gs_scheduling(self, timeout: float = 10.0) -> None:
        """
        等待GS的loop线程完成调度

        通过检查 `_scheduling_complete` 事件来判断调度是否完成

        Args:
            timeout: 超时时间（秒）
        """
        start_time = time.time()

        # 等待调度完成（scheduling_complete被设置）
        while not self._scheduling_complete.is_set():
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"GS scheduling timeout after {timeout}s")
            remaining = timeout - elapsed
            # 等待一小段时间，避免CPU空转
            time.sleep(0.01)

    def _handle_revoke_invoke(self, task_id: str, num_instances: int) -> TaskStateReport:
        """
        处理GS发起的revoke请求

        返回TaskStateReport，包含voluntary_reclaim信息
        """
        # 获取任务信息以获取TP*PP
        task_info = self.gs.tasks.get_task(task_id)
        workers_per_instance = task_info.tp * task_info.pp if task_info else NPUS_PER_NODE

        # 获取任务的idle workers
        if self.get_idle_workers_callback:
            idle_workers = self.get_idle_workers_callback(task_id)
        else:
            idle_workers = []

        # 确定要回收的workers数量
        workers_to_reclaim = min(len(idle_workers), num_instances * workers_per_instance)
        reclaimed_workers = idle_workers[:workers_to_reclaim]

        # 【修复】如果没有可回收的worker，返回不含voluntary_reclaim的状态报告
        if not reclaimed_workers:
            state_report = TaskStateReport(
                task_id=task_id,
                done_samples=task_info.done_samples or 0,
                done_rounds=task_info.done_rounds or 0,
                elapsed_time_sec=task_info.elapsed_time_sec or 0.0,
                remaining_samples=task_info.remaining_samples or 0,
                current_instances=task_info.current_instances or 0,
                idle_instances=task_info.idle_instances or 0,
                busy_instances=task_info.busy_instances or 0,
                in_rollout_phase=task_info.in_rollout_phase or False,
                voluntary_reclaim=None
            )
            return state_report

        # 【修复】先通知仿真器执行reclaim（更新仿真器状态）
        if self.reclaim_callback:
            self.reclaim_callback(task_id, reclaimed_workers)

        # 更新allocated_workers
        if task_id in self.allocated_workers:
            worker_set = set(reclaimed_workers)
            self.allocated_workers[task_id] = [
                w for w in self.allocated_workers[task_id] if w not in worker_set
            ]

        # 创建ReclaimConfirm
        worker_info_list = []
        for worker_id in reclaimed_workers:
            wi = self.gs.workers._worker_table.get(worker_id)
            if wi:
                worker_info_list.append(wi)

        actual_reclaimed_instances = len(reclaimed_workers) // workers_per_instance
        reclaim_confirm = ReclaimConfirm(
            task_id=task_id,
            reclaimed_instances=actual_reclaimed_instances,
            reclaimed_workers=worker_info_list
        )

        # 【修复】从仿真器获取实际状态，而不是GS内部状态
        if self.get_state_callback:
            state_report = self.get_state_callback(task_id)
            # 添加voluntary_reclaim信息
            state_report.voluntary_reclaim = reclaim_confirm
        else:
            # Fallback: 使用GS内部状态（不应发生）
            task_info = self.gs.tasks.get.get(task_id)
            state_report = TaskStateReport(
                task_id=task_id,
                done_samples=task_info.done_samples or 0,
                done_rounds=task_info.done_rounds or 0,
                elapsed_time_sec=task_info.elapsed_time_sec or 0.0,
                remaining_samples=task_info.remaining_samples or 0,
                current_instances=task_info.current_instances or 0,
                idle_instances=task_info.idle_instances or 0,
                busy_instances=task_info.busy_instances or 0,
                in_rollout_phase=task_info.in_rollout_phase or False,
                voluntary_reclaim=reclaim_confirm
            )

        return state_report

    def _handle_assign_invoke(self, task_id: str, instance_placements: List[List[str]]) -> TaskStateReport:
        """
        处理GS发起的assign请求

        instance_placements: 每个实例的worker ID列表
        """
        # 收集所有worker IDs
        all_worker_ids = []
        for placement in instance_placements:
            all_worker_ids.extend(placement)

        # 【修复】更新allocated_workers（而不是累加）
        # 原因：extend会导致重复分配，造成total_instances计算错误
        if task_id not in self.allocated_workers:
            self.allocated_workers[task_id] = []
        self.allocated_workers[task_id].extend(all_worker_ids)
        # 去重：可能GS会多次分配相同的worker
        self.allocated_workers[task_id] = list(set(self.allocated_workers[task_id]))

        # 通知仿真器执行分配 - 为每个实例分别调用
        if self.expand_callback:
            for placement in instance_placements:
                if placement:  # 跳过空placement
                    self.expand_callback(task_id, placement)
                else:
                    print(f"  警告: 任务{task_id}收到空placement，跳过")

        # 从GS的idle列表中移除新分配的worker
        self.gs.workers.del_workers_from_idle(all_worker_ids)

        # 【修复】从仿真器获取实际状态，而不是GS内部状态
        if self.get_state_callback:
            state_report = self.get_state_callback(task_id)
        else:
            # Fallback: 使用GS内部状态（不应发生）
            task_info = self.gs.tasks.get_task(task_id)
            state_report = TaskStateReport(
                task_id=task_id,
                done_samples=task_info.done_samples or 0,
                done_rounds=task_info.done_rounds or 0,
                elapsed_time_sec=task_info.elapsed_time_sec or 0.0,
                remaining_samples=task_info.remaining_samples or 0,
                current_instances=task_info.current_instances or 0,
                idle_instances=task_info.idle_instances or 0,
                busy_instances=task_info.busy_instances or 0,
                in_rollout_phase=task_info.in_rollout_phase or False,
                voluntary_reclaim=None
            )

        return state_report

    def get_task_workers(self, task_id: str) -> List[str]:
        """
        获取任务当前分配的worker IDs

        Args:
            task_id: 任务ID

        Returns:
            worker ID列表
        """
        return self.allocated_workers.get(task_id, [])

    def cleanup(self):
        """
        清理GS资源，停止后台线程

        必须在仿真结束时调用，否则loop_thread会一直运行导致程序无法退出
        """
        print("[GSAdapter] Cleaning up GroupScheduler resources...")
        if hasattr(self.gs, 'destroy'):
            self.gs.destroy()

        # 等待loop线程结束（最多等待2秒）
        if hasattr(self.gs, 'loop_thread') and self.gs.loop_thread.is_alive():
            self.gs.loop_thread.join(timeout=2.0)
            if self.gs.loop_thread.is_alive():
                print("[GSAdapter] Warning: loop thread did not exit gracefully")
            else:
                print("[GSAdapter] loop thread stopped")
        else:
            print("[GSAdapter] loop thread already stopped")
