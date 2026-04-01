"""
调度决策日志器

支持 JSON + 控制台双输出，记录 Group调度器的完整决策过程
"""

import json
import time
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional


@dataclass
class TaskSnapshot:
    """任务状态快照"""
    task_id: str
    base_instances: int
    current_instances: int
    idle_instances: int
    busy_instances: int
    tp: int
    pp: int
    cards_per_instance: int
    remaining_samples: int
    in_rollout_phase: bool


@dataclass
class DeltaCardRange:
    """卡数调整范围"""
    task_id: str
    min_cards: int  # 必须=回收，正数=必须增加
    max_cards: int  # 最多可以调整的卡数


@dataclass
class SchedulingDecision:
    """调度决策"""
    task_id: str
    plan_delta: int  # 卡数调整量（正数=增加，负数=回收）


@dataclass
class PlacementDecision:
    """放置决策"""
    task_id: str
    instance_placements: List[List[str]]  # 每个实例的worker_id列表
    total_instances: int


@dataclass
class SchedulingCycleLog:
    """完整调度周期日志"""
    cycle_id: int
    timestamp_sec: float
    timestamp_str: str

    # 输入状态
    task_snapshots: List[TaskSnapshot]

    # 算度结果
    delta_card_ranges: List[DeltaCardRange]

    # 调度结果
    plan: List[int]

    # 放置结果
    placement_decisions: List[PlacementDecision]

    # 性能指标
    phase_durations: Dict[str, float]
    total_duration_sec: float


class SchedulerLogger:
    """调度决策日志器"""

    def __init__(
        self,
        log_dir: str = "results",
        log_prefix: str = "scheduler_trace",
        enable_console: bool = False  # 默认关闭控制台打印
    ):
        """
        Args:
            log_dir: 日志文件目录
            log_prefix: 日志文件前缀
            enable_console: 是否在控制台打印
        """
        self.log_dir = log_dir
        self.log_prefix = log_prefix
        self.enable_console = enable_console

        # 确保目录存在
        os.makedirs(log_dir, exist_ok=True)

        # 创建日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%M")
        self.log_file = os.path.join(log_dir, f"{log_prefix}_{timestamp}.jsonl")

        self.cycle_count = 0
        self.effective_scheduling_count = 0  # 有效调度决策计数（有实际分配或回收）
        self._current_cycle_start: Optional[float] = None
        self._phase_start_times: Dict[str, float] = {}

    def start_cycle(self) -> int:
        """开始一个新的调度周期，返回 cycle_id"""
        self.cycle_count += 1
        self._current_cycle_start = time.time()
        self._phase_start_times = {}
        return self.cycle_count

    def log_phase_start(self, phase_name: str) -> None:
        """记录阶段开始时间"""
        self._phase_start_times[phase_name] = time.time()

    def log_phase_end(self, phase_name: str) -> float:
        """记录阶段结束时间，返回耗时"""
        if phase_name not in self._phase_start_times:
            return 0.0
        duration = time.time() - self._phase_start_times[phase_name]
        del self._phase_start_times[phase_name]
        return duration

    def log_scheduling_cycle(self, log: SchedulingCycleLog) -> None:
        """记录完整的调度周期"""
        self._write_jsonl(asdict(log))

    def log_simulation_step(self, current_time: float, step: int):
        """记录仿真步（时间推进）"""
        log_entry = {
            'timestamp': self.get_timestamp(),
            'event': 'simulation_step',
            'step': step,
            'current_time': current_time
        }
        self._write_jsonl(log_entry)

    def log_task_initial_allocation(self, task_id: str, base_instances: int, worker_ids: List[str], gpus_per_instance: int):
        """记录任务初始分配（注册时）"""
        log_entry = {
            'timestamp': self.get_timestamp(),
            'event': 'task_initial_allocation',
            'task_id': task_id,
            'base_instances': base_instances,
            'allocated_instances': len(worker_ids) // gpus_per_instance,
            'worker_ids': worker_ids,
            'gpus_per_instance': gpus_per_instance
        }
        self._write_jsonl(log_entry)

    def log_instance_expand(self, task_id: str, worker_ids: List[str], total_instances: int, gpus_per_instance: int):
        """记录实例扩容（GS分配）"""
        log_entry = {
            'timestamp' : self.get_timestamp(),
            'event': 'instance_expand',
            'task_id': task_id,
            'worker_ids': worker_ids,
            'added_instances': len(worker_ids) // gpus_per_instance,
            'total_instances': total_instances,
            'gpus_per_instance': gpus_per_instance
        }
        self._write_jsonl(log_entry)

    def log_instance_reclaim(self, task_id: str, worker_ids: List[str], total_instances: int, gpus_per_instance: int):
        """记录实例缩容（GS回收）"""
        log_entry = {
            'timestamp': self.get_timestamp(),
            'event': 'instance_reclaim',
            'task_id': task_id,
            'worker_ids': worker_ids,
            'removed_instances': len(worker_ids) // gpus_per_instance,
            'total_instances': total_instances,
            'gpus_per_instance': gpus_per_instance
        }
        self._write_jsonl(log_entry)

    def get_timestamp(self) -> str:
        """获取当前时间戳字符串"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def _write_jsonl(self, data: Dict) -> None:
        """写入日志条目到JSONL文件"""
        import json
        with open(self.log_file, "a") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def log_report_state(self, task_id, state, need_schedule: bool, simulation_step: Optional[int] = None, current_time: Optional[float] = None, num_rounds: Optional[int] = None) -> None:
        """记录任务状态上报"""
        log_entry = {
            'timestamp': self.get_timestamp(),
            'event': 'report_state',
            'task_id': task_id,
            'state': {
                'done_samples': state.done_samples,
                'remaining_samples': state.remaining_samples,
                'current_instances': state.current_instances,
                'idle_instances': state.idle_instances,
                'busy_instances': state.busy_instances,
                'in_rollout_phase': state.in_rollout_phase,
                'has_voluntary_reclaim': state.voluntary_reclaim is not None,
                # 新增轮次进度
                'done_rounds': state.done_rounds,
                'total_rounds': num_rounds if num_rounds is not None else 'unknown',
                'current_round': state.done_rounds + 1 if num_rounds is not None else 'unknown'
            },
            'need_schedule': need_schedule
        }
        # 只有在值不为 None 时才添加这些字段
        if simulation_step is not None:
            log_entry['simulation_step'] = simulation_step
        if current_time is not None:
            log_entry['current_time'] = current_time
        self._write_jsonl(log_entry)

    def log_round_progress(self, task_id: str, done_rounds: int, num_rounds: int,
                           done_samples: int, remaining_samples: int,
                           current_round: int, samples_in_current_round: int,
                           simulation_step: Optional[int] = None,
                           current_time: Optional[float] = None) -> None:
        """记录轮次进度状态"""
        log_entry = {
            'timestamp': self.get_timestamp(),
            'event': 'round_progress',
            'task_id': task_id,
            'round_status': {
                'current_round': current_round,           # 当前轮次 (1-based)
                'total_rounds': num_rounds,               # 总轮数
                'done_rounds': done_rounds,               # 已完成轮数
                'remaining_rounds': num_rounds - done_rounds,  # 剩余轮数
                'done_samples': done_samples,             # 已完成样本总数
                'remaining_samples': remaining_samples,   # 剩余样本总数
                'samples_in_current_round': samples_in_current_round  # 当前轮次剩余样本
            }
        }
        if simulation_step is not None:
            log_entry['simulation_step'] = simulation_step
        if current_time is not None:
            log_entry['current_time'] = current_time
        self._write_jsonl(log_entry)

    def log_round_transition(self, task_id: str, from_round: int, to_round: int,
                             num_rounds: int, instance_count: int,
                             simulation_step: Optional[int] = None,
                             current_time: Optional[float] = None) -> None:
        """记录轮次切换事件"""
        log_entry = {
            'timestamp': self.get_timestamp(),
            'event': 'round_transition',
            'task_id': task_id,
            'transition': {
                'from_round': from_round,
                'to_round': to_round,
                'total_rounds': num_rounds,
                'remaining_rounds': num_rounds - to_round,
                'instance_count': instance_count  # 进入下一轮时的实例数
            }
        }
        if simulation_step is not None:
            log_entry['simulation_step'] = simulation_step
        if current_time is not None:
            log_entry['current_time'] = current_time
        self._write_jsonl(log_entry)

    def log_task_table_snapshot(self, task_table, cycle_id=None, trigger_source=None, num_rounds_map: Optional[Dict[str, int]] = None):
        """记录 task_table 快照状态"""
        snapshot = {
            'timestamp': self.get_timestamp(),
            'cycle_id': cycle_id if cycle_id is not None else self.cycle_count,
            'event': 'task_table_snapshot',
            'task_table_snapshot': {
                task_id: {
                    'task_id': task.task_id,
                    'base_instances': task.base_instances,
                    'current_instances': task.current_instances,
                    'idle_instances': task.idle_instances,
                    'busy_instances': task.busy_instances,
                    'in_rollout_phase': task.in_rollout_phase,
                    'remaining_samples': task.remaining_samples,
                    'done_samples': task.done_samples,
                    'done_rounds': task.done_rounds,
                    # 新增轮次进度
                    'total_rounds': num_rounds_map.get(task_id, 'unknown') if num_rounds_map else 'unknown',
                    'current_round': task.done_rounds + 1,
                    'remaining_rounds': (num_rounds_map.get(task_id, 0) - task.done_rounds) if num_rounds_map else 'unknown'
                }
                for task_id, task in task_table._task_table.items()
            }
        }
        # 添加触发来源（如果提供了）
        if trigger_source is not None:
            snapshot['trigger_source'] = trigger_source
        self._write_jsonl(snapshot)

    def log_assess_range_result(self, delta_card_ranges, cycle_id=None):
        """记录 assess_range 阶段结果"""
        log_entry = {
            'timestamp': self.get_timestamp(),
            'cycle_id': cycle_id if cycle_id is not None else self.cycle_count,
            'event': 'assess_range_result',
            'delta_card_ranges': delta_card_ranges  # List of (min_cards, max_cards)
        }
        self._write_jsonl(log_entry)

    def log_dont_starve_result(self, plan, excess_cards, cycle_id=None):
        """记录 dont_starve 阶段结果"""
        log_entry = {
            'timestamp': self.get_timestamp(),
            'cycle_id': cycle_id if cycle_id is not None else self.cycle_count,
            'event': 'dont_starve_result',
            'plan': plan,  # List of card adjustments
            'excess_cards': excess_cards
        }
        self._write_jsonl(log_entry)

    def log_feed_more_result(self, plan, cycle_id=None):
        """记录 feed_more 阶段结果"""
        log_entry = {
            'timestamp': self.get_timestamp(),
            'cycle_id': cycle_id if cycle_id is not None else self.cycle_count,
            'event': 'feed_more_result',
            'plan': plan,  # Final plan after feed_more
        }
        self._write_jsonl(log_entry)

    def log_compute_allocation_result(self, tasks_snapshot, workers_snapshot, plan, cycle_id=None):
        """记录 compute_card_allocation 最终结果"""
        total_instances = sum(t.current_instances for t in tasks_snapshot._task_table.values())
        total_cards = sum(t.current_instances * t.tp * t.pp for t in tasks_snapshot._task_table.values())
        idle_workers = workers_snapshot.num_idle_worker()
        total_workers = workers_snapshot.num_worker()

        # 计算分配和回收卡数
        allocate_cards = sum(p for p in plan if p > 0)
        reclaim_cards = sum(-p for p in plan if p < 0)

        # 统计有效调度决策（有实际分配或回收）
        if allocate_cards > 0 or reclaim_cards > 0:
            self.effective_scheduling_count += 1

        log_entry = {
            'timestamp': self.get_timestamp(),
            'cycle_id': cycle_id if cycle_id is not None else self.cycle_count,
            'event': 'compute_allocation_result',
            'total_tasks': len(tasks_snapshot._task_table),
            'idle_workers': idle_workers,
            'total_workers': total_workers,
            'plan': plan,
            'summary': {
                'total_plan_cards': sum(abs(p) for p in plan),
                'allocate_cards': allocate_cards,
                'reclaim_cards': reclaim_cards,
            }
        }
        self._write_jsonl(log_entry)

    def log_find_best_placement_global(self, allocation_requests, placements, cycle_id=None):
        """
        记录 find_best_placement_global 结果

        allocation_requests: [(task_id, num_instances), ...]
        placements: [(task_id, [[worker_id], ...], ...]]

        详细日志：记录每个实例的 worker IDs
        """
        # 详细日志：记录每个实例的 worker IDs
        placements_detail = {}
        for tid, placements_list in placements:
            placements_detail[tid] = []
            for instance_idx, placement in enumerate(placements_list):
                placements_detail[tid].append({
                    'instance_idx': instance_idx,
                    'num_gpus': len(placement),
                    'worker_ids': placement
                })

        log_entry = {
            'timestamp': self.get_timestamp(),
            'cycle_id': cycle_id if cycle_id is not None else self.cycle_count,
            'event': 'find_best_placement_global',
            'allocation_requests': allocation_requests,
            'placements': placements,  # 保存原始的placements数据
            'placements_detail': placements_detail
        }
        self._write_jsonl(log_entry)

    def close(self) -> None:
        """关闭日志器"""
        print(f"\n调度日志已保存到: {self.log_file}")
