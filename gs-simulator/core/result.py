"""SimulationResult模块 - 仿真结果"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any


@dataclass
class SchedulingTrace:
    """调度决策轨迹"""
    timestamp: float
    task_id: str
    action: str  # "assign" or "reclaim"
    instances_count: int
    reason: str


@dataclass
class SimulationResult:
    """完整仿真结果"""
    test_case_name: str
    total_simulation_time: float
    task_completion_times: Dict[str, float] = field(default_factory=dict)
    scheduling_traces: List[SchedulingTrace] = field(default_factory=list)
    scheduling_trace_count: int = 0  # 有效调度决策数（可从logger获取）
    gpu_utilization_curve: List[Tuple[float, float]] = field(default_factory=list)
    task_states_history: Dict[str, List[Tuple[float, Any]]] = field(default_factory=dict)
    event_history: List[Any] = field(default_factory=list)

    @property
    def total_completion_time(self):
        return self.total_simulation_time

    @property
    def avg_task_completion_time(self):
        if not self.task_completion_times:
            return 0.0
        return sum(self.task_completion_times.values()) / len(self.task_completion_times)

    @property
    def slowest_task(self):
        if not self.task_completion_times:
            return ("", 0.0)
        return max(self.task_completion_times.items(), key=lambda x: x[1])

    @property
    def fastest_task(self):
        if not self.task_completion_times:
            return ("", 0.0)
        return min(self.task_completion_times.items(), key=lambda x: x[1])

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，便于JSON序列化"""
        return {
            "test_case_name": self.test_case_name,
            "total_simulation_time": self.total_simulation_time,
            "task_completion_times": self.task_completion_times,
            "avg_task_completion_time": self.avg_task_completion_time,
            "slowest_task": self.slowest_task,
            "fastest_task": self.fastest_task,
            "scheduling_trace_count": self.scheduling_trace_count,  # 使用直接统计的计数
            "avg_utilization": self._calc_avg_utilization(),
        }

    def _calc_avg_utilization(self) -> float:
        """计算平均GPU利用率"""
        if not self.gpu_utilization_curve:
            return 0.0
        return sum(u for _, u in self.gpu_utilization_curve) / len(self.gpu_utilization_curve)

    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=2)

    def save_to_file(self, filepath: str) -> None:
        """保存到文件"""
        with open(filepath, 'w') as f:
            f.write(self.to_json())
