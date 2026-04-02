"""Metrics模块 - 指标收集"""

from typing import List, Tuple, Dict, Any


class MetricsCollector:
    """指标收集器"""

    def __init__(self):
        self.scheduling_traces: List[Any] = []
        self.utilization_samples: List[Tuple[float, float]] = []
        self.task_states: Dict[str, List[Tuple[float, Any]]] = {}
        self.events: List[Any] = []

    def record_scheduling_decision(self, trace: Any) -> None:
        """记录调度决策"""
        self.scheduling_traces.append(trace)

    def record_utilization(self, time: float, utilization: float) -> None:
        """记录GPU利用率"""
        self.utilization_samples.append((time, utilization))

    def record_task_state(self, time: float, task_id: str, state: Any) -> None:
        """记录任务状态"""
        if task_id not in self.task_states:
            self.task_states[task_id] = []
        self.task_states[task_id].append((time, state))

    def record_event(self, event: Any) -> None:
        """记录事件"""
        self.events.append(event)

    def get_scheduling_traces(self) -> List[Any]:
        return self.scheduling_traces

    def get_utilization_curve(self) -> List[Tuple[float, float]]:
        return self.utilization_samples

    def get_avg_utilization(self) -> float:
        """计算平均利用率"""
        if not self.utilization_samples:
            return 0.0
        return sum(u for _, u in self.utilization_samples) / len(self.utilization_samples)

    def get_task_states_history(self) -> Dict[str, List[Tuple[float, Any]]]:
        return self.task_states

    def get_events(self) -> List[Any]:
        return self.events
