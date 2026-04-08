from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from .models import TaskConfig, TaskStateReport


class InferScheduler(ABC):
    @abstractmethod
    def reclaim(self, num_instances: int) -> TaskStateReport:
        """Reclaim instances and return the latest task state snapshot."""

    @abstractmethod
    def assign(self, placements: List[List[str]]) -> TaskStateReport:
        """Assign new workers and return the latest task state snapshot."""


class GroupSchedulerProtocol(ABC):
    @abstractmethod
    def register_task(self, config: TaskConfig, scheduler: InferScheduler) -> bool:
        """Register a task and its task-local scheduler."""

    @abstractmethod
    def report_state(self, report: TaskStateReport) -> None:
        """Accept an asynchronous state report from a task."""

    @abstractmethod
    def unregister_task(self, task_id: str) -> None:
        """Remove a task and release its workers."""
