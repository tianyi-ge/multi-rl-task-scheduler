from typing import Any
class WorkerInfo:
    def __init__(self, node_type: str, node_id: int, local_rank: int):
        self.machine_id = f"{node_type}_{node_id}"
        self.node_type = node_type
        self.node_id = node_id
        self.local_rank = local_rank
        self._id = None

    def set_id(self, worker_id: Any):
        self._id = worker_id

    @property
    def id(self):
        return self._id

class WorkerTable:
    def __init__(self):
        self._worker_table: dict[Any, WorkerInfo] = {}
        self._idle_workers: list[Any] = []
        # 似乎find_best_placement_global 调用频率低于add/del_workers,
        # 可以考虑不长期维护 idle_workers_per_machine，而是临时构建
        self._idle_workers_per_machine: dict[int, list[Any]] = {}
    
    @property
    def idle_workers(self):
        return self._idle_workers

    def get_worker_list(self):
        return list(self._worker_table.keys())

    def num_worker(self):
        return len(self._worker_table)
    
    def num_idle_worker(self):
        return len(self._idle_workers)
    
    def idle_workers_per_machine(self) -> dict[int, list[Any]]:
        return self._idle_workers_per_machine.copy()

    def register(self, worker_info: WorkerInfo):
        worker_id = worker_info.id
        if worker_id in self._worker_table:
            return False
        machine_id = worker_info.machine_id
        self._worker_table[worker_id] = worker_info
        self._idle_workers.append(worker_id)
        if machine_id not in self._idle_workers_per_machine:
            self._idle_workers_per_machine[machine_id] = []
        self._idle_workers_per_machine[machine_id].append(worker_id)
        return True
    
    def add_workers_to_idle(self, workers: list):
        """将workers添加到idle列表
        Args:
            workers: 要添加的worker列表
        """
        for worker_id in workers:
            if not (worker_id in self._worker_table and worker_id not in self._idle_workers):
                continue
            self._idle_workers.append(worker_id)
            # 更新机器维度的idle列表
            machine_id = self._worker_table[worker_id].machine_id
            if worker_id not in self._idle_workers_per_machine[machine_id]:
                self._idle_workers_per_machine[machine_id].append(worker_id)

    def del_workers_from_idle(self, workers: list, enable_placement_optimization: bool = False):
        """从idle列表删除指定的workers
        Args:
            workers: 要删除的worker列表
            enable_placement_optimization: 是否启用放置优化（暂时未使用）
        """
        workers_set = set(workers)

        # 从全局idle列表中移除 - 使用列表推导式
        self._idle_workers = [w for w in self._idle_workers if w not in workers_set]

        # 从机器维度的idle列表中移除
        for worker_id in workers:
            if not (worker_id in self._worker_table):
                continue
            machine_id = self._worker_table[worker_id].machine_id
            if worker_id in self._idle_workers_per_machine[machine_id]:
                self._idle_workers_per_machine[machine_id].remove(worker_id)