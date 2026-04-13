import threading
from types import SimpleNamespace
from typing import Any
import copy
import yr
import pickle
import heapq
from mindspeed_llm.tasks.posttrain.rlxf.group_scheduler.config import (
    acceleration_limit_ratio,
    catch_up_ratio,
    max_consecutive_reclaims,
    max_free_gpu_ratio,
)

from mindspeed_llm.tasks.posttrain.rlxf.group_scheduler.task import TaskTable
from mindspeed_llm.tasks.posttrain.rlxf.group_scheduler.worker import WorkerTable, WorkerInfo
from mindspeed_llm.tasks.posttrain.rlxf.group_scheduler.data_class import TaskConfig, TaskStateReport

NPUS_PER_NODE = 8

# TODO: 指定线程数
@yr.instance
class GroupScheduler:
    """组调度器类，负责管理Worker和Job之间的资源分配"""
    def __init__(self, num_node: int):
        # set config
        self.config = {}
        self.config["npus_per_node"] = NPUS_PER_NODE
        # self.config["num_node"] = len(yr.resources())
        self.config["num_node"] = num_node
        self.config = SimpleNamespace(**self.config)
        
        # workers
        self.workers: WorkerTable = WorkerTable()

        # tasks
        self.tasks: TaskTable = TaskTable()

        # scheduler
        self.update_lock = threading.Lock()
        self.loop_cv = threading.Condition()
        self.running_loop: bool = True
        self.schedule_tag: bool = False
        self.consecutive_reclaim_count: int = 0
        self.start_loop()
        

        
        
# ------------- Public API ---------------
    def create(self):
        """创建 DsV3Instance workers, master节点调用
        """
        # yr 集群获取 config
        self.num_workers = self.config.npus_per_node * self.config.num_node
        for id in range(self.num_workers):
            # 指定到shared节点
            local_rank = id % self.config.npus_per_node
            node_type: str = "shared"
            node_id: int = id // self.config.npus_per_node
            worker_info = WorkerInfo(node_type, node_id, local_rank)
            worker_info.set_id(str(id))
            self.workers.register(worker_info)
            yr.kv_set(worker_info.id, pickle.dumps(worker_info))


    def destroy(self, timeout_ms = None ) -> list:
        self.running_loop = False
        worker_ids = self.workers.get_worker_list()
        for worker_id in worker_ids:
            yr.kv_del(worker_id)
        # 通知task释放？
        pass

    def register_task(self, config: TaskConfig) -> bool:
        if self.tasks.register(config):
            self.trigger_schedule()
            return True
        return False

    def report_state(self, state: TaskStateReport, need_schedule: bool = True) -> bool:
        task_id = state.task_id
        if not self.tasks.check_task_exist(task_id):
            return False

        with self.update_lock:
            if state.voluntary_reclaim:
                # A task releases workers
                reclaimed_workers = state.voluntary_reclaim.reclaimed_workers
                worker_ids = [worker_info.id for worker_info in reclaimed_workers]
                self.workers.add_workers_to_idle(worker_ids)
                self.tasks.del_workers_from_used(task_id, worker_ids)

            self.tasks.update_task_info(state)
        if need_schedule:
            self.trigger_schedule()
        return True

# ------------- Private API ---------------

    def trigger_schedule(self):
        with self.loop_cv:
            self.set_schedule_tag(True)
            self.loop_cv.notify()

    def start_loop(self):
        self.loop_thread = threading.Thread(target=self.loop)
        self.loop_thread.start()

    def loop(self):
        while self.running_loop:
            with self.loop_cv:
                self.loop_cv.wait_for(lambda: self.get_schedule_tag())
                self.set_schedule_tag(False)
            tasks_snapshot, workers_snapshot, plan = self.compute_card_allocation()
            self.execute(tasks_snapshot, workers_snapshot, plan)
    
    def get_schedule_tag(self) -> bool:
        return self.schedule_tag

    def set_schedule_tag(self, tag: bool):
        self.schedule_tag = tag
    

    # TODO: 优化 execute逻辑，后续可改为边回收边释放，减少空泡
    def execute(self, tasks_snapshot, workers_snapshot, plan):
        """
        执行调度plan

        分两种情况：
        1. 有回收：执行回收，然后根据回收限制决定是触发新调度还是继续做分配
        2. 只有分配：直接执行分配
        """
        # ---------- 第一步: 判断是否有回收 ----------
        has_reclaim = any(p < 0 for p in plan)
        has_assign = any(p > 0 for p in plan)

        tasks = tasks_snapshot.get_all_tasks()
        n_tasks = len(tasks)

        if has_reclaim:
            # ---------- 情况A: 有回收 ----------
            # 收集要回收的任务
            reclaim_tasks = []
            for i in range(n_tasks):
                if plan[i] < 0:
                    task = tasks[i]
                    # 回收前验证：确保仍有可回收的实例
                    if task.idle_instances > 0 or task.busy_instances > task.base_instances:
                        cards_to_reclaim = -plan[i]
                        instances_to_reclaim = cards_to_reclaim // (task.tp * task.pp)
                        reclaim_tasks.append( (task.task_id, instances_to_reclaim) )

            # 并发发送回收请求，等待全部完成
            # 这是最耗时的步骤（2～10秒左右）
            reclaimed_gpus_list = self.concurrent_reclaim(reclaim_tasks)

            # 更新连续回收计数器
            self.consecutive_reclaim_count += 1

            # ---------- 检查回收限制，决定下一步 ----------
            # TODO: 传入worker快照
            free_gpus = workers_snapshot.num_idle_worker()
            total_gpus = workers_snapshot.num_worker()
            free_gpu_ratio = free_gpus / total_gpus
            force_assign = (
                self.consecutive_reclaim_count >= max_consecutive_reclaims
                or free_gpu_ratio >= max_free_gpu_ratio
            )

            new_tasks_snapshot, new_workers_snapshot, plan = self.compute_card_allocation()
            if force_assign:
                # ---------- 安全阀触发：强制做分配，不触发新调度 ----------
                # 重置连续回收计数器
                self.consecutive_reclaim_count = 0

                # 执行分配（复用情况B的逻辑）
                self.do_assign(new_tasks_snapshot, new_workers_snapshot, plan)
            else:
                # ---------- 正常情况：触发新的完整调度 ----------
                self.execute(new_tasks_snapshot, new_workers_snapshot, plan)
            return

        if has_assign:
            # ---------- 情况B: 只有分配，直接执行 ----------
            # 重置连续回收计数器
            self.consecutive_reclaim_count = 0
            self.do_assign(tasks_snapshot, workers_snapshot, plan)

    def compute_card_allocation(self):
        with self.update_lock:
            tasks_snapshot = copy.deepcopy(self.tasks)
            workers_snapshot = copy.deepcopy(self.workers)
        delta_card_ranges = self.assess_range(tasks_snapshot)
        plan, excess_cards = self.dont_starve(tasks_snapshot, delta_card_ranges, workers_snapshot.num_idle_worker())
        if excess_cards:
            plan = self.feed_more(tasks_snapshot, delta_card_ranges, plan, excess_cards)
        return tasks_snapshot, workers_snapshot, plan
    
    def assess_range(self, tasks_snapshot):
        """
        计算每个任务的卡数调整范围 [min_cards, max_cards]

        返回:
        delta_card_ranges[i] = (min_cards, max_cards)
            - min_cards: 必须调整的卡数（负数表示必须回收，正数表示必须增加）
            - max_cards: 最多可以调整的卡数
        """
        delta_card_ranges = []

        for task in tasks_snapshot.get_all_tasks():
            cards_per_instance = task.tp * task.pp
            current_cards = task.current_instances * cards_per_instance # 当前任务实际占用的总卡数
            max_total_cards = acceleration_limit_ratio * task.base_instances * cards_per_instance # 该任务允许占用的最大卡数上限。
            # 情况0: 不在rollout阶段（包括刚注册还没开始的任务）→ 所有空闲实例都可以回收，且不再分配
            min_cards = max_cards = 0
            if not task.in_rollout_phase:
                min_cards = -task.idle_instances * cards_per_instance
                max_cards = 0
                
            # 情况1: 有剩余样本，但忙实例数没到基线 → 必须增加
            elif task.busy_instances < task.base_instances and task.remaining_samples > 0:
                min_cards = (catch_up_ratio * task.base_instances - task.busy_instances) * cards_per_instance
                max_cards = max_total_cards - current_cards

            # 情况2: 有空闲实例，且没有剩余样本 → 必须回收（包括空闲实例和超额忙实例）
            elif task.remaining_samples == 0:
                must_reclaim_instances = task.idle_instances
                if task.busy_instances > task.base_instances:
                    must_reclaim_instances += (task.busy_instances - task.base_instances)
                min_cards = -must_reclaim_instances * cards_per_instance
                max_cards = 0

            # 情况3: 忙实例数超过基线 → 可以回收超额部分
            elif task.busy_instances > task.base_instances:
                min_cards = -(task.busy_instances - task.base_instances) * cards_per_instance
                max_cards = max_total_cards - current_cards

            # 情况4: 其他 → 不强制调整
            else:
                min_cards = 0
                max_cards = max_total_cards - current_cards

            min_cards = min(min_cards, max_cards)
            delta_card_ranges.append((min_cards, max_cards) )
        return delta_card_ranges


    def dont_starve(self, tasks_snapshot, delta_card_ranges, free_card_count):
        """
        优先满足min > 0的任务，生成初步plan

        返回:
        plan: 每个任务的卡数调整量（正数=增加，负数=回收）
        excess_cards: 富余的卡数（可以继续分配）
        """
        tasks = tasks_snapshot.get_all_tasks()
        n_tasks = len(tasks)
        plan = [0] * n_tasks

        # ---------- 第一步: 收集需求 ----------
        needy_tasks = []  # 需要增加卡的任务 (min > 0)
        reclaimable_tasks = []  # 可以回收卡的任务
        for i in range(n_tasks):
            min_cards, max_cards = delta_card_ranges[i]

            if min_cards > 0:
                needy_tasks.append( (i, min_cards) )
            elif min_cards < 0:
                # 可以回收，按优先级排序
                task = tasks[i]
                cards_per_instance = task.tp * task.pp
                assert min_cards % cards_per_instance == 0
                if task.idle_instances > 0:
                    priority = 0  # 最高优先级：有空闲实例
                else:
                    priority = 1  # 次优先级：忙实例超基线
                reclaimable_cards = -min_cards
                # 按 priority 降序排序（0 最高优先级， 排在最后），其次按 reclaimable_cards 降序排序
                heapq.heappush(reclaimable_tasks, (priority, -reclaimable_cards, i) )

        # ---------- 第二步: 满足 needy_tasks ----------
        for (i, needed_cards) in needy_tasks:
            while needed_cards > 0:
                cards_per_instance = tasks[i].tp * tasks[i].pp

                # 先看有没有足够的空闲卡
                if free_card_count >= cards_per_instance:
                    plan[i] += cards_per_instance
                    free_card_count -= cards_per_instance
                    needed_cards -= cards_per_instance
                else:
                    # 需要回收
                    if not reclaimable_tasks:
                        break

                    priority, reclaimable_cards, j = heapq.heappop(reclaimable_tasks)
                    reclaimable_cards = -reclaimable_cards
                    cards_per_instance_j = tasks[j].tp * tasks[j].pp

                    # 回收一个实例
                    reclaim_amount = min(reclaimable_cards, cards_per_instance_j)
                    plan[j] -= reclaim_amount
                    free_card_count += reclaim_amount

                    # 如果还能回收，放回到队尾重新排序
                    remaining = reclaimable_cards - reclaim_amount
                    if remaining > 0:
                        heapq.heappush(reclaimable_tasks, (priority, -remaining, j) )

        # ---------- 第三步: 计算富余卡 ----------
        excess_cards = free_card_count

        return plan, excess_cards


    def feed_more(self, tasks_snapshot, delta_card_ranges, plan, excess_cards):
        """
        把富余卡分配给收益最大的任务
        """
        if excess_cards <= 0:
            return plan
        # 计算每个任务的收益分
        tasks = tasks_snapshot.get_all_tasks()
        n_tasks = len(tasks)

        task_scores = []
        for i in range(n_tasks):
            min_cards, max_cards = delta_card_ranges[i]
            task = tasks[i]

            # 跳过已经满足min的任务，或者不能再增加的任务
            if max_cards <= 0:
                continue

            # 计算收益分
            score = tasks_snapshot.compute_allocation_score(task.task_id, plan[i])
            if score > 0:
                heapq.heappush(task_scores, (-score, i) ) # 负号用于降序


        # 分配富余卡
        while excess_cards > 0 and task_scores:
            neg_score, i = heapq.heappop(task_scores)
            task = tasks[i]
            cards_per_instance = task.tp * task.pp
            min_cards, max_cards = delta_card_ranges[i]

            # 检查上限
            current_cards = task.current_instances * cards_per_instance + plan[i]
            max_allowed = task.current_instances * cards_per_instance + max_cards
            if current_cards + cards_per_instance > max_allowed:
                continue
            
            # 如果富余卡足够分配可以处理一个实例的卡数
            if excess_cards >= cards_per_instance:
                plan[i] += cards_per_instance
                excess_cards -= cards_per_instance

                # 如果还有收益，继续排队
                new_score = tasks_snapshot.compute_allocation_score(task.task_id, plan[i])
                if new_score > 0:
                    new_current_cards = current_cards + cards_per_instance
                    if new_current_cards < max_allowed:
                        heapq.heappush(task_scores, (-new_score, i) )
        return plan



    def do_assign(self, tasks_snapshot, workers_snapshot, plan):
        """
        执行分配（被情况A的强制分配和情况B复用）
        """
        # 收集分配需求
        tasks = tasks_snapshot.get_all_tasks()
        n_tasks = len(tasks)

        allocation_requests = []
        for i in range(n_tasks):
            if plan[i] > 0:
                task = tasks[i]
                # 验证：必须在rollout阶段且有剩余样本
                if task.in_rollout_phase and task.remaining_samples > 0:
                    cards_to_allocate = plan[i]
                    instances_to_allocate = cards_to_allocate // (task.tp * task.pp)
                    allocation_requests.append( (task.task_id, instances_to_allocate) )

        placements = self.find_best_placement_global(tasks_snapshot, workers_snapshot, allocation_requests)

        # 并发发送分配请求
        self.concurrent_assign(placements)

    def find_best_placement_global(self, tasks_snapshot, workers_snapshot, allocation_requests) -> list[tuple[Any, list[WorkerInfo]]]:
        """
        全局优化GPU放置

        allocation_requests: [(task_id, num_instances), ...]
        free_by_machine: {machine_id: [worker_id, worker_id ...], ...}

        返回: [(task_id, [placement1, placement2, ...]), ...]
            其中每个placement是[worker_id, ...]，长度为tp*pp
        """
        placements = []
        used_gpus = set()
        # TODO: 传入worker和task快照
        free_by_machine = workers_snapshot.get_idle_worker_per_machine()
        left_allocation_requests = []
        # 第一轮：优先单机器放置
        for (task_id, num_instances) in allocation_requests:
            task = tasks_snapshot.get_task(task_id)
            worker_per_instance = task.tp * task.pp
            instance_placements = []
            for _ in range(num_instances):
                # 找有足够GPU的机器
                placed = False
                for machine_id in free_by_machine:
                    available = [worker_id for worker_id in free_by_machine[machine_id]
                                if worker_id not in used_gpus]
                    if len(available) >= worker_per_instance:
                        # 选择这个机器
                        selected_gpus = available[:worker_per_instance]
                        instance_placements.append(selected_gpus)
                        used_gpus.update(selected_gpus)
                        break
                else:
                    left_allocation_requests.append( (task_id, num_instances - len(instance_placements) ) )

                    break

            placements.append( (task_id, instance_placements) )

        # 第二轮：处理剩余的跨机器放置
        # TODO: 改进为启发式算法，尽量减少GPU碎片化
        start = 0
        left_idle_workers = [worker_id for worker_id in workers_snapshot.idle_workers if worker_id not in used_gpus]
        for (task_id, num_instances) in left_allocation_requests:
            task = tasks_snapshot.get_task(task_id)
            worker_per_instance = task.tp * task.pp
            instance_placements = []
            for  _ in range(num_instances):
                if start + worker_per_instance > len(left_idle_workers):
                    break
                selected_gpus = left_idle_workers[start: start + worker_per_instance]
                start += worker_per_instance
                instance_placements.append(selected_gpus)
            placements.append( (task_id, instance_placements) )
        return placements

    # assume that task_id.assign/revoke() returns TaskStateReport 
    def concurrent_reclaim(self, reclaim_tasks):
        reclaim_jobs = [task_id.reclaim.invoke(num_instances) for task_id, num_instances in reclaim_tasks]
        reclaim_results = yr.get(reclaim_jobs)
        for reclaim_result in reclaim_results:
            # 维护本地的tasks状态信息
            self.report_state(reclaim_result, need_schedule=False)

    def concurrent_assign(self, placements):
        assign_jobs = [task_id.assign.invoke(instance_placements) for task_id, instance_placements in placements]
        assign_results = yr.get(assign_jobs)
        for assign_result in assign_results:
            # 维护本地的tasks状态信息
            self.report_state(assign_result, need_schedule=False)



def test():
    """测试函数，用于测试GroupScheduler的基本功能"""
    yr.init()
    gs = GroupScheduler.invoke(len(yr.resources()))
    print(gs, type(gs))
    yr.get(gs.create.invoke())
    num_workers = len(yr.get(gs.workers.invoke()))
    print(f"Created {num_workers} workers")
    yr.finalize()
    return

if __name__ == "__main__":
    test()