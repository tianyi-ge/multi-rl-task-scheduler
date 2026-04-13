---
name: GRPO GPU调度器仿真器设计
description: 基于真实调度器实现的仿真器，用于验证调度算法正确性和性能基准测试
type: design
---

# GRPO GPU调度器仿真器设计

## 1. 概述

### 1.1 设计目标

设计一个基于真实调度器实现的仿真器，用于：

1. **算法正确性验证**：验证4阶段调度逻辑（assess_range、dont_starve、feed_more、execute）的正确性
2. **性能基准测试**：对比不同调度策略的端到端完成时间
3. **调度效果评估**：评估调度器在处理长尾问题上的效果

### 1.2 设计原则

- **基于真实实现**：直接调用GroupScheduler和RlxfScheduler的真实代码
- **固定步长+事件驱动**：固定1秒步长推进，中间穿插事件处理，保证仿真效率
- **可复现性**：所有随机性使用固定种子，推理时间预计算，确保仿真结果可重复
- **可配置性**：支持多种任务测试场景和参数配置
- **真实性**：模拟真实的资源竞争、长尾行为、延迟等

---

## 2. 整体架构

### 2.1 核心组件

```
                    ┌─────────────────────────────────────┐
                    │          Simulator                    │
                    │  (固定步长引擎，时间推进控制器)          │
                    └──────────────┬──────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
          ▼                        ▼                        ▼
┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
│   ClusterModel     │   │  TaskModel (xN)   │   │ GroupScheduler    │
│  - GPU资源池        │   │  - 模拟RL推理任务  │   │  (真实实现)       │
│  - 拓扑结构         │   │  - rollout/train │   │  - 全局调度决策    │
│  - 分配/回收操作   │   │  - 状态上报      │   │  - 4阶段算法      │
└───────────────────┘   └───────────────────┘   └───────────────────┘
          │                        │                        │
          │                        │                        │
          └────────────────────────┼────────────────────────┘
                                   │
                            事件交互层
                    ┌──────────────▼──────────────────────┐
                    │          EventLoop                   │
                    │  - GPU分配事件 (耗时~60s)             │
                    │  - GPU回收事件 (耗时2-10s)            │
                    │  - 样本完成事件                       │
                    │  - 状态上报事件                       │
                    │  - 调度触发事件                       │
                    └─────────────────────────────────────┘
```

### 2.2 仿真场景流程

```
1. 初始化阶段
   - 创建ClusterModel：配置机器数量、每机GPU数、拓扑结构
   - 创建N个TaskModel：配置tp、pp、base_instances、样本数
   - 初始化分配：按base_instances分配GPU给各任务
   - 验证初始约束：所有任务base_instances刚好占满集群，无剩余GPU

2. 仿真运行
   ┌─────────────────────────────────────────────────────────────┐
   │  while 有未完成任务或有待处理事件:                           │
   │    1. 推进一个固定步长（1秒）                                │
   │       - 在推进过程中，如果到达了事件时间，立即处理事件               │
   │       - 事件处理会触发调度决策                                 │
   │    2. 各TaskModel在当前时间点执行推理                        │
   │       - 实例处理样本（使用预计算的推理时间）                   │
   │       - 长尾发生：部分实例空闲，部分实例忙                      │
   │       - 实例从共享样本队列取样本                               │
   │    3. 记录当前时刻的GPU利用率                                 │
   │    4. GroupScheduler触发调度决策（真实实现）                      │
   │       - assess_range → dont_starve → feed_more → execute     │
   │       - reclaim事件（并发，延迟2-10s后完成）                    │
   │       - assign事件（并发，延迟~60s冷启动后才能工作）            │
   └─────────────────────────────────────────────────────────────┘

3. 结束阶段
   - 输出：每个任务的端到端完成时间
   - 输出：调度决策历史
   - 输出：资源利用率曲线
   - 输出：长尾行为分析
```

---

## 3. 核心数据结构

### 3.1 集群模型

```python
@dataclass
class Machine:
    """单个机器模型"""
    machine_id: int
    gpu_count: int
    gpu_states: List[int] = field(default_factory=list)  # 0=空闲, 1=已分配

@dataclass
class GPUPlacement:
    """GPU位置信息"""
    machine_id: int
    gpu_id: int

@dataclass
class ClusterModel:
    """集群模型，维护GPU资源池"""
    machines: List[Machine]
    free_gpus: List[GPUPlacement] = field(default_factory=list)
    
    @classmethod
    def from_config(cls, config: ClusterConfig) -> ClusterModel:
        """从配置创建集群"""
        machines = []
        for i in range(config.machine_count):
            machines.append(Machine(
                machine_id=i,
                gpu_count=config.gpus_per_machine,
                gpu_states=[0] * config.gpus_per_machine
            ))
        cluster = cls(machines=machines)
        cluster._init_free_gpus()
        return cluster
    
    def _init_free_gpus(self) -> None:
        """初始化空闲GPU列表"""
        for machine in self.machines:
            for gpu_id in range(machine.gpu_count):
                self.free_gpus.append(GPUPlacement(machine.machine_id, gpu_id))
    
    def allocate_instance(self, tp: int, pp: int) -> Optional[List[GPUPlacement]]:
        """
        分配一个实例的GPU（tp*pp张卡）
        
        优先级：
        1. 同机分配（拓扑感知）
        2. 跨机分配
        
        返回：GPU列表（成功）或None（失败）
        """
        cards_needed = tp * pp
        return self._try_same_machine(cards_needed) or self._try_cross_machine(cards_needed)
    
    def _try_same_machine(self, cards_needed: int) -> Optional[List[GPUPlacement]]:
        """尝试同机分配"""
        for machine in self.machines:
            available_gpus = [
                GPUPlacement(machine.machine_id, i)
                for i in range(machine.gpu_count)
                if machine.gpu_states[i] == 0
            ]
            if len(available_gpus) >= cards_needed:
                selected = available_gpus[:cards_needed]
                for gpu in selected:
                    machine.gpu_states[gpu.gpu_id] = 1
                    self.free_gpus.remove(gpu)
                return selected
        return None
    
    def _try_cross_machine(self, cards_needed: int) -> Optional[List[GPUPlacement]]:
        """尝试跨机分配"""
        if len(self.free_gpus) < cards_needed:
            return None
        selected = self.free_gpus[:cards_needed]
        for gpu in selected:
            self.machines[gpu.machine_id].gpu_states[gpu.gpu_id] = 1
        self.free_gpus = self.free_gpus[cards_needed:]
        return selected
    
    def reclaim_gpus(self, gpus: List[GPUPlacement]) -> None:
        """回收GPU"""
        for gpu in gpus:
            self.machines[gpu.machine_id].gpu_states[gpu.gpu_id] = 0
            self.free_gpus.append(gpu)
    
    def get_utilization(self) -> float:
        """计算GPU利用率"""
        total = sum(m.gpu_count for m in self.machines)
        used = sum(sum(m.gpu_states) for m in self.machines)
        return used / total if total > 0 else 0.0
```

### 3.2 任务模型

```python
class InstanceState(Enum):
    """实例状态"""
    INIT = 0          # 初始状态
    COLD_STARTING = 1  # 冷启动中（分配后~60s）
    IDLE = 2          # 空闲（没有样本可处理）
    BUSY = 3          # 忟（正在推理）

class TaskPhase(Enum):
    """任务阶段"""
    REGISTERED = 0  # 已注册，未开始
    ROLLOUT = 1     # 推理阶段
    TRAIN = 2       # 训练阶段
    DONE = 3        # 完成

@dataclass
class Instance:
    """任务实例（DP切片）"""
    instance_id: int
    gpus: List[GPUPlacement]  # 分配的GPU
    state: InstanceState = InstanceState.INIT
    speed_factor: float = 1.0  # 速度因子（基于固定种子）
    current_sample_start_time: Optional[float] = None
    samples_processed: int = 0
    cold_start_end_time: Optional[float] = None

    # 预计算的推理时间表
    cards_per_instance: int = field(init=False)  # tp * pp（固定）
    inference_time_table: List[float] = field(default_factory=list)
    current_sample_index: int = 0

@dataclass
class SampleQueue:
    """每轮迭代的样本队列"""
    samples_per_round: int  # 本轮总样本数
    round_id: int = 0
    available_samples: int = field(init=False)
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
class TaskModel:
    """任务模型，模拟RL推理任务"""
    task_id: str
    tp: int
    pp: int
    base_instances: int
    samples_per_round: int
    total_samples: int
    random_seed: int
    
    # 长尾参数
    long_tail_ratio: float = 0.3  # 长尾实例比例
    slow_factor_range: Tuple[float, float] = (0.2, 0.5)  # 慢实例速度因子范围
    
    # 状态
    phase: TaskPhase = TaskPhase.REGISTERED
    instances: List[Instance] = field(default_factory=list)
    sample_queue: SampleQueue = field(init=False)
    done_samples: int = 0
    done_rounds: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    def __post_init__(self):
        self.sample_queue = SampleQueue(samples_per_round=self.samples_per_round)
    
    def init_instances(self, base_count: int, cluster: ClusterModel) -> None:
        """初始化实例，分配GPU并预计算推理时间"""
        for i in range(base_count):
            gpus = cluster.allocate_instance(self.tp, self.pp)
            if gpus:
                cards_per_instance = self.tp * self.pp
                inst = Instance(
                    instance_id=i,
                    gpus=gpus,
                    state=InstanceState.COLD_STARTING
                )
                inst.cards_per_instance = cards_per_instance

                # 预计算所有样本的推理时间
                inst.precompute_inference_times(
                    total_samples=self.total_samples,
                    random_seed=self.random_seed,
                    long_tail_ratio=self.long_tail_ratio,
                    slow_factor_range=self.slow_factor_range
                )

                self.instances.append(inst)
    
    def precompute_inference_times(self, total_samples: int, random_seed: int,
                                  long_tail_ratio: float, slow_factor_range: Tuple[float, float]) -> None:
        """
        预计算所有样本的推理时间

        推理时间 = 基础时间(1 / cards_per_instance) * 速度因子

        速度因子基于固定随机种子计算：
        - 正常实例：speed_factor = 1.0
        - 慢实例（长尾）：speed_factor ∈ [slow_factor_range]
        """
        rng = random.Random(random_seed + self.instance_id)
        is_slow_instance = rng.random() < long_tail_ratio
        speed_factor = rng.uniform(*slow_factor_range) if is_slow_instance else 1.0
        self.speed_factor = speed_factor

        base_time = 1.0 / self.cards_per_instance

        for _ in range(total_samples):
            inference_time = base_time * speed_factor
            self.inference_time_table.append(inference_time)

    def get_next_inference_time(self) -> Optional[float]:
        """获取下一个样本的预计算推理时间"""
        if self.current_sample_index < len(self.inference_time_table):
            time = self.inference_time_table[self.current_sample_index]
            self.current_sample_index += 1
            return time
        return None
    
    def step(self, current_time: float, time_delta: float) -> None:
        """推进时间，模拟推理执行"""
        for inst in self.instances:
            self._step_instance(inst, current_time)
        
        # 检查rollout阶段是否完成
        if self.phase == TaskPhase.ROLLOUT:
            if self.done_samples >= self.total_samples:
                self.phase = TaskPhase.DONE
                self.end_time = current_time
            elif self._all_instances_idle() and self._no_more_samples():
                self.phase = TaskPhase.TRAIN
    
    def _step_instance(self, inst: Instance, current_time: float) -> None:
        """推进单个实例的状态"""
        if inst.state == InstanceState.COLD_STARTING:
            # 检查冷启动是否完成
            if inst.cold_start_end_time and current_time >= inst.cold_start_end_time:
                inst.state = InstanceState.IDLE  # 冷启动完成，先空闲
            return
        
        if inst.state == InstanceState.IDLE:
            # 尝试从样本队列取一个样本
            if self.sample_queue.try_lock_samples(1):
                inst.state = InstanceState.BUSY
                inst.current_sample_start_time = current_time
            # 取不到样本，继续空闲
            return
        
        elif inst.state == InstanceState.BUSY:
            # 检查样本是否完成 - 使用预计算的推理时间
            inference_time = inst.get_next_inference_time()  # 获取预计算值
            if inference_time is None:
                inst.state = InstanceState.IDLE
                return

            elapsed = current_time - inst.current_sample_start_time
            if elapsed >= inference_time:
                # 样本完成
                self.sample_queue.unlock_samples(1)
                self.done_samples += 1
                inst.samples_processed += 1

                # 检查本轮是否完成
                if (self.sample_queue.available_samples == 0 and
                    self.sample_queue.locked_samples == 0):
                    # 本轮完成
                    self.done_rounds += 1
                    if self.done_rounds * self.samples_per_round >= self.total_samples:
                        self.phase = TaskPhase.DONE
                        self.end_time = current_time
                    else:
                        self.sample_queue.next_round()

                # 尝试取下一个样本
                if (self.phase == TaskPhase.ROLLOUT and
                    self.sample_queue.try_lock_samples(1)):
                    inst.current_sample_start_time = current_time
                else:
                    inst.state = InstanceState.IDLE
    
    def _all_instances_idle(self) -> bool:
        return all(i.state == InstanceState.IDLE for i in self.instances)
    
    def _no_more_samples(self) -> bool:
        return (self.done_rounds * self.samples_per_round >= self.total_samples)
    
    def get_state_report(self) TaskStateReport:
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
    
    def _elapsed_time(self) -> float:
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time
```

### 3.3 事件定义

```python
@dataclass
class Event:
    """基类事件"""
    event_id: str
    timestamp: float
    event_type: str
    task_id: str

@dataclass
class AssignEvent(Event):
    """GPU分配事件"""
    event_type: str = "assign"
    instances_count: int
    gpus: List[List[GPUPlacement]]  # 每个实例的GPU分配
    cold_start_duration: float = 60.0  # 冷启动时长

@dataclass
class AssignCompleteEvent(Event):
    """分配完成事件"""
    event_type: str = "assign_complete"
    instances_count: int
    gpus: List[List[GPUPlacement]]

@dataclass
class ReclaimEvent(Event):
    """GPU回收事件"""
    event_type: str = "reclaim"
    instances_count: int
    gpus: List[GPUPlacement]  # 要回收的GPU
    reclaim_duration: float = 5.0  # 回收时长（可配置2-10s）

@dataclass
class ReclaimCompleteEvent(Event):
    """回收完成事件"""
    event_type: str = "reclaim_complete"
    gpus: List[GPUPlacement]
```

### 3.4 仿真时钟

```python
class SimulationClock:
    """仿真时钟，固定步长推进 + 事件驱动混合模式"""
    current_time: float = 0.0
    time_step: float = 1.0  # 固定步长1秒
    events: PriorityQueue[Tuple[float, Event]] = field(default_factory=PriorityQueue)

    def schedule_event(self, event: Event) -> None:
        """调度事件"""
        self.events.put((event.timestamp, event))

    def peek_time(self) -> float:
        """获取当前时间"""
        return self.current_time

    def advance_one_step(self, event_handler: Callable[[Event], None]) -> None:
        """
        推进一个固定步长（1秒）

        在推进过程中，如果到达了事件时间，立即处理事件

        Args:
            event_handler: 事件处理回调函数
        """
        self.current_time += self.time_step

        # 处理在当前时间点到达的所有事件
        while not self.events.empty() and self.events.queue[0][0] <= self.current_time:
            _, event = self.events.get()
            event_handler(event)

    def has_pending_events(self) -> bool:
        """检查是否有待处理的事件"""
        return not self.events.empty()
```

---

## 4. 仿真器核心

### 4.1 仿真器主类

```python
@dataclass
class SimulationResult:
    """完整仿真结果"""
    test_case_name: str
    total_simulation_time: float
    task_completion_times: Dict[str, float]
    scheduling_traces: List[SchedulingTrace]
    gpu_utilization_curve: List[Tuple[float, float]]
    task_states_history: Dict[str, List[Tuple[float, TaskStateReport]]
    event_history: List[Event]
    
    @property
    def total_completion_time(self) -> float:
        return self.total_simulation_time
    
    @property
    def avg_task_completion_time(self) -> float:
        if not self.task_completion_times:
            return 0.0
        return sum(self.task_completion_times.values()) / len(self.task_completion_times)
    
    @property
    def slowest_task(self) -> Tuple[str, float]:
        return max(self.task_completion_times.items(), key=lambda x: x[1])

class Simulator:
    """仿真器主类"""
    
    def __init__(self, cluster_config: ClusterConfig, global_random_seed: int = 42):
        self.cluster = ClusterModel.from_config(cluster_config)
        self.clock = SimulationClock()
        self.tasks: Dict[str, TaskModel] = {}
        self.scheduler: Optional[GroupScheduler] = None  # 真实调度器
        self.metrics = MetricsCollector()
        self.global_seed = global_random_seed
        self.random = random.Random(global_random_seed)
    
    def run(self, test_case: TestCase) -> SimulationResult:
        """运行仿真 - 固定步长推进"""
        self._init_from_test_case(test_case)

        while self._has_active_tasks() or self.clock.has_pending_events():
            # 1. 推进一个固定步长（1秒）
            #    - 推进过程中会自动处理到达的事件
            #    - 事件处理会触发调度决策
            self.clock.advance_one_step(self._handle_event)

            # 2. 让所有任务在当前时间点执行推理
            #    - 使用预计算的推理时间
            current_time = self.clock.peek_time()
            for task in self.tasks.values():
                task.step(current_time, self.clock.time_step)

            # 3. 记录当前时刻的GPU利用率
            self.metrics.record_utilization(current_time, self.cluster.get_utilization())

        return self._collect_results(test_case.name)
    
    def _init_from_test_case(self, test_case: TestCase) -> None:
        """从测试用例初始化"""
        # 验证初始约束
        test_case.validate_initial_constraints()
        
        # 创建调度器（真实实现）
        self.scheduler = GroupScheduler(...)
        
        # 创建任务
        for task_config in test_case.tasks:
            task = TaskModel.from_config(task_config, random_seed=self.global_seed)
            self.tasks[task.task_id] = task
            
            # 向调度器注册
            self.scheduler.register_task(task_config)
        
        # 初始分配：按base_instances分配GPU
        for task in self.tasks.values():
            task.init_instances(task.base_instances, self.cluster)
            
            # 为每个实例创建冷启动完成事件
            for inst in task.instances:
                end_time = self.clock.current_time + 60.0  # 冷启动60s
                inst.cold_start_end_time = end_time
                event = AssignCompleteEvent(
                    event_id=f"{task.task_id}_init_{inst.instance_id}",
                    timestamp=end_time,
                    task_id=task.task_id,
                    instances_count=1,
                    gpus=[inst.gpus]
                )
                self.clock.schedule_event(event)
    
    def _handle_event(self, event: Event) -> None:
        """处理事件"""
        if isinstance(event, AssignCompleteEvent):
            self._handle_assign_complete(event)
        elif isinstance(event, ReclaimCompleteEvent):
            self._handle_reclaim_complete(event)
    
    def _handle_assign_complete(self, event: AssignCompleteEvent) -> None:
        """处理分配完成事件"""
        task = self.tasks[event.task_id]
        
        # 上报状态，触发调度
        report = task.get_state_report()
        self._report_to_scheduler(report)
    
    def _handle_reclaim_complete(self, event: ReclaimCompleteEvent) -> None:
        """处理回收完成事件"""
        task = self.tasks[event.task_id]
        
        # 回收GPU
        self.cluster.reclaim_gpus(event.gpus)
        
        # 移除对应的实例
        gpus_to_remove = set(event.gpus)
        task.instances = [
            inst for inst in task.instances
            if not any(g in gpus_to_remove for g in inst.gpus)
        ]
        
        # 上报状态，触发调度
        report = task.get_state_report()
        self._report_to_scheduler(report)
    
    def _report_to_scheduler(self, report: TaskStateReport) -> None:
        """上报状态到调度器，触发调度决策"""
        # 调度器执行4阶段算法
        self.scheduler.report_state(report)
        # 调度器内部会调用reclaim/assign，这些操作通过回调创建事件
    
    def _has_active_tasks(self) -> bool:
        return any(t.phase != TaskPhase.DONE for t in self.tasks.values())
    
    def _collect_results(self, test_case_name: str) -> SimulationResult:
        """收集仿真结果"""
        completion_times = {
            task_id: task.end_time or 0
            for task_id, task in self.tasks.items()
        }
        return SimulationResult(
            test_case_name=test_case_name,
            total_simulation_time=self.clock.current_time,
            task_completion_times=completion_times,
            scheduling_traces=self.metrics.get_scheduling_traces(),
            gpu_utilization_curve=self.metrics.get_utilization_curve(),
            task_states_history=self.metrics.get_task_states_history(),
            event_history=self.metrics.get_events()
        )
```

### 4.2 调度器接口适配

```python
class SchedulerAdapter:
    """调度器接口适配层，用于仿真器与真实调度器交互"""
    
    def __init__(self, simulator: Simulator):
        self.simulator = simulator
    
    def reclaim(self, task_id: str, instances_count: int, gpus: List[GPUPlacement]) -> None:
        """调度器请求回收实例（来自调度器的execute阶段）"""
        # 创建回收事件
        current_time = self.simulator.clock.current_time
        reclaim_duration = self.simulator.random.uniform(2.0, 10.0)  # 2-10s
        
        event = ReclaimCompleteEvent(
            event_id=f"{task_id}_reclaim_{uuid4()}",
            timestamp=current_time + reclaim_duration,
            task_id=task_id,
            gpus=gpus
        )
        self.simulator.clock.schedule_event(event)
    
    def assign(self, task_id: str, placements: List[List[GPUPlacement]]) -> None:
        """调度器请求分配实例（来自调度器的execute阶段）"""
        # 并发创建多个分配完成事件
        current_time = self.simulator.clock.current_time
        cold_start_duration = 60.0
        
        for i, gpu_set in enumerate(placements):
            event = AssignCompleteEvent(
                event_id=f"{task_id}_assign_{uuid4()}_{i}",
                timestamp=current_time + cold_start_duration,
                task_id=task_id,
                instances_count=1,
                gpus=[gpu_set]
            )
            self.simulator.clock.schedule_event(event)
```

---

## 5. 测试用例设计

### 5.1 测试用例配置

```python
@dataclass
class ClusterConfig:
    machine_count: int
    gpus_per_machine: int

@dataclass
class TaskConfig:
    task_id: str
    tp: int
    pp: int
    base_instances: int
    samples_per_round: int
    total_samples: int
    long_tail_ratio: Optional[float] = None
    slow_factor_range: Optional[Tuple[float, float]] = None

@dataclass
class TestCase:
    name: str
    description: str
    cluster: ClusterConfig
    tasks: List[TaskConfig]
    
    def validate_initial_constraints(self) -> bool:
        """验证初始约束：
        1. 所有任务base_instances刚好占满集群
        2. 不同任务有不同的tp/pp组合
        """
        total_cluster_gpus = self.cluster.machine_count * self.cluster.gpus_per_machine
        total_required = sum(
            t.base_instances * t.tp * t.pp
            for t in self.tasks
        )
        
        if total_required != total_cluster_gpus:
            raise ValueError(
                f"初始分配必须正好占用集群："
                f"集群{total_cluster_gpus}张卡，任务需要{total_required}张卡"
            )
        return True
```

### 5.2 测试用例列表（20个）

详见完整测试用例实现，包括：

1. **TC1**: 基础场景 - 正常调度
2. **TC2**: 资源竞争 - 中等规模
3. **TC3**: 严重长尾 - 多任务
4. **TC4**: 混合并行策略 - 大规模
5. **TC5**: 集群满载 - 高压力
6. **TC6**: 快速+慢速混合 - 极化差异
7. **TC7**: 极端长尾 - 超慢实例
8. **TC8**: 渐变长尾 - 分层速度
9. **TC9**: 拓扑感知 - 跨机挑战
10. **TC10**: 大规模混合工作负载
11. **TC11**: 高密度小粒度任务
12. **TC12**: 高并发长尾
13. **TC13**: 样本量差异巨大
14. **TC14**: 复杂拓扑 + 多种并行策略
15. **TC15**: 超大规模集群压力
16. **TC16**: 边界 - 单任务独占
17. **TC17**: PP并行策略为主
18. **TC18**: 回收限制测试
19. **TC19**: 长尾程度递增
20. **TC20**: 真实生产场景模拟

### 5.3 按复杂度分类

```python
TEST_CASES_BY_COMPLEXITY = {
    "basic": [tc1],
    "medium": [tc2, tc3, tc6, tc11],
    "high": [tc4, tc5, tc10, tc12, tc14],
    "extreme": [tc15, tc18, tc20],
}
```

---

## 6. 指标收集与分析

### 6.1 指标收集器

```python
class MetricsCollector:
    """收集仿真过程中的各种指标"""
    
    def __init__(self):
        self.scheduling_traces: List[SchedulingTrace] = []
        self.utilization_samples: List[Tuple[float, float]] = []
        self.task_states: Dict[str, List[Tuple[float, TaskStateReport]]] = {}
        self.events: List[Event] = []
    
    def record_scheduling_decision(self, trace: SchedulingTrace) -> None:
        self.scheduling_traces.append(trace)
    
    def record_utilization(self, time: float, utilization: float) -> None:
        self.utilization_samples.append((time, utilization))
    
    def record_task_state(self, time: float, report: TaskStateReport) -> None:
        if report.task_id not in self.task_states:
            self.task_states[report.task_id] = []
        self.task_states[report.task_id].append((time, report))
    
    def record_event(self, event: Event) -> None:
        self.events.append(event)
    
    def get_average_utilization(self) -> float:
        if not self.utilization_samples:
            return 0.0
        return sum(u for _, u in self.utilization_samples) / len(self.utilization_samples)
```

### 6.2 结果分析器

```python
class ResultAnalyzer:
    """分析仿真结果，生成报告"""
    
    @staticmethod
    def generate_report(result: SimulationResult) -> str:
        """生成文本报告"""
        # 输出：总时间、平均完成时间、最慢任务、平均利用率
        # 输出：任务完成时间详情
        # 输出：调度决策统计
        # 输出：长尾行为分析
        pass
    
    @staticmethod
    def compare_results(results: List[SimulationResult]) -> str:
        """对比多个仿真结果"""
        pass
    
    @staticmethod
    def _analyze_longtail(result: SimulationResult) -> str:
        """分析长尾行为"""
        # 检测idle>0且busy>0的时刻
        pass
```

### 6.3 验证器

```python
class SimulatorValidator:
    """验证仿真结果的正确性"""
    
    @staticmethod
    def validate(result: SimulationResult, test_case: TestCase) -> List[str]:
        """验证仿真结果，返回错误列表"""
        # 1. 验证所有任务都完成了
        # 2. 验证完成时间非负
        # 3. 验证利用率在[0,1]范围内
        # 4. 验证调度决策的因果性
        pass
```

---

## 7. 使用示例

### 7.1 运行单个测试用例

```python
# 加载测试用例
test_case = load_test_case("tc5_cluster_full_load_pressure")

# 创建仿真器
simulator = Simulator(
    cluster_config=test_case.cluster,
    global_random_seed=42
)

# 运行仿真
result = simulator.run(test_case)

# 验证结果
errors = SimulatorValidator.validate(result, test_case)
if errors:
    print("验证失败:", errors)
else:
    print("验证通过")

# 生成报告
report = ResultAnalyzer.generate_report(result)
print(report)

# 保存结果
save_result(result, f"results/{test_case.name}.json")
```

### 7.2 批量运行测试用例

```python
# 按复杂度运行
for complexity, test_cases in TEST_CASES_BY_COMPLEXITY.items():
    print(f"运行复杂度 {complexity} 的测试用例...")
    for test_case in test_cases:
        result = run_simulation(test_case)
        save_result(result)
        print(f"  {test_case.name}: 完成")

# 对比结果
results = load_all_results()
comparison = ResultAnalyzer.compare_results(results)
print(comparison)
```

---

## 8. 输出说明

### 8.1 仿真结果输出

- `task_completion_times`: 每个任务的端到端完成时间
- `total_simulation_time`: 总仿真时间
- `gpu_utilization_curve`: GPU利用率曲线
- `scheduling_traces`: 调度决策历史
- `task_states_history`: 任务状态历史

### 8.2 分析报告输出

- 基本指标：总时间、平均完成时间、最慢任务、平均利用率
- 任务完成时间详情
- 调度决策统计：触发次数、回收回收次数、分配次数
- 长尾行为分析：每个任务的长尾事件次数和持续时间

---

## 9. 关键设计要点

### 9.1 初始约束

- 所有任务base_instances * tp * pp 之和 = 集群总GPU数
- 初始分配无剩余GPU

### 9.2 并发和延迟

- 回收操作：并发，延迟2-10s
- 分配操作：并发，延迟~60s（冷启动）
- 冷启动完成后实例才能开始取样本

### 9.3 长尾模拟

- 基于固定随机种子计算速度因子
- 长尾比例可配置（long_tail_ratio）
- 慢实例速度因子范围可配置（slow_factor_range）
- 推理时间在实例初始化时预计算，提高仿真效率和可复现性
- 推理时间 = 基础时间(1 / cards_per_instance) * 速度因子

### 9.4 样本队列

- 每轮迭代有共享样本队列
- 实例从队列中取样本（锁定）
- 样本推理完成后解锁
- 本轮所有样本完成后进入下一轮

### 9.5 时间推进机制

- 固定步长推进：每步1秒
- 混合模式：步长推进过程中，如果到达了事件时间，立即处理事件
- 事件处理会触发调度决策
- 记录每个时间点的GPU利用率

---

## 10. 遗留问题

### 10.1 动态任务加入/退出

当前设计支持静态任务集合。动态任务场景需要扩展：
- 仿真脚本控制任务加入/退出时机
- 动态调整TestCase配置

### 10.2 网络延迟模拟

当前设计中调度器调用是同步的。可以扩展：
- 调度器状态上报增加网络延迟
- 调度决策返回增加网络延迟

### 10.3 真实调度器集成

设计基于真实调度器，需要：
- 实现GroupSchedulerProxy适配层
- 调度器接口与仿真器事件系统的绑定
