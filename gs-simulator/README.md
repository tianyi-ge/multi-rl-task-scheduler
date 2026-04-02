# GroupScheduler 仿真器

基于真实 `GroupScheduler` 实现的仿真器，用于验证调度算法正确性和性能基准测试。

## 特性

- **真实 GS 事件驱动**：使用真实 GroupScheduler 的事件驱动机制（loop 线程 + 条件变量）
- **预计算推理时间**：实例初始化时预计算所有样本的推理时间，提高仿真效率和可复现性
- **固定步长推进**：每步推进 0.5 秒，中间穿插事件处理
- **拓扑支持分配**：优先同机分配 GPU，其次跨机分配
- **长尾模拟**：基于固定随机种子模拟长尾行为
- **可复现性**：所有随机性使用固定种子
- **跨节点通信建模**：跨节点分配的实例有通信开销（1.2x）

## 项目结构

```
gs-simulator/
├── __init__.py
├── README.md
├── requirements.txt
├── test_scheduler_effectiveness.py  # 调度器效果验证
├── core/                    # 核心模块
│   ├── __init__.py
│   ├── simulator.py             # 仿真器主类
│   ├── gs_adapter.py            # GroupScheduler 适配层
│   ├── result.py                # 仿真结果数据结构
│   └── event.py                 # 事件系统
├── models/                   # 数据模型
│   ├── __init__.py
│   ├── instance.py              # 实例类（含预计算推理时间）
│   ├── task.py                 # 任务模型和样本队列
│   ├── cluster.py               # 集群模型和机器模型
│   ├── cluster_config.py        # 集群配置
│   ├── task_config.py          # 任务配置
│   └── test_case.py            # 测试用例
├── test_cases/               # 测试用例定义
│   ├── __init__.py
│   └── benchmark.py           # Benchmark测试用例（BB1-BB12）
└── results/                   # 测试结果输出目录
```

## 架构概览

仿真器通过 `GSAdapter` 适配层与真实的 `GroupScheduler` 交互，实现事件驱动的调度机制。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Simulator (仿真器)                          │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    SimulationLoop                           │  │
│  │                                                               │  │
│  │  ┌─────────────────────────────────────────────────────────────┐   │  │
│  │  │  1. advance_one_step() → 推进时钟 0.5s        │   │  │
│  │  └─────────────────────────────────────────────────────────────┘   │  │
│  │                                                               │  │
│  │  ┌─────────────────────────────────────────────────────────────┐   │  │
│  │  │  2. task.step() → 执行推理（所有任务）           │   │  │
│  │  └─────────────────────────────────────────────────────────────┘   │  │
│  │                                                               │  │
│  │  ┌─────────────────────────────────────────────────────────────┐   │  │
│  │  │  3. report_state() → 上报状态到 GS            │   │  │
│  │  └─────────────────────────────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────┘  │
│         ↓                                                          │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │              GSAdapter (适配层)                        │  │
│  │                                                               │  │
│  │  ┌─────────────────────────────────────────────────────────────┐   │  │
│  │  │ 1. gs.report_state() → 触发 GS 调度        │   │  │
│  │  └─────────────────────────────────────────────────────────────┘   │  │
│  │         ↓                                                    │  │
│  │  ┌─────────────────────────────────────────────────────────────┐   │  │
│  │  │ 2. _wait_for_gs_scheduling() → 等待完成     │   │  │
│  │  └─────────────────────────────────────────────────────────────┘   │  │
│  │         ↑                                                    │  │
│  │         └──────────────────────┐                               │  │
│  │                              ↓                               │  │
│  │  ┌─────────────────────────────┴─────────────────────────────┐   │  │
│  │  │  Mock Callbacks (硬件交互模拟)                  │   │  │
│  │  │                                                               │  │
│  │  │  ┌─────────────────────────────────────────────────────────┐ │  │  │
│  │  │  │ MockRevokeCallable.invoke(num_instances)        │ │  │  │
│  │  │  │   ↓                                         │ │  │  │
│  │  │  │   返回 MockFuture                            │ │  │  │
│  │  │  └─────────────────────────────────────────────────────────┘ │  │  │
│  │  │                                                               │  │  │
│  │  │  ┌─────────────────────────────────────────────────────────┐ │  │  │
│  │  │  │ MockAssignCallable.invoke(placements)         │ │  │  │
│  │  │  │   ↓                                         │ │  │  │
│  │  │  │   返回 MockFuture                            │ │  │  │
│  │  │  └─────────────────────────────────────────────────────────┘ │  │  │
│  │  └─────────────────────────────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│              GroupScheduler (真实调度器)                      │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Loop Thread (事件驱动循环)                          │  │
│  │                                                               │  │
│  │  ┌─────────────────────────────────────────────────────────────┐   │  │
│  │  │ while running_loop:                                  │   │  │
│  │  │   cv.wait_for(need_schedule)  [阻塞等待]            │   │  │
│  │  │   compute_card_allocation()                          │   │  │
│  │  │   execute(task_table_snapshot, plan)                   │   │  │
│  │  └─────────────────────────────────────────────────────────────┘   │. │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  concurrent_assign(placements)                        │  │
│  │  ┌───────────────────────────────────────────────────────────┐│  │
│  │  │ task.assign(placements) → 返回 MockFuture      ││  │
│  │  │ yr.get(futures) → 阻塞等待结果               ││  │
│  │  └───────────────────────────────────────────────────────────┘│  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  concurrent_reclaim(reclaim_tasks)                      │  │
│  │  ┌───────────────────────────────────────────────────────────┐│  │
│  │  │ task.revoke(instances) → 返回 MockFuture        ││  │
│  │  │ yr.get(futures, timeout) → 阻塞等待          ││  │
│  │  └───────────────────────────────────────────────────────────┘│  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 完整仿真流程图

```
初始化阶段:
┌─────────────────────────────────────────────────────────────────┐
│ 1. 创建 Simulator                                      │
│    - 创建 ClusterModel (32张卡)                          │
│    - 创建 GSAdapter                                       │
│    - GSAdapter 初始化 GroupScheduler                        │
│      - GS.start_loop() → 启动 loop 线程 (running_loop=True) │
│      - GS.create() → 创建 Worker 实例                      │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. 初始化任务                                          │
│    - 创建 TaskModel                                        │
│    - 初始化分配 Instance (base_instances)                   │
│    - 预计算推理时间                                     │
│    - 注册到 GS (register_task)                            │
│      - GS.tasks.register()                                 │
│      - Patch TaskInfo.revoke → MockRevokeCallable            │
│      - Patch TaskInfo.assign → MockAssignCallable              │
└─────────────────────────────────────────────────────────────────┘

仿真循环 (每 0.5 秒):
┌─────────────────────────────────────────────────────────────────┐
│ ┌─────────────────────────────────────────────────────────────┐│
│ │ 时间步进                                             ││
│ │ - clock.advance_one_step() → current_time += 0.5s        ││
│ └─────────────────────────────────────────────────────────────┘│
│    ↓
│ ┌─────────────────────────────────────────────────────────────┐│
│ │ 任务执行                                             ││
│ │ for task in tasks:                                     ││
│ │   task.step(current_time)                                ││
│ │     for instance in instances:                             ││
│ │       if instance.state == BUSY:                            ││
│ │         执行推理 (预计算时间表)                          ││
│ │         完成样本后更新状态                                ││
│ └─────────────────────────────────────────────────────────────┘│
│    ↓
│ ┌─────────────────────────────────────────────────────────────┐│
│ │ 上报状态到 GS                                         ││
│ │ for task in tasks:                                     ││
│ │   state = task.get_state_report_for_gs()                  ││
│ │   gs_adapter.report_state(state, need_schedule=True)       ││
│ │     ↓                                                  ││
│ │     gs.report_state(need_schedule=True)                    ││
│ │       - 设置 need_schedule = True                        ││
│ │       - cv.notify() → 唤醒 loop 线程                    ││
│ │     ↓                                                  ││
│ │     _wait_for_gs_scheduling() → 轮询等待完成            ││
│ │       while gs.need_schedule:                             ││
│ │         sleep(0.01)  [10ms]                              ││
│ └─────────────────────────────────────────────────────────────┘│
│    ↓
│ ┌─────────────────────────────────────────────────────────────┐│
│ │ GS loop 线程执行调度 (并行)                          ││
│ │ while running_loop:                                     ││
│ │   cv.wait_for(need_schedule)  [被notify唤醒]          ││
│ │   task_table_snapshot = compute_card_allocation()           ││
│ │     - assess_range() → 计算每个任务的调整范围          ││
│ │     - dont_starve() → 满足必须增加的任务              ││
│ │     - feed_more() → 分配富余卡给收益最大的任务            ││
│ │   execute(task_table_snapshot, plan)                     ││
│ │     - 如果有 reclaim:                                   ││
│ │       concurrent_reclaim()                               ││
│ │         task.revoke() → 返回 MockFuture                ││
│ │         yr.get() → 等待仿真器回调执行                   ││
│ │         [MockRevokeCallable.invoke 执行]                 ││
│ │           _handle_revoke_invoke()                         ││
│ │             获取 idle workers                            ││
│ │             更新 allocated_workers                         ││
│ │             返回 TaskStateReport (含voluntary_reclaim)    ││
│ │       检查连续回收限制                               ││
│ │       - 如果强制分配: do_assign()                      ││
│ │       - 否则: execute() (递归)                        ││
│ │     - 如果有 assign:                                   ││
│ │       do_assign()                                       ││
│ │         concurrent_assign()                                ││
│ │           task.assign() → 返回 MockFuture                 ││
│ │           yr.get() → 等待仿真器回调执行                   ││
│ │           [MockAssignCallable.invoke 执行]                  ││
│ │             _handle_assign_invoke()                        ││
│ │               创建新 Instance                              ││
│ │               分配 GPU (仿真器)                          ││
│ │               更新 allocated_workers                       ││
│ │               返回 TaskStateReport                        ││
│ │     - need_schedule = False                               ││
│ └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 检查是否所有任务完成                                    │
│ if all(task.phase == DONE for task in tasks):                  │
│   退出仿真循环                                           │
└─────────────────────────────────────────────────────────────────┘
```

## 关键组件说明

### 1. GSAdapter (GroupScheduler 适配层)

连接仿真器与真实 GS 的桥接层：

```python
class GSAdapter:
    def __init__(self, num_nodes: int):
        # 初始化真实 GS
        self.gs = GroupScheduler(num_nodes)
        self.gs.create()
        # GS.start_loop() 内部启动 loop 线程

    def report_state(self, state: TaskStateReport, need_schedule: bool = True):
        # 调用 GS.report_state，触发事件驱动
        self.gs.report_state(state, need_schedule=need_schedule)

        if need_schedule:
            # 等待 GS loop 线程完成调度
            self._wait_for_for_gs_scheduling()

    def _wait_for_gs_scheduling(self, timeout: float = 10.0):
        # 轮询等待 need_schedule 变为 False
        while self.gs.need_schedule:
            time.sleep(0.01)

    # Mock callbacks (硬件交互模拟)
    def _handle_revoke_invoke(self, task_id: str, num_instances: int):
        # 获取空闲 workers
        idle_workers = self.get_idle_workers_callback(task_id)
        # 创建 ReclaimConfirm
        return TaskStateReport(..., voluntary_reclaim=reclaim_confirm)

    def _handle_assign_invoke(self, task_id: str, instance_placements):
        # 创建新 Instance
        # 分配 GPU
        return TaskStateReport(...)
```

### 2. yr_mock (yr 框架模拟)

模拟 yr 框架的异步行为：

```python
class MockFuture:
    def __init__(self):
        self._future = Future()

    def set_result(self, result):
        self._future.set_result(result)

    def get(self, timeout=None):
        return self._future.result(timeout)

class MockInstance:
    def invoke(self, *args, **kwargs):
        future = MockFuture()
        result = self._cls(*args, **kwargs)
        future.set_result(result)
        return future  # 返回 Future，不是直接结果

def mock_get(object_ref_or_refs, timeout=None):
    # 阻塞等待 Future 完成
    if isinstance(object_ref_or_refs, list):
        return [ref.get(timeout=timeout) for ref in object_ref_or_refs]
    else:
        return object_ref_or_refs.get(timeout=timeout)
```

### 3. Instance (任务实例)

```python
class Instance:
    # 跨节点通信建模
    placement_nodes: Set[int]  # 实例跨越的节点
    communication_factor: float = 1.0  # 通信因子

    def precompute_inference_times(self, ...):
        # 预计算推理时间
        base_time = 10.0 * (8.0 / cards_per_instance) * self.communication_factor
        for i in range(total_samples):
            speed = generate_speed_factor(...)
            self.inference_time_table.append(base_time * speed)
```

### 4. Simulator (仿真器主类)

```python
class Simulator:
    def run(self):
        while self._has_active_tasks():
            current_time = self.clock.advance_one_step()

            # 1. 任务执行
            for task in self.tasks.values():
                task.step(current_time)

            # 2. 上报状态（触发 GS 调度）
            if self.enable_gs:
                self._reporter_all_task_states(current_time)

    def _reporter_all_task_states(self, current_time):
        # 释放已完成任务的资源
        for task in self.tasks.values():
            if task.phase == DONE:
                self.cluster.reclaim_gpus(...)
                self.gs_adapter.gs.workers.add_workers_to_idle(...)

        # 上报所有任务状态
        for task in self.tasks.values():
            state = task.get_state_report_for_gs()
            self.gs_adapter.report_state(state, need_schedule=True)
```

## 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- pytest: 单元测试框架
- pytest-cov: 测试覆盖率

## 快速开始

### 运行调度器效果测试

```bash
# 交互式选择测试用例
python test_scheduler_effectiveness.py

# 按索引选择（如选择BB12）
python test_scheduler_effectiveness.py -i 12

# 按名称选择
python test_scheduler_effectiveness.py -c BB12_multi_task_diverse_longtail

# 只运行GS模式
python test_scheduler_effectiveness.py -i 13 --gs-only

# 只运行无GS模式
python test_scheduler_effectiveness.py -i 13 --no-gs-only

# 列出所有测试用例
python test_scheduler_effectiveness.py --list
```

该测试会运行 Benchmark 测试用例并对比有/无 GroupScheduler 的性能差异。

目前共有 12 个测试用例（BB1-BB12），详见下表。

### 编程方式运行

```python
from core.simulator import Simulator
from models.test_case import TestCase
from models.task_config import TaskConfig
from models.cluster_config import ClusterConfig

# 创建测试用例
test_case = TestCase(
    name="custom_test",
    cluster=ClusterConfig(machine_count=4, gpus_per_machine=8),
    tasks=[
        TaskConfig(
            task_id="task1",
            tp=4, pp=2,
            base_instances=2,
            samples_per_round=16,
            total_samples=100,
            time_distribution="longtail_normal",
            distribution_params={"slow_ratio": 0.3}
        ),
    ]
)

# 创建仿真器（启用 GS）
simulator = Simulator(test_case, enable_gs=True)

# 运行仿真
result = simulator.run()

# 输出结果
print(f"总时间: {result.total_simulation_time:.2f}s")
print(f"任务完成时间: {result.task_completion_times}")
```

## 调度效果基准测试

### 设计目的

本基准测试套件设计用于验证 GroupScheduler 在不同场景下的调度效果：

1. **资源利用率提升**：动态分配空闲资源给繁忙任务
2. **任务完成时间优化**：快任务释放资源帮助慢任务
3. **调度决策有效性**：验证调度算法的正确性

### 测试用例分类

**基础场景（BB1-BB4）**：验证基本调度功能在不同任务规模下的表现

**复杂场景（BB5-BB11）**：验证调度器在复杂拓扑、高并发、资源碎片等场景的处理能力

**高调度密度场景（BB12）**：专门设计用于展示GS调度的显著效果，长尾差异大，调度决策频繁

### Benchmark测试用例详情

| 用例名 | 集群规模 | 任务数 | 并行策略(TP×PP) | 描述 | GS优化效果 |
|--------|----------|--------|-----------------|------|------------|
| BB1_two_tasks_competition | 32卡 | 2 | 4×2 (8卡/实例) | 双任务竞争，验证资源共享 | 13.6% |
| BB2_three_tasks_mixed_parallelism | 32卡 | 3 | 混合(4×2, 8×1, 2×4) | 三任务混合并行策略 | 39.8% |
| BB3_large_scale_two_tasks | 64卡 | 2 | 8×2 (16卡/实例) | 大规模双任务场景 | 10.3% |
| BB4_four_tasks_small_scale | 32卡 | 4 | 4×2 (8卡/实例) | 四任务小规模调度 | 12.3% |
| BB5_small_granularity_dense | 32卡 | 8 | 2×1, 4×1 | 小粒度密集任务（8任务竞争） | 13.5% |
| BB6_mixed_parallelism | 32卡 | 4 | 2×1, 4×1, 4×2, 8×1 | 混合并行粒度（多种卡数混合） | 33.3% |
| BB7_large_instance | 64卡 | 2 | 8×2 (16卡/实例) | 大实例场景（16卡/实例） | 8.2% |
| BB8_fragmentation_test | 32卡 | 5 | 8×1, 4×1, 4×2 | 资源碎片化测试 | 7.8% |
| BB9_strong_competition | 32卡 | 4 | 8×1 (8卡/实例) | 饱和竞争（4任务100%占用，相同并行策略） | 15.3% |
| BB10_dynamic_scaling | 64卡 | 3 | 8×1 (8卡/实例) | 动态扩缩容（base差异大） | 12.9% |
| BB11_heterogeneous_cluster | 64卡 | 6 | 混合(2, 4, 8, 16) | 异构集群（多种TP×PP混合） | 22.0% |
| BB12_multi_task_diverse_longtail | 64卡 | 4 | 4×2 (8卡/实例) | 高调度密度（长尾差异大） | 39.9% |

**注**: GS优化效果 = (无GS时间 - 有GS时间) / 无GS时间 × 100%

### 长尾分布参数说明

每个测试用例使用不同的长尾分布参数来模拟真实场景：

- `slow_ratio`: 长尾样本比例（如0.5表示50%样本是长尾）
- `slow_min`: 长尾最小倍数（如1.0表示至少比正常慢1倍）
- `slow_max`: 长尾最大倍数（如5.0表示最多比正常慢5倍）

例如 BB12 中：
- 快任务：slow_ratio=0.2/0.3（20%/30%长尾），长尾轻微
- 慢任务：slow_ratio=0.7/0.8（70%/80%长尾），长尾严重

## 仿真结果示例

```python
{
    "test_case_name": "BB3_three_tasks_mixed_parallelism",
    "total_simulation_time": 343.5,
    "task_completion_times": {
        "task1": 51.0,
        "task2": 323.5,
        "task3": 343.5
    },
    "avg_task_completion_time": 239.33,
    "slowest_task": ("task3", 343.5),
    "fastest_task": ("task1", 51.0),
    "scheduling_trace_count": 0,
    "avg_utilization": 0.95
}
```

## 开发说明

### 添加新的测试用例

在 `test_cases/benchmark.py` 中添加：

```python
def _create_bb13_custom() -> TestCase:
    """BB13: 自定义测试用例（新增时使用）"""
    return TestCase(
        name="BB13_custom",
        description="自定义测试用例",
        cluster=ClusterConfig(machine_count=2, gpus_per_machine=8),
        tasks=[
            TaskConfig(
                task_id="task1",
                tp=4, pp=2,
                base_instances=1,
                samples_per_round=16,
                total_samples=100,
                time_distribution="longtail_normal",
                distribution_params={"slow_ratio": 0.2, "slow_min": 1.0, "slow_max": 1.5}
            ),
        ]
    )
```

然后在 `get_benchmark_cases()` 函数中添加到返回列表。

## 许可证

本项目用于学术研究和性能测试。
