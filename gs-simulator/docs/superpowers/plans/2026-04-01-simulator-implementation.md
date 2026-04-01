# GRPO GPU调度器仿真器实现计划

> **Goal:** 实现基于真实调度器的仿真器，用于验证调度算法正确性和性能基准测试

> **Architecture:** 事件驱动 + 固定步长推进，调用真实GroupScheduler的4阶段算法

> **Tech Stack:** Python 3.10+, pytest (测试), dataclasses (数据结构)

---

## Task 1: 创建项目结构和基础文件

**Files:**
- Create: `gs-simulator/__init__.py`
- Create: `gs-simulator/__main__.py`
- Create: `gs-simulator/core/__init__.py`
- Create: `gs-simulator/models/__init__.py`
- Create: `gs-simulator/events/__init__.py`
- Create: `gs-simulator/test_cases/__init__.py`
- Create: `gs-simulator/requirements.txt`

---

## Task 2: 实现核心数据结构

**Files:**
- Create: `gs-simulator/models/instance.py` - Instance类，包含预计算推理时间
- Create: `gs-simulator/models/task.py` - TaskModel和SampleQueue类
- Create: `gs-simulator/models/cluster.py` - ClusterModel和Machine类
- Create: `gs-simulator/models/cluster_config.py` - ClusterConfig类
- Create: `gs-simulator/models/task_config.py` - TaskConfig类
- Create: `gs-simulator/models/test_case.py` - TestCase类
- Create: `gs-simulator/models/__init__.py` - 统一导出

**Implementation detail:**
- Instance类：包含`cards_per_instance`、`inference_time_table`、`precompute_inference_times()`、`get_next_inference_time()`方法
- TaskModel类：修改`_step_instance()`使用预计算的推理时间表
- ClusterModel类：实现拓扑感知分配和回收
- `allocate_instance()`: 优先同机分配，其次跨机分配
- `reclaim_gpus()`: 回收GPU到空闲池
- `get_utilization()`: 计算GPU利用率

---

## Task 3: 实现仿真时钟和事件系统

**Files:**
- Create: `gs-simulator/events/event.py` - 基类Event和子类
- Create: `gs-simulator/events/clock.py` - SimulationClock类，固定步长推进

**Implementation detail:**
- SimulationClock：新增`time_step`字段、`peek_time()`方法、`advance_one_step()`方法
- 移除`get_next_event_time()`和`advance_to()`方法
- Event基类和子类：AssignEvent, AssignCompleteEvent, ReclaimEvent, ReclaimCompleteEvent

---

## Task 4: 实现仿真器主类

**Files:**
- Create: `gs-simulator/core/simulator.py` - Simulator主类
- Create: `gs-simulator/core/scheduler_adapter.py` - SchedulerAdapter适配层

**Implementation detail:**
- Simulator类：新增`time_step`字段，修改`run()`方法为固定步长循环
- SchedulerAdapter：实现`reclaim()`和`assign()`方法，创建ReclaimCompleteEvent和AssignCompleteEvent

---

## Task 5: 实现指标收集器和分析器

**Files:**
- Create: `gs-simulator/core/metrics_collector.py` - MetricsCollector类
- Create: `gs-simulator/core/result_analyzer.py` - ResultAnalyzer类
- Create: `gs-simulator/core/result.py` - SimulationResult类
- Create: `gs-simulator/core/validator.py` - SimulatorValidator类

**Implementation detail:**
- MetricsCollector：记录调度决策、利用率、任务状态、事件历史
- ResultAnalyzer：生成报告、对比结果、分析长尾行为
- SimulatorValidator：验证结果正确性

---

## Task 6: 实现测试用例

**Files:**
- Create: `gs-simulator/test_cases/test_cases.py` - 20个测试用例定义

**Implementation detail:**
- TC1-TC20：按照设计文档定义20个测试用例
- 按复杂度分类：basic/medium/high/extreme
- 验证初始约束：所有任务base_instances刚好占满集群

---

## Task 7: 编写单元测试

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_instance.py` - 测试Instance预计算推理时间
- Create: `tests/test_cluster.py` - 测试集群GPU分配
- Create: `tests/test_clock.py` - 测试固定步长推进
- Create: `tests/test_task.py` - 测试任务推理执行

**Implementation detail:**
- 使用pytest框架
- 测试正常流程和边界条件
- 测试长尾行为

---

## Task 8: 创建入口脚本

**Files:**
- Create: `gs-simulator/__main__.py` - 主入口
- Create: `gs-simulator/cli.py` - 命令行接口

**Implementation detail:**
- 支持运行单个测试用例
- 支持批量运行测试用例
- 支持不同步长配置
- 输出JSON格式的仿真结果

---

## Task 9: 集成真实调度器

**Files:**
- Modify: 从真实调度器代码导入GroupScheduler
- Create: `gs-simulator/core/mock_scheduler.py` - 调度器Mock（如果真实调度器未完成）

**Implementation detail:**
- 先实现调度器Mock版本用于仿真
- 后续可可切换为真实调度器实现

---

## Task 10: 集成和测试

1. 运行所有单元测试确保通过
2. 运行TC1基础场景验证整个流程
3. 检查生成的仿真结果JSON格式正确
