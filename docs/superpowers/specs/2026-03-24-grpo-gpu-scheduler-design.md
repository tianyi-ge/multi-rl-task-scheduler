---
name: GRPO GPU 集群调度器设计
description: 多GRPO任务在公共GPU池中时分复用的调度系统，解决推理长尾问题
type: design
---

# GRPO GPU 集群调度器设计

## 1. 问题背景

GRPO（Group Relative Policy Optimization）多轮强化学习的推理阶段存在严重的长尾问题，导致GPU利用率低。多个任务运行在同一个集群时，需要一个全局调度器来：

1. 让多个任务时分复用公共GPU池
2. 分配粒度为 `tp * pp`（一个完整推理实例所需的卡数）
3. 接收任务上报指标，动态决定：
   - 从哪些任务回收多少张卡
   - 将回收的空闲卡分配给哪些任务

### 1.1 重要设计约束

1. **历史数据不确定性**：每个任务的历史平均每轮耗时、总耗时受到总卡数、并行策略等配置的影响，不一定能拿到准确值
2. **并行策略多样性**：每个任务的并行策略（tp/pp）不一样，但都是2的倍数
3. **拓扑感知**：一个tp组尽可能要放在同一台机器上，否则推理时延会非常高
4. **回收决策下放**：当中心调度器决策要回收某个任务的几张卡时，让任务本身来决定回收哪几张卡

### 1.2 核心设计原则

**关键洞察**：
1. 历史平均端到端耗时往往缺失，无法用它准确衡量任务是否落后
2. `EstimateFutureRoundTime` 很难估计准确，会导致边际损失/收益预估不准确
3. 但我们可以明确知道每个任务的基线 `K_i^base`

**可靠信号**：
- 当任务剩余样本用不满 `K_i^base` 个实例时，实例可以释放
- 当任务的每个实例都很忙、且还有剩余样本没推完、且当前实例数 `< K_i^base`，任务必须上报 `needs_more_instances = true`，调度器必须给它增加实例到 `K_i^base`

**长尾的正确定义**：
- 不是"实例间完成样本数差异大"（因为每个样本推理时间本来就不同）
- 而是"部分实例已经没有样本可取了，而其他实例还在执行推理"
- 注意：此时给长尾任务增加实例是**不合理**的，因为本轮样本已经分配完了

## 2. 优化目标

1. **约束条件**：每个强化学习训练任务的端到端耗时不能比固定分配时更长
2. **优化目标**：尽可能让所有训练任务的平均端到端耗时最低

## 3. 核心概念

### 3.1 符号定义

**任务级指标**：
- `T_i`：第i个任务
- `K_i^base`：任务i的固定分配基准实例数（核心基线，可靠）
- `K_i(t)`：时刻t任务i的实际实例数
- `tp_i`：任务i的tensor parallelism（2的幂）
- `pp_i`：任务i的pipeline parallelism（2的幂）
- `cards_per_instance_i = tp_i * pp_i`：每个实例需要的卡数
- `K_i^min = 1`：最小保证实例数
- `S_per_round_i`：每轮固定处理的样本数（任务启动时确定）

**集群拓扑**：
- `M_m`：第m台机器
- `G_{m,g}`：机器m上的第g张GPU
- `Placement_{i,j}`：任务i的第j个实例的GPU放置（机器ID列表，长度为tp_i）
- `ColocalityScore(Placement)`：放置的拓扑分数（同一机器的GPU越多分数越高）

**进度指标**：
- `S_total_i`：任务i的总样本数
- `S_done_i`：已完成样本数
- `S_rem_i = S_total_i - S_done_i`：剩余样本数
- `R_done_i`：已完成轮次数
- `R_total_i = ceil(S_total_i / S_per_round_i)`：总轮次数
- `R_rem_i = R_total_i - R_done_i`：剩余轮次数
- `InRolloutPhase(i)`：是否处于推理阶段（任务上报）

**当前状态指标**：
- `K_i_idle(t)`：时刻t任务i的空闲实例数（已完成当前轮次的推理）
- `K_i_busy(t) = K_i(t) - K_i_idle(t)`：忙实例数（仍在推理中）
- `needs_more_instances`：任务主动上报的信号（需要更多实例到K_i^base）
- `can_release_instances`：任务主动上报的信号（有空闲实例可以释放）

### 3.2 任务侧信号生成逻辑

**任务Agent自己决定何时上报信号**：

```
函数 TaskShouldRequestMoreInstances():
  # 当满足以下所有条件时，设置 needs_more_instances = true
  if InRolloutPhase(i) and
     K_i < K_i^base and
     K_i_idle == 0 and           # 所有实例都在忙
     S_rem_i > 0:                # 还有样本没推完
    return true
  return false

函数 TaskCanReleaseInstances():
  # 当有空闲实例，且剩余样本用不满 K_i^base 时
  if K_i_idle > 0:
    # 估计需要多少实例：假设每轮样本均匀分配
    needed_instances = min(K_i^base, ceil(S_rem_i / S_per_instance))
    if K_i > needed_instances:
      return true
  return false
```

**重要**：这些信号由任务Agent生成，调度器只负责响应。

## 4. 调度决策流程

### 4.1 调度触发事件

以下任一事件触发调度决策：
1. **实例完成事件**：某个任务的一个推理实例完成了当前batch
2. **轮次完成事件**：某个任务完成了一整轮（推理+训练）
3. **任务加入事件**：新任务提交到集群
4. **任务退出事件**：某个任务完成退出
5. **任务状态上报**：任务上报 `needs_more_instances` 或 `can_release_instances`

### 4.2 调度决策算法

**核心问题**：没有 `EstimateFutureRoundTime()`，如何评估给任务分配实例的收益？

**答案**：用**当前可观测状态**作为收益代理信号。

**收益信号优先级**（从高到低）：
1. **P0**: 满足 `needs_more_instances = true` 的任务（必须给它们 `K_i^base`）
2. **P1**: 在推理阶段、`K_i < K_i^base`、有剩余样本的任务（次优先）
3. **P2**: 在推理阶段、`K_i >= K_i^base`、但剩余样本还很多的任务（最后考虑）
4. **无收益**: 在训练阶段的任务（给更多实例也没用）

**长尾处理**：
- 不给长尾任务增加实例（因为本轮样本已分配完，加实例也没用）
- 长尾只是一个**观测信号**，用于理解发生了什么，但不用于分配决策

```
函数 Schedule():
  """
  调度决策主函数
  1. 先收集所有需求
  2. 模拟分配/回收，检查避免回收后马上归还
  3. 最终一起发起命令
  """

  # ========== 阶段1: 收集所有可靠信号 ==========
  P0_tasks = []           # 需要恢复到 K_i^base 的任务（必须满足）
  AllCandidates = []      # 所有可以分配的任务（P1+P2合并）
  Reclaimable = []        # 可以回收的任务（有空闲实例）

  for each task i:
    # ---------- P0: 任务明确说需要更多实例 ----------
    if i.needs_more_instances and K_i < K_i^base:
      deficit = K_i^base - K_i
      P0_tasks.append( (i, deficit) )

    # ---------- 收集所有分配候选 ----------
    score = ComputeAllocationScore(i)
    if score > 0:
      AllCandidates.append( (-score, i) )  # 负号用于降序排序

    # ---------- 可以回收的任务（只回收空闲实例） ----------
    if K_i_idle > 0:
      # 优先回收有空闲实例的任务
      reclaim_priority = K_i_idle
      Reclaimable.append( (-reclaim_priority, i) )

  # 按优先级排序
  Reclaimable.sort()     # 空闲多的在前
  AllCandidates.sort()    # 分数高的在前

  # ========== 阶段2: 模拟分配，避免回收后马上归还 ==========
  # 用临时状态模拟，不真正执行
  temp_state = CopyCurrentState()
  pending_reclaims = []   # 待回收列表
  pending_allocations = [] # 待分配列表

  # ---------- 先处理 P0 任务（必须满足） ----------
  for (i, deficit) in P0_tasks:
    while deficit > 0:
      cards_needed = cards_per_instance_i

      # 先看有没有空闲GPU
      if temp_state.C_free >= cards_needed:
        placement = FindBestPlacement(i, temp_state.free_gpus)
        if placement is not None:
          pending_allocations.append( (i, 1, placement) )
          temp_state.Allocate(i, 1, placement)
          deficit -= 1
        else:
          break
      else:
        # 需要从其他任务回收（只回收有空闲实例的任务）
        if not Reclaimable:
          break

        (neg_prio, j) = Reclaimable.pop(0)

        # 关键检查：不要从 P0 任务回收！
        if j in [t for (t, d) in P0_tasks]:
          continue

        # 模拟回收j的空闲实例
        reclaim_count = min(j.K_i_idle, 1)
        if reclaim_count > 0:
          pending_reclaims.append( (j, reclaim_count) )
          temp_state.Reclaim(j, reclaim_count)

  # ---------- 再处理其他任务（把所有空闲GPU都分配完） ----------
  # 合并剩余的候选，继续分配直到没有空闲GPU
  while temp_state.C_free > 0 and AllCandidates:
    (neg_score, i) = AllCandidates.pop(0)

    # 检查上限
    max_allowed = 1.5 * K_i^base
    if temp_state.GetK(i) >= max_allowed:
      continue

    cards_needed = cards_per_instance_i
    if temp_state.C_free >= cards_needed:
      placement = FindBestPlacement(i, temp_state.free_gpus)
      if placement is not None:
        pending_allocations.append( (i, 1, placement) )
        temp_state.Allocate(i, 1, placement)
        # 重新计算分数，如果还有收益继续排队
        new_score = ComputeAllocationScoreWithState(i, temp_state)
        if new_score > 0 and temp_state.GetK(i) < max_allowed:
          insert (-new_score, i) into AllCandidates in sorted order

  # ========== 阶段3: 最终一起发起命令 ==========
  # 先回收，再分配
  for (j, count) in pending_reclaims:
    Reclaim(j, count)

  for (i, count, placement) in pending_allocations:
    Allocate(i, count, placement)
```

**设计说明**：
1. **不主动回收非空闲实例**：只回收 `K_i_idle > 0` 的任务
2. **模拟分配避免抖动**：先用临时状态模拟，检查避免从P0任务回收
3. **批量执行命令**：决策完成后，先回收再分配，一起发起命令
4. **空闲卡全部分配完**：一直分配直到 `C_free == 0`

### 4.3 收益评分函数

**关键思想**：不预测未来，只用**当前可观测状态**来评分。

```
函数 ComputeAllocationScore(task_i):
  """
  计算给 task_i 分配一个实例的收益分数（0-100）
  分数越高越值得分配
  """

  # 情况1: 在训练阶段，给实例没用
  if not task_i.in_rollout_phase:
    return 0

  # 情况2: 没有剩余样本了
  if task_i.S_rem_i <= 0:
    return 0

  score = 0

  # ---------- 信号1: 当前实例数 vs 基准 ----------
  if K_i < K_i^base:
    # 还没到基准，分数很高
    deficit_ratio = (K_i^base - K_i) / K_i^base
    score += 50 * deficit_ratio  # 最多+50分

  # ---------- 信号2: 忙实例比例 ----------
  if K_i > 0:
    busy_ratio = K_i_busy / K_i
    score += 30 * busy_ratio  # 都在忙的话+30分

  # ---------- 信号3: 剩余样本充足度 ----------
  # 假设每个实例每轮能处理 S_per_instance 个样本
  # 剩余样本够不够当前实例吃好几轮？
  if K_i > 0:
    samples_per_instance_per_round = S_per_round_i / K_i
    if samples_per_instance_per_round > 0:
      rounds_left_at_current = task_i.S_rem_i / samples_per_instance_per_round / K_i
      # 剩余样本够吃越久，越值得给更多实例
      sample_sufficiency = min(rounds_left_at_current / 5, 1.0)  # 够吃5轮就满分
      score += 20 * sample_sufficiency  # 最多+20分

  return score
```

**设计理由**：
- 不需要预测未来
- 所有信号都是当前可观测的
- 分数是启发式的，但方向正确
- 优先帮助那些在推理阶段、缺实例、所有实例都在忙、剩余样本还很多的任务

### 4.4 拓扑感知的放置算法

**FindBestPlacement(task_i, free_gpus)**: 为任务i找到最佳GPU放置

```
函数 FindBestPlacement(task_i, free_gpus):
  # 需要 tp_i 张GPU组成一个实例
  required = task_i.tp_i

  # 按机器分组空闲GPU
  free_by_machine = GroupByMachine(free_gpus)

  best_placement = None
  best_score = -infinity

  # 策略1: 优先找同一台机器上有足够GPU的
  for machine m in free_by_machine:
    if len(free_by_machine[m]) >= required:
      # 同一机器有足够GPU，完美放置
      placement = PickGPUsFromMachine(m, required)
      score = ColocalityScore(placement)  # 满分
      return placement

  # 策略2: 找跨机器但机器数最少的
  for candidate_placement in GenerateCandidatePlacements(free_gpus, required):
    score = ColocalityScore(candidate_placement)
    if score > best_score:
      best_score = score
      best_placement = candidate_placement

  return best_placement

函数 ColocalityScore(placement):
  # 计算放置的拓扑分数
  # 同一机器的GPU越多分数越高
  machine_counts = CountMachines(placement)
  max_in_one_machine = max(machine_counts.values())
  return max_in_one_machine / len(placement)  # 0-1之间，越高越好
```

### 4.5 回收决策下放

当调度器决定从任务j回收1个实例时：

1. 调度器只发送：`请回收1个实例`
2. 任务Agent自己决定回收哪1个实例：
   - 优先回收空闲实例
   - 如果没有空闲实例，优先回收最慢的实例
   - 确保剩下的实例有良好的拓扑放置

**调度器不关心具体回收哪几张卡，只关心数量。**

### 4.6 长尾检测

**HasLongTail(i)**: 判断任务i是否正在经历长尾

```
函数 HasLongTail(i):
  """
  长尾的正确定义：
  不是"实例间样本数差异大"（因为每个样本推理时间本来就不同）
  而是"部分实例已经没有样本可取了，而其他实例还在执行推理"
  """
  if not InRolloutPhase(i):
    return False

  # 检查是否有实例已经空闲（没样本可取了），同时还有实例在忙
  has_idle_instances = (K_i_idle > 0)
  has_busy_instances = (K_i_busy > 0)

  # 部分实例闲了、部分还在忙 → 长尾
  return has_idle_instances and has_busy_instances
```

**注意**：检测到长尾只是用于观测，**不用于**分配决策（因为给长尾任务加实例没用，本轮样本已分配完）。

## 5. 系统架构与接口

### 5.1 系统架构

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           全局调度器                                       │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐ │
│  │  状态管理器      │  │  调度决策引擎    │  │  收益评分器          │ │
│  │  - 任务状态      │  │  - P0任务处理     │  │  - Compute-         │ │
│  │  - GPU池状态     │  │  - P1/P2任务分配  │  │    AllocationScore()│ │
│  │  - 机器拓扑      │  │  - 回收决策       │  │                      │ │
│  └──────────────────┘  └──────────────────┘  └──────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    拓扑放置管理器                                  │   │
│  │  - FindBestPlacement()                                          │   │
│  │  - ColocalityScore()                                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└────────┬─────────────────────────────────────────────────────────────────┘
         │
         │  RPC / gRPC
         │
┌────────▼──────────┐  ┌────────▼──────────┐  ┌────────▼──────────┐
│  任务Agent A      │  │  任务Agent B      │  │  任务Agent C      │
│  ┌─────────────┐  │  │  ┌─────────────┐  │  │  ┌─────────────┐  │
│  │  信号生成器 │  │  │  │  信号生成器 │  │  │  │  信号生成器 │  │
│  │  - needs-   │  │  │  │  - needs-   │  │  │  │  - needs-   │  │
│  │    more-    │  │  │  │    more-    │  │  │  │    more-    │  │
│  │    instances│  │  │  │    instances│  │  │  │    instances│  │
│  │  - can-     │  │  │  │  - can-     │  │  │  │  - can-     │  │
│  │    release-  │  │  │  │    release-  │  │  │  │    release-  │  │
│  │    instances│  │  │  │    instances│  │  │  │    instances│  │
│  │  - 长尾检测 │  │  │  │  - 长尾检测 │  │  │  │  - 长尾检测 │  │
│  └─────────────┘  │  │  └─────────────┘  │  │  └─────────────┘  │
│  ┌─────────────┐  │  │  ┌─────────────┐  │  │  ┌─────────────┐  │
│  │ GPU管理器   │  │  │  │ GPU管理器   │  │  │  │ GPU管理器   │  │
│  │ - 权重加载  │  │  │  │ - 权重加载  │  │  │  │ - 权重加载  │  │
│  │ - 实例启停  │  │  │  │ - 实例启停  │  │  │  │ - 实例启停  │  │
│  │ - 回收决策  │  │  │  │ - 回收决策  │  │  │  │ - 回收决策  │  │
│  └─────────────┘  │  │  └─────────────┘  │  │  └─────────────┘  │
└────────────────────┘  └────────────────────┘  └────────────────────┘
```

### 5.2 Protobuf 接口定义

```protobuf
// GPU位置信息
message GPUPlacement {
  int32 gpu_id = 1;
  int32 machine_id = 2;
}

// 单个推理实例的完整状态
message InstanceState {
  int32 instance_id = 1;
  bool is_busy = 2;

  // 推理阶段的实时指标
  double elapsed_time_sec = 3;    // τ_elapsed_j
  int32 done_samples = 4;          // S_done_j
  int32 remaining_samples = 5;     // S_rem_j

  // GPU放置信息
  repeated GPUPlacement gpus = 6;   // 这个实例占用的GPU
}

// 任务配置
message TaskConfig {
  string task_id = 1;
  int32 base_instances = 2;      // K_i^base
  int32 tp = 3;                    // 2的幂
  int32 pp = 4;                    // 2的幂
  int32 samples_per_round = 5;   // S_per_round_i
  int32 total_samples = 6;        // S_total_i
}

// 各阶段耗时指标（可选，仅用于观测）
message PhaseMetrics {
  double weight_transfer_sec = 1;
  double rollout_gen_sec = 2;
  double rollout_tool_sec = 3;
  double ref_log_prob_sec = 4;
  double reward_sec = 5;
  double adv_sec = 6;
  double update_sec = 7;
  double total_round_sec = 8;
}

// 任务状态上报
message TaskStateReport {
  string task_id = 1;

  // 进度
  int32 done_samples = 2;
  int32 done_rounds = 3;
  double elapsed_time_sec = 4;

  // 当前分配
  int32 current_instances = 5;
  int32 idle_instances = 6;

  // ========== 核心调度信号 ==========
  bool needs_more_instances = 7;   // 需要增加实例到 K_i^base
  bool can_release_instances = 8;   // 有空闲实例可以释放
  bool in_rollout_phase = 9;        // 是否处于推理阶段
  int32 remaining_samples = 10;      // S_rem_i

  // 各实例状态
  repeated InstanceState instances = 11;

  // 本轮最新的阶段耗时（可选，用于观测）
  PhaseMetrics latest_metrics = 12;
}

// 调度决策：为单个任务的分配
message TaskAllocation {
  string task_id = 1;

  // 目标实例数（调度器只决定数量）
  int32 target_instances = 2;

  // 推荐的GPU放置（任务可以选择不遵守）
  repeated GPUPlacement recommended_gpus = 3;
}

// 调度决策
message SchedulingDecision {
  repeated TaskAllocation allocations = 1;

  // 决策时间戳
  double timestamp_sec = 2;
}

// 任务Agent上报：回收确认
message ReclaimConfirm {
  string task_id = 1;
  int32 reclaimed_instances = 2;
  repeated GPUPlacement reclaimed_gpus = 3;  // 具体回收了哪些GPU
}

// 调度器服务
service GlobalScheduler {
  // 任务注册
  rpc RegisterTask(TaskConfig) returns (RegisterResponse);

  // 任务状态上报（触发调度）
  rpc ReportState(TaskStateReport) returns (SchedulingDecision);

  // 任务退出
  rpc UnregisterTask(UnregisterRequest) returns (Empty);
}
```

## 6. 调优参数

| 参数名 | 推荐初始值 | 说明 |
|--------|-----------|------|
| `max_extra_instances_ratio` | 1.5 | 最多给任务额外分配多少倍 `K_i^base`（P2阶段） |

## 7. 边界情况处理

**新任务加入**：
- 初始分配给它 `K_i^base` 个实例（如果有足够GPU）
- 如果GPU不够，先给 `K_i^min` 个实例，有空闲时再补上

**任务退出**：
- 回收该任务的所有GPU
- 重新触发一次调度，把GPU分配给其他任务

**某个任务严重落后**：
- 任务自己会检测到并上报 `needs_more_instances = true`
- 调度器优先满足这类任务，恢复到 `K_i^base`

**GPU不足以满足所有 `needs_more_instances`**：
- 按任务提交顺序或者优先级分配
- 或者临时启用"抢占模式"：从 `can_release_instances` 的任务回收

## 8. 可观测性

**调度器级别监控**：
- 调度决策频率
- GPU利用率（总、平均）
- 各任务的GPU分配变化历史
- 切换次数/频率
- P0/P1任务队列长度

**任务级别监控**：
- 每个任务的 `needs_more_instances` 事件计数
- 每个任务的 `can_release_instances` 事件计数
- 每个任务的长尾因子时间序列
- 每个任务的 `K_i(t)` vs `K_i^base`

## 9. 降级模式

如果调度器出现问题，可以降级到：
1. **固定分配模式**：所有任务保持 `K_i^base` 不变
2. **简单均分模式**：所有任务平分GPU
