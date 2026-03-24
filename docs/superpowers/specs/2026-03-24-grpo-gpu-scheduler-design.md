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

### 1.1 重要设计约束补充

1. **历史数据不确定性**：每个任务的历史平均每轮耗时、总耗时受到总卡数、并行策略等配置的影响，不一定能拿到准确值
2. **并行策略多样性**：每个任务的并行策略（tp/pp）不一样，但都是2的倍数
3. **拓扑感知**：一个tp组尽可能要放在同一台机器上，否则推理时延会非常高，而且预测也不准
4. **回收决策下放**：当中心调度器决策要回收某个任务的几张卡时，让任务本身来决定回收哪几张卡

## 2. 优化目标

1. **约束条件**：每个强化学习训练任务的端到端耗时不能比固定分配时更长
2. **优化目标**：尽可能让所有训练任务的平均端到端耗时最低

## 3. 核心概念与数学模型

### 3.1 符号定义

**任务级指标**：
- `T_i`：第i个任务
- `K_i^base`：任务i的固定分配基准实例数
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

**时间指标**（按阶段）：
- `τ_wt_i`：权重传输时间
- `τ_rg_i`：rollout生成时间（包含old_log_prob，单个实例，单个batch）
- `τ_rt_i`：rollout工具调用时间
- `τ_rlp_i`：ref_log_prob计算时间
- `τ_rwd_i`：reward计算时间
- `τ_adv_i`：advantage计算时间
- `τ_upd_i`：update训练时间

**进度指标**：
- `S_total_i`：任务i的总样本数
- `S_done_i`：已完成样本数
- `S_rem_i = S_total_i - S_done_i`：剩余样本数
- `R_done_i`：已完成轮次数
- `R_total_i = ceil(S_total_i / S_per_round_i)`：总轮次数
- `R_rem_i = R_total_i - R_done_i`：剩余轮次数
- `T_start_i`：任务i的启动时间
- `T_now`：当前时间
- `T_elapsed_i = T_now - T_start_i`：已用时间

**当前状态指标**（实时指标）：
- `K_i_idle(t)`：时刻t任务i的空闲实例数（已完成当前轮次的推理）
- `K_i_busy(t) = K_i(t) - K_i_idle(t)`：忙实例数（仍在推理中）
- 对每个忙实例j：
  - `τ_elapsed_j`：已用时间
  - `S_done_j`：已完成样本数
  - `S_rem_j`：剩余样本数

### 3.2 基准耗时模型（固定分配时）

**重要说明**：由于历史平均每轮耗时受到总卡数、并行策略等配置影响，不一定能拿到准确值。我们采用**保守估计策略**：

1. 如果有基准分配下的历史数据，用历史数据
2. 如果没有，用当前观测到的最快轮次作为基准下界
3. 当数据不足时，倾向于不回收GPU（保守策略）

单轮完整耗时（基准分配 `K_i^base` 个实例）：

```
τ_round_i^base = τ_wt_i + τ_rollout_round_i^base + τ_non_rollout_i
```

其中：
- `τ_rollout_round_i^base`：基准分配下，整轮rollout的耗时（所有实例完成的时间，即长尾时间）
- `τ_non_rollout_i = max(τ_rlp_i, τ_rwd_i) + τ_adv_i + τ_upd_i`

基准总耗时：
```
如果有历史数据:
  τ_round_avg_i^base = 历史平均每轮耗时（固定分配下）
否则:
  τ_round_avg_i^base = min(观测到的所有轮次耗时)  # 保守估计
T_total_i^base = R_total_i * τ_round_avg_i^base
```

基准剩余耗时：
```
T_rem_i^base = R_rem_i * τ_round_avg_i^base
```

**数据不足时的降级策略**：
- 当历史轮次 < 3：不回收任何GPU，只分配空闲GPU
- 当历史轮次 < 5：回收时加倍边际损失（更保守）

### 3.3 欠债追踪 - 性能约束保证

**松弛时间**（slack）：
```
Slack_i = T_total_i^base - T_elapsed_i - T_rem_i(K_i^base)
```
- `Slack_i > 0`：任务i有时间缓冲，可以承受变慢
- `Slack_i ≤ 0`：任务i正在落后，需要加速

**欠债追踪器**（简化版，用轮次而非积分）：

每当完成一轮：
```
Δ_debt = (τ_round_actual_i - τ_round_avg_i^base)
Debt_i += Δ_debt
```

如果这一轮用了 `K_i < K_i^base`，`Δ_debt` 通常为正（欠债）；
如果用了 `K_i > K_i^base`，`Δ_debt` 可能为负（还债）。

**约束条件**：
在任何时刻，满足：
```
T_elapsed_i + T_rem_i(K_i^base) + max(Debt_i, 0) ≤ T_total_i^base
```

### 3.4 Rollout耗时估计（基于实时指标）

当处于推理阶段中，当前轮次的剩余推理时间估计：
```
τ_rollout_rem_i ≈ max_{j in busy_instances} [ (τ_elapsed_j / S_done_j) * S_rem_j ]
```
（假设每个实例继续以当前的平均速度处理剩余样本）

对于不同K值的历史rollout时间，用插值或历史记录估计。

## 4. 调度决策流程

### 4.1 调度触发事件

以下任一事件触发调度决策：
1. **实例完成事件**：某个任务的一个推理实例完成了当前batch
2. **轮次完成事件**：某个任务完成了一整轮（推理+训练）
3. **任务加入事件**：新任务提交到集群
4. **任务退出事件**：某个任务完成退出

### 4.2 调度决策算法（贪心边际收益 + 拓扑感知）

```
函数 Schedule():
  # 步骤0: 检查历史数据充分性
  for each task i:
    i.has_sufficient_history = (i.done_rounds >= 3)
    i.very_sufficient_history = (i.done_rounds >= 5)

  # 步骤1: 约束检查 - 确保所有任务即使现在开始只保留基准分配也能按时完成
  for each task i:
    if not CheckConstraint(i):
      # 这个任务不能再被回收GPU了，标记为不可回收
      i.can_reclaim = False
      # 如果它的 K_i < K_i^base，需要先还回来
      if K_i < K_i^base:
        NeedToRestore[i] = K_i^base - K_i
    else:
      # 历史数据不足时更保守
      if i.has_sufficient_history:
        i.can_reclaim = (K_i > K_i^min)  # 至少保留1个实例
      else:
        i.can_reclaim = (K_i > K_i^base)  # 数据不足时只回收超过基准的部分

  # 步骤2: 收集可回收的GPU
  Reclaimable = []
  for each task i where i.can_reclaim:
    # 计算从这个任务回收1个实例的边际"损失"
    loss_i = ComputeMarginalLoss(i)

    # 历史数据不足时加倍损失（更保守）
    if not i.very_sufficient_history:
      loss_i *= 2.0

    Reclaimable.append( (loss_i, i) )

  # 按损失从小到大排序（先回收损失最小的）
  Reclaimable.sort()

  # 步骤3: 计算各任务增加1个实例的边际收益
  GainCandidates = []
  for each task i:
    # 检查是否可以接受更多GPU（可以是K_i > K_i^base来还债）
    gain_i = ComputeMarginalGain(i)
    GainCandidates.append( (-gain_i, i) )  # 负号用于排序

  # 按收益从大到小排序
  GainCandidates.sort()

  # 步骤4: 贪心分配 - 先满足需要恢复基准分配的任务
  for each task i in NeedToRestore:
    while NeedToRestore[i] > 0 and (C_free > 0 or Reclaimable not empty):
      if C_free >= cards_per_instance_i:
        # 有空闲GPU，直接分配（考虑拓扑）
        placement = FindBestPlacement(i, free_gpus)
        Allocate(i, 1, placement)
        NeedToRestore[i] -= 1
      else:
        # 需要回收
        (loss, j) = Reclaimable.pop(0)
        Reclaim(j, 1)  # 只是通知任务回收1个实例，任务自己决定回收哪几个

  # 步骤5: 贪心分配 - 把回收/空闲的GPU分配给收益最高的任务
  while C_free > 0 or Reclaimable not empty:
    if GainCandidates is empty:
      break

    (neg_gain, i) = GainCandidates.pop(0)
    gain_i = -neg_gain

    # 收益必须为正才值得分配
    if gain_i <= 0:
      continue

    cards_needed = cards_per_instance_i

    if C_free >= cards_needed:
      # 有空闲GPU，直接分配（考虑拓扑）
      placement = FindBestPlacement(i, free_gpus)
      if placement is not None:
        Allocate(i, 1, placement)
        # 重新计算这个任务的下一个边际收益
        new_gain = ComputeMarginalGain(i)
        if new_gain > 0:
          insert (-new_gain, i) into GainCandidates in sorted order
    else:
      # 需要看回收是否划算
      if Reclaimable is empty:
        break

      (loss_j, j) = Reclaimable[0]

      # 只有当 gain_i > loss_j + 切换开销 时才值得做
      if gain_i > loss_j + τ_wt_i:
        (loss_j, j) = Reclaimable.pop(0)
        # 通知任务j回收1个实例 - 任务自己决定回收哪几个
        Reclaim(j, 1)
        # 现在应该有GPU了，分配给i
        placement = FindBestPlacement(i, free_gpus)
        if placement is not None:
          Allocate(i, 1, placement)
          # 重新计算i的下一个边际收益
          new_gain = ComputeMarginalGain(i)
          if new_gain > 0:
            insert (-new_gain, i) into GainCandidates in sorted order
          # 重新计算j的下一个边际损失
          if j.can_reclaim:
            new_loss = ComputeMarginalLoss(j)
            if not j.very_sufficient_history:
              new_loss *= 2.0
            insert (new_loss, j) into Reclaimable in sorted order
      else:
        # 不划算，停止
        break
```

### 4.3 拓扑感知的放置算法

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

### 4.4 回收决策下放

当调度器决定从任务j回收1个实例时：

1. 调度器只发送：`请回收1个实例`
2. 任务Agent自己决定回收哪1个实例：
   - 优先回收空闲实例
   - 如果没有空闲实例，优先回收最慢的实例
   - 确保剩下的实例有良好的拓扑放置

**调度器不关心具体回收哪几张卡，只关心数量。**

### 4.3 关键函数定义

**CheckConstraint(i)**: 检查任务i的约束是否满足
```
函数 CheckConstraint(i):
  # 即使现在开始只保留基准分配，也能按时完成吗？
  # 假设我们立即把K_i恢复到K_i^base
  T_remaining_needed = T_rem_i^base + max(Debt_i, 0)
  return (T_elapsed_i + T_remaining_needed) ≤ T_total_i^base
```

**ComputeMarginalLoss(i)**: 计算从任务i回收1个实例的边际损失
```
函数 ComputeMarginalLoss(i):
  if K_i == K_i^min:
    return infinity  # 不能再少了

  # 当前K_i个实例时，估计剩余时间
  τ_rem_current = EstimateRemainingTime(i, K_i)

  # 如果变成K_i-1个实例，估计剩余时间
  τ_rem_after = EstimateRemainingTime(i, K_i - 1)

  # 边际损失 = 耗时增加量
  loss = τ_rem_after - τ_rem_current

  # 如果这个任务当前有空闲实例，损失为0或很小（因为回收的是空闲的）
  if K_i_idle > 0:
    loss *= α_idle  # α_idle = 0.1

  return max(loss, 0)
```

**ComputeMarginalGain(i)**: 计算给任务i增加1个实例的边际收益
```
函数 ComputeMarginalGain(i):
  # 当前K_i个实例时，估计剩余时间
  τ_rem_current = EstimateRemainingTime(i, K_i)

  # 如果变成K_i+1个实例，估计剩余时间
  τ_rem_after = EstimateRemainingTime(i, K_i + 1)

  # 边际收益 = 耗时减少量
  gain = τ_rem_current - τ_rem_after

  # 如果这个任务正在经历长尾，收益更高
  if HasLongTail(i):
    gain *= α_longtail  # α_longtail = 1.5

  # 如果这个任务欠债，还债有额外收益
  if Debt_i > 0:
    gain *= α_debt  # α_debt = 1.2

  return max(gain, 0)
```

**EstimateRemainingTime(i, K)**: 估计任务i在K个实例下的剩余时间
```
函数 EstimateRemainingTime(i, K):
  # 情况1: 当前正在推理阶段
  if InRolloutPhase(i):
    # 估计当前轮次的剩余推理时间
    τ_rollout_rem = EstimateRolloutRemaining(i, K)

    # 加上非推理阶段时间
    τ_non_rollout = τ_non_rollout_i  # 历史值

    # 当前轮剩余时间
    τ_round_rem = τ_rollout_rem + τ_non_rollout

    # 后续轮次
    if K > 0:
      # 用历史数据估计后续轮次时间
      τ_round_future = EstimateFutureRoundTime(i, K)
    else:
      τ_round_future = infinity

    τ_rem = τ_round_rem + (R_rem_i - 1) * τ_round_future

  # 情况2: 当前正在训练阶段
  else:
    # 当前轮剩余时间（训练剩下的部分）
    τ_upd_rem = EstimateUpdateRemaining(i)

    # 后续轮次
    τ_round_future = EstimateFutureRoundTime(i, K)
    τ_rem = τ_upd_rem + R_rem_i * τ_round_future

  return τ_rem
```

**EstimateRolloutRemaining(i, K)**: 估计当前推理阶段的剩余时间
```
函数 EstimateRolloutRemaining(i, K):
  # K是假设的实例数，但我们当前有K_i个实例
  # 如果K == K_i，用实时指标估计
  if K == K_i and K_i_busy > 0:
    # 对每个忙实例，估计它的剩余时间
    max_rem = 0
    for each busy instance j:
      if S_done_j > 0:
        # 当前平均速度
        speed = S_done_j / τ_elapsed_j
        if speed > 0:
          τ_rem_j = S_rem_j / speed
          max_rem = max(max_rem, τ_rem_j)
    return max_rem

  # 如果K != K_i，或者没有实时数据，用历史估计
  return EstimateRolloutTimeFromHistory(i, K)
```

**HasLongTail(i)**: 判断任务i是否正在经历长尾
```
函数 HasLongTail(i):
  if not InRolloutPhase(i) or K_i_busy == 0:
    return False

  # 计算各忙实例的已完成样本数
  S_done_list = [j.S_done_j for j in i.instances if j.is_busy]

  if len(S_done_list) < 2:
    return False

  # 如果最快和最慢的差距超过阈值，认为有长尾
  S_done_list.sort()
  fastest = S_done_list[-1]
  slowest = S_done_list[0]

  return fastest > tail_ratio_threshold * slowest
```

## 5. 补充设计说明

### 5.1 处理历史数据不确定性

由于历史平均耗时受配置影响，不一定准确，我们采用**保守策略**：

| 历史轮次数 | 策略 |
|-----------|------|
| 0-2轮 | 不回收GPU，只分配空闲GPU |
| 3-4轮 | 可以回收，但边际损失 × 2 |
| ≥5轮 | 正常调度 |

**设计理由**：数据不足时，错误决策的风险很高，宁愿不优化也不要让任务变慢。

---

### 5.2 处理并行策略多样性

每个任务的tp/pp可以不同，但都是2的幂：

- 调度器支持异构任务
- 分配时检查任务i的 `cards_per_instance_i = tp_i * pp_i`
- GPU池必须能被各种 `cards_per_instance_i` 分配

---

### 5.3 拓扑感知设计

**核心原则**：一个tp组尽可能放在同一台机器上。

**设计决策**：
1. 调度器维护集群拓扑状态：哪些GPU在同一台机器
2. 分配时调用 `FindBestPlacement()` 找最佳放置
3. 优先同一机器 → 最少跨机器 → 任意放置
4. 如果找不到满足拓扑要求的放置，拒绝分配（而不是分配坏的放置）

**ColocalityScore计算**：
```
score = (同一机器上的最大GPU数) / tp_i
范围: [0, 1]，1表示所有GPU在同一机器
```

---

### 5.4 回收决策下放

**责任划分**：

| 组件 | 决策内容 |
|------|---------|
| 全局调度器 | 从哪个任务回收、回收几个实例 |
| 任务Agent | 回收具体哪几个实例、哪几张卡 |

**任务Agent的回收优先级**：
1. 优先回收**空闲实例**（idle_instances）
2. 其次回收**最慢的实例**（长尾制造者）
3. 确保剩下的实例有良好的拓扑

**通信流程**：
```
调度器 → 任务Agent: "请回收1个实例"
任务Agent → 调度器: "已回收，具体是这几张卡：[...]"
```

---

## 6. 系统架构与接口

### 6.1 系统架构

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           全局调度器                                       │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐ │
│  │  状态管理器      │  │  调度决策引擎    │  │  欠债追踪器          │ │
│  │  - 任务状态      │  │  - 约束检查      │  │  - Debt_i            │ │
│  │  - GPU池状态     │  │  - 收益/损失计算 │  │  - Slack_i           │ │
│  │  - 机器拓扑      │  │  - 历史数据检查   │  │                      │ │
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
│  │  指标收集器 │  │  │  │  指标收集器 │  │  │  │  指标收集器 │  │
│  │  - 阶段耗时 │  │  │  │  - 阶段耗时 │  │  │  │  - 阶段耗时 │  │
│  │  - 实例状态 │  │  │  │  - 实例状态 │  │  │  │  - 实例状态 │  │
│  │  - 样本进度 │  │  │  │  - 样本进度 │  │  │  │  - 样本进度 │  │
│  │  - GPU位置   │  │  │  │  - GPU位置   │  │  │  │  - GPU位置   │  │
│  └─────────────┘  │  │  └─────────────┘  │  │  └─────────────┘  │
│  ┌─────────────┐  │  │  ┌─────────────┐  │  │  ┌─────────────┐  │
│  │ GPU管理器   │  │  │  │ GPU管理器   │  │  │  │ GPU管理器   │  │
│  │ - 权重加载  │  │  │  │ - 权重加载  │  │  │  │ - 权重加载  │  │
│  │ - 实例启停  │  │  │  │ - 实例启停  │  │  │  │ - 实例启停  │  │
│  │ - 回收决策  │  │  │  │ - 回收决策  │  │  │  │ - 回收决策  │  │
│  └─────────────┘  │  │  └─────────────┘  │  │  └─────────────┘  │
└────────────────────┘  └────────────────────┘  └────────────────────┘
```

### 6.2 Protobuf 接口定义

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

// 各阶段耗时指标
message PhaseMetrics {
  double weight_transfer_sec = 1;    // τ_wt_i
  double rollout_gen_sec = 2;         // τ_rg_i (包含old_log_prob)
  double rollout_tool_sec = 3;        // τ_rt_i
  double ref_log_prob_sec = 4;        // τ_rlp_i
  double reward_sec = 5;               // τ_rwd_i
  double adv_sec = 6;                  // τ_adv_i
  double update_sec = 7;               // τ_upd_i

  // 整轮耗时
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

  // 各实例状态
  repeated InstanceState instances = 7;

  // 本轮最新的阶段耗时
  PhaseMetrics latest_metrics = 8;

  // 历史平均每轮耗时（基准分配下）
  double avg_round_sec_base = 9;
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

## 7. 调优参数

| 参数名 | 推荐初始值 | 说明 |
|--------|-----------|------|
| `α_idle` | 0.1 | 空闲实例的边际损失权重 |
| `α_longtail` | 1.5 | 长尾任务的收益加成 |
| `α_debt` | 1.2 | 还债任务的收益加成 |
| `switch_cost_threshold` | 2.0 | 收益需要是切换成本的多少倍才值得切换 |
| `min_allocation_duration` | 300秒 | 最小分配持续时间 |
| `tail_ratio_threshold` | 2.0 | 判断长尾的快慢比例阈值 |

## 8. 边界情况处理

**新任务加入**：
- 初始分配给它 `K_i^base` 个实例（如果有足够GPU）
- 如果GPU不够，先给 `K_i^min` 个实例，有空闲时再补上
- 新任务的欠债初始化为0

**任务退出**：
- 回收该任务的所有GPU
- 重新触发一次调度，把GPU分配给其他任务

**某个任务严重落后**：
- `Slack_i < 0` 且 `Debt_i` 很大
- 触发紧急模式：
  - 优先给这个任务分配GPU
  - 可以从其他 `Slack_j > 0` 的任务回收GPU

**收益计算抖动**：
- 用指数移动平均平滑历史指标
- 设置最小分配持续时间

**切换开销过大**：
- 在 `ComputeMarginalGain` 中减去切换开销 `τ_wt_i`
- 设置"切换成本阈值"

## 9. 可观测性

**调度器级别监控**：
- 调度决策频率
- GPU利用率（总、平均）
- 各任务的GPU分配变化历史
- 切换次数/频率

**任务级别监控**：
- 每个任务的 `Debt_i` 时间序列
- 每个任务的 `Slack_i` 时间序列
- 每个任务的实际耗时 vs 基准耗时
- 每个任务的长尾因子时间序列

## 10. 降级模式

如果调度器出现问题，可以降级到：
1. **固定分配模式**：回到基准，不再动态调整
2. **简单均分模式**：所有任务平分GPU
