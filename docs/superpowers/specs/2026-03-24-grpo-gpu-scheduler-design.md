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

### 1.2 核心不变量

```python
assert not (S_rem_i > 0 and K_i^idle > 0)
```

**含义**：任务自己会管理好实例，当还有剩余样本时不会让实例空闲。

---

## 2. 调度核心流程

每次触发group scheduler调度会有以下四个阶段：

```python
delta_card_ranges = assess_range(task_states)
plan, excess_cards = dont_starve(delta_card_ranges)
if excess_cards:
    plan = feed_more(delta_card_ranges, plan, excess_cards)
execute(plan)
```

**关键点**：
- `assess_range`、`dont_starve`、`feed_more` 只计算**卡数调整量**，不涉及具体GPU放置
- `execute` 阶段才处理具体GPU分配，使用调度器维护的 `free_gpus` 表

### 2.1 符号定义

**任务级指标**：
- `T_i`：第i个任务
- `K_i^base`：任务i的固定分配基准实例数
- `K_i(t)`：时刻t任务i的实际实例数
- `K_i^idle(t)`：空闲实例数
- `K_i^busy(t)`：忙实例数（`K_i - K_i^idle`）
- `tp_i`：任务i的tensor parallelism（2的幂）
- `pp_i`：任务i的pipeline parallelism（2的幂）
- `cards_per_instance_i = tp_i * pp_i`：每个实例需要的卡数
- `S_rem_i`：剩余样本数

**集群状态**：
- `free_gpus`：调度器维护的空闲GPU表，包含具体哪台机器的哪几张卡
  - 格式：`[(machine_id, gpu_id), ...]`

---

## 3. 阶段一：assess_range() - 评估增减范围

计算每个任务的GPU卡数调整范围。

```python
def assess_range(task_states):
    """
    计算每个任务的卡数调整范围 [min_cards, max_cards]

    返回:
      delta_card_ranges[i] = (min_cards, max_cards)
        - min_cards: 必须调整的卡数（负数表示必须回收，正数表示必须增加）
        - max_cards: 最多可以调整的卡数（inf表示无上限）
    """
    delta_card_ranges = []

    for i in range(n_tasks):
        task = task_states[i]
        cards_per_instance = task.tp * task.pp

        # 情况0: 不在rollout阶段 → 所有空闲实例都可以回收，且不再分配
        if not task.in_rollout_phase:
            min_cards = -task.K_i^idle * cards_per_instance
            max_cards = 0
            delta_card_ranges.append( (min_cards, max_cards) )

        # 情况1: 有剩余样本，但忙实例数没到基线 → 必须增加
        elif task.K_i^busy < task.K_i^base and task.S_rem_i > 0:
            min_cards = (catch_up_ratio * task.K_i^base - task.K_i^busy) * cards_per_instance
            max_cards = float('inf')
            delta_card_ranges.append( (min_cards, max_cards) )

        # 情况2: 有空闲实例，且没有剩余样本 → 必须回收
        elif task.K_i^idle > 0 and task.S_rem_i == 0:
            min_cards = -task.K_i^idle * cards_per_instance
            max_cards = 0
            delta_card_ranges.append( (min_cards, max_cards) )

        # 情况3: 忙实例数超过基线 → 可以回收超额部分
        elif task.K_i^busy > task.K_i^base:
            min_cards = -(task.K_i^busy - task.K_i^base) * cards_per_instance
            max_cards = float('inf')
            delta_card_ranges.append( (min_cards, max_cards) )

        # 情况4: 其他 → 不强制调整
        else:
            min_cards = 0
            max_cards = float('inf')
            delta_card_ranges.append( (min_cards, max_cards) )

    return delta_card_ranges
```

---

## 4. 阶段二：dont_starve() - 优先满足"没小康"的任务

优先满足所有 `min > 0` 的任务（也就是"没小康"的任务：有剩余样本但忙实例数没到基线）。

**回收优先级**：
1. 优先从本就空闲的卡中选
2. 如果没有空闲卡，先从"富得流油"的任务回收（`K_i^idle > 0`）
3. 再从"遥遥领先"的任务回收（`K_i^busy > K_i^base`）

**注意**：这一阶段只计算卡数，不做具体GPU放置。

```python
def dont_starve(delta_card_ranges):
    """
    优先满足min > 0的任务，生成初步plan

    返回:
      plan: 每个任务的卡数调整量（正数=增加，负数=回收）
      excess_cards: 富余的卡数（可以继续分配）
    """
    plan = [0] * n_tasks
    free_card_count = get_current_free_card_count()  # 只统计数量，不关心具体GPU

    # ---------- 第一步: 收集需求 ----------
    needy_tasks = []  # 需要增加卡的任务 (min > 0)
    reclaimable_tasks = []  # 可以回收卡的任务

    for i in range(n_tasks):
        min_cards, max_cards = delta_card_ranges[i]

        if min_cards > 0:
            needy_tasks.append( (i, min_cards) )
        elif min_cards < 0:
            # 可以回收，按优先级排序
            task = task_states[i]
            if task.K_i^idle > 0:
                priority = 0  # 最高优先级：有空闲实例
            else:
                priority = 1  # 次优先级：忙实例超基线
            reclaimable_tasks.append( (priority, i, -min_cards) )

    # 按优先级排序可回收任务
    reclaimable_tasks.sort()

    # ---------- 第二步: 满足 needy_tasks ----------
    for (i, needed_cards) in needy_tasks:
        while needed_cards > 0:
            cards_per_instance = task_states[i].tp * task_states[i].pp

            # 先看有没有足够的空闲卡
            if free_card_count >= cards_per_instance:
                plan[i] += cards_per_instance
                free_card_count -= cards_per_instance
                needed_cards -= cards_per_instance
            else:
                # 需要回收
                if not reclaimable_tasks:
                    break

                priority, j, reclaimable_cards = reclaimable_tasks.pop(0)
                cards_per_instance_j = task_states[j].tp * task_states[j].pp

                # 回收一个实例
                reclaim_amount = min(reclaimable_cards, cards_per_instance_j)
                plan[j] -= reclaim_amount
                free_card_count += reclaim_amount

                # 如果还能回收，放回去
                remaining = reclaimable_cards - reclaim_amount
                if remaining > 0:
                    reclaimable_tasks.append( (priority, j, remaining) )
                    reclaimable_tasks.sort()

    # ---------- 第三步: 计算富余卡 ----------
    excess_cards = free_card_count

    return plan, excess_cards
```

---

## 5. 阶段三：feed_more() - 富余卡喂给收益最大的任务

如果 `excess_cards` 不为空，则选择收益最大的任务分发，分发数量可以按剩余样本量估计。

**注意**：这一阶段同样只计算卡数，不做具体GPU放置。

```python
def feed_more(delta_card_ranges, plan, excess_cards):
    """
    把富余卡分配给收益最大的任务
    """
    # 计算每个任务的收益分
    task_scores = []
    for i in range(n_tasks):
        min_cards, max_cards = delta_card_ranges[i]
        task = task_states[i]

        # 跳过已经满足min的任务，或者不能再增加的任务
        if max_cards <= 0:
            continue

        # 计算收益分
        score = compute_allocation_score(task)
        if score > 0:
            task_scores.append( (-score, i) )  # 负号用于降序

    # 按分数排序
    task_scores.sort()

    # 分配富余卡
    while excess_cards > 0 and task_scores:
        neg_score, i = task_scores.pop(0)
        task = task_states[i]
        cards_per_instance = task.tp * task.pp

        # 检查上限
        current_cards = task.K_i * cards_per_instance + plan[i]
        max_allowed = acceleration_limit_ratio * task.K_i^base * cards_per_instance
        if current_cards >= max_allowed:
            continue

        if excess_cards >= cards_per_instance:
            plan[i] += cards_per_instance
            excess_cards -= cards_per_instance

            # 如果还有收益，继续排队
            new_score = compute_allocation_score_with_plan(task, plan[i])
            if new_score > 0:
                new_current = current_cards + cards_per_instance
                if new_current < acceleration_limit_ratio * task.K_i^base * cards_per_instance:
                    task_scores.append( (-new_score, i) )
                    task_scores.sort()

    return plan
```

### 5.1 收益评分函数

```python
def compute_allocation_score(task):
    """
    计算给任务分配一个实例的收益分数（0-100）
    """
    # 在训练阶段，给实例没用
    if not task.in_rollout_phase:
        return 0

    # 没有剩余样本了
    if task.S_rem_i <= 0:
        return 0

    score = 0

    # 信号1: 当前实例数 vs 基准
    if task.K_i < task.K_i^base:
        deficit_ratio = (task.K_i^base - task.K_i) / task.K_i^base
        score += 50 * deficit_ratio

    # 信号2: 忙实例比例
    if task.K_i > 0:
        busy_ratio = task.K_i^busy / task.K_i
        score += 30 * busy_ratio

    # 信号3: 剩余样本充足度
    if task.K_i > 0:
        samples_per_instance_per_round = task.S_per_round_i / task.K_i
        if samples_per_instance_per_round > 0:
            rounds_left = task.S_rem_i / samples_per_instance_per_round / task.K_i
            sample_sufficiency = min(rounds_left / 5, 1.0)
            score += 20 * sample_sufficiency

    return score
```

---

## 6. 阶段四：execute() - 执行plan

决策完成后，一起发起命令。先回收，再分配。

**关键点**：这一阶段才处理具体GPU放置，使用调度器维护的 `free_gpus` 表。

```python
def execute(plan):
    """
    执行调度plan

    plan[i]: 任务i的卡数调整量
      - 正数: 分配这么多卡
      - 负数: 回收这么多卡
      - 0: 不调整

    调度器维护 free_gpus 表：[(machine_id, gpu_id), ...]
    """
    # ---------- 第一步: 先执行回收 ----------
    for i in range(n_tasks):
        if plan[i] < 0:
            cards_to_reclaim = -plan[i]
            instances_to_reclaim = cards_to_reclaim // (task_states[i].tp * task_states[i].pp)

            # 让任务自己决定回收哪些GPU
            reclaimed_gpus = reclaim(task_states[i], instances_to_reclaim)

            # 更新 free_gpus 表
            free_gpus.extend(reclaimed_gpus)

    # ---------- 第二步: 再执行分配 ----------
    for i in range(n_tasks):
        if plan[i] > 0:
            cards_to_allocate = plan[i]
            instances_to_allocate = cards_to_allocate // (task_states[i].tp * task_states[i].pp)
            task = task_states[i]

            for _ in range(instances_to_allocate):
                # 使用 free_gpus 表找拓扑最佳的放置
                placement = find_best_placement(task, free_gpus)
                if placement:
                    allocate(task, 1, placement)

                    # 从 free_gpus 表中移除已分配的GPU
                    for (machine_id, gpu_id) in placement:
                        free_gpus.remove( (machine_id, gpu_id) )
```

---

## 7. 拓扑感知放置

调度器维护 `free_gpus` 表，记录哪些GPU是空闲的，具体到哪台机器的哪几张卡。

```python
def find_best_placement(task, free_gpus):
    """
    为任务找到最佳GPU放置
    优先同一机器 → 最少跨机器 → 任意放置

    free_gpus: [(machine_id, gpu_id), ...]
    """
    required = task.tp

    # 按机器分组空闲GPU
    free_by_machine = {}
    for (machine_id, gpu_id) in free_gpus:
        if machine_id not in free_by_machine:
            free_by_machine[machine_id] = []
        free_by_machine[machine_id].append(gpu_id)

    # 策略1: 优先找同一台机器上有足够GPU的
    for machine in free_by_machine:
        if len(free_by_machine[machine]) >= required:
            gpu_ids = free_by_machine[machine][:required]
            return [(machine, gpu_id) for gpu_id in gpu_ids]

    # 策略2: 找跨机器但机器数最少的
    best_placement = None
    best_score = -float('inf')

    for candidate in generate_candidate_placements(free_gpus, required):
        score = colocality_score(candidate)
        if score > best_score:
            best_score = score
            best_placement = candidate

    return best_placement

def colocality_score(placement):
    """计算放置的拓扑分数（同一机器的GPU越多分数越高）"""
    machine_counts = {}
    for (machine_id, gpu_id) in placement:
        machine_counts[machine_id] = machine_counts.get(machine_id, 0) + 1

    max_in_one_machine = max(machine_counts.values()) if machine_counts else 0
    return max_in_one_machine / len(placement)
```

---

## 8. 系统架构与接口

### 8.1 Protobuf 接口定义

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
  double elapsed_time_sec = 3;
  int32 done_samples = 4;
  int32 remaining_samples = 5;
  repeated GPUPlacement gpus = 6;
}

// 任务配置
message TaskConfig {
  string task_id = 1;
  int32 base_instances = 2;
  int32 tp = 3;
  int32 pp = 4;
  int32 samples_per_round = 5;
  int32 total_samples = 6;
}

// 任务状态上报
message TaskStateReport {
  string task_id = 1;

  // 进度
  int32 done_samples = 2;
  int32 done_rounds = 3;
  double elapsed_time_sec = 4;
  int32 remaining_samples = 5;

  // 当前分配
  int32 current_instances = 6;
  int32 idle_instances = 7;
  int32 busy_instances = 8;

  // 阶段
  bool in_rollout_phase = 9;

  // 各实例状态
  repeated InstanceState instances = 10;
}

// 调度决策：为单个任务的分配
message TaskAllocation {
  string task_id = 1;
  int32 instance_delta = 2;  // 正数=增加，负数=回收，0=不变
  repeated GPUPlacement recommended_gpus = 3;  // 只有分配时会填，回收时不会填
}

// 调度决策
message SchedulingDecision {
  repeated TaskAllocation allocations = 1;
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
  rpc RegisterTask(TaskConfig) returns (RegisterResponse);
  rpc ReportState(TaskStateReport) returns (SchedulingDecision);
  rpc UnregisterTask(UnregisterRequest) returns (Empty);
}
```

---

## 9. 调优参数

| 参数名 | 推荐初始值 | 说明 |
|--------|-----------|------|
| `catch_up_ratio` | 1.2 | `assess_range` 阶段使用：给落后任务多一些卡让它赶上进度（min_cards 计算） |
| `acceleration_limit_ratio` | 1.5 | `feed_more` 阶段使用：控制不给某个可加速任务分配太多卡 |

---

## 10. 边界情况处理

**新任务加入**：
- 初始分配给它 `K_i^base` 个实例（如果有足够GPU）
- 如果GPU不够，先给1个实例，有空闲时再补上

**任务退出**：
- 回收该任务的所有GPU，更新 `free_gpus` 表
- 重新触发一次调度

**GPU不足以满足所有 needy_tasks**：
- 按任务提交顺序或者优先级分配

---

## 11. 可观测性

**调度器级别监控**：
- 调度决策频率
- GPU利用率（总、平均）
- 各任务的GPU分配变化历史
- 切换次数/频率
- needy_tasks队列长度
- `free_gpus` 表大小和拓扑分布

**任务级别监控**：
- 每个任务的 `K_i^busy` vs `K_i^base`
- 每个任务的 `K_i^idle` 时间序列
- 每个任务的长尾检测状态

---

## 12. 降级模式

如果调度器出现问题，可以降级到：
1. **固定分配模式**：所有任务保持 `K_i^base` 不变
2. **简单均分模式**：所有任务平分GPU
