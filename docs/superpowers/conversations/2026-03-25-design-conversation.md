# GRPO GPU 调度器设计对话记录（2026-03-25）

## 本次对话的关键更新

### 用户反馈 1：长尾定义修正

> 因为每个样本的推理时间是不同的，实例之间完成的样本数差异大不代表真的有长尾。只有部分实例已经没有样本可以取了，而其他实例还在执行推理，才能明确这是有长尾现象。

**响应**：更新了 `HasLongTail()` 函数的定义：

```python
函数 HasLongTail(i):
  if not InRolloutPhase(i):
    return False
  # 检查是否有实例已经空闲（没样本可取了），同时还有实例在忙
  has_idle_instances = (K_i_idle > 0)
  has_busy_instances = (K_i_busy > 0)
  # 部分实例闲了、部分还在忙 → 长尾
  return has_idle_instances and has_busy_instances
```

---

### 用户反馈 2：不给长尾任务加实例

> 给长尾的任务增加实例可能是不合理的，因为长尾代表着我即使增加实例也无法分担本轮的样本了。所以根据长尾增加实例这个步骤可以去除，只需考虑先满足需要实例的任务，其次把富余的空闲实例分配给收益最大化的任务。现在没有了未来剩余时间的预测，还有什么其他方法来评估给任务i分配k个实例能带来的收益？

**响应**：

1. **移除长尾任务分配逻辑**：长尾只用于观测，不用于分配决策
2. **设计新的收益评估函数** `ComputeAllocationScore()`，基于当前可观测状态：
   - 信号1: `K_i < K_i^base` → 最多+50分
   - 信号2: 忙实例比例 → 最多+30分
   - 信号3: 剩余样本充足度 → 最多+20分

调度优先级简化为：
- **P0**: `needs_more_instances = true`（必须满足）
- **P1**: 推理阶段 + `K_i < K_i^base` + 有剩余样本
- **P2**: 推理阶段 + `K_i >= K_i^base` + 剩余样本多（只用空闲GPU）

---

### 用户反馈 3：文档整理

> 整理删除不需要的design和plan，保持文档简洁明了。

**响应**：
- ✅ 删除旧的实现计划：`plans/2026-03-24-grpo-gpu-scheduler.md`
- ✅ 删除旧的对话记录：`specs/2026-03-24-design-conversation.md`
- ✅ 重写设计文档，删除所有不需要保留的已删除设计内容
- ✅ 更新系统架构图

---

## 最终设计文档状态

保留文件：
- `docs/superpowers/specs/2026-03-24-grpo-gpu-scheduler-design.md`（最新设计）

已删除文件：
- `docs/superpowers/plans/2026-03-24-grpo-gpu-scheduler.md`
- `docs/superpowers/specs/2026-03-24-design-conversation.md`
