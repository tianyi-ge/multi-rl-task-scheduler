"""基准测试用例定义 - 扩展版本

约束：
1. 所有任务的base instance所需卡数之和 = 集群卡数
2. 每个任务的时间分布参数不同，确保差异化测试
"""

from models.test_case import TestCase
from models.cluster_config import ClusterConfig
from models.task_config import TaskConfig


def get_benchmark_cases() -> list:
    """基准测试套件：验证调度效果"""
    return [
        # ========== 基础场景（BB1-BB4）==========
        _create_bb1_two_tasks_competition(),
        _create_bb2_three_tasks_mixed_parallelism(),
        _create_bb3_large_scale_two_tasks(),
        _create_bb4_four_tasks_small_scale(),

        # ========== 复杂场景（BB5-BB11）==========
        _create_bb5_small_granularity_dense(),      # 小粒度密集任务
        _create_bb6_mixed_parallelism(),            # 混合并行粒度
        _create_bb7_large_instance(),               # 大实例场景
        _create_bb8_fragmentation_test(),           # 资源碎片化测试
        _create_bb9_strong_competition(),           # 多任务强竞争
        _create_bb10_dynamic_scaling(),             # 动态扩缩容
        _create_bb11_heterogeneous_cluster(),       # 异构集群场景

        # ========== 高调度密度场景（BB12）==========
        _create_bb12_multi_task_diverse_longtail(), # 多任务差异化长尾场景
    ]


# ========== 辅助函数 ==========

def _make_time_params(slow_ratio: float, slow_min: float, slow_max: float) -> dict:
    """创建长尾分布参数"""
    return {
        "slow_ratio": slow_ratio,
        "slow_min": slow_min,
        "slow_max": slow_max
    }


def _make_diverse_time_params(task_idx: int, num_tasks: int) -> dict:
    """
    为不同任务创建差异化时间分布参数

    策略：基于任务索引生成不同的 slow_ratio, slow_min, slow_max
    确保每个任务的时间分布特征不同

    长尾效应设置：
    - slow_ratio: 50%~70% 的样本是长尾
    - slow_min: 0.8倍~2.0倍
    - slow_max: 3.0倍~5.0倍
    """
    # slow_ratio: 0.5 ~ 0.7 (50%~70%长尾)
    slow_ratio = 0.5 + (task_idx % 3) * 0.1

    # slow_min: 0.8 ~ 2.0
    slow_min = 0.8 + (task_idx % 5) * 0.3

    # slow_max: 3.0 ~ 5.0
    slow_max = 3.0 + (task_idx % 5) * 0.5

    return {
        "slow_ratio": slow_ratio,
        "slow_min": slow_min,
        "slow_max": slow_max
    }


# ========== 基础场景 ==========

def _create_bb1_two_tasks_competition() -> TestCase:
    """BB1: 双任务竞争 - 基础对称场景

    集群: 32卡, 基线总和: 32卡 (满载)

    ┌─────────────────────────────────────────────────────────────────┐
    │  BB1 (双任务对称竞争)                                            │
    │                                                                 │
    │  task1: base=2实例(16卡), 64样本, 50%慢样本(0.8-3倍)             │
    │  task2: base=2实例(16卡), 64样本, 60%慢样本(1.5-4.5倍)           │
    │                                                                 │
    │  特点:                                                          │
    │  - 两任务base相同、样本相同                                       │
    │  - 长尾差异: task2比task1更慢(60% vs 50%, 4.5倍 vs 3倍)          │
    │  - task1先完成 → 释放16卡 → GS分配给task2                        │
    │                                                                 │
    │  GS调度机会:                                                     │
    │  - 长尾差异导致task1先完成                                        │
    │  - task1释放的16卡可给task2扩容                                  │
    │  - task2获得额外1实例(16卡) → 加速处理剩余样本                    │
    │                                                                 │
    │  验证目标: GS在基础双任务竞争场景的动态调度能力                   │
    └─────────────────────────────────────────────────────────────────┘
    """
    return TestCase(
        name="BB1_two_tasks_competition",
        description="双任务竞争场景 - 基础验证",
        cluster=ClusterConfig(machine_count=4, gpus_per_machine=8),  # 32张卡
        tasks=[
            TaskConfig(
                task_id="task1",
                tp=4, pp=2,  # 8卡/实例
                base_instances=2,  # 16张卡
                samples_per_round=8,
                num_rounds=8,  # 总样本64
                time_distribution="longtail_normal",
                distribution_params=_make_time_params(0.5, 0.8, 3.0)
            ),
            TaskConfig(
                task_id="task2",
                tp=4, pp=2,  # 8卡/实例
                base_instances=2,  # 16张卡
                samples_per_round=8,
                num_rounds=8,  # 总样本64
                time_distribution="longtail_normal",
                distribution_params=_make_time_params(0.6, 1.5, 4.5)
            )
        ]
    )


def _create_bb2_three_tasks_mixed_parallelism() -> TestCase:
    """BB2: 三任务不同并行策略 - 差异化长尾强调度场景

    集群: 32卡, 基线总和: 32卡 (满载)

    ┌─────────────────────────────────────────────────────────────────┐
    │  BB2 (三任务快慢分离)                                            │
    │                                                                 │
    │  fast_task1: base=1实例(8卡), 48样本, 20%慢样本(1-1.5倍)         │
    │            → 极快完成 → 释放8卡                                  │
    │                                                                 │
    │  fast_task2: base=1实例(8卡), 48样本, 25%慢样本(1-1.8倍)         │
    │            → 极快完成 → 释放8卡                                  │
    │                                                                 │
    │  slow_task: base=2实例(16卡), 256样本, 70%慢样本(2-5倍)          │
    │           → 持续处理 → 接收释放的16卡                            │
    │                                                                 │
    │  GS调度优势:                                                     │
    │  - 2个快任务完成后释放16卡                                        │
    │  - slow_task从16卡增至32卡(4实例)                                │
    │  - 获得100%额外算力, 加速处理200+样本                            │
    │  - 显著缩短总时间                                                │
    │                                                                 │
    │  设计要点:                                                       │
    │  - 不同并行策略(4×2, 8×1, 2×4)验证GS适应性                       │
    │  - 快任务样本少(48), 慢任务样本多(256)                            │
    │  - 快慢分离最大化GS调度收益                                       │
    │                                                                 │
    │  验证目标: 多任务、混合并行策略下的动态调度能力                   │
    └─────────────────────────────────────────────────────────────────┘
    """
    return TestCase(
        name="BB2_three_tasks_mixed_parallelism",
        description="三任务 - 混合并行策略(快慢分离)",
        cluster=ClusterConfig(machine_count=4, gpus_per_machine=8),  # 32张卡
        tasks=[
            TaskConfig(
                task_id="fast_task1",
                tp=4, pp=2,  # 8卡/实例
                base_instances=1,  # 8张卡
                samples_per_round=8,
                num_rounds=6,  # 总样本48，快速完成
                time_distribution="longtail_normal",
                distribution_params=_make_time_params(0.2, 1.0, 1.5)  # 20%长尾，轻微
            ),
            TaskConfig(
                task_id="fast_task2",
                tp=8, pp=1,  # 8卡/实例
                base_instances=1,  # 8张卡
                samples_per_round=8,
                num_rounds=6,  # 总样本48，快速完成
                time_distribution="longtail_normal",
                distribution_params=_make_time_params(0.25, 1.0, 1.8)  # 25%长尾，轻微
            ),
            TaskConfig(
                task_id="slow_task",
                tp=2, pp=4,  # 8卡/实例
                base_instances=2,  # 16张卡
                samples_per_round=16,
                num_rounds=16,  # 总样本256，持续处理
                time_distribution="longtail_normal",
                distribution_params=_make_time_params(0.7, 2.0, 5.0)  # 70%长尾，严重
            )
        ]
    )


def _create_bb3_large_scale_two_tasks() -> TestCase:
    """BB3: 大规模双任务 - 不对称设计(与BB7对称对比)

    集群: 64卡, 基线总和: 64卡 (满载)

    ┌─────────────────────────────────────────────────────────────────┐
    │  BB3 (不对称设计) - 与BB7对比                                     │
    │                                                                 │
    │  task1(快): base=1实例(16卡), 32样本, 10%慢样本(1-2倍)           │
    │           → ~2000s完成 → 释放16卡                               │
    │                                                                 │
    │  task2(慢): base=3实例(48卡), 384样本, 50%慢样本(2-5倍)          │
    │           → 需要更多时间                                         │
    │                                                                 │
    │  GS动态调度优势:                                                 │
    │  - task1完成后释放16卡                                           │
    │  - GS立即将16卡分配给task2                                       │
    │  - task2从48卡增至64卡(4实例), 获得33%额外算力                   │
    │  - task2加速处理剩余300+样本                                     │
    │                                                                 │
    │  对比BB7 (对称设计):                                             │
    │  - task1: base=2(32卡), 128样本                                  │
    │  - task2: base=2(32卡), 128样本                                  │
    │  - 两任务同时进行，无"早完成早释放"场景                           │
    │  - GS调度机会少，主要依靠长尾差异微调                             │
    └─────────────────────────────────────────────────────────────────┘
    """
    return TestCase(
        name="BB3_large_scale_two_tasks",
        description="大规模 - 不对称设计(早释放→动态调度)",
        cluster=ClusterConfig(machine_count=8, gpus_per_machine=8),  # 64张卡
        tasks=[
            TaskConfig(
                task_id="task1",
                tp=8, pp=2,  # 16卡/实例
                base_instances=1,  # base=16卡 (少样本快任务)
                samples_per_round=8,
                num_rounds=4,  # 总样本32 (极快任务)
                time_distribution="longtail_normal",
                distribution_params=_make_time_params(0.1, 1.0, 2.0)  # 10%慢样本(极快)
            ),
            TaskConfig(
                task_id="task2",
                tp=8, pp=2,  # 16卡/实例
                base_instances=3,  # base=48卡 (多样本慢任务)
                samples_per_round=48,
                num_rounds=8,  # 总样本384 (慢任务)
                time_distribution="longtail_normal",
                distribution_params=_make_time_params(0.5, 2.0, 5.0)  # 50%慢样本, 2-5倍长尾
            )
        ]
    )


def _create_bb4_four_tasks_small_scale() -> TestCase:
    """BB4: 四任务小规模 - 多任务竞争调度

    集群: 32卡, 基线总和: 32卡 (满载)

    ┌─────────────────────────────────────────────────────────────────┐
    │  BB4 (四任务差异化竞争)                                          │
    │                                                                 │
    │  task0-3: 各base=1实例(8卡), 各64样本                            │
    │  时间分布参数差异化:                                              │
    │  - task0: 50%慢样本(0.8-3倍)                                     │
    │  - task1: 60%慢样本(1.1-3.5倍)                                   │
    │  - task2: 70%慢样本(1.4-4倍)                                     │
    │  - task3: 80%慢样本(1.7-4.5倍)                                   │
    │                                                                 │
    │  特点:                                                          │
    │  - 4任务base相同、样本相同                                        │
    │  - 长尾参数逐级递增, 完成时间差异化                               │
    │  - task0最快完成 → 释放8卡 → 分给最慢的task3                      │
    │  - task1次快完成 → 释放8卡 → 分给剩余慢任务                       │
    │                                                                 │
    │  GS调度机会:                                                     │
    │  - 多次顺序释放和分配(4次机会)                                    │
    │  - 每次释放8卡, 分给仍在处理的长尾任务                            │
    │  - 级联调度效应                                                  │
    │                                                                 │
    │  验证目标: 多任务顺序完成场景的级联调度能力                       │
    └─────────────────────────────────────────────────────────────────┘
    """
    return TestCase(
        name="BB4_four_tasks_small_scale",
        description="四任务 - 小规模级联调度",
        cluster=ClusterConfig(machine_count=4, gpus_per_machine=8),  # 32张卡
        tasks=[
            TaskConfig(
                task_id=f"task{i}",
                tp=4, pp=2,  # 8卡/实例
                base_instances=1,  # 8张卡
                samples_per_round=8,
                num_rounds=8,  # 总样本64
                time_distribution="longtail_normal",
                distribution_params=_make_diverse_time_params(i, 4)
            )
            for i in range(4)
        ]
    )


# ========== 复杂场景 ==========

def _create_bb5_small_granularity_dense() -> TestCase:
    """BB5: 小粒度密集任务 - 8任务高频调度

    集群: 32卡, 基线总和: 32卡 (满载)

    ┌─────────────────────────────────────────────────────────────────┐
    │  BB5 (小粒度密集调度)                                            │
    │                                                                 │
    │  6个TP=2×1任务(2卡/实例):                                         │
    │  - 各base=2实例(4卡), 各32样本                                    │
    │  - 小粒度, 完成快, 调度频率高                                     │
    │                                                                 │
    │  2个TP=4×1任务(4卡/实例):                                         │
    │  - 各base=1实例(4卡), 各32样本                                    │
    │  - 中粒度                                                        │
    │                                                                 │
    │  特点:                                                          │
    │  - 8任务密集竞争, 调度密度最高                                    │
    │  - 小粒度(2卡/实例) → 快完成 → 快释放                            │
    │  - 高频释放/分配循环                                             │
    │                                                                 │
    │  GS调度机会:                                                     │
    │  - 多任务交替完成, 频繁释放小粒度资源                             │
    │  - 8个任务 → 最多8次顺序调度机会                                  │
    │  - 小卡数实例便于灵活组合分配                                     │
    │                                                                 │
    │  验证目标: 小粒度、高调度密度场景的调度效率                       │
    └─────────────────────────────────────────────────────────────────┘
    """
    return TestCase(
        name="BB5_small_granularity_dense",
        description="小粒度密集 - 8任务高频调度",
        cluster=ClusterConfig(machine_count=4, gpus_per_machine=8),  # 32张卡
        tasks=[
            # 6个TP=2,PP=1任务 (2卡/实例)
            TaskConfig(
                task_id=f"task{i}",
                tp=2, pp=1,  # 2卡/实例
                base_instances=2,  # 4张卡
                samples_per_round=4,
                num_rounds=8,  # 总样本32
                time_distribution="longtail_normal",
                distribution_params=_make_diverse_time_params(i, 8)
            )
            for i in range(6)
        ] + [
            # 2个TP=4,PP=1任务 (4卡/实例)
            TaskConfig(
                task_id=f"task{i+6}",
                tp=4, pp=1,  # 4卡/实例
                base_instances=1,  # 4张卡
                samples_per_round=4,
                num_rounds=8,  # 总样本32
                time_distribution="longtail_normal",
                distribution_params=_make_diverse_time_params(i+6, 8)
            )
            for i in range(2)
        ]
    )


def _create_bb6_mixed_parallelism() -> TestCase:
    """BB6: 混合并行粒度 - 多粒度资源分配

    集群: 32卡, 基线总和: 32卡 (满载)

    ┌─────────────────────────────────────────────────────────────────┐
    │  BB6 (混合粒度调度)                                              │
    │                                                                 │
    │  task_small(2×1): base=2实例(4卡), 32样本, 50%慢样本(0.8-3倍)     │
    │                 → 小粒度 → 快完成 → 释放4卡                      │
    │                                                                 │
    │  task_medium1(4×1): base=2实例(8卡), 64样本, 50%慢样本(1.4-4倍)   │
    │                   → 中粒度                                       │
    │                                                                 │
    │  task_medium2(2×2): base=1实例(4卡), 64样本, 60%慢样本(1.2-4倍)   │
    │                   → 中粒度                                       │
    │                                                                 │
    │  task_large(8×1): base=2实例(16卡), 128样本, 60%慢样本(1.8-5倍)   │
    │                  → 大粒度 → 持续处理 → 接收释放资源               │
    │                                                                 │
    │  GS调度优势:                                                     │
    │  - 小粒度任务先完成, 释放4卡                                      │
    │  - 4卡可分配给同粒度的task_medium2扩容                            │
    │  - 或累积后分配给大粒度task_large                                 │
    │  - 多粒度混合, 验证GS资源分配灵活性                               │
    │                                                                 │
    │  设计要点:                                                       │
    │  - 4种不同并行粒度(2, 4, 4, 8卡)                                  │
    │  - 样本数递增(32, 64, 64, 128)                                    │
    │  - 小粒度快完成, 大粒度持续                                       │
    │                                                                 │
    │  验证目标: 多粒度混合场景的资源匹配能力                           │
    └─────────────────────────────────────────────────────────────────┘
    """
    return TestCase(
        name="BB6_mixed_parallelism",
        description="混合粒度 - 多粒度资源分配",
        cluster=ClusterConfig(machine_count=4, gpus_per_machine=8),  # 32张卡
        tasks=[
            TaskConfig(
                task_id="task_small",
                tp=2, pp=1,  # 2卡/实例
                base_instances=2,  # 4张卡
                samples_per_round=4,
                num_rounds=8,  # 总样本32
                time_distribution="longtail_normal",
                distribution_params=_make_time_params(0.5, 0.8, 3.0)
            ),
            TaskConfig(
                task_id="task_medium1",
                tp=4, pp=1,  # 4卡/实例
                base_instances=2,  # 8张卡
                samples_per_round=8,
                num_rounds=8,  # 总样本64
                time_distribution="longtail_normal",
                distribution_params=_make_time_params(0.5, 1.4, 4.0)
            ),
            TaskConfig(
                task_id="task_medium2",
                tp=2, pp=2,  # 4卡/实例
                base_instances=1,  # 4张卡
                samples_per_round=8,
                num_rounds=8,  # 总样本64
                time_distribution="longtail_normal",
                distribution_params=_make_time_params(0.6, 1.2, 4.0)
            ),
            TaskConfig(
                task_id="task_large",
                tp=8, pp=1,  # 8卡/实例
                base_instances=2,  # 16张卡
                samples_per_round=16,
                num_rounds=8,  # 总样本128
                time_distribution="longtail_normal",
                distribution_params=_make_time_params(0.6, 1.8, 5.0)
            )
        ]
    )


def _create_bb7_large_instance() -> TestCase:
    """BB7: 大实例场景 - 对称设计(与BB3不对称对比)

    集群: 64卡, 基线总和: 64卡 (满载)

    ┌─────────────────────────────────────────────────────────────────┐
    │  BB7 (对称设计) - 与BB3对比                                      │
    │                                                                 │
    │  task1: base=2实例(32卡), 128样本, 50%慢样本(0.8-3倍)            │
    │  task2: base=2实例(32卡), 128样本, 30%慢样本(0.35-0.65倍)        │
    │                                                                 │
    │  特点:                                                          │
    │  - 两任务base相同 (各32卡)                                       │
    │  - 两任务样本数相同 (各128样本)                                   │
    │  - 两任务同时进行，完成时间相近                                   │
    │                                                                 │
    │  GS调度机会少:                                                   │
    │  - 无"早完成早释放"场景                                          │
    │  - 主要依靠长尾差异进行微调                                       │
    │                                                                 │
    │  对比BB3 (不对称设计):                                           │
    │  - task1: base=1(16卡), 32样本 → ~2000s完成 → 释放16卡          │
    │  - task2: base=3(48卡), 384样本 → 获得额外16卡 → 加速处理        │
    │  - GS有明显的动态调度机会                                        │
    └─────────────────────────────────────────────────────────────────┘
    """
    return TestCase(
        name="BB7_large_instance",
        description="大实例场景 - 对称设计(无早释放调度机会)",
        cluster=ClusterConfig(machine_count=8, gpus_per_machine=8),  # 64张卡
        tasks=[
            TaskConfig(
                task_id="task1",
                tp=8, pp=2,  # 16卡/实例
                base_instances=2,  # base=32卡
                samples_per_round=16,
                num_rounds=8,  # 总样本128
                time_distribution="longtail_normal",
                distribution_params=_make_time_params(0.5, 0.8, 3.0)  # 50%慢样本
            ),
            TaskConfig(
                task_id="task2",
                tp=8, pp=2,  # 16卡/实例
                base_instances=2,  # base=32卡
                samples_per_round=16,
                num_rounds=8,  # 总样本128
                time_distribution="longtail_normal",
                distribution_params=_make_time_params(0.3, 0.35, 0.65)  # 30%慢样本
            )
        ]
    )


def _create_bb8_fragmentation_test() -> TestCase:
    """BB8: 资源碎片化测试 - 多粒度碎片重组

    集群: 32卡, 基线总和: 32卡 (满载)

    ┌─────────────────────────────────────────────────────────────────┐
    │  BB8 (碎片化调度)                                                │
    │                                                                 │
    │  task1(8×1): base=1实例(8卡), 64样本                             │
    │  task2(4×1): base=2实例(8卡), 64样本                             │
    │  task3(4×2): base=1实例(8卡), 64样本                             │
    │  task4(4×1): base=1实例(4卡), 32样本 → 快完成 → 释放4卡          │
    │  task5(4×1): base=1实例(4卡), 32样本 → 快完成 → 释放4卡          │
    │                                                                 │
    │  特点:                                                          │
    │  - 5任务, 3种粒度(4卡, 8卡, 8卡)                                  │
    │  - 小粒度任务(4卡)样本少, 快完成                                  │
    │  - 大粒度任务(8卡)样本多, 持续                                    │
    │  - 碎片化释放: 4卡+4卡=8卡 → 可组合给8卡任务                      │
    │                                                                 │
    │  GS调度优势:                                                     │
    │  - 小粒度释放4卡碎片                                             │
    │  - GS碎片重组: 累积碎片分配给合适粒度任务                         │
    │  - 验证资源碎片化场景的调度能力                                   │
    │                                                                 │
    │  验证目标: 多粒度碎片重组能力                                     │
    └─────────────────────────────────────────────────────────────────┘
    """
    return TestCase(
        name="BB8_fragmentation_test",
        description="碎片化测试 - 多粒度碎片重组",
        cluster=ClusterConfig(machine_count=4, gpus_per_machine=8),  # 32张卡
        tasks=[
            TaskConfig(
                task_id="task1",
                tp=8, pp=1,  # 8卡/实例
                base_instances=1,  # 8张卡
                samples_per_round=8,
                num_rounds=8,  # 总样本64
                time_distribution="longtail_normal",
                distribution_params=_make_diverse_time_params(0, 5)
            ),
            TaskConfig(
                task_id="task2",
                tp=4, pp=1,  # 4卡/实例
                base_instances=2,  # 8张卡
                samples_per_round=8,
                num_rounds=8,  # 总样本64
                time_distribution="longtail_normal",
                distribution_params=_make_diverse_time_params(1, 5)
            ),
            TaskConfig(
                task_id="task3",
                tp=4, pp=2,  # 8卡/实例
                base_instances=1,  # 8张卡
                samples_per_round=8,
                num_rounds=8,  # 总样本64
                time_distribution="longtail_normal",
                distribution_params=_make_diverse_time_params(2, 5)
            ),
            TaskConfig(
                task_id="task4",
                tp=4, pp=1,  # 4卡/实例
                base_instances=1,  # 4张卡
                samples_per_round=4,
                num_rounds=8,  # 总样本32
                time_distribution="longtail_normal",
                distribution_params=_make_diverse_time_params(3, 5)
            ),
            TaskConfig(
                task_id="task5",
                tp=4, pp=1,  # 4卡/实例
                base_instances=1,  # 4张卡
                samples_per_round=4,
                num_rounds=8,  # 总样本32
                time_distribution="longtail_normal",
                distribution_params=_make_diverse_time_params(4, 5)
            )
        ]
    )


def _create_bb9_strong_competition() -> TestCase:
    """BB9: 饱和竞争场景 - 同策略差异化竞争

    集群: 32卡, 基线总和: 32卡 (满载)

    ┌─────────────────────────────────────────────────────────────────┐
    │  BB9 (饱和竞争)                                                  │
    │                                                                 │
    │  task0-3: 各base=1实例(8卡), 各64样本, 相同并行策略(8×1)          │
    │                                                                 │
    │  时间分布差异化:                                                  │
    │  - task0: 50%慢样本(0.8-3倍) → 最快完成                          │
    │  - task1: 60%慢样本(1.1-3.5倍)                                   │
    │  - task2: 70%慢样本(1.4-4倍)                                     │
    │  - task3: 80%慢样本(1.7-4.5倍) → 最慢完成                        │
    │                                                                 │
    │  特点:                                                          │
    │  - 4任务100%占用, 相同并行策略                                    │
    │  - 完全饱和竞争                                                  │
    │  - 只有长尾差异驱动调度                                          │
    │                                                                 │
    │  GS调度机会:                                                     │
    │  - task0最快完成 → 释放8卡 → 分给task3                            │
    │  - task1次快完成 → 释放8卡 → 分给task2/task3                      │
    │  - 级联调度, 最慢任务获得最多额外资源                             │
    │                                                                 │
    │  对比无GS:                                                       │
    │  - 无GS: 每任务固定8卡, 先完成任务的8卡闲置                       │
    │  - 有GS: 先完成任务释放资源给慢任务                               │
    │                                                                 │
    │  验证目标: 饱和竞争、同策略场景的长尾调度能力                     │
    └─────────────────────────────────────────────────────────────────┘
    """
    return TestCase(
        name="BB9_strong_competition",
        description="饱和竞争 - 4任务同策略差异化",
        cluster=ClusterConfig(machine_count=4, gpus_per_machine=8),  # 32张卡
        tasks=[
            TaskConfig(
                task_id=f"task{i}",
                tp=8, pp=1,  # 8卡/实例
                base_instances=1,  # 8张卡
                samples_per_round=8,
                num_rounds=8,  # 总样本64
                time_distribution="longtail_normal",
                distribution_params=_make_diverse_time_params(i, 4)
            )
            for i in range(4)
        ]
    )


def _create_bb10_dynamic_scaling() -> TestCase:
    """BB10: 动态扩缩容 - base差异大强调度场景

    集群: 64卡, 基线总和: 64卡 (满载)

    ┌─────────────────────────────────────────────────────────────────┐
    │  BB10 (动态扩缩容)                                               │
    │                                                                 │
    │  task_large: base=4实例(32卡), 128样本, 50%慢样本(0.8-3倍)        │
    │             → 大任务, 持续处理                                   │
    │                                                                 │
    │  task_medium: base=2实例(16卡), 64样本, 60%慢样本(1.6-4.5倍)      │
    │              → 中任务                                            │
    │                                                                 │
    │  task_small: base=2实例(16卡), 64样本, 25%慢样本(0.35-0.65倍)     │
    │             → 小任务, 极快完成 → 释放16卡                         │
    │                                                                 │
    │  特点:                                                          │
    │  - base差异大(32 vs 16 vs 16卡)                                  │
    │  - task_small长尾轻(25%), 极快完成                               │
    │  - task_medium和task_large需要更多时间                           │
    │                                                                 │
    │  GS调度优势:                                                     │
    │  - task_small先完成释放16卡                                       │
    │  - 16卡可给task_medium扩容(从2→3实例)                            │
    │  - 或给task_large扩容(从4→5实例)                                 │
    │  - 大base任务获得额外资源, 显著加速                               │
    │                                                                 │
    │  验证目标: 大base差异场景的动态扩容能力                           │
    └─────────────────────────────────────────────────────────────────┘
    """
    return TestCase(
        name="BB10_dynamic_scaling",
        description="动态扩缩容 - base差异大",
        cluster=ClusterConfig(machine_count=8, gpus_per_machine=8),  # 64张卡
        tasks=[
            TaskConfig(
                task_id="task_large",
                tp=8, pp=1,  # 8卡/实例
                base_instances=4,  # 32张卡
                samples_per_round=16,
                num_rounds=8,  # 总样本128
                time_distribution="longtail_normal",
                distribution_params=_make_time_params(0.5, 0.8, 3.0)
            ),
            TaskConfig(
                task_id="task_medium",
                tp=8, pp=1,  # 8卡/实例
                base_instances=2,  # 16张卡
                samples_per_round=8,
                num_rounds=8,  # 总样本64
                time_distribution="longtail_normal",
                distribution_params=_make_time_params(0.6, 1.6, 4.5)
            ),
            TaskConfig(
                task_id="task_small",
                tp=8, pp=1,  # 8卡/实例
                base_instances=2,  # 16张卡
                samples_per_round=8,
                num_rounds=8,  # 总样本64
                time_distribution="longtail_normal",
                distribution_params=_make_time_params(0.25, 0.35, 0.65)
            )
        ]
    )


def _create_bb11_heterogeneous_cluster() -> TestCase:
    """BB11: 异构集群场景 - 6任务多粒度混合

    集群: 64卡, 基线总和: 64卡 (满载)

    ┌─────────────────────────────────────────────────────────────────┐
    │  BB11 (异构集群)                                                 │
    │                                                                 │
    │  6任务, 4种并行粒度(2, 4, 8, 16卡/实例):                          │
    │                                                                 │
    │  task_2card(2×1): base=4实例(8卡), 32样本 → 小粒度, 快完成        │
    │  task_4card1(4×1): base=2实例(8卡), 64样本                       │
    │  task_4card2(2×2): base=2实例(8卡), 64样本                       │
    │  task_8card1(8×1): base=2实例(16卡), 96样本                      │
    │  task_8card2(4×2): base=1实例(8卡), 64样本                       │
    │  task_16card(8×2): base=1实例(16卡), 128样本 → 大粒度, 持续      │
    │                                                                 │
    │  特点:                                                          │
    │  - 粒度多样性最强(2/4/8/16卡)                                     │
    │  - 样本数递增(32/64/64/96/64/128)                                │
    │  - 小粒度任务样本少, 快完成                                       │
    │  - 大粒度任务样本多, 持续处理                                     │
    │                                                                 │
    │  GS调度优势:                                                     │
    │  - 小粒度(2卡)释放后可组合分配                                    │
    │  - 4个2卡任务完成释放32卡 → 可分配给16卡任务                      │
    │  - 异构粒度验证GS资源匹配灵活性                                   │
    │                                                                 │
    │  验证目标: 多粒度异构场景的综合调度能力                           │
    └─────────────────────────────────────────────────────────────────┘
    """
    return TestCase(
        name="BB11_heterogeneous_cluster",
        description="异构集群 - 6任务多粒度混合",
        cluster=ClusterConfig(machine_count=8, gpus_per_machine=8),  # 64张卡
        tasks=[
            TaskConfig(
                task_id="task_2card",
                tp=2, pp=1,  # 2卡/实例
                base_instances=4,  # 8张卡
                samples_per_round=4,
                num_rounds=8,  # 总样本32
                time_distribution="longtail_normal",
                distribution_params=_make_diverse_time_params(0, 6)
            ),
            TaskConfig(
                task_id="task_4card1",
                tp=4, pp=1,  # 4卡/实例
                base_instances=2,  # 8张卡
                samples_per_round=8,
                num_rounds=8,  # 总样本64
                time_distribution="longtail_normal",
                distribution_params=_make_diverse_time_params(1, 6)
            ),
            TaskConfig(
                task_id="task_4card2",
                tp=2, pp=2,  # 4卡/实例
                base_instances=2,  # 8张卡
                samples_per_round=8,
                num_rounds=8,  # 总样本64
                time_distribution="longtail_normal",
                distribution_params=_make_diverse_time_params(2, 6)
            ),
            TaskConfig(
                task_id="task_8card1",
                tp=8, pp=1,  # 8卡/实例
                base_instances=2,  # 16张卡
                samples_per_round=12,
                num_rounds=8,  # 总样本96
                time_distribution="longtail_normal",
                distribution_params=_make_diverse_time_params(3, 6)
            ),
            TaskConfig(
                task_id="task_8card2",
                tp=4, pp=2,  # 8卡/实例
                base_instances=1,  # 8张卡
                samples_per_round=8,
                num_rounds=8,  # 总样本64
                time_distribution="longtail_normal",
                distribution_params=_make_diverse_time_params(4, 6)
            ),
            TaskConfig(
                task_id="task_16card",
                tp=8, pp=2,  # 16卡/实例
                base_instances=1,  # 16张卡
                samples_per_round=16,
                num_rounds=8,  # 总样本128
                time_distribution="longtail_normal",
                distribution_params=_make_diverse_time_params(5, 6)
            )
        ]
    )


# ========== 高调度密度场景 ==========

def _create_bb12_multi_task_diverse_longtail() -> TestCase:
    """BB12: 多任务差异化长尾场景 - 高调度密度最佳场景

    集群: 64卡, 基线总和: 64卡 (满载)

    ┌─────────────────────────────────────────────────────────────────┐
    │  BB12 (高调度密度 - 最佳调度场景)                                 │
    │                                                                 │
    │  fast_task1: base=2实例(16卡), 96样本, 20%慢样本(1-1.5倍)         │
    │            → 极快完成 → 释放16卡                                  │
    │                                                                 │
    │  fast_task2: base=2实例(16卡), 96样本, 30%慢样本(1-2倍)           │
    │            → 快完成 → 释放16卡                                    │
    │                                                                 │
    │  slow_task1: base=2实例(16卡), 256样本, 70%慢样本(2-5倍)          │
    │            → 持续处理 → 接收释放资源                              │
    │                                                                 │
    │  slow_task2: base=2实例(16卡), 320样本, 80%慢样本(2.5-6倍)        │
    │            → 最慢完成 → 接收最多额外资源                          │
    │                                                                 │
    │  特点:                                                          │
    │  - 长尾差异最极端(20% vs 80%, 1.5倍 vs 6倍)                       │
    │  - 样本数差异最极端(96 vs 320)                                    │
    │  - 快任务早完成释放32卡                                           │
    │  - 慢任务持续处理576样本                                          │
    │                                                                 │
    │  GS调度优势(最显著):                                              │
    │  - 2个快任务完成后释放32卡                                        │
    │  - slow_task1从16卡增至32卡(4实例), 100%额外算力                  │
    │  - slow_task2从16卡增至48卡(6实例), 200%额外算力                  │
    │  - 加速处理576样本, 显著缩短总时间                                │
    │                                                                 │
    │  设计要点:                                                       │
    │  - 快慢分离最大化                                                │
    │  - 长尾差异最大化                                                │
    │  - 样本数差异最大化                                               │
    │  - GS调度效果最显著                                               │
    │                                                                 │
    │  验证目标: 最佳调度场景下的GS最大调度效果                          │
    └─────────────────────────────────────────────────────────────────┘
    """
    return TestCase(
        name="BB12_multi_task_diverse_longtail",
        description="高调度密度 - 长尾差异最大",
        cluster=ClusterConfig(machine_count=8, gpus_per_machine=8),  # 64张卡
        tasks=[
            # 快任务1：20%长尾，轻微，样本少
            TaskConfig(
                task_id="fast_task1",
                tp=4, pp=2,  # 8卡/实例
                base_instances=2,  # 16张卡
                samples_per_round=8,
                num_rounds=12,  # 总样本96，快速完成
                time_distribution="longtail_normal",
                distribution_params=_make_time_params(0.2, 1.0, 1.5)  # 20%长尾，轻微
            ),
            # 快任务2：30%长尾，样本少
            TaskConfig(
                task_id="fast_task2",
                tp=4, pp=2,  # 8卡/实例
                base_instances=2,  # 16张卡
                samples_per_round=8,
                num_rounds=12,  # 总样本96，快速完成
                time_distribution="longtail_normal",
                distribution_params=_make_time_params(0.3, 1.0, 2.0)  # 30%长尾
            ),
            # 慢任务1：70%长尾，严重，样本多
            TaskConfig(
                task_id="slow_task1",
                tp=4, pp=2,  # 8卡/实例
                base_instances=2,  # 16张卡
                samples_per_round=16,
                num_rounds=16,  # 总样本256，持续处理
                time_distribution="longtail_normal",
                distribution_params=_make_time_params(0.7, 2.0, 5.0)  # 70%长尾，严重
            ),
            # 慢任务2：80%长尾，最严重，样本最多
            TaskConfig(
                task_id="slow_task2",
                tp=4, pp=2,  # 8卡/实例
                base_instances=2,  # 16张卡
                samples_per_round=16,
                num_rounds=20,  # 总样本320，最慢完成
                time_distribution="longtail_normal",
                distribution_params=_make_time_params(0.8, 2.5, 6.0)  # 80%长尾，最严重
            ),
        ]
    )


def get_test_case_by_name(name: str) -> TestCase:
    """根据名称获取测试用例"""
    cases = get_benchmark_cases()
    for case in cases:
        if case.name == name:
            return case
    raise ValueError(f"找不到测试用例: {name}")


def get_test_case_summary() -> dict:
    """获取所有测试用例的摘要信息"""
    cases = get_benchmark_cases()
    summary = {}
    for case in cases:
        total_cards = sum(task.total_cards() for task in case.tasks)
        cluster_cards = case.cluster.total_gpus()
        summary[case.name] = {
            "description": case.description,
            "cluster_size": cluster_cards,
            "num_tasks": len(case.tasks),
            "total_base_cards": total_cards,
            "utilization": total_cards / cluster_cards * 100,
            "tp_pp_combinations": sorted(set(f"{task.tp}*{task.pp}" for task in case.tasks)),
            "base_instances_range": [task.base_instances for task in case.tasks]
        }
    return summary


def validate_all_cases() -> bool:
    """验证所有测试用例是否满足约束条件"""
    cases = get_benchmark_cases()
    all_valid = True
    for case in cases:
        total_cards = sum(task.total_cards() for task in case.tasks)
        cluster_cards = case.cluster.total_gpus()
        if total_cards != cluster_cards:
            print(f"❌ {case.name}: 总基线{total_cards}卡 != 集群{cluster_cards}卡")
            all_valid = False
        else:
            print(f"✓ {case.name}: 总基线{total_cards}卡 = 集群{cluster_cards}卡")
    return all_valid


def check_time_diversity() -> dict:
    """检查每个测试用例中任务的时间分布参数是否不同"""
    cases = get_benchmark_cases()
    diversity_report = {}
    for case in cases:
        params_list = []
        for task in case.tasks:
            params = task.distribution_params
            params_tuple = (params["slow_ratio"], params["slow_min"], params["slow_max"])
            params_list.append(params_tuple)

        # 检查是否有重复
        has_duplicate = len(params_list) != len(set(params_list))
        diversity_report[case.name] = {
            "num_tasks": len(case.tasks),
            "unique_params": len(set(params_list)),
            "has_duplicate": has_duplicate,
            "params_list": params_list
        }
    return diversity_report


if __name__ == "__main__":
    print("验证所有测试用例的约束条件...")
    print("-" * 60)
    valid = validate_all_cases()
    print("-" * 60)
    if valid:
        print("✓ 所有测试用例满足约束条件：总基线卡数 = 集群卡数")
    else:
        print("❌ 存在不满足约束的测试用例")

    print("\n检查时间分布参数多样性...")
    print("-" * 60)
    diversity = check_time_diversity()
    all_diverse = True
    for case_name, report in diversity.items():
        if report["has_duplicate"]:
            print(f"❌ {case_name}: 有重复的时间分布参数")
            print(f"   参数列表: {report['params_list']}")
            all_diverse = False
        else:
            print(f"✓ {case_name}: {report['unique_params']}个独特参数")
    print("-" * 60)
    if all_diverse:
        print("✓ 所有测试用例的时间分布参数各不相同")