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
    """BB1: 双任务竞争

    集群: 32卡, 基线: 32卡 (2任务*2实例*8卡/实例)
    """
    return TestCase(
        name="BB1_two_tasks_competition",
        description="双任务竞争场景",
        cluster=ClusterConfig(machine_count=4, gpus_per_machine=8),  # 32张卡
        tasks=[
            TaskConfig(
                task_id="task1",
                tp=4, pp=2,  # 8卡/实例
                base_instances=2,  # 16张卡
                samples_per_round=16,
                total_samples=64,
                time_distribution="longtail_normal",
                distribution_params=_make_time_params(0.5, 0.8, 3.0)
            ),
            TaskConfig(
                task_id="task2",
                tp=4, pp=2,  # 8卡/实例
                base_instances=2,  # 16张卡
                samples_per_round=16,
                total_samples=64,
                time_distribution="longtail_normal",
                distribution_params=_make_time_params(0.6, 1.5, 4.5)
            )
        ]
    )


def _create_bb2_three_tasks_mixed_parallelism() -> TestCase:
    """BB2: 三任务不同并行策略 - 差异化长尾

    集群: 32卡, 基线: 32卡
    task1: 1实例(8卡) - 快任务，样本少，长尾轻
    task2: 1实例(8卡) - 快任务，样本少，长尾轻
    task3: 2实例(16卡) - 慢任务，样本多，长尾重

    设计目标：task1/task2快速完成后释放16卡，GS分配给task3
    """
    return TestCase(
        name="BB2_three_tasks_mixed_parallelism",
        description="三任务 - 混合并行策略（差异化长尾）",
        cluster=ClusterConfig(machine_count=4, gpus_per_machine=8),  # 32张卡
        tasks=[
            TaskConfig(
                task_id="fast_task1",
                tp=4, pp=2,  # 8卡/实例
                base_instances=1,  # 8张卡
                samples_per_round=16,
                total_samples=48,  # 样本少，快速完成
                time_distribution="longtail_normal",
                distribution_params=_make_time_params(0.2, 1.0, 1.5)  # 20%长尾，轻微
            ),
            TaskConfig(
                task_id="fast_task2",
                tp=8, pp=1,  # 8卡/实例
                base_instances=1,  # 8张卡
                samples_per_round=16,
                total_samples=48,  # 样本少，快速完成
                time_distribution="longtail_normal",
                distribution_params=_make_time_params(0.25, 1.0, 1.8)  # 25%长尾，轻微
            ),
            TaskConfig(
                task_id="slow_task",
                tp=2, pp=4,  # 8卡/实例
                base_instances=2,  # 16张卡
                samples_per_round=32,
                total_samples=256,  # 样本多，持续处理
                time_distribution="longtail_normal",
                distribution_params=_make_time_params(0.7, 2.0, 5.0)  # 70%长尾，严重
            )
        ]
    )


def _create_bb3_large_scale_two_tasks() -> TestCase:
    """BB3: 大规模双任务

    集群: 64卡, 基线: 64卡 (2任务*2实例*16卡/实例)
    """
    return TestCase(
        name="BB3_large_scale_two_tasks",
        description="大规模 - 双任务场景",
        cluster=ClusterConfig(machine_count=8, gpus_per_machine=8),  # 64张卡
        tasks=[
            TaskConfig(
                task_id="task1",
                tp=8, pp=2,  # 16卡/实例
                base_instances=2,  # 32张卡
                samples_per_round=32,
                total_samples=128,
                time_distribution="longtail_normal",
                distribution_params=_make_time_params(0.5, 0.8, 3.0)
            ),
            TaskConfig(
                task_id="task2",
                tp=8, pp=2,  # 16卡/实例
                base_instances=2,  # 32张卡
                samples_per_round=32,
                total_samples=128,
                time_distribution="longtail_normal",
                distribution_params=_make_time_params(0.6, 1.2, 4.0)
            )
        ]
    )


def _create_bb4_four_tasks_small_scale() -> TestCase:
    """BB4: 四任务小规模

    集群: 32卡, 基线: 32卡 (4任务*1实例*8卡/实例)
    每个任务有不同的时间分布参数
    """
    return TestCase(
        name="BB4_four_tasks_small_scale",
        description="四任务 - 小规模场景",
        cluster=ClusterConfig(machine_count=4, gpus_per_machine=8),  # 32张卡
        tasks=[
            TaskConfig(
                task_id=f"task{i}",
                tp=4, pp=2,  # 8卡/实例
                base_instances=1,  # 8张卡
                samples_per_round=16,
                total_samples=64,
                time_distribution="longtail_normal",
                distribution_params=_make_diverse_time_params(i, 4)
            )
            for i in range(4)
        ]
    )


# ========== 复杂场景 ==========

def _create_bb5_small_granularity_dense() -> TestCase:
    """BB5: 小粒度密集任务

    集群: 32卡
    基线: 6*TP=2,PP=1 (6任务*2实例*2卡=24卡) + 2*TP=4,PP=1 (2任务*1实例*4卡=8卡) = 32卡
    """
    return TestCase(
        name="BB5_small_granularity_dense",
        description="小粒度密集任务 - TP*PP=2/4, 8任务竞争",
        cluster=ClusterConfig(machine_count=4, gpus_per_machine=8),  # 32张卡
        tasks=[
            # 6个TP=2,PP=1任务 (2卡/实例)
            TaskConfig(
                task_id=f"task{i}",
                tp=2, pp=1,  # 2卡/实例
                base_instances=2,  # 4张卡
                samples_per_round=8,
                total_samples=32,
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
                samples_per_round=8,
                total_samples=32,
                time_distribution="longtail_normal",
                distribution_params=_make_diverse_time_params(i+6, 8)
            )
            for i in range(2)
        ]
    )
    # 总基线: 6*4 + 2*4 = 32卡


def _create_bb6_mixed_parallelism() -> TestCase:
    """BB6: 混合并行粒度

    集群: 32卡
    基线: task_small(4卡) + task_medium1(8卡) + task_medium2(4卡) + task_large(16卡) = 32卡
    """
    return TestCase(
        name="BB6_mixed_parallelism",
        description="混合并行粒度 - TP*PP=2/4/8混合",
        cluster=ClusterConfig(machine_count=4, gpus_per_machine=8),  # 32张卡
        tasks=[
            TaskConfig(
                task_id="task_small",
                tp=2, pp=1,  # 2卡/实例
                base_instances=2,  # 4张卡
                samples_per_round=8,
                total_samples=32,
                time_distribution="longtail_normal",
                distribution_params=_make_time_params(0.5, 0.8, 3.0)
            ),
            TaskConfig(
                task_id="task_medium1",
                tp=4, pp=1,  # 4卡/实例
                base_instances=2,  # 8张卡
                samples_per_round=16,
                total_samples=64,
                time_distribution="longtail_normal",
                distribution_params=_make_time_params(0.5, 1.4, 4.0)
            ),
            TaskConfig(
                task_id="task_medium2",
                tp=2, pp=2,  # 4卡/实例
                base_instances=1,  # 4张卡
                samples_per_round=16,
                total_samples=64,
                time_distribution="longtail_normal",
                distribution_params=_make_time_params(0.6, 1.2, 4.0)
            ),
            TaskConfig(
                task_id="task_large",
                tp=8, pp=1,  # 8卡/实例
                base_instances=2,  # 16张卡
                samples_per_round=32,
                total_samples=128,
                time_distribution="longtail_normal",
                distribution_params=_make_time_params(0.6, 1.8, 5.0)
            )
        ]
    )
    # 总基线: 4 + 8 + 4 + 16 = 32张卡


def _create_bb7_large_instance() -> TestCase:
    """BB7: 大实例场景

    集群: 64卡
    基线: 2任务*2实例*16卡/实例 = 64卡
    """
    return TestCase(
        name="BB7_large_instance",
        description="大实例场景 - TP*PP=16, 64卡集群",
        cluster=ClusterConfig(machine_count=8, gpus_per_machine=8),  # 64张卡
        tasks=[
            TaskConfig(
                task_id="task1",
                tp=8, pp=2,  # 16卡/实例
                base_instances=2,  # 32张卡
                samples_per_round=32,
                total_samples=128,
                time_distribution="longtail_normal",
                distribution_params=_make_time_params(0.5, 0.8, 3.0)
            ),
            TaskConfig(
                task_id="task2",
                tp=8, pp=2,  # 16卡/实例
                base_instances=2,  # 32张卡
                samples_per_round=32,
                total_samples=128,
                time_distribution="longtail_normal",
                distribution_params=_make_time_params(0.3, 0.35, 0.65)
            )
        ]
    )
    # 总基线: 32 + 32 = 64张卡


def _create_bb8_fragmentation_test() -> TestCase:
    """BB8: 资源碎片化测试

    集群: 32卡
    基线: 8 + 8 + 8 + 4 + 4 = 32卡
    """
    return TestCase(
        name="BB8_fragmentation_test",
        description="资源碎片化测试 - TP*PP=4/8混合",
        cluster=ClusterConfig(machine_count=4, gpus_per_machine=8),  # 32张卡
        tasks=[
            TaskConfig(
                task_id="task1",
                tp=8, pp=1,  # 8卡/实例
                base_instances=1,  # 8张卡
                samples_per_round=16,
                total_samples=64,
                time_distribution="longtail_normal",
                distribution_params=_make_diverse_time_params(0, 5)
            ),
            TaskConfig(
                task_id="task2",
                tp=4, pp=1,  # 4卡/实例
                base_instances=2,  # 8张卡
                samples_per_round=16,
                total_samples=64,
                time_distribution="longtail_normal",
                distribution_params=_make_diverse_time_params(1, 5)
            ),
            TaskConfig(
                task_id="task3",
                tp=4, pp=2,  # 8卡/实例
                base_instances=1,  # 8张卡
                samples_per_round=16,
                total_samples=64,
                time_distribution="longtail_normal",
                distribution_params=_make_diverse_time_params(2, 5)
            ),
            TaskConfig(
                task_id="task4",
                tp=4, pp=1,  # 4卡/实例
                base_instances=1,  # 4张卡
                samples_per_round=8,
                total_samples=32,
                time_distribution="longtail_normal",
                distribution_params=_make_diverse_time_params(3, 5)
            ),
            TaskConfig(
                task_id="task5",
                tp=4, pp=1,  # 4卡/实例
                base_instances=1,  # 4张卡
                samples_per_round=8,
                total_samples=32,
                time_distribution="longtail_normal",
                distribution_params=_make_diverse_time_params(4, 5)
            )
        ]
    )
    # 总基线: 8 + 8 + 8 + 4 + 4 = 32张卡


def _create_bb9_strong_competition() -> TestCase:
    """BB9: 饱和竞争场景

    集群: 32卡
    基线: 4任务*1实例*8卡/实例 = 32卡 (100%占用)
    每个任务使用相同并行策略(TP=8, PP=1)，只有完成后才能释放资源
    测试GS在饱和竞争场景下的调度能力
    """
    return TestCase(
        name="BB9_strong_competition",
        description="饱和竞争 - 4任务100%占用，相同并行策略",
        cluster=ClusterConfig(machine_count=4, gpus_per_machine=8),  # 32张卡
        tasks=[
            TaskConfig(
                task_id=f"task{i}",
                tp=8, pp=1,  # 8卡/实例
                base_instances=1,  # 8张卡
                samples_per_round=16,
                total_samples=64,
                time_distribution="longtail_normal",
                distribution_params=_make_diverse_time_params(i, 4)
            )
            for i in range(4)
        ]
    )
    # 总基线: 4 * 8 = 32张卡


def _create_bb10_dynamic_scaling() -> TestCase:
    """BB10: 动态扩缩容

    集群: 64卡
    基线: task_large(32卡) + task_medium(16卡) + task_small(16卡) = 64卡
    """
    return TestCase(
        name="BB10_dynamic_scaling",
        description="动态扩缩容 - base_instances差异大",
        cluster=ClusterConfig(machine_count=8, gpus_per_machine=8),  # 64张卡
        tasks=[
            TaskConfig(
                task_id="task_large",
                tp=8, pp=1,  # 8卡/实例
                base_instances=4,  # 32张卡
                samples_per_round=32,
                total_samples=128,
                time_distribution="longtail_normal",
                distribution_params=_make_time_params(0.5, 0.8, 3.0)
            ),
            TaskConfig(
                task_id="task_medium",
                tp=8, pp=1,  # 8卡/实例
                base_instances=2,  # 16张卡
                samples_per_round=16,
                total_samples=64,
                time_distribution="longtail_normal",
                distribution_params=_make_time_params(0.6, 1.6, 4.5)
            ),
            TaskConfig(
                task_id="task_small",
                tp=8, pp=1,  # 8卡/实例
                base_instances=2,  # 16张卡
                samples_per_round=16,
                total_samples=64,
                time_distribution="longtail_normal",
                distribution_params=_make_time_params(0.25, 0.35, 0.65)
            )
        ]
    )
    # 总基线: 32 + 16 + 16 = 64张卡


def _create_bb11_heterogeneous_cluster() -> TestCase:
    """BB11: 异构集群场景

    集群: 64卡
    基线: 8 + 8 + 8 + 16 + 8 + 16 = 64卡
    """
    return TestCase(
        name="BB11_heterogeneous_cluster",
        description="异构集群场景 - TP*PP=2/4/8/16混合",
        cluster=ClusterConfig(machine_count=8, gpus_per_machine=8),  # 64张卡
        tasks=[
            TaskConfig(
                task_id="task_2card",
                tp=2, pp=1,  # 2卡/实例
                base_instances=4,  # 8张卡
                samples_per_round=8,
                total_samples=32,
                time_distribution="longtail_normal",
                distribution_params=_make_diverse_time_params(0, 6)
            ),
            TaskConfig(
                task_id="task_4card1",
                tp=4, pp=1,  # 4卡/实例
                base_instances=2,  # 8张卡
                samples_per_round=16,
                total_samples=64,
                time_distribution="longtail_normal",
                distribution_params=_make_diverse_time_params(1, 6)
            ),
            TaskConfig(
                task_id="task_4card2",
                tp=2, pp=2,  # 4卡/实例
                base_instances=2,  # 8张卡
                samples_per_round=16,
                total_samples=64,
                time_distribution="longtail_normal",
                distribution_params=_make_diverse_time_params(2, 6)
            ),
            TaskConfig(
                task_id="task_8card1",
                tp=8, pp=1,  # 8卡/实例
                base_instances=2,  # 16张卡
                samples_per_round=24,
                total_samples=96,
                time_distribution="longtail_normal",
                distribution_params=_make_diverse_time_params(3, 6)
            ),
            TaskConfig(
                task_id="task_8card2",
                tp=4, pp=2,  # 8卡/实例
                base_instances=1,  # 8张卡
                samples_per_round=16,
                total_samples=64,
                time_distribution="longtail_normal",
                distribution_params=_make_diverse_time_params(4, 6)
            ),
            TaskConfig(
                task_id="task_16card",
                tp=8, pp=2,  # 16卡/实例
                base_instances=1,  # 16张卡
                samples_per_round=32,
                total_samples=128,
                time_distribution="longtail_normal",
                distribution_params=_make_diverse_time_params(5, 6)
            )
        ]
    )
    # 总基线: 8 + 8 + 8 + 16 + 8 + 16 = 64张卡


# ========== 高调度密度场景 ==========

def _create_bb12_multi_task_diverse_longtail() -> TestCase:
    """BB12: 多任务差异化长尾场景 - 高调度密度

    设计目标：让GS调度效果更明显
    - 快任务早早完成释放资源
    - 慢任务持续处理，接收释放的资源
    - 多次有效调度机会

    集群: 64卡，基线: 64卡
    4个任务，长尾效应差异明显：
    - 2快任务（20%/30%长尾，样本少）早早完成释放32张卡
    - 2慢任务（70%/80%长尾，样本多）持续处理，接收释放的资源
    """
    return TestCase(
        name="BB12_multi_task_diverse_longtail",
        description="多任务差异化 - 长尾差异大（高调度密度）",
        cluster=ClusterConfig(machine_count=8, gpus_per_machine=8),  # 64张卡
        tasks=[
            # 快任务1：20%长尾，轻微，样本少
            TaskConfig(
                task_id="fast_task1",
                tp=4, pp=2,  # 8卡/实例
                base_instances=2,  # 16张卡
                samples_per_round=32,
                total_samples=96,  # 较少样本，快速完成
                time_distribution="longtail_normal",
                distribution_params=_make_time_params(0.2, 1.0, 1.5)  # 20%长尾，轻微
            ),
            # 快任务2：30%长尾，样本少
            TaskConfig(
                task_id="fast_task2",
                tp=4, pp=2,  # 8卡/实例
                base_instances=2,  # 16张卡
                samples_per_round=32,
                total_samples=96,  # 较少样本，快速完成
                time_distribution="longtail_normal",
                distribution_params=_make_time_params(0.3, 1.0, 2.0)  # 30%长尾
            ),
            # 慢任务1：70%长尾，严重，样本多
            TaskConfig(
                task_id="slow_task1",
                tp=4, pp=2,  # 8卡/实例
                base_instances=2,  # 16张卡
                samples_per_round=32,
                total_samples=256,  # 更多样本，持续处理
                time_distribution="longtail_normal",
                distribution_params=_make_time_params(0.7, 2.0, 5.0)  # 70%长尾，严重
            ),
            # 慢任务2：80%长尾，最严重，样本最多
            TaskConfig(
                task_id="slow_task2",
                tp=4, pp=2,  # 8卡/实例
                base_instances=2,  # 16张卡
                samples_per_round=32,
                total_samples=320,  # 样本最多，最慢完成
                time_distribution="longtail_normal",
                distribution_params=_make_time_params(0.8, 2.5, 6.0)  # 80%长尾，最严重
            ),
        ]
    )
    # 总基线: 16+16+16+16 = 64张卡 ✓


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