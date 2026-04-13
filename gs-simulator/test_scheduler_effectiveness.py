#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scheduler Effectiveness Test

Test GroupScheduler effectiveness:
1. Compare results with/without scheduling (Real GroupScheduler vs No Scheduling)
2. Print key metrics during simulation
3. Record scheduling decisions
4. Calculate scheduling benefits (time saved, utilization improved)

Usage:
    # 交互式选择
    python test_scheduler_effectiveness.py

    # 按索引选择
    python test_scheduler_effectiveness.py -i 3
    python test_scheduler_effectiveness.py --index 3

    # 按名称选择
    python test_scheduler_effectiveness.py -c BB3_three_tasks_mixed_parallelism
    python test_scheduler_effectiveness.py --case BB3_three_tasks_mixed_parallelism

    # 只运行无GS模式
    python test_scheduler_effectiveness.py -i 3 --no-gs-only

    # 只运行有GS模式
    python test_scheduler_effectiveness.py -c BB3_three_tasks_mixed_parallelism --gs-only

    # 列出所有用例
    python test_scheduler_effectiveness.py --list
"""

import sys
import os
import json
import argparse
from typing import Dict, List, Tuple

# Add path
sys.path.insert(0, os.path.dirname(__file__))

from core.simulator import Simulator
from test_cases.benchmark import get_benchmark_cases, get_test_case_summary


def print_test_cases_table(test_cases) -> None:
    """打印测试用例表格"""
    print("\n" + "="*100)
    print(f"{'序号':<4} {'用例名称':<35} {'集群':<8} {'任务数':<6} {'TP*PP':<15} {'描述':<30}")
    print("="*100)

    for i, tc in enumerate(test_cases, 1):
        tp_pp = ", ".join(sorted(set(f"{task.tp}*{task.pp}" for task in tc.tasks)))
        print(f"{i:<4} {tc.name:<35} {tc.cluster.total_gpus():<8} "
              f"{len(tc.tasks):<6} {tp_pp:<15} {tc.description:<30}")

    print("="*100)


def select_test_case(test_cases, args) -> object:
    """
    选择测试用例

    支持多种选择方式：
    1. 命令行参数：--index 或 --case
    2. 交互式输入
    """
    # 列出所有用例
    print_test_cases_table(test_cases)

    # 方式1: 命令行指定
    if args.index:
        idx = args.index - 1
        if 0 <= idx < len(test_cases):
            tc = test_cases[idx]
            print(f"\n[已选择] 用例 {args.index}: {tc.name}")
            print(f"         描述: {tc.description}")
            return tc
        else:
            print(f"\n[错误] 索引 {args.index} 超出范围 (1-{len(test_cases)})")
            return None

    if args.case:
        # 支持模糊匹配
        matches = [tc for tc in test_cases if args.case.lower() in tc.name.lower()]
        if len(matches) == 1:
            tc = matches[0]
            print(f"\n[已选择] 用例: {tc.name}")
            print(f"         描述: {tc.description}")
            return tc
        elif len(matches) > 1:
            print(f"\n[错误] 匹配到多个用例，请更精确:")
            for tc in matches:
                print(f"       - {tc.name}")
            return None
        else:
            print(f"\n[错误] 找不到匹配的用例: {args.case}")
            print(f"[提示] 使用 --list 查看所有可用用例")
            return None

    # 方式2: 交互式输入
    print("\n[选择用例]")
    print("  输入方式:")
    print("    1. 数字序号 (如: 3)")
    print("    2. 用例名称 (如: BB3_three_tasks_mixed_parallelism)")
    print("    3. 按Tab键自动补全（如果终端支持）")

    while True:
        try:
            choice = input("\n请选择 (1-{} 或 名称): ".format(len(test_cases))).strip()

            if not choice:
                print("[提示] 输入不能为空")
                continue

            # 尝试作为数字解析
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(test_cases):
                    tc = test_cases[idx]
                    print(f"\n[已选择] 用例 {choice}: {tc.name}")
                    print(f"         描述: {tc.description}")
                    return tc
                else:
                    print(f"[错误] 索引超出范围 (1-{len(test_cases)})")
                    continue

            # 作为名称解析
            # 支持模糊匹配
            matches = [tc for tc in test_cases if choice.lower() in tc.name.lower()]
            if len(matches) == 1:
                tc = matches[0]
                print(f"\n[已选择] 用例: {tc.name}")
                print(f"         描述: {tc.description}")
                return tc
            elif len(matches) > 1:
                print(f"[错误] 匹配到多个用例，请更精确:")
                for tc in matches:
                    print(f"       - {tc.name}")
            else:
                print(f"[错误] 找不到匹配的用例: {choice}")
                print(f"[提示] 尝试使用数字序号，或查看上方表格")

        except (EOFError, KeyboardInterrupt):
            print("\n[取消] 用户中断")
            return None


def run_simulation(test_case, enable_gs: bool, log_dir: str = "results") -> Dict:
    """
    Run simulation

    Args:
        test_case: Test case
        enable_gs: Enable GS scheduling
        log_dir: Directory for scheduling logs

    Returns:
        Simulation result dictionary
    """
    print(f"\n{'='*80}")
    print(f"运行测试用例: {test_case.name}")
    print(f"描述: {test_case.description}")
    print(f"GS调度: {'启用' if enable_gs else '禁用'}")
    print(f"{'='*80}")

    import traceback
    try:
        simulator = Simulator(
            test_case=test_case,
            enable_gs=enable_gs,
            log_dir=log_dir
        )

        result = simulator.run()

        return {
            "enable_gs": enable_gs,
            "test_case_name": test_case.name,
            "total_time": result.total_simulation_time,
            "task_completion_times": result.task_completion_times,
            "scheduling_trace_count": result.scheduling_trace_count  # 使用直接统计的计数
        }
    except Exception as e:
        print(f"[错误] 仿真失败: {e}")
        traceback.print_exc()
        raise


def print_comparison(result_no_gs: Dict, result_with_gs: Dict) -> None:
    """
    Print comparison results

    Args:
        result_no_gs: Result without scheduling
        result_with_gs: Result with scheduling
    """
    print(f"\n{'='*80}")
    print(f"调度效果对比 - {result_no_gs['test_case_name']}")
    print(f"{'='*80}")

    # Basic info
    print("\n[测试用例信息]")
    print(f"  名称: {result_no_gs['test_case_name']}")

    # Comparison results
    print("\n[仿真结果对比]")

    print(f"  无调度:")
    print(f"    总耗时: {result_no_gs['total_time']:.2f}s")
    print(f"    调度决策数: {result_no_gs['scheduling_trace_count']}")

    for task_id, time in result_no_gs['task_completion_times'].items():
        print(f"    {task_id}: {time:.2f}s")

    print(f"\n  使用GroupScheduler:")
    print(f"    总耗时: {result_with_gs['total_time']:.2f}s")
    print(f"    调度决策数: {result_with_gs['scheduling_trace_count']}")

    for task_id, time in result_with_gs['task_completion_times'].items():
        print(f"    {task_id}: {time:.2f}s")

    # Calculate benefits
    time_saved = result_no_gs['total_time'] - result_with_gs['total_time']

    if result_no_gs['total_time'] > 0:
        time_speedup = (time_saved / result_no_gs['total_time']) * 100
    else:
        time_speedup = 0.0

    print("\n[调度效果]")
    if time_saved > 0:
        print(f"  ✓ 节省时间: {time_saved:.2f}s ({time_speedup:.1f}% 加速)")
    elif time_saved < 0:
        print(f"  ✗ 时间增加: {-time_saved:.2f}s ({-time_speedup:.1f}% 变慢)")
    else:
        print(f"  = 时间无变化")


def save_results(results: List[Dict], filename: str = 'scheduler_benchmark_results.json') -> None:
    """
    Save results to file

    Args:
        results: All test results
        filename: File name
    """
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    filepath = os.path.join(results_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[保存] 结果已保存至: {filepath}")


def main():
    """Main test flow"""
    parser = argparse.ArgumentParser(
        description='GroupScheduler调度效果测试',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 交互式选择用例
  python test_scheduler_effectiveness.py

  # 按索引选择用例
  python test_scheduler_effectiveness.py -i 3

  # 按名称选择用例
  python test_scheduler_effectiveness.py -c BB3_three_tasks_mixed_parallelism

  # 只运行无GS模式
  python test_scheduler_effectiveness.py -i 3 --no-gs-only

  # 只运行有GS模式
  python test_scheduler_effectiveness.py -c BB3 --gs-only

  # 列出所有用例
  python test_scheduler_effectiveness.py --list
        """
    )

    parser.add_argument(
        '--case', '-c',
        type=str,
        metavar='NAME',
        help='按名称选择测试用例 (如: BB3_three_tasks_mixed_parallelism)'
    )
    parser.add_argument(
        '--index', '-i',
        type=int,
        metavar='NUM',
        help='按索引选择测试用例 (1-based)'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='列出所有可用测试用例'
    )
    parser.add_argument(
        '--no-gs-only',
        action='store_true',
        help='只运行无GS调度模式'
    )
    parser.add_argument(
        '--gs-only',
        action='store_true',
        help='只运行有GS调度模式'
    )

    args = parser.parse_args()

    # Get all test cases
    test_cases = get_benchmark_cases()

    print("="*100)
    print("GroupScheduler 调度效果测试")
    print("="*100)

    # 只列出用例
    if args.list:
        print_test_cases_table(test_cases)
        return

    # 选择测试用例
    test_case = select_test_case(test_cases, args)
    if test_case is None:
        return

    # Run tests
    results = []

    # 检查运行模式
    run_no_gs = not args.gs_only
    run_with_gs = not args.no_gs_only

    if not run_no_gs and not run_with_gs:
        print("\n[错误] --no-gs-only 和 --gs-only 不能同时使用")
        return

    # 1. No scheduling
    if run_no_gs:
        print("\n[1/2] 运行无调度模式...")
        result_no_gs = run_simulation(test_case, enable_gs=False, log_dir="results/no_gs")
        results.append(result_no_gs)
    else:
        result_no_gs = None

    # 2. With scheduling
    if run_with_gs:
        print("\n[2/2] 运行GroupScheduler模式...")
        try:
            result_with_gs = run_simulation(test_case, enable_gs=True, log_dir="results/with_gs")
            results.append(result_with_gs)

            # Print comparison if both modes ran
            if result_no_gs is not None:
                print_comparison(result_no_gs, result_with_gs)
        except Exception as e:
            print(f"\n[错误] GroupScheduler运行失败: {e}")
            import traceback
            traceback.print_exc()
            if result_no_gs is not None:
                print("\n只保存无调度模式的结果")
    else:
        result_with_gs = None

    # Save results
    if results:
        save_results(results)

    print("\n" + "="*80)
    print("测试完成!")
    print("="*80)

    if result_with_gs is not None:
        print("\n调度轨迹日志保存在:")
        print("  - results/with_gs/scheduler_trace_*.jsonl")
        print("\n使用jq分析日志:")
        print("  cat results/with_gs/scheduler_trace_*.jsonl | jq -s '.' | less")


if __name__ == '__main__':
    main()
