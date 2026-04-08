# multi-rl-task-scheduler

Multi-task RL scheduler design and code skeleton for a shared GPU worker pool.

## Current status

This repository now contains:
- design spec in `docs/superpowers/specs/2026-03-24-grpo-gpu-scheduler-design.md`
- Python interface skeleton in `src/multi_rl_task_scheduler/`

## Code / spec alignment

The current Python skeleton aligns with issue #2 and the latest spec on these points:
- `InferScheduler.reclaim()` returns `TaskStateReport`
- `InferScheduler.assign()` returns `TaskStateReport`
- `TaskStateReport` includes `state_version`
- `GroupScheduler` drops stale reports and rebuilds free worker state from `assigned_workers`
- `compute_allocation_score()` follows the issue #2 exponential scoring rule
- `find_best_placement_global()` follows the issue #2 two-pass placement rule
