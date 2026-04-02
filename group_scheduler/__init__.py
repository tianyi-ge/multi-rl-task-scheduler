"""
GroupScheduler mock setup
Must be imported before using group_scheduler module
"""
import sys
import os
from types import ModuleType

from .yr_mock import mock_yr as yr

# Check if already initialized
if 'mindspeed_llm' in sys.modules:
    pass
else:
    # Create mock mindspeed_llm module structure
    mindspeed_llm = ModuleType('mindspeed_llm')

    tasks = ModuleType('tasks')
    posttrain = ModuleType('posttrain')
    rlxf = ModuleType('rlxf')
    group_scheduler_gs = ModuleType('group_scheduler')

    # Build module structure first
    rlxf.group_scheduler = group_scheduler_gs
    posttrain.rlxf = rlxf
    tasks.posttrain = posttrain

    tasks_utils = ModuleType('tasks_utils')
    global_vars = ModuleType('global_vars')
    global_vars.NPUS_PER_NODE = 8
    tasks_utils.global_vars = global_vars

    mindspeed_llm.tasks = tasks
    mindspeed_llm.tasks_utils = tasks_utils

    # Register in sys.modules
    sys.modules['mindspeed_llm'] = mindspeed_llm
    sys.modules['mindspeed_llm.tasks'] = tasks
    sys.modules['mindspeed_llm.tasks.posttrain'] = posttrain
    sys.modules['mindspeed_llm.tasks.posttrain.rlxf'] = rlxf
    sys.modules['mindspeed_llm.tasks.posttrain.rlxf.group_scheduler'] = group_scheduler_gs
    sys.modules['mindspeed_llm.tasks.utils'] = tasks_utils
    sys.modules['mindspeed_llm.tasks.utils.global_vars'] = global_vars

    # Now import and set up actual modules
    pkg_dir = os.path.dirname(__file__)
    old_path = list(sys.path)

    # Add package directory to path
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)

    try:
        import config as _config
        import task as _task
        import worker as _worker
        import data_class as _data_class

        group_scheduler_gs.acceleration_limit_ratio = _config.acceleration_limit_ratio
        group_scheduler_gs.catch_up_ratio = _config.catch_up_ratio
        group_scheduler_gs.max_consecutive_reclaims = _config.max_consecutive_reclaims
        group_scheduler_gs.max_free_gpu_ratio = _config.max_free_gpu_ratio
        group_scheduler_gs.TaskTable = _task.TaskTable
        group_scheduler_gs.TaskConfig = _task.TaskConfig
        group_scheduler_gs.WorkerTable = _worker.WorkerTable
        group_scheduler_gs.WorkerInfo = _worker.WorkerInfo

        group_scheduler_gs.task = _task
        group_scheduler_gs.worker = _worker
        group_scheduler_gs.data_class = _data_class
    finally:
        sys.path[:] = old_path
