"""
Setup mindspeed_llm mocks before importing group_scheduler
"""
import sys
import os
from types import ModuleType

# Add group_scheduler directory to path
_gs_dir = os.path.dirname(__file__)
if _gs_dir not in sys.path:
    sys.path.insert(0, _gs_dir)

# Import local modules
import config
import task
import worker

# Create mock mindspeed_llm module structure
mindspeed_llm = ModuleType('mindspeed_llm')

tasks = ModuleType('tasks')
posttrain = ModuleType('posttrain')
rlxf = ModuleType('rlxf')
group_scheduler_gs = ModuleType('group_scheduler')

# Assign to the mock structure
group_scheduler_gs.acceleration_limit_ratio = config.acceleration_limit_ratio
group_scheduler_gs.catch_up_ratio = config.catch_up_ratio
group_schedulerScheduler.max_consecutive_reclaims = config.max_consecutive_reclaims
group_scheduler_gs.max_free_gpu_ratio = config.max_free_gpu_ratio
group_scheduler_gs.TaskTable = task.TaskTable
group_scheduler_gs.TaskConfig = task.TaskConfig
group_scheduler_gs.WorkerTable = worker.WorkerTable
group_scheduler_gs.WorkerInfo = worker.WorkerInfo

# Build the module tree
group_scheduler_gs.task = task
group_scheduler_gs.worker = worker

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

print("mindspeed_llm mocks loaded")
