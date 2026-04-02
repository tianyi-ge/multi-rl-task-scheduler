# Mock mindspeed_llm 模块

tasks = type('tasks', (), {
    'posttrain': type('posttrain', (), {
        'rlxf': type('rlxf', (), {
            'group_scheduler': type('group_scheduler', (), {
                'config': None,
                'task': None,
                'worker': None
            })
        })
    })
})

tasks_utils = type('tasks_utils', (), {
    'global_vars': type('global_vars', (), {
        'NPUS_PER_NODE': 8  # 默认值
    })
})

def __getattr__(name):
    raise ImportError(f"Mock无法模拟: {name}")
