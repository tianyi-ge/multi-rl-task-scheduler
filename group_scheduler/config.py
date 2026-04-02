"""
Group Scheduler 配置模块
"""
import yaml
from pathlib import Path
from pydantic import BaseModel, Field


# 定义配置结构
class SchedulerConfig(BaseModel):
    """Group Scheduler 配置类"""
    acceleration_limit_ratio: float = Field(default=2.0, gt=0, description="任务允许占用的最大卡数上限比例")
    catch_up_ratio: float = Field(default=1.5, gt=0, description="追赶比例")
    max_consecutive_reclaims: int = Field(default=3, ge=0, description="最大连续回收次数")
    max_free_gpu_ratio: float = Field(default=0.3, ge=0, le=1.0, description="最大空闲GPU比例")


# 读取配置文件
def load_scheduler_config() -> SchedulerConfig:
    """从 config.yaml 加载配置"""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        raw_data = yaml.safe_load(f)
    return SchedulerConfig(**raw_data)


# 导出配置变量
scheduler_config = load_scheduler_config()
acceleration_limit_ratio = scheduler_config.acceleration_limit_ratio
catch_up_ratio = scheduler_config.catch_up_ratio
max_consecutive_reclaims = scheduler_config.max_consecutive_reclaims
max_free_gpu_ratio = scheduler_config.max_free_gpu_ratio