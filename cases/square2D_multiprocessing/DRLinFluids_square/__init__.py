from gym.envs.registration import register
from DRLinFluids import environments_tianshou
from DRLinFluids import environment_tensorforce
from DRLinFluids import utils,cfd
from DRLinFluids.environments_tianshou import OpenFoam_tianshou
from DRLinFluids.environment_tensorforce import OpenFoam_tensorforce

__all__ = [
    "OpenFoam_tianshou",
    "OpenFoam_tensorforce",
]

register(
    id="OpenFoam-v0",
    entry_point="square2D.DRLinFluids:OpenFoam_tianshou", # 第一个myenv是文件夹名字，第二个myenv是文件名字，MyEnv是文件内类的名字
    max_episode_steps=100,
    #reward_threshold=100.0,
)


