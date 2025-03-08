from hanshu.pendulum import PendulumEnv

from hanshu.ddpg import DDPGPolicy
from hanshu.worker import DummyEnvWorker,RayEnvWorker,SubprocEnvWorker,EnvWorker
"""Data package."""
# isort:skip_file
from gym.envs.registration import register
from hanshu.openfoam import environments
from hanshu.openfoam import utils
from hanshu.openfoam.environments import OpenFoam

from hanshu.base import BasePolicy
from hanshu.ddpg import DDPGPolicy
from hanshu.student_policy import student_Net,mlp,teacher_mlp,student_mlp,student_Net
# from hanshu.registration import register
# from hanshu.core import Env

# import ReplayBuffer_base
__all__ = [
    "ReplayBuffer",
    "OpenFoam",
    "environments",
    ]


register(
    id="OpenFoam-v0",
    entry_point="hanshu.openfoam:OpenFoam", # 第一个myenv是文件夹名字，第二个myenv是文件名字，MyEnv是文件内类的名字
    max_episode_steps=80,
    # reward_threshold=100.0,
)
register(
    id='Pendulum-v3',
    entry_point='hanshu:PendulumEnv',
    max_episode_steps=200,
)

