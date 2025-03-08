from hanshu.pendulum import PendulumEnv
from hanshu.openfoam import cfd
from hanshu.openfoam import environments
from hanshu.openfoam import utils
from hanshu.tensorfoam.openfoam.offpolicy import offpolicy_trainer
from hanshu.openfoam.environments import OpenFoam
"""Data package."""
# isort:skip_file
from gym import register

# import ReplayBuffer_base
__all__ = [
    "ReplayBuffer",
    "OpenFoam",
    "environments",
    ]


register(
    id='Pendulum-v3',
    entry_point='hanshu:PendulumEnv',
    max_episode_steps=200,
)

