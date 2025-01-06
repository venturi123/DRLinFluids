from gym.envs.registration import register
from DRLinFluids import environments
from DRLinFluids import cfd, utils
from DRLinFluids.environments import OpenFoam

__all__ = [
    "OpenFoam",
]


register(
    id="OpenFoam-v0",
    entry_point="DRLinFluids.environments:OpenFoam",
    max_episode_steps=100,
)
