# read version from installed package
from importlib.metadata import version
import os

from drlinfluids import runner, wrapper, extractor, logger, utils

__version__ = version("drlinfluids")
__srcpath__ = os.path.dirname(__file__)

__all__ = [
    "utils",
    "environment",
    "extractor",
    "logger",
    "runner",
    "wrapper",
]
