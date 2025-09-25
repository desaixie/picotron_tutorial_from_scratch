import builtins
import fcntl
import random
import numpy as np
import torch


def print(*args, is_print_rank=True, **kwargs):
    """Serialize stdout across ranks to keep pipeline-parallel logs coherent."""
    raise NotImplementedError("Implement rank-aware printing using file locking.")


def set_all_seed(seed):
    """Seed Python, NumPy, and PyTorch RNGs (CPU and CUDA) with the provided value."""
    raise NotImplementedError("Propagate the seed to each RNG backend.")


def to_readable_format(num, precision=3):
    """Convert integer counters to compact strings with units (K/M/B/T)."""
    raise NotImplementedError("Format large integers into human-readable tokens.")
