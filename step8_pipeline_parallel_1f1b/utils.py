import builtins
import fcntl
import random
import numpy as np
import torch


def print(*args, is_print_rank=True, **kwargs):
    """Synchronize stdout across ranks to keep 1F1B pipeline logs readable."""
    raise NotImplementedError("Implement rank-aware printing with file locking.")


def set_all_seed(seed):
    """Seed Python, NumPy, and PyTorch (CPU/CUDA) RNGs for deterministic runs."""
    raise NotImplementedError("Propagate the seed to each RNG backend.")


def to_readable_format(num, precision=3):
    """Format integer counters into short strings with suffixes (K/M/B/T)."""
    raise NotImplementedError("Convert large counts into compact human-readable strings.")
