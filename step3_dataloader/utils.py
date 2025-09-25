import builtins
import fcntl
import random
import numpy as np
import torch


def print(*args, is_print_rank=True, **kwargs):
    """Serialize stdout across distributed ranks so multi-process prints stay readable."""
    raise NotImplementedError("Implement lock-based synchronized printing across ranks.")


def set_all_seed(seed):
    """Seed Python, NumPy, and PyTorch RNGs (CPU and CUDA) for deterministic runs."""
    raise NotImplementedError("Propagate the provided seed to each supported RNG backend.")


def to_readable_format(num, precision=3):
    """Turn large integers into compact strings with suffixes such as K, M, or B."""
    raise NotImplementedError("Format integer counters into human-friendly strings.")
