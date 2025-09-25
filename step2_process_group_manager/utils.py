import builtins
import fcntl
import random
import numpy as np
import torch


def print(*args, is_print_rank=True, **kwargs):
    """Synchronize stdout across processes to avoid interleaved logging."""
    raise NotImplementedError("Implement cross-rank print with file locking.")


def set_all_seed(seed):
    """Seed Python, NumPy, and PyTorch (CPU/CUDA) RNGs for reproducibility."""
    raise NotImplementedError("Propagate the seed to each framework-specific RNG.")


def to_readable_format(num, precision=3):
    """Convert large integer counters to shorthand strings like 1.2K or 3.4M."""
    raise NotImplementedError("Format integer values into human-readable strings for logging.")
