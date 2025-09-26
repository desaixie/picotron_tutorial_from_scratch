import builtins
import fcntl
import random
import numpy as np
import torch


def print(*args, is_print_rank=True, **kwargs):
    """Synchronize stdout so bucketed data-parallel logs stay readable."""
    raise NotImplementedError("Implement rank-aware printing with file locking.")


def set_all_seed(seed):
    """Seed all RNGs (Python, NumPy, PyTorch CPU/CUDA) for reproducibility."""
    raise NotImplementedError(
        "Propagate the seed to each RNG backend."  # CPU/MPS NOTE: Guard torch.cuda.manual_seed_all() and optionally seed torch.mps.
    )


def to_readable_format(num, precision=3):
    """Format integers into strings with units (K/M/B/T) for compact logging."""
    raise NotImplementedError("Convert counts into human-friendly strings.")
