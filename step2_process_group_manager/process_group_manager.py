import os
import torch
import torch.distributed as dist


class ProcessGroupManager:
    """Utility for organizing tensor, pipeline, and data parallel process groups on a 3D grid."""

    def __init__(self, dp_size, pp_size, tp_size):
        """Derive ranks, create subgroups, and expose convenience attributes for every axis."""
        raise NotImplementedError(
            "Compute rank metadata and instantiate the distributed subgroups."  # CPU/MPS NOTE: Use backend="gloo" and guard CUDA-only collectives when running without GPUs.
        )

    def __str__(self):
        """Return a short string describing the world topology for this rank."""
        raise NotImplementedError("Format the DP/PP/TP configuration for logging.")


def setup_process_group_manager(dp_size, pp_size, tp_size):
    """Instantiate and cache a `ProcessGroupManager` singleton for other modules to use."""
    raise NotImplementedError("Create the global process group manager instance.")
