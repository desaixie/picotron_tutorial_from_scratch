import os
import torch
import torch.distributed as dist


class ProcessGroupManager:
    """Manage tensor, pipeline, and data parallel groups for the naive data-parallel step."""

    def __init__(self, dp_size, pp_size, tp_size):
        """Compute rank metadata, build the parallelism grid, and create subgroup handles."""
        raise NotImplementedError(
            "Populate attributes for DP, PP, and TP groups and related helpers."  # CPU/MPS NOTE: Prefer backend="gloo" and guard CUDA-only collectives when GPUs are absent.
        )

    def __str__(self):
        """Return a concise textual summary of the topology for the current rank."""
        raise NotImplementedError("Format DP/PP/TP sizes and ranks for logging.")


def setup_process_group_manager(dp_size, pp_size, tp_size):
    """Instantiate the global process group manager singleton."""
    raise NotImplementedError("Create and cache the process group manager instance.")
