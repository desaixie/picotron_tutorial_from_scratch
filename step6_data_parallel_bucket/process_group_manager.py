import os
import torch
import torch.distributed as dist


class ProcessGroupManager:
    """Manage DP/PP/TP process groups for the gradient-bucket data-parallel tutorial step."""

    def __init__(self, dp_size, pp_size, tp_size):
        """Populate rank metadata, construct the parallelism grid, and create subgroup handles."""
        raise NotImplementedError("Initialize DP, PP, and TP groups along with helper attributes.")

    def __str__(self):
        """Summarize the current rank's position and group sizes."""
        raise NotImplementedError("Format DP/PP/TP sizes and ranks for logging.")


def setup_process_group_manager(dp_size, pp_size, tp_size):
    """Instantiate and cache the global process group manager."""
    raise NotImplementedError("Create the singleton ProcessGroupManager instance.")
