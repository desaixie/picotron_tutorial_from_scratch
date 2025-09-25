import contextlib
from typing import List

import torch
import torch.distributed as dist
from torch import nn

import process_group_manager as pgm


class DataParallelNaive(nn.Module):
    """Minimal data-parallel wrapper that synchronizes gradients via explicit all-reduce hooks."""

    def __init__(self, module):
        """Wrap the provided module and register gradient hooks for naive data parallelism."""
        super().__init__()
        raise NotImplementedError("Store the wrapped module and set up backward hooks for gradient sync.")

    def forward(self, *inputs, **kwargs):
        """Delegate the forward pass to the wrapped module."""
        raise NotImplementedError("Call the underlying module with the provided inputs.")

    def register_backward_hook(self, hook):
        """Attach the gradient synchronization hook to every parameter requiring gradients."""
        raise NotImplementedError("Iterate over parameters and register the provided hook where needed.")

    def _allreduce_grads(self, grad):
        """All-reduce gradients across data-parallel ranks when synchronization is enabled."""
        raise NotImplementedError("Perform the gradient all-reduce and averaging across the DP group.")
