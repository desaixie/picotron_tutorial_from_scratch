import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import process_group_manager as pgm

STEP, VERBOSE = 0, os.environ.get("VERBOSE", "0") == "1"


def pipeline_communicate(operation, device, dtype, tensor=None, shapes=None):
    """Perform point-to-point pipeline communication for the requested direction."""
    raise NotImplementedError("Implement send/recv logic between pipeline stages for the specified operation.")


def bidirectional_pipeline_communicate(operation, send_tensor, recv_shapes, device, dtype):
    """Send and receive tensors simultaneously during 1F1B steady-state transitions."""
    raise NotImplementedError("Implement paired send/recv operations for bidirectional pipeline communication.")


class PipelineParallel(nn.Module):
    """Pipeline-parallel shard of the transformer used in the 1F1B tutorial step."""

    def __init__(self, model, config):
        """Assign layers to this stage and keep only the relevant submodules."""
        super().__init__()
        raise NotImplementedError("Distribute layers across pipeline ranks and extract local modules.")

    def distribute_layers(self, num_layers):
        """Return the layer indices owned by this pipeline stage."""
        raise NotImplementedError("Compute per-stage layer allocations based on world size and rank.")

    def forward(self, input_ids, position_ids, hidden_states):
        """Run the local embedding/decoder slice and return the stage output."""
        raise NotImplementedError("Implement the forward pass through the stage-resident modules.")

    def backward(self, input_tensor, output_tensor, output_tensor_grad):
        """Backpropagate through the local stage and return gradients for the previous stage."""
        raise NotImplementedError("Perform backward pass and return input gradients when applicable.")


def train_step_pipeline_afab(model, data_loader, tensor_shapes, device, dtype):
    """Execute an AFAB-style pipeline step: all forwards followed by all backwards."""
    raise NotImplementedError("Implement the AFAB schedule including communications and loss accumulation.")


def train_step_pipeline_1f1b(model, data_loader, tensor_shapes, device, dtype):
    """Run a 1F1B pipeline schedule with warmup, steady state, and cooldown phases."""
    raise NotImplementedError("Implement forward/backward interleaving and bidirectional comms for 1F1B.")
