import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import process_group_manager as pgm

STEP, VERBOSE = 0, os.environ.get("VERBOSE", "0") == "1"


def pipeline_communicate(operation, device, dtype, tensor=None, shapes=None):
    """Handle point-to-point pipeline communications for forward/backward micro-batches."""
    raise NotImplementedError("Implement send/recv logic for pipeline stages based on the operation type.")


class PipelineParallel(nn.Module):
    """Pipeline-parallel wrapper that owns a shard of decoder layers for AFAB scheduling."""

    def __init__(self, model, config):
        """Distribute layers across pipeline stages and keep stage-specific modules."""
        super().__init__()
        raise NotImplementedError("Determine layer ownership for this stage and extract the relevant submodules.")

    def distribute_layers(self, num_layers):
        """Return the layer indices assigned to this pipeline stage."""
        raise NotImplementedError("Compute balanced layer assignment across pipeline ranks.")

    def forward(self, input_ids, position_ids, hidden_states):
        """Run the local embedding/decoder slice and return the stage output."""
        raise NotImplementedError("Implement the forward pass for the pipeline stage.")

    def backward(self, input_tensor, output_tensor, output_tensor_grad):
        """Backpropagate through the local stage and return gradients for the previous stage."""
        raise NotImplementedError("Perform backward pass on the stage and return input gradients when needed.")


def train_step_pipeline_afab(model, data_loader, tensor_shapes, device, dtype):
    """Execute one training iteration using Activation Forward / Activation Backward pipeline scheduling."""
    raise NotImplementedError("Implement AFAB forward/backward passes, pipeline communications, and loss reporting.")
