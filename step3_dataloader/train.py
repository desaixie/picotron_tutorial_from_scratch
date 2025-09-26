"""torchrun --nproc_per_node 1 train.py --micro_batch_size 4 --gradient_accumulation_steps 8 --seq_len 128 --max_tokens 40960 --num_proc 16 --run_name dataloader --use_wandb"""
import argparse
import datetime
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from torch.optim import AdamW
from transformers import AutoConfig

import lovely_tensors as lt
lt.monkey_patch()

from dataloader import MicroBatchDataLoader
from model import Llama
import process_group_manager as pgm
from process_group_manager import setup_process_group_manager
from utils import print, set_all_seed, to_readable_format


def train_step(model, dataloader, device):
    """Consume `grad_acc_steps` micro-batches, accumulate gradients, and return the scalar loss."""
    raise NotImplementedError(
        "Iterate over the micro-batch loader, compute loss, and accumulate gradients."  # CPU/MPS NOTE: Ensure tensors are moved to the active device and guard CUDA-only ops.
    )


def parse_args():
    """Expose CLI options covering environment, model, dataset, parallelism, and logging settings."""
    raise NotImplementedError(
        "Define CLI arguments for dataloader training step."  # CPU/MPS NOTE: Allow selection of backend/device so CPU runs can request gloo and cpu/mps devices.
    )


def main():
    """Set up distributed context, data loader, and run the token-budgeted training loop."""
    raise NotImplementedError(
        "Implement initialization, logging, training loop, and tear-down for step 3."  # CPU/MPS NOTE: Replace torch.cuda.* usage with device-agnostic guards and prefer backend="gloo" when CUDA is unavailable.
    )


if __name__ == "__main__":
    main()
