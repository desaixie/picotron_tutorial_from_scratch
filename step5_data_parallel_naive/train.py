"""torchrun --nproc_per_node 4 train.py --dp_size 4 --micro_batch_size 1 --gradient_accumulation_steps 8 --seq_len 128 --max_tokens 40960 --num_proc 16 --run_name dp_naive --use_wandb"""
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
from data_parallel import DataParallelNaive
from model import Llama
import process_group_manager as pgm
from process_group_manager import setup_process_group_manager
from tensor_parallel import apply_tensor_parallel
from utils import print, set_all_seed, to_readable_format


def train_step(model, dataloader, device):
    """Iterate over micro-batches, optionally defer grad sync, and return accumulated loss."""
    raise NotImplementedError(
        "Implement gradient accumulation with optional all-reduce on the final micro-batch."  # CPU/MPS NOTE: Guard CUDA-only APIs and use gloo for gradient sync on CPU.
    )


def parse_args():
    """Expose CLI options for data-parallel training configuration and logging."""
    raise NotImplementedError(
        "Define argparse arguments covering environment, model, dataset, and parallelism settings."  # CPU/MPS NOTE: Include device/backend flags for CPU/MPS execution.
    )


def main():
    """Initialize DP/TP groups, wrap the model, run the training loop, and handle logging."""
    raise NotImplementedError(
        "Implement setup, optional tensor/data parallel wrapping, training loop, and teardown."  # CPU/MPS NOTE: Prefer backend="gloo" and guard torch.cuda.* usage when GPUs are absent.
    )


if __name__ == "__main__":
    main()
