"""torchrun --nproc_per_node 4 train.py --tp_size 4 --micro_batch_size 4 --gradient_accumulation_steps 8 --seq_len 128 --max_tokens 40960 --num_proc 16 --run_name tp_naive --use_wandb"""
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
from tensor_parallel import apply_tensor_parallel
from utils import print, set_all_seed, to_readable_format


def train_step(model, dataloader, device):
    """Iterate over micro-batches, accumulate gradients, and return the averaged loss."""
    raise NotImplementedError("Consume the dataloader, compute loss, scale by grad accumulation, and backpropagate.")


def parse_args():
    """Expose CLI flags for tensor-parallel training including environment, data, and logging options."""
    raise NotImplementedError("Define argparse arguments mirroring the reference implementation.")


def main():
    """Initialize tensor parallel groups, wrap the model, and execute the training loop."""
    raise NotImplementedError("Implement setup, tensor-parallel application, training loop, and logging for step 4.")


if __name__ == "__main__":
    main()
