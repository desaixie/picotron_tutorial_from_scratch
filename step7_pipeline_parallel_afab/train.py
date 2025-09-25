"""torchrun --nproc_per_node 4 train.py --pp_size 4 --pp_engine afab --micro_batch_size 4 --gradient_accumulation_steps 8 --seq_len 128 --max_tokens 40960 --num_proc 16 --run_name pp_afab --use_wandb"""
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

from data_parallel import DataParallelBucket
from dataloader import MicroBatchDataLoader
from model import Llama
import process_group_manager as pgm
from process_group_manager import setup_process_group_manager
from tensor_parallel import apply_tensor_parallel
from pipeline_parallel import PipelineParallel, train_step_pipeline_afab
from utils import print, set_all_seed, to_readable_format


def all_reduce_loss_across_dp_ranks(loss, device):
    """Average the scalar loss across data-parallel ranks, only acting on the last pipeline stage."""
    raise NotImplementedError("Implement all-reduce of the loss across the DP group when on the last stage.")


def train_step(model, dataloader, device):
    """Run one non-pipelined training step with gradient accumulation and return the averaged loss."""
    raise NotImplementedError("Iterate over micro-batches, control gradient sync, and accumulate loss.")


def parse_args():
    """Define CLI options for the AFAB pipeline-parallel training script."""
    raise NotImplementedError("Expose environment, model, dataset, parallelism, and logging arguments.")


def main():
    """Initialize TP/PP/DP stacks, wrap the model, execute training loop, and report metrics."""
    raise NotImplementedError("Implement orchestration for AFAB training including setup, loop, and teardown.")


if __name__ == "__main__":
    main()
