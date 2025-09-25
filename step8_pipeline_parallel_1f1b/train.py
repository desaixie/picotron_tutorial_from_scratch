"""torchrun --nproc_per_node 4 train.py --pp_size 4 --pp_engine 1f1b --micro_batch_size 4 --gradient_accumulation_steps 8 --seq_len 128 --max_tokens 40960 --num_proc 16 --run_name pp_1f1b --use_wandb"""
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
from pipeline_parallel import PipelineParallel, train_step_pipeline_afab, train_step_pipeline_1f1b
from utils import print, set_all_seed, to_readable_format


def all_reduce_loss_across_dp_ranks(loss, device):
    """Average the loss across data-parallel ranks, acting only on the last pipeline stage."""
    raise NotImplementedError("Implement DP loss reduction when running on the last pipeline stage.")


def train_step(model, dataloader, device):
    """Run a non-pipelined training step with gradient accumulation and return the averaged loss."""
    raise NotImplementedError("Iterate over micro-batches, handle grad sync gating, and accumulate loss.")


def parse_args():
    """Define CLI options for the 1F1B pipeline training script."""
    raise NotImplementedError("Expose environment, model, data, parallelism, and logging flags.")


def main():
    """Set up TP/PP/DP configurations, wrap the model, and execute the training loop."""
    raise NotImplementedError("Implement initialization, pipeline scheduling selection, training loop, and teardown.")


if __name__ == "__main__":
    main()
