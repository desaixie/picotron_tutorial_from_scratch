"""torchrun --nproc_per_node 2 train.py --tp_size 2 --run_name process_group_manager --use_wandb"""
import argparse
import datetime
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from torch.optim import AdamW
from transformers import AutoConfig

from model import Llama
import process_group_manager as pgm
from process_group_manager import setup_process_group_manager
from utils import print, set_all_seed


def parse_args():
    """Define CLI for configuring tensor/pipeline/data parallel sizes and run metadata."""
    raise NotImplementedError("Add arguments for environment, model, training, and logging options.")


def main():
    """Initialize distributed process groups, build the model, run one dummy step, and report metrics."""
    raise NotImplementedError("Implement process group setup, seeding, model instantiation, training step, and logging.")


if __name__ == "__main__":
    main()
