"""2D classification training entry point.

Usage:
    python train_2d.py --config configs/cls_2d_efficientnet.yaml [--gpu 0]
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from models import build_model
from data import build_dataset_2d, null_collate_2d
from engine.trainer import Trainer
from utils.utils import set_all_random_seed, load_config


def main():
    parser = argparse.ArgumentParser("2D CRC T-stage Classification")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Load config
    config = load_config(args.config)
    seed = config.get("training", {}).get("seed", 42)
    set_all_random_seed(seed)

    # Build model
    model = build_model(config)
    model.cuda()

    # Build datasets
    train_dataset = build_dataset_2d(config, mode="train")
    val_dataset = build_dataset_2d(config, mode="test")

    train_cfg = config.get("training", {})
    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=train_cfg.get("batch_size", 32),
        drop_last=True,
        num_workers=train_cfg.get("num_workers", 8),
        pin_memory=False,
        worker_init_fn=lambda wid: np.random.seed(torch.initial_seed() // 2**32 + wid),
        collate_fn=null_collate_2d,
    )
    val_loader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=train_cfg.get("val_batch_size", 64),
        drop_last=False,
        num_workers=train_cfg.get("num_workers", 8),
        pin_memory=False,
        collate_fn=null_collate_2d,
    )

    print(f"Train dataset:\n{train_dataset}")
    print(f"Val dataset:\n{val_dataset}")

    # Train
    trainer = Trainer(model, train_loader, val_loader, config, label_key="label")
    trainer.train()


if __name__ == "__main__":
    main()
