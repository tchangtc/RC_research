"""3D classification training entry point.

Usage:
    python train_3d.py --config configs/cls_3d_ynet.yaml [--gpu 0]
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from models import build_model
from data import build_dataset_3d, null_collate_3d
from data.augmentation_3d import (
    GaussianNoiseTransform, GaussianBlurTransform, BrightnessTransform,
    GammaTransform, RandomRotFlip, MirrorTransform,
)
from engine.trainer import Trainer
from utils.utils import set_all_random_seed, load_config


def get_train_transforms():
    """Default 3D augmentation pipeline (nnU-Net style)."""
    return transforms.Compose([
        GaussianNoiseTransform(noise_variance=(0, 0.07), p_per_sample=0.25),
        GaussianBlurTransform(blur_sigma=(1, 2.5), different_sigma_per_channel=False,
                              p_per_channel=0.25, p_per_sample=0.25),
        BrightnessTransform(mu=0, sigma=1, per_channel=False, p_per_channel=0.25,
                            p_per_sample=0.25),
        GammaTransform(gamma_range=(0.25, 1), per_channel=False, p_per_sample=0.25),
        RandomRotFlip(p_per_sample=0.25),
        MirrorTransform(p_per_sample=0.25),
    ])


def main():
    parser = argparse.ArgumentParser("3D CRC T-stage Classification")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # Load config
    config = load_config(args.config)
    seed = config.get("training", {}).get("seed", 42)
    set_all_random_seed(seed)

    # Build model
    model = build_model(config)
    model.cuda()

    # Build datasets
    train_dataset = build_dataset_3d(config, mode="train")
    train_dataset.transforms = get_train_transforms()
    val_dataset = build_dataset_3d(config, mode="test")

    train_cfg = config.get("training", {})
    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=train_cfg.get("batch_size", 8),
        drop_last=False,
        num_workers=train_cfg.get("num_workers", 16),
        pin_memory=False,
        worker_init_fn=lambda wid: np.random.seed(torch.initial_seed() // 2**32 + wid),
        collate_fn=null_collate_3d,
    )
    val_loader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=train_cfg.get("val_batch_size", 8),
        drop_last=False,
        num_workers=train_cfg.get("num_workers", 16),
        pin_memory=False,
        collate_fn=null_collate_3d,
    )

    print(f"Train dataset:\n{train_dataset}")
    print(f"Val dataset:\n{val_dataset}")

    # Train
    trainer = Trainer(model, train_loader, val_loader, config, label_key="T_stage")
    trainer.train()


if __name__ == "__main__":
    main()
