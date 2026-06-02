"""Evaluation script for trained models.

Supports both 2D and 3D models, with rectal filling subset analysis.

Usage:
    python evaluate.py --config configs/cls_2d_efficientnet.yaml \\
                       --checkpoint outputs/cls_2d/checkpoint/099_model.pth \\
                       --gpu 0
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.cuda.amp as amp
from sklearn import metrics as sklearn_metrics
from torch.utils.data import DataLoader, SequentialSampler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from models import build_model
from data import build_dataset_2d, build_dataset_3d, null_collate_2d, null_collate_3d
from metrics.metrics import calculate_metrics, specificity, calculate_acc_pre_rec
from utils.utils import time_to_str, load_config


@torch.no_grad()
def run_evaluation(model, dataloader, label_key, mixed_precision=True):
    """Generic evaluation for both 2D and 3D models."""
    model.eval()
    valid_num = 0
    valid_truth = []
    valid_probability = []
    start_time = time.time()

    output_key = "label" if label_key == "label" else "T_stage"

    for t, batch in enumerate(dataloader):
        batch_size = len(batch["index"])
        for k in ["image", label_key]:
            batch[k] = batch[k].cuda()

        model.output_type = ["inference"]
        with amp.autocast(enabled=mixed_precision):
            output = model(batch)

        valid_num += batch_size
        valid_truth.append(batch[label_key].data.cpu().numpy())
        valid_probability.append(output[output_key].data.cpu().numpy())

        print(
            f"\r {valid_num:8d} / {len(dataloader.dataset)}  "
            f"{time_to_str(time.time() - start_time, 'sec')}",
            end="", flush=True,
        )
    print()

    assert valid_num == len(dataloader.dataset)

    truth = np.concatenate(valid_truth)
    probability = np.concatenate(valid_probability)

    auc = sklearn_metrics.roc_auc_score(truth, probability)
    spec = specificity(truth, probability)
    metric = calculate_acc_pre_rec(truth, probability)

    acc, prec, rec = metric[0], metric[1], metric[2]

    print(f"\nResults:")
    print(f"  AUC:         {auc:.4f}")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Precision:   {prec:.4f}")
    print(f"  Recall:      {rec:.4f}")
    print(f"  Specificity: {spec:.4f}")

    return {"AUC": auc, "Accuracy": acc, "Precision": prec, "Recall": rec, "Specificity": spec}


def main():
    parser = argparse.ArgumentParser("CRC T-stage Model Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    config = load_config(args.config)

    # Build model and load checkpoint
    model = build_model(config)
    f = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
    model.load_state_dict(f["state_dict"], strict=True)
    model.cuda()

    # Determine 2D or 3D
    is_3d = config.get("model", {}).get("name", "").endswith("_3d")

    if is_3d:
        dataset = build_dataset_3d(config, mode="test")
        dataloader = DataLoader(
            dataset, sampler=SequentialSampler(dataset),
            batch_size=config.get("training", {}).get("val_batch_size", 8),
            drop_last=False, num_workers=4, pin_memory=False,
            collate_fn=null_collate_3d,
        )
        run_evaluation(model, dataloader, label_key="T_stage")
    else:
        dataset = build_dataset_2d(config, mode="test")
        dataloader = DataLoader(
            dataset, sampler=SequentialSampler(dataset),
            batch_size=config.get("training", {}).get("val_batch_size", 64),
            drop_last=False, num_workers=4, pin_memory=False,
            collate_fn=null_collate_2d,
        )
        run_evaluation(model, dataloader, label_key="label")


if __name__ == "__main__":
    main()
