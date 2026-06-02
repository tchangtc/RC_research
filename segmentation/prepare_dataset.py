"""Convert raw NIfTI data to nnU-Net v2 dataset format.

Usage:
    python prepare_dataset.py \\
        --raw_dir data/raw \\
        --nnunet_dir data/nnunet \\
        --dataset_id 100 \\
        --dataset_name RectalCancer \\
        --train_ratio 0.8
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np


def create_dataset_json(output_dir, dataset_name, channel_names, labels,
                        num_training, num_test):
    """Create nnU-Net v2 dataset.json metadata file."""
    dataset_json = {
        "channel_names": channel_names,
        "labels": labels,
        "numTraining": num_training,
        "numTest": num_test,
        "name": dataset_name,
        "description": f"{dataset_name} dataset for rectal cancer tumor segmentation",
        "reference": "Deep Learning Models for Preoperative T-stage Assessment in Rectal Cancer Using MRI",
        "licence": "CC BY 4.0",
        "release": "1.0",
    }

    json_path = os.path.join(output_dir, "dataset.json")
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=2)
    print(f"Created {json_path}")


def prepare_dataset(raw_dir, nnunet_dir, dataset_id, dataset_name,
                    train_ratio=0.8, seed=42):
    """Convert raw NIfTI data to nnU-Net v2 format.

    Expected raw structure:
        raw_dir/
            images/   -> *.nii.gz
            labels/   -> *.nii.gz (matching filenames)
    """
    images_dir = os.path.join(raw_dir, "images")
    labels_dir = os.path.join(raw_dir, "labels")

    dataset_dir = os.path.join(nnunet_dir, f"Dataset{dataset_id:03d}_{dataset_name}")
    images_tr = os.path.join(dataset_dir, "imagesTr")
    labels_tr = os.path.join(dataset_dir, "labelsTr")
    images_ts = os.path.join(dataset_dir, "imagesTs")
    labels_ts = os.path.join(dataset_dir, "labelsTs")

    for d in [images_tr, labels_tr, images_ts, labels_ts]:
        os.makedirs(d, exist_ok=True)

    # Get all case names
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".nii.gz")])
    case_names = [f.replace(".nii.gz", "") for f in image_files]

    # Shuffle and split
    np.random.seed(seed)
    np.random.shuffle(case_names)
    split_idx = int(len(case_names) * train_ratio)
    train_cases = case_names[:split_idx]
    test_cases = case_names[split_idx:]

    print(f"Total cases: {len(case_names)}")
    print(f"Train cases: {len(train_cases)}")
    print(f"Test cases:  {len(test_cases)}")

    # Copy training data (nnU-Net naming: case_0000.nii.gz)
    for case in train_cases:
        src_img = os.path.join(images_dir, f"{case}.nii.gz")
        src_lab = os.path.join(labels_dir, f"{case}.nii.gz")

        dst_img = os.path.join(images_tr, f"{case}_0000.nii.gz")
        dst_lab = os.path.join(labels_tr, f"{case}.nii.gz")

        shutil.copy2(src_img, dst_img)
        if os.path.exists(src_lab):
            shutil.copy2(src_lab, dst_lab)

    # Copy test data
    for case in test_cases:
        src_img = os.path.join(images_dir, f"{case}.nii.gz")
        src_lab = os.path.join(labels_dir, f"{case}.nii.gz")

        dst_img = os.path.join(images_ts, f"{case}_0000.nii.gz")
        dst_lab = os.path.join(labels_ts, f"{case}.nii.gz")

        shutil.copy2(src_img, dst_img)
        if os.path.exists(src_lab):
            shutil.copy2(src_lab, dst_lab)

    # Create dataset.json
    create_dataset_json(
        output_dir=dataset_dir,
        dataset_name=dataset_name,
        channel_names={"0": "T2WI"},
        labels={"background": 0, "tumor": 1},
        num_training=len(train_cases),
        num_test=len(test_cases),
    )

    print(f"\nDataset prepared at: {dataset_dir}")


def main():
    parser = argparse.ArgumentParser("Prepare nnU-Net v2 Dataset")
    parser.add_argument("--raw_dir", type=str, required=True, help="Raw data directory")
    parser.add_argument("--nnunet_dir", type=str, required=True, help="nnU-Net output directory")
    parser.add_argument("--dataset_id", type=int, default=100, help="Dataset ID (3-digit)")
    parser.add_argument("--dataset_name", type=str, default="RectalCancer")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_dataset(
        args.raw_dir, args.nnunet_dir,
        args.dataset_id, args.dataset_name,
        args.train_ratio, args.seed,
    )


if __name__ == "__main__":
    main()
