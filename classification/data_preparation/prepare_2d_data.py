"""Convert NIfTI volumes to 2D PNG slices for 2D classification.

This script extracts 2D slices from 3D NIfTI volumes and their corresponding
segmentation masks, producing PNG images suitable for the 2D classifier.

Usage:
    python prepare_2d_data.py \\
        --input_dir data/3d/imagesTr \\
        --mask_dir data/3d/labelsTr \\
        --output_img_dir data/2d/images_train \\
        --output_mask_dir data/2d/masks_train \\
        --output_csv data/2d/train.csv
"""
import argparse
import os
import numpy as np
import cv2
import SimpleITK as sitk
import pandas as pd
from pathlib import Path


def extract_slices(image_path, mask_path, output_img_dir, output_mask_dir):
    """Extract non-zero 2D slices from a 3D volume."""
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)

    img_array = sitk.GetArrayFromImage(image)
    mask_array = sitk.GetArrayFromImage(mask)

    case_name = Path(image_path).stem.replace(".nii", "")

    slices_info = []
    for i in range(img_array.shape[0]):
        img_slice = img_array[i]
        mask_slice = mask_array[i]

        # Skip empty slices (all background in mask)
        if mask_slice.max() == 0:
            continue

        # Normalize image to [0, 255] for PNG
        img_norm = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
        img_uint8 = (img_norm * 255).astype(np.uint8)

        # Mask to binary [0, 255]
        mask_uint8 = (mask_slice > 0).astype(np.uint8) * 255

        slice_name = f"{case_name}_{i:03d}.png"
        img_path_out = os.path.join(output_img_dir, case_name, slice_name)
        mask_path_out = os.path.join(output_mask_dir, case_name, slice_name)

        os.makedirs(os.path.dirname(img_path_out), exist_ok=True)
        os.makedirs(os.path.dirname(mask_path_out), exist_ok=True)

        cv2.imwrite(img_path_out, img_uint8)
        cv2.imwrite(mask_path_out, mask_uint8)

        slices_info.append({
            "img_dir": case_name,
            "img_name": slice_name,
            "img_path": img_path_out,
            "mask_path": mask_path_out,
        })

    return slices_info


def main():
    parser = argparse.ArgumentParser("Prepare 2D data from NIfTI volumes")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--output_img_dir", type=str, required=True)
    parser.add_argument("--output_mask_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--labels_csv", type=str, default=None,
                        help="CSV with case_name,T_Stage mapping")
    args = parser.parse_args()

    os.makedirs(args.output_img_dir, exist_ok=True)
    os.makedirs(args.output_mask_dir, exist_ok=True)

    # Load T-stage labels if provided
    label_map = {}
    if args.labels_csv:
        labels_df = pd.read_csv(args.labels_csv)
        for _, row in labels_df.iterrows():
            label_map[row["img_name"]] = row["T_Stage"]

    # Process all cases
    all_slices = []
    image_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith(".nii.gz")])

    for img_file in image_files:
        case_name = img_file.replace(".nii.gz", "")
        mask_file = f"{case_name}.nii.gz"
        mask_path = os.path.join(args.mask_dir, mask_file)

        if not os.path.exists(mask_path):
            print(f"Warning: mask not found for {case_name}, skipping")
            continue

        img_path = os.path.join(args.input_dir, img_file)
        print(f"Processing {case_name}...")

        slices = extract_slices(img_path, mask_path,
                               args.output_img_dir, args.output_mask_dir)

        # Add T-stage label
        t_stage = label_map.get(case_name, 0)
        for s in slices:
            s["T_Stage"] = t_stage

        all_slices.extend(slices)

    # Save CSV
    df = pd.DataFrame(all_slices)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved {len(all_slices)} slices to {args.output_csv}")
    print(f"T-stage distribution: {dict(df.T_Stage.value_counts())}")


if __name__ == "__main__":
    main()
