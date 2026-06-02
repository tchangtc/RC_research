"""Prepare 3D data for classification.

Resamples NIfTI volumes to isotropic spacing using B-spline interpolation,
matching the preprocessing used in the paper.

Usage:
    python prepare_3d_data.py \\
        --input_dir data/raw/images \\
        --mask_dir data/raw/labels \\
        --output_dir data/3d \\
        --target_spacing 0.36 0.36 0.36
"""
import argparse
import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
from pathlib import Path


def resample_image(image, target_spacing, interp=sitk.sitkBSpline):
    """Resample image to target isotropic spacing."""
    identity = sitk.Transform(3, sitk.sitkIdentity)
    sp = image.GetSpacing()
    sz = image.GetSize()

    new_sz = (
        int(round(sz[0] * sp[0] / target_spacing[0])),
        int(round(sz[1] * sp[1] / target_spacing[1])),
        int(round(sz[2] * sp[2] / target_spacing[2])),
    )

    ref = sitk.Image(new_sz, image.GetPixelIDValue())
    ref.SetSpacing(target_spacing)
    ref.SetOrigin(image.GetOrigin())
    ref.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])

    return sitk.Resample(image, ref, identity, interp)


def resample_label(label, target_spacing, ref_image):
    """Resample label using nearest neighbor to preserve label values."""
    identity = sitk.Transform(3, sitk.sitkIdentity)
    np_label = sitk.GetArrayFromImage(label)
    labels = np.unique(np_label)
    resampled_list = []

    for lbl_val in labels:
        tmp = (np_label == lbl_val).astype(np.uint8)
        tmp_img = sitk.GetImageFromArray(tmp)
        tmp_img.CopyInformation(label)
        tmp_img.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
        tmp_resampled = sitk.Resample(tmp_img, ref_image, identity, sitk.sitkNearestNeighbor)
        resampled_list.append(sitk.GetArrayFromImage(tmp_resampled))

    one_hot = np.stack(resampled_list, axis=0)
    result = np.argmax(one_hot, axis=0).astype(np.uint8)
    out = sitk.GetImageFromArray(result)
    out.CopyInformation(ref_image)
    return out


def main():
    parser = argparse.ArgumentParser("Prepare 3D data for classification")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--target_spacing", type=float, nargs=3, default=[0.36, 0.36, 0.36])
    parser.add_argument("--labels_csv", type=str, default=None)
    args = parser.parse_args()

    output_img = os.path.join(args.output_dir, "imagesTr")
    output_mask = os.path.join(args.output_dir, "labelsTr")
    os.makedirs(output_img, exist_ok=True)
    os.makedirs(output_mask, exist_ok=True)

    target_spacing = tuple(args.target_spacing)

    # Load labels
    label_map = {}
    if args.labels_csv:
        labels_df = pd.read_csv(args.labels_csv)
        for _, row in labels_df.iterrows():
            label_map[row["img_name"]] = row["T_Stage"]

    image_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith(".nii.gz")])
    csv_rows = []

    for img_file in image_files:
        case_name = img_file.replace(".nii.gz", "")
        mask_file = f"{case_name}.nii.gz"
        mask_path = os.path.join(args.mask_dir, mask_file)

        if not os.path.exists(mask_path):
            print(f"Warning: mask not found for {case_name}, skipping")
            continue

        print(f"Processing {case_name}...")

        image = sitk.ReadImage(os.path.join(args.input_dir, img_file))
        mask = sitk.ReadImage(mask_path)

        # Resample
        resampled_img = resample_image(image, target_spacing, sitk.sitkBSpline)
        resampled_mask = resample_label(mask, target_spacing, resampled_img)

        # Save
        sitk.WriteImage(resampled_img, os.path.join(output_img, f"{case_name}.nii.gz"))
        sitk.WriteImage(resampled_mask, os.path.join(output_mask, f"{case_name}.nii.gz"))

        t_stage = label_map.get(case_name, 0)
        csv_rows.append({
            "img_name": case_name,
            "img_path": f"imagesTr/{case_name}.nii.gz",
            "mask_path": f"labelsTr/{case_name}.nii.gz",
            "T_Stage": t_stage,
        })

    # Save CSV
    df = pd.DataFrame(csv_rows)
    csv_path = os.path.join(args.output_dir, "train.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(csv_rows)} cases to {csv_path}")


if __name__ == "__main__":
    main()
