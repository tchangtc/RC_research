#!/bin/bash
# Plan and preprocess nnU-Net dataset
# Usage: bash plan_and_preprocess.sh <dataset_id>

DATASET_ID=${1:-100}
DATASET_NAME="Dataset${DATASET_ID}_RectalCancer"

# Set nnU-Net paths (customize these for your environment)
export nnUNet_raw="${nnUNet_raw:-data/nnunet}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-data/nnunet_preprocessed}"
export nnUNet_results="${nnUNet_results:-data/nnunet_results}"

echo "=============================================="
echo "nnU-Net v2 Plan & Preprocess"
echo "=============================================="
echo "Dataset: ${DATASET_NAME}"
echo "nnUNet_raw: ${nnUNet_raw}"
echo "nnUNet_preprocessed: ${nnUNet_preprocessed}"
echo "nnUNet_results: ${nnUNet_results}"
echo "=============================================="

# Verify dataset exists
if [ ! -d "${nnUNet_raw}/${DATASET_NAME}" ]; then
    echo "ERROR: Dataset directory not found: ${nnUNet_raw}/${DATASET_NAME}"
    echo "Run prepare_dataset.py first."
    exit 1
fi

# Plan and preprocess 3D full resolution
echo ""
echo "Planning 3d_fullres..."
nnUNetv2_plan_and_preprocess -d ${DATASET_ID} -pl nnUNetPlannerResEncL -c 3d_fullres

# Plan and preprocess 3D low resolution (optional, for ensemble)
echo ""
echo "Planning 3d_lowres..."
nnUNetv2_plan_and_preprocess -d ${DATASET_ID} -pl nnUNetPlannerResEncL -c 3d_lowres

echo ""
echo "Preprocessing complete!"
echo "Preprocessed data at: ${nnUNet_preprocessed}/${DATASET_NAME}"
