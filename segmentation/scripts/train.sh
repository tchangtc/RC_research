#!/bin/bash
# Train nnU-Net model
# Usage: bash train.sh <dataset_id> <config> [fold]

DATASET_ID=${1:-100}
CONFIG=${2:-3d_fullres}
FOLD=${3:-all}

export nnUNet_raw="${nnUNet_raw:-data/nnunet}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-data/nnunet_preprocessed}"
export nnUNet_results="${nnUNet_results:-data/nnunet_results}"

echo "=============================================="
echo "nnU-Net v2 Training"
echo "=============================================="
echo "Dataset ID: ${DATASET_ID}"
echo "Configuration: ${CONFIG}"
echo "Fold: ${FOLD}"
echo "=============================================="

if [ "${FOLD}" = "all" ]; then
    # Train all 5 folds
    for fold in 0 1 2 3 4; do
        echo ""
        echo "Training fold ${fold}..."
        nnUNetv2_train ${DATASET_ID} ${CONFIG} ${fold} -tr nnUNetTrainer
    done
else
    echo ""
    echo "Training fold ${FOLD}..."
    nnUNetv2_train ${DATASET_ID} ${CONFIG} ${FOLD} -tr nnUNetTrainer
fi

echo ""
echo "Training complete!"
echo "Results at: ${nnUNet_results}"
