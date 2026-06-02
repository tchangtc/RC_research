#!/bin/bash
# Run nnU-Net prediction
# Usage: bash predict.sh <dataset_id> <config> [fold]

DATASET_ID=${1:-100}
CONFIG=${2:-3d_fullres}
FOLD=${3:-0}

export nnUNet_raw="${nnUNet_raw:-data/nnunet}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-data/nnunet_preprocessed}"
export nnUNet_results="${nnUNet_results:-data/nnunet_results}"

DATASET_NAME="Dataset${DATASET_ID}_RectalCancer"
INPUT_DIR="${nnUNet_raw}/${DATASET_NAME}/imagesTs"
OUTPUT_DIR="data/segmentation_output"

echo "=============================================="
echo "nnU-Net v2 Prediction"
echo "=============================================="
echo "Dataset ID: ${DATASET_ID}"
echo "Configuration: ${CONFIG}"
echo "Fold: ${FOLD}"
echo "Input: ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "=============================================="

mkdir -p ${OUTPUT_DIR}

# Find trained model
MODEL_DIR="${nnUNet_results}/${DATASET_NAME}/nnUNetTrainer__${CONFIG}__nnUNetPlans/${FOLD}"

if [ ! -d "${MODEL_DIR}" ]; then
    echo "ERROR: Model directory not found: ${MODEL_DIR}"
    echo "Run train.sh first."
    exit 1
fi

# Predict test set
echo ""
echo "Running prediction..."
nnUNetv2_predict \
    -i ${INPUT_DIR} \
    -o ${OUTPUT_DIR} \
    -d ${DATASET_ID} \
    -c ${CONFIG} \
    -f ${FOLD} \
    -tr nnUNetTrainer

echo ""
echo "Prediction complete!"
echo "Segmentation masks saved to: ${OUTPUT_DIR}"
echo ""
echo "Next step: Run classification/data_preparation/prepare_3d_data.py"
echo "to convert these masks into classification-ready format."
