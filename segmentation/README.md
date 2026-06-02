# Stage I: Rectal Tumor Segmentation (nnU-Net v2)

This module handles automatic segmentation of rectal tumors from T2-weighted MRI
using [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet).

## Overview

The segmentation stage produces binary tumor masks that are used as input for the
downstream T-stage classification (Stage II).

**Input:** T2-weighted MRI volumes (NIfTI format, `.nii.gz`)
**Output:** Binary tumor segmentation masks (NIfTI format, `.nii.gz`)
**Labels:** 0 = background, 1 = tumor

## Prerequisites

```bash
pip install nnunetv2
```

## Dataset Preparation

### Step 1: Organize Raw Data

Place your original NIfTI files in the following structure:

```
data/raw/
├── images/          # T2WI MRI volumes (e.g., crc_001.nii.gz)
└── labels/          # Manual tumor annotations (e.g., crc_001.nii.gz)
```

### Step 2: Convert to nnU-Net Format

Run the preparation script:

```bash
python segmentation/prepare_dataset.py \
    --raw_dir data/raw \
    --nnunet_dir data/nnunet \
    --dataset_id 100 \
    --dataset_name RectalCancer
```

This creates the nnU-Net-compatible structure:

```
data/nnunet/Dataset100_RectalCancer/
├── imagesTr/        # Training images (crc_001_0000.nii.gz)
├── labelsTr/        # Training labels (crc_001.nii.gz)
├── imagesTs/        # Test images (crc_266_0000.nii.gz)
├── labelsTs/        # Test labels (crc_266.nii.gz, optional)
└── dataset.json     # Dataset metadata
```

**Note:** nnU-Net v2 requires image filenames to follow the pattern
`<case>_<modality>.nii.gz` (e.g., `crc_001_0000.nii.gz` for the first modality).

### Step 3: Plan and Preprocess

```bash
# Set nnU-Net environment variables
export nnUNet_raw=data/nnunet
export nnUNet_preprocessed=data/nnunet_preprocessed
export nnUNet_results=data/nnunet_results

# Plan and preprocess (automatically determines optimal settings)
bash segmentation/scripts/plan_and_preprocess.sh 100
```

### Step 4: Train

```bash
bash segmentation/scripts/train.sh 100 3d_fullres
```

### Step 5: Predict (Generate Masks for Classification)

```bash
bash segmentation/scripts/predict.sh 100 3d_fullres
```

## Using Segmentation Output for Classification

After prediction, the generated masks can be used directly by the classification module.
The classification datasets expect:

- **2D:** PNG slices extracted from the NIfTI volumes + masks
- **3D:** NIfTI volumes + masks (resampled to isotropic spacing)

See `classification/data_preparation/` for scripts that convert segmentation outputs
into classification-ready formats.

## Tips

1. **3D vs 2D configurations:** For rectal cancer MRI, `3d_fullres` typically performs
   best. Consider `3d_lowres` for faster experimentation.
2. **Ensemble:** You can ensemble predictions from multiple configurations:
   ```bash
   nnUNetv2_ensemble -tr nnUNetTrainer -i data/nnunet_results/Dataset100/...
   ```
3. **Cross-validation:** nnU-Net automatically runs 5-fold cross-validation.
   Use the results to select the best configuration.
