# Stage I: Rectal Tumor Segmentation (nnU-Net v2)

Automatic segmentation of rectal tumors from T2-weighted MRI using [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet).

## Overview

The segmentation stage produces **binary tumor masks** used as input for the downstream T-stage classification (Stage II).

| Item | Details |
|------|---------|
| **Input** | T2-weighted MRI volumes (NIfTI, `.nii.gz`) |
| **Output** | Binary tumor segmentation masks (NIfTI, `.nii.gz`) |
| **Labels** | 0 = background, 1 = tumor |
| **Framework** | nnU-Net v2 (install via `pip install nnunetv2`) |

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

```bash
python segmentation/prepare_dataset.py \
    --raw_dir data/raw \
    --nnunet_dir data/nnunet \
    --dataset_id 100 \
    --dataset_name RectalCancer \
    --train_ratio 0.8
```

This creates the nnU-Net-compatible structure:

```
data/nnunet/Dataset100_RectalCancer/
├── imagesTr/        # Training images  (crc_001_0000.nii.gz)
├── labelsTr/        # Training labels  (crc_001.nii.gz)
├── imagesTs/        # Test images      (crc_266_0000.nii.gz)
├── labelsTs/        # Test labels      (crc_266.nii.gz, optional)
└── dataset.json     # Dataset metadata
```

> **Note:** nnU-Net v2 requires image filenames to follow the pattern `<case>_<modality>.nii.gz` (e.g., `crc_001_0000.nii.gz` for the first modality).

### Step 3: Plan and Preprocess

```bash
export nnUNet_raw=data/nnunet
export nnUNet_preprocessed=data/nnunet_preprocessed
export nnUNet_results=data/nnunet_results

bash segmentation/scripts/plan_and_preprocess.sh 100
```

nnU-Net will automatically:
- Analyze dataset properties (spacing, intensity, size)
- Determine optimal network architecture
- Create preprocessing plans for each configuration

### Step 4: Train

```bash
# Train all 5 folds (recommended for cross-validation)
bash segmentation/scripts/train.sh 100 3d_fullres

# Or train a single fold
bash segmentation/scripts/train.sh 100 3d_fullres 0
```

### Step 5: Predict

```bash
bash segmentation/scripts/predict.sh 100 3d_fullres
```

Output masks are saved to `data/segmentation_output/`.

## Using Segmentation Output for Classification

After prediction, the generated masks feed directly into the classification module:

| Classification Mode | Preparation Script |
|---------------------|--------------------|
| **2D** | `classification/data_preparation/prepare_2d_data.py` — extracts PNG slices |
| **3D** | `classification/data_preparation/prepare_3d_data.py` — resamples to isotropic spacing |

## Script Reference

### `prepare_dataset.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--raw_dir` | required | Directory with `images/` and `labels/` subfolders |
| `--nnunet_dir` | required | Output directory for nnU-Net format |
| `--dataset_id` | 100 | nnU-Net dataset ID (3-digit number) |
| `--dataset_name` | RectalCancer | Dataset name suffix |
| `--train_ratio` | 0.8 | Train/test split ratio |
| `--seed` | 42 | Random seed for reproducibility |

### `plan_and_preprocess.sh`

| Argument | Default | Description |
|----------|---------|-------------|
| `$1` (dataset_id) | 100 | Dataset ID to preprocess |

### `train.sh`

| Argument | Default | Description |
|----------|---------|-------------|
| `$1` (dataset_id) | 100 | Dataset ID |
| `$2` (config) | 3d_fullres | nnU-Net configuration |
| `$3` (fold) | all | Fold number (0-4) or `all` |

### `predict.sh`

| Argument | Default | Description |
|----------|---------|-------------|
| `$1` (dataset_id) | 100 | Dataset ID |
| `$2` (config) | 3d_fullres | nnU-Net configuration |
| `$3` (fold) | 0 | Fold number for prediction |

## Tips

1. **Configuration selection:** For rectal cancer MRI, `3d_fullres` typically performs best. Use `3d_lowres` for faster experimentation or ensembling.

2. **Ensemble predictions:** Combine multiple configurations for better results:
   ```bash
   nnUNetv2_ensemble \
       -i data/nnunet_results/Dataset100/.../fold_0/predicted \
       -i data/nnunet_results/Dataset100/.../fold_1/predicted \
       -o data/ensemble_output
   ```

3. **Cross-validation:** nnU-Net automatically runs 5-fold cross-validation. Use the aggregated results to select the best configuration before final training.

4. **Environment variables:** Always set the three nnU-Net paths before running any nnU-Net command:
   ```bash
   export nnUNet_raw=data/nnunet
   export nnUNet_preprocessed=data/nnunet_preprocessed
   export nnUNet_results=data/nnunet_results
   ```
