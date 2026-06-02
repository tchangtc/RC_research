# Stage II: T-stage Classification

Binary classification of rectal cancer T-stage (T2 vs T3) from MRI data.

## Models

| Model | Dimension | Backbone | Input |
|-------|-----------|----------|-------|
| EfficientNet-B0 | 2D | EfficientNet-B0 (ImageNet pretrained) | 256×256 PNG slices |
| YNet 2D | 2D | Custom V-Net encoder | 256×256 PNG slices |
| YNet 3D | 3D | Custom 3D V-Net encoder | 128³ NIfTI volumes |

## Quick Start

### 2D Classification

```bash
# 1. Prepare 2D data from NIfTI volumes (if not already done)
python data_preparation/prepare_2d_data.py \
    --input_dir data/3d/imagesTr \
    --mask_dir data/3d/labelsTr \
    --output_img_dir data/2d/images_train \
    --output_mask_dir data/2d/masks_train \
    --output_csv data/2d/train.csv \
    --labels_csv data/labels.csv

# 2. Train
python train_2d.py --config ../configs/cls_2d_efficientnet.yaml --gpu 0

# 3. Evaluate
python evaluate.py \
    --config ../configs/cls_2d_efficientnet.yaml \
    --checkpoint outputs/cls_2d/checkpoint/099_model.pth \
    --gpu 0
```

### 3D Classification

```bash
# 1. Prepare 3D data (resample to isotropic spacing)
python data_preparation/prepare_3d_data.py \
    --input_dir data/raw/images \
    --mask_dir data/raw/labels \
    --output_dir data/3d \
    --target_spacing 0.36 0.36 0.36 \
    --labels_csv data/labels.csv

# 2. Train
python train_3d.py --config ../configs/cls_3d_ynet.yaml --gpu 0

# 3. Evaluate
python evaluate.py \
    --config ../configs/cls_3d_ynet.yaml \
    --checkpoint outputs/cls_3d/checkpoint/297_model.pth \
    --gpu 0
```

## Project Structure

```
classification/
├── src/
│   ├── models/         # Model definitions (EfficientNet, YNet 2D/3D)
│   ├── data/           # Datasets and augmentations
│   ├── losses/         # Loss functions (BCE + SmoothL1 + Dice)
│   ├── metrics/        # Evaluation metrics
│   ├── utils/          # Utilities (seeding, Lookahead, etc.)
│   └── engine/         # Training and evaluation loops
├── data_preparation/   # Data conversion scripts
├── train_2d.py         # 2D training entry point
├── train_3d.py         # 3D training entry point
└── evaluate.py         # Evaluation script
```

## Configuration

All hyperparameters are controlled via YAML files in `../configs/`:

- `cls_2d_efficientnet.yaml` - 2D EfficientNet training
- `cls_3d_ynet.yaml` - 3D YNet training

Key configurable parameters:
- **Data paths**: All input/output directories
- **Model**: Architecture selection and parameters
- **Training**: Epochs, batch size, learning rate, scheduler
- **Loss**: Weights for BCE, SmoothL1, Dice components
- **Preprocessing** (3D only): Resampling spacing, crop sizes, margins

## Rectal Filling Analysis

To evaluate the impact of rectal filling on model performance, prepare
separate CSV files for well-filled and poorly-filled cases:

```
data/2d/
├── train.csv
├── test.csv
├── test_filling_well.csv       # Cases with adequate rectal filling
└── test_filling_not_well.csv   # Cases with inadequate rectal filling
```

Then evaluate on each subset by modifying the config or using the evaluate
script with appropriate dataset selection.
