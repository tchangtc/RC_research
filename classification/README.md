# Stage II: T-stage Classification

Binary classification of rectal cancer T-stage (**T2** vs **T3**) from MRI data.

## Models

| Model | Dimension | Backbone | Input Size | Input Channels |
|-------|-----------|----------|------------|----------------|
| **EfficientNet-B0** | 2D | EfficientNet-B0 (ImageNet) | 256×256 | 3 (image + mask + zeros) |
| **YNet 2D** | 2D | Custom V-Net encoder | 256×256 | 1 |
| **YNet 3D** | 3D | Custom 3D V-Net encoder | 128³ | 2 (image + mask) |

## Quick Start

### 2D Classification

```bash
# 1. Prepare 2D data from NIfTI volumes
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

## Source Code Architecture

```
classification/src/
├── models/
│   ├── __init__.py            # build_model() factory function
│   ├── efficientnet_2d.py     # EffNet: RGB norm + EfficientNet encoder + FC head
│   ├── ynet_2d.py             # YNet2D, Classification2D: 2D V-Net encoder
│   ├── ynet_3d.py             # YNet3D, Classification3D, YNetCls3D: 3D V-Net
│   └── layers.py              # Shared building blocks (LUConv, DownTransition, etc.)
├── data/
│   ├── __init__.py            # build_dataset_2d/3d() factory functions
│   ├── dataset_2d.py          # CRCDataset2D: PNG slices with bbox crop
│   ├── dataset_3d.py          # CRCDataset3D: NIfTI volumes with resampling
│   ├── augmentation_2d.py     # 2D augmentations (flip, affine, elastic, cutout, etc.)
│   └── augmentation_3d.py     # 3D nnU-Net style augmentations
├── engine/
│   ├── __init__.py
│   ├── trainer.py             # Unified Trainer class (2D & 3D)
│   └── evaluator.py           # evaluate() function (2D & 3D)
├── losses/
│   ├── __init__.py
│   └── losses.py              # DiceLoss, DiceBCELoss
├── metrics/
│   ├── __init__.py
│   └── metrics.py             # precision, recall, specificity, F-scores, etc.
└── utils/
    ├── __init__.py
    └── utils.py               # seeding, Lookahead, time_to_str, load_config
```

## Configuration

All hyperparameters are controlled via YAML files in `../configs/`.

### 2D Config (`cls_2d_efficientnet.yaml`)

| Section | Key Parameters |
|---------|---------------|
| `model` | `name: efficientnet_b0`, `image_size: 256`, `pretrained: true` |
| `training` | `epochs: 600`, `batch_size: 32`, `seed: 982742` |
| `loss` | `bce_weight: 0.5`, `smooth_l1_weight: 0.5`, `dice_weight: 0.0` |
| `optimizer` | `lr: 1e-5`, `betas: [0.9, 0.99]`, `weight_decay: 0.01` |
| `scheduler` | `step_size: 48`, `gamma: 0.5` |

### 3D Config (`cls_3d_ynet.yaml`)

| Section | Key Parameters |
|---------|---------------|
| `model` | `name: ynet_3d`, `image_size: 128`, `input_channels: 2` |
| `preprocessing` | `target_spacing: [0.36, 0.36, 0.36]`, `padding: [456³]`, `final: [128³]` |
| `training` | `epochs: 300`, `batch_size: 8`, `seed: 42` |
| `loss` | `bce_weight: 0.5`, `smooth_l1_weight: 0.5`, `dice_weight: 0.0` |
| `optimizer` | `lr: 1e-5`, `betas: [0.9, 0.999]`, `weight_decay: 1e-4` |
| `scheduler` | `step_size: 25`, `gamma: 0.5` |

## Data Pipeline

### 2D Pipeline

```
PNG image + mask
  → mask_to_bbox (contour extraction)
  → crop with 50px margin
  → resize to 256×256
  → normalize to [0, 1]
  → augmentation (flip, affine, elastic, cutout, contrast/noise)
  → concat [image, mask, zeros] → 3 channels
```

### 3D Pipeline

```
NIfTI volume + label
  → read with SimpleITK
  → Z-score normalization: (x - mean) / (std + 1e-8)
  → augmentation (Gaussian noise/blur, brightness, gamma, rotation, mirror)
  → pad/crop to 456³
  → center crop around tumor bbox (margin: z±10, y±25, x±25)
  → pad/crop to 128³
  → concat [image, label] → 2 channels
```

### 3D Augmentation Pipeline (nnU-Net style)

| Transform | Parameters | Probability |
|-----------|-----------|-------------|
| GaussianNoise | variance: 0~0.07 | 25% |
| GaussianBlur | sigma: 1~2.5 | 25% |
| Brightness | mu=0, sigma=1 | 25% |
| Gamma | range: 0.25~1 | 25% |
| RandomRotFlip | 90° rotation + flip | 25% |
| Mirror | along all axes | 25% |

## Rectal Filling Analysis

To evaluate the impact of rectal filling on model performance, prepare separate CSV files:

```
data/2d/
├── train.csv
├── test.csv
├── test_filling_well.csv       # Cases with adequate rectal filling
└── test_filling_not_well.csv   # Cases with inadequate rectal filling
```

Then run evaluation on each subset and compare metrics.

## Reference Notebooks

Located in `notebooks/`:

| Notebook | Description |
|----------|-------------|
| `3d_augmentation_visualization.ipynb` | Interactive visualization of all 3D augmentation transforms with mid-slice views |
| `3d_resampling.ipynb` | B-spline resampling to isotropic 0.36mm spacing with visual comparison |
