# RC Research

**Deep Learning Models for Preoperative T-stage Assessment in Rectal Cancer Using MRI: Exploring the Impact of Rectal Filling**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/DOI-10.3389/fmed.2023.1326324-blue)](https://doi.org/10.3389/fmed.2023.1326324)

## Overview

This repository implements a **two-stage deep learning pipeline** for automated preoperative T-stage assessment of rectal cancer from T2-weighted MRI:

| Stage | Task | Method |
|-------|------|--------|
| **Stage I** | Tumor Segmentation | [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) |
| **Stage II** | T-stage Classification (T2 vs T3) | EfficientNet-B0 (2D) / YNet (3D) |

The key research contribution explores how **rectal filling** (adequate vs inadequate) impacts deep learning model performance for T-stage assessment. Our findings demonstrate that models trained on well-filled cases achieve significantly better segmentation (DSC, p=0.006; ASD, p=0.017) and classification performance.

## Project Structure

```
RC_research/
├── configs/                          # YAML configuration files
│   ├── cls_2d_efficientnet.yaml      #   2D EfficientNet training config
│   ├── cls_3d_ynet.yaml              #   3D YNet training config
│   └── seg_nnunet.yaml               #   Segmentation config reference
│
├── segmentation/                     # Stage I: Tumor Segmentation
│   ├── README.md                     #   nnU-Net v2 usage guide
│   ├── prepare_dataset.py            #   Raw data → nnU-Net format
│   └── scripts/                      #   Plan, train, predict shell scripts
│
├── classification/                   # Stage II: T-stage Classification
│   ├── README.md                     #   Classification module guide
│   ├── train_2d.py                   #   2D training entry point
│   ├── train_3d.py                   #   3D training entry point
│   ├── evaluate.py                   #   Evaluation script (2D & 3D)
│   ├── notebooks/                    #   Reference notebooks
│   ├── data_preparation/             #   Data conversion scripts
│   └── src/                          #   Modular source code
│       ├── models/                   #     EffNet, YNet 2D/3D, shared layers
│       ├── data/                     #     Datasets & augmentations
│       ├── engine/                   #     Unified Trainer & Evaluator
│       ├── losses/                   #     BCE + SmoothL1 + Dice
│       ├── metrics/                  #     AUC, Acc, Prec, Recall, Spec
│       └── utils/                    #     Seeding, Lookahead, config loader
│
├── data/                             # Data directory (gitignored)
├── pyproject.toml                    # Python project metadata
└── requirements.txt                  # Dependencies
```

## Prerequisites

- Python ≥ 3.8
- CUDA-capable GPU (recommended)

```bash
# Clone repository
git clone https://github.com/tchangtc/RC_research.git
cd RC_research

# Install dependencies
pip install -r requirements.txt

# For segmentation (Stage I), also install nnU-Net v2
pip install nnunetv2
```

## Quick Start

### Stage I — Segmentation

```bash
# 1. Prepare nnU-Net dataset
python segmentation/prepare_dataset.py \
    --raw_dir data/raw \
    --nnunet_dir data/nnunet \
    --dataset_id 100 \
    --dataset_name RectalCancer

# 2. Plan & preprocess
export nnUNet_raw=data/nnunet
export nnUNet_preprocessed=data/nnunet_preprocessed
export nnUNet_results=data/nnunet_results
bash segmentation/scripts/plan_and_preprocess.sh 100

# 3. Train
bash segmentation/scripts/train.sh 100 3d_fullres

# 4. Predict (generates tumor masks for Stage II)
bash segmentation/scripts/predict.sh 100 3d_fullres
```

See [segmentation/README.md](segmentation/README.md) for details.

### Stage II — Classification

#### 2D Classification (EfficientNet-B0)

```bash
# 1. Extract 2D slices from NIfTI volumes
python classification/data_preparation/prepare_2d_data.py \
    --input_dir data/3d/imagesTr \
    --mask_dir data/3d/labelsTr \
    --output_img_dir data/2d/images_train \
    --output_mask_dir data/2d/masks_train \
    --output_csv data/2d/train.csv \
    --labels_csv data/labels.csv

# 2. Train
python classification/train_2d.py \
    --config configs/cls_2d_efficientnet.yaml --gpu 0

# 3. Evaluate
python classification/evaluate.py \
    --config configs/cls_2d_efficientnet.yaml \
    --checkpoint outputs/cls_2d/checkpoint/099_model.pth \
    --gpu 0
```

#### 3D Classification (YNet)

```bash
# 1. Prepare 3D data (resample to isotropic 0.36mm)
python classification/data_preparation/prepare_3d_data.py \
    --input_dir data/raw/images \
    --mask_dir data/raw/labels \
    --output_dir data/3d \
    --target_spacing 0.36 0.36 0.36 \
    --labels_csv data/labels.csv

# 2. Train
python classification/train_3d.py \
    --config configs/cls_3d_ynet.yaml --gpu 0

# 3. Evaluate
python classification/evaluate.py \
    --config configs/cls_3d_ynet.yaml \
    --checkpoint outputs/cls_3d/checkpoint/297_model.pth \
    --gpu 0
```

See [classification/README.md](classification/README.md) for details.

## Model Architectures

### EfficientNet-B0 (2D)

| Component | Details |
|-----------|---------|
| Backbone | EfficientNet-B0 (ImageNet pretrained) |
| Input | 256×256 PNG slices, 3 channels (image + mask + zeros) |
| Normalization | ImageNet mean/std |
| Head | AdaptiveAvgPool2d → Linear(1280, 1) |
| Loss | 0.5 × BCE + 0.5 × SmoothL1 |
| Scheduler | StepLR (step=48, γ=0.5), 600 epochs |

### YNet 3D

| Component | Details |
|-----------|---------|
| Backbone | Custom 3D V-Net encoder (1→64→128→256→512) |
| Input | 128³ NIfTI volumes, 2 channels (image + mask) |
| Normalization | Z-score per volume |
| Head | Conv3d→BN→PReLU→AdaptiveAvgPool3d→Dropout(0.4)→FC(1024,1) |
| Loss | 0.5 × BCE + 0.5 × SmoothL1 |
| Scheduler | StepLR (step=25, γ=0.5), 300 epochs |

### Preprocessing Pipeline

```
NIfTI Volume → Resample (B-spline, 0.36mm isotropic)
             → Pad/Crop to 456³
             → Center crop around tumor bbox (margin: z±10, y±25, x±25)
             → Pad/Crop to 128³
             → Z-score normalization
```

## Rectal Filling Analysis

A core contribution of this work is the analysis of how rectal filling quality affects model performance. Cases are stratified into:

| Group | Abbreviation | Description |
|-------|-------------|-------------|
| Filling Well | `fw` | Adequate rectal distension |
| Filling Not Well | `fnw` | Inadequate rectal distension |

To reproduce the filling analysis, prepare separate CSV files:

```
data/2d/
├── train.csv
├── test.csv
├── test_filling_well.csv
└── test_filling_not_well.csv
```

Then evaluate on each subset and compare AUC, accuracy, and other metrics.

## Configuration

All hyperparameters are managed through YAML files in `configs/`. Key sections:

| Section | Parameters |
|---------|-----------|
| `data` | All input/output directory paths |
| `model` | Architecture selection and parameters |
| `preprocessing` | Resampling, crop sizes, margins (3D only) |
| `training` | Epochs, batch size, learning rate, workers, seed |
| `loss` | Weights for BCE, SmoothL1, Dice components |
| `optimizer` | Adam parameters (lr, betas, eps, weight_decay) |
| `scheduler` | StepLR parameters (step_size, gamma) |
| `output` | Save directory and checkpoint frequency |

## Reference Notebooks

Two interactive notebooks are provided in `classification/notebooks/`:

| Notebook | Purpose |
|----------|---------|
| `3d_augmentation_visualization.ipynb` | Visualize all 3D augmentation transforms (noise, blur, brightness, gamma, rotation, mirror) with individual and full pipeline examples |
| `3d_resampling.ipynb` | B-spline resampling to isotropic spacing (0.36mm) with single-case exploration and batch processing |

## Citation

If you use this code in your research, please cite:

```bibtex
@article{tian2023deep,
  title     = {Deep learning models for preoperative T-stage assessment in
               rectal cancer using MRI: exploring the impact of rectal filling},
  author    = {Tian, Chang et al.},
  journal   = {Frontiers in Medicine},
  volume    = {10},
  pages     = {1326324},
  year      = {2023},
  publisher = {Frontiers},
  doi       = {10.3389/fmed.2023.1326324}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
