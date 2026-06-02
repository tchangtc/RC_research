# RC Research

Deep Learning Models for Preoperative T-stage Assessment in Rectal Cancer Using MRI:
Exploring the Impact of Rectal Filling.

## Overview

This repository implements a two-stage deep learning pipeline for automated
preoperative T-stage assessment of rectal cancer from T2-weighted MRI:

- **Stage I — Segmentation:** Automatic rectal tumor segmentation using
  [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet)
- **Stage II — Classification:** Binary T-stage classification (T2 vs T3)
  using EfficientNet (2D) and custom YNet architectures (2D/3D)

The key research contribution is exploring how **rectal filling** (adequate vs
inadequate) impacts model performance for T-stage assessment.

## Project Structure

```
RC_research/
├── configs/               # YAML configuration files
├── segmentation/          # Stage I: nnU-Net tumor segmentation
│   ├── README.md          # nnU-Net usage guide
│   ├── prepare_dataset.py # Data format conversion
│   └── scripts/           # Training & inference shell scripts
├── classification/        # Stage II: T-stage classification
│   ├── README.md          # Classification module guide
│   ├── src/               # Source code (models, data, engine)
│   ├── data_preparation/  # Data conversion scripts
│   ├── train_2d.py        # 2D training
│   ├── train_3d.py        # 3D training
│   └── evaluate.py        # Model evaluation
├── data/                  # Data directory (gitignored)
├── configs/               # YAML configs for all stages
├── requirements.txt       # Python dependencies
└── pyproject.toml         # Project metadata
```

## Setup

```bash
# Clone repository
git clone https://github.com/tchangtc/RC_research.git
cd RC_research

# Install dependencies
pip install -r requirements.txt

# For segmentation (Stage I), also install nnU-Net v2
pip install nnunetv2
```

## Pipeline

### Stage I: Segmentation

1. Prepare your T2-weighted MRI data in NIfTI format with tumor annotations
2. Convert to nnU-Net format: `python segmentation/prepare_dataset.py ...`
3. Train: `bash segmentation/scripts/train.sh 100 3d_fullres`
4. Predict: `bash segmentation/scripts/predict.sh 100 3d_fullres`

See [segmentation/README.md](segmentation/README.md) for details.

### Stage II: Classification

1. Prepare 2D or 3D data from segmentation output
2. Train classifier: `python classification/train_2d.py --config configs/cls_2d_efficientnet.yaml`
3. Evaluate: `python classification/evaluate.py --config ... --checkpoint ...`

See [classification/README.md](classification/README.md) for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{tian2023deep,
  title={Deep learning models for preoperative T-stage assessment in rectal cancer
         using MRI: exploring the impact of rectal filling},
  author={Tian, ... and Ma, ...},
  journal={Frontiers in Medicine},
  volume={10},
  year={2023},
  doi={10.3389/fmed.2023.1326324}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.
