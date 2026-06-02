"""Dataset definitions for 2D and 3D classification."""

from .dataset_2d import CRCDataset2D, null_collate_2d
from .dataset_3d import CRCDataset3D, null_collate_3d


def build_dataset_2d(config, mode="train"):
    """Build 2D dataset from config."""
    import pandas as pd
    from .dataset_2d import train_augment_v00

    data_cfg = config["data"]
    model_cfg = config["model"]

    if mode == "train":
        df = pd.read_csv(data_cfg["train_csv"])
        return CRCDataset2D(
            df, mode="train", transforms=train_augment_v00,
            image_dir=data_cfg["image_train_dir"],
            mask_dir=data_cfg["mask_train_dir"],
            image_size=model_cfg["image_size"],
        )
    elif mode == "test":
        df = pd.read_csv(data_cfg["test_csv"])
        return CRCDataset2D(
            df, mode="test", transforms=None,
            image_dir=data_cfg["image_test_dir"],
            mask_dir=data_cfg["mask_test_dir"],
            image_size=model_cfg["image_size"],
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


def build_dataset_3d(config, mode="train"):
    """Build 3D dataset from config."""
    import pandas as pd

    data_cfg = config["data"]
    model_cfg = config["model"]
    prep_cfg = config.get("preprocessing", {})

    if mode == "train":
        df = pd.read_csv(data_cfg["train_csv"])
        return CRCDataset3D(
            df, mode="train",
            image_dir=data_cfg["image_train_dir"],
            mask_dir=data_cfg["mask_train_dir"],
            image_size=model_cfg["image_size"],
            target_spacing=tuple(prep_cfg.get("target_spacing", (0.36, 0.36, 0.36))),
            padding_size=prep_cfg.get("padding_size", [456, 456, 456]),
            cropping_size=prep_cfg.get("cropping_size", [456, 456, 456]),
            center_crop_distance=prep_cfg.get("center_crop_distance", [10, 25, 25]),
            final_size=prep_cfg.get("final_size", [128, 128, 128]),
        )
    elif mode == "test":
        df = pd.read_csv(data_cfg["test_csv"])
        return CRCDataset3D(
            df, mode="test",
            image_dir=data_cfg["image_test_dir"],
            mask_dir=data_cfg["mask_test_dir"],
            image_size=model_cfg["image_size"],
            target_spacing=tuple(prep_cfg.get("target_spacing", (0.36, 0.36, 0.36))),
            padding_size=prep_cfg.get("padding_size", [456, 456, 456]),
            cropping_size=prep_cfg.get("cropping_size", [456, 456, 456]),
            center_crop_distance=prep_cfg.get("center_crop_distance", [10, 25, 25]),
            final_size=prep_cfg.get("final_size", [128, 128, 128]),
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
