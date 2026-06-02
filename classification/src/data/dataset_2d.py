"""2D Dataset for rectal cancer T-stage classification from MRI slices."""

import cv2
import numpy as np
import pandas as pd
import torch
from skimage.measure import find_contours, label, regionprops
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from .augmentation_2d import train_augment_v00


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def mask_to_border(mask):
    """Extract contour border from binary mask."""
    h, w = mask.shape
    border = np.zeros((h, w))
    contours = find_contours(mask, 128)
    for contour in contours:
        for c in contour:
            x, y = int(c[0]), int(c[1])
            border[x][y] = 255
    return border


def mask_to_bbox(mask):
    """Extract bounding box from binary mask using contours."""
    border = mask_to_border(mask)
    lbl = label(border)
    props = regionprops(lbl)
    bboxes = []
    for prop in props:
        y1, x1, y2, x2 = prop.bbox
        bboxes.append([x1, y1, x2, y2])
    return bboxes


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CRCDataset2D(Dataset):
    """2D CRC Dataset for T-stage classification.

    Reads PNG image + mask pairs, crops around tumor bbox, and normalizes.

    Args:
        df: DataFrame with columns [img_dir, img_name, img_path, mask_path, T_Stage]
        mode: 'train' or 'test'
        transforms: augmentation function (image, mask) -> (image, mask)
        image_dir: directory containing images
        mask_dir: directory containing masks
        image_size: target resize dimension
    """

    def __init__(self, df, mode="train", transforms=None,
                 image_dir="", mask_dir="", image_size=256):
        df = df.copy()
        df.loc[:, "i"] = np.arange(len(df))

        self.df = df
        self.length = len(df)
        self.mode = mode
        self.transforms = transforms
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size

    def __len__(self):
        return self.length

    def __str__(self):
        num_patient = len(set(self.df.img_dir))
        num_image = len(self.df)
        string = f"\tlen = {self.length}\n"
        string += f"\tnum_patient = {num_patient}\n"
        string += f"\tnum_image = {num_image}\n"

        count = dict(self.df.T_Stage.value_counts())
        for k in [0, 1]:
            c = count.get(k, 0)
            string += f"\t\tT_Stage{k} = {c:5d} ({c/len(self.df):0.3f})\n"
        return string

    def _read_data(self, dd):
        """Read image and mask from disk."""
        image = cv2.imread(
            f"{self.image_dir}/{dd.img_dir}/{dd.img_name}", cv2.IMREAD_UNCHANGED
        )
        mask = cv2.imread(
            f"{self.mask_dir}/{dd.img_dir}/{dd.img_name}", cv2.IMREAD_GRAYSCALE
        )
        return image, mask

    def __getitem__(self, index):
        dd = self.df.iloc[index]

        # Read image and mask
        image, mask = self._read_data(dd)

        # Extract bbox from mask and crop with margin
        bboxes = mask_to_bbox(mask)
        x1, y1, x2, y2 = bboxes[0]
        margin = 50

        crop_img = image[max(0, y1 - margin): y2 + margin, max(0, x1 - margin): x2 + margin]
        crop_mask = mask[max(0, y1 - margin): y2 + margin, max(0, x1 - margin): x2 + margin]

        # Resize
        image = cv2.resize(crop_img, dsize=(self.image_size, self.image_size),
                          interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(crop_mask, dsize=(self.image_size, self.image_size),
                         interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255
        mask = mask / 255

        # Apply augmentations
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        # Convert to tensors: [image, mask, zeros] -> 3 channels
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)
        zeros_tensor = torch.zeros(1, self.image_size, self.image_size)

        combined = torch.cat([image_tensor, mask_tensor, zeros_tensor], dim=0)

        label = torch.FloatTensor([dd.T_Stage])

        return {
            "index": index,
            "patient_id": dd.img_dir,
            "image": combined,
            "label": label,
        }


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

TENSOR_KEYS = ["image", "label"]


def null_collate_2d(batch):
    """Custom collate function that stacks tensor keys."""
    d = {}
    keys = batch[0].keys()
    for k in keys:
        v = [b[k] for b in batch]
        if k in TENSOR_KEYS:
            v = torch.stack(v, dim=0)
        d[k] = v

    d["label"] = d["label"].reshape(-1)
    return d
