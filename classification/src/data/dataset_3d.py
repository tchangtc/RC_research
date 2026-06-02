"""3D Dataset for rectal cancer T-stage classification from MRI volumes."""

import copy
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Default preprocessing parameters
# ---------------------------------------------------------------------------

DEFAULT_TARGET_SPACING = (0.36, 0.36, 0.36)
DEFAULT_PADDING_SIZE = [456, 456, 456]
DEFAULT_CROPPING_SIZE = [456, 456, 456]
DEFAULT_DISTANCE = [10, 25, 25]
DEFAULT_FINAL_SIZE = [128, 128, 128]


class CRCDataset3D(Dataset):
    """3D CRC Dataset for T-stage classification.

    Reads NIfTI volumes + labels, resamples, normalizes, and crops around tumor.

    Args:
        df: DataFrame with columns [img_name, img_path, mask_path, T_Stage]
        mode: 'train' or 'test'
        image_dir: directory containing NIfTI images
        mask_dir: directory containing NIfTI labels
        image_size: not used directly (kept for API compatibility)
        transforms: torchvision-style transform Compose (operates on sample dict)
        target_spacing: isotropic resampling target (mm)
        padding_size: pad to this size before center crop
        cropping_size: crop to this size before center crop
        center_crop_distance: margin around tumor bbox [z, y, x]
        final_size: final volume size after all preprocessing
    """

    def __init__(
        self,
        df,
        mode="train",
        image_dir="",
        mask_dir="",
        image_size=128,
        transforms=None,
        target_spacing=DEFAULT_TARGET_SPACING,
        padding_size=None,
        cropping_size=None,
        center_crop_distance=None,
        final_size=None,
    ):
        df = df.copy()
        df.loc[:, "i"] = np.arange(len(df))

        self.df = df
        self.length = len(df)
        self.mode = mode
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.transforms = transforms
        self.target_spacing = target_spacing
        self.padding_size = padding_size or DEFAULT_PADDING_SIZE
        self.cropping_size = cropping_size or DEFAULT_CROPPING_SIZE
        self.distance = center_crop_distance or DEFAULT_DISTANCE
        self.final_size = final_size or DEFAULT_FINAL_SIZE

    def __len__(self):
        return self.length

    def __str__(self):
        num_patient = len(set(self.df.img_name))
        string = f"\tlen = {self.length}\n"
        string += f"\tnum_patient = {num_patient}\n"

        count = dict(self.df.T_Stage.value_counts())
        for k in [0, 1]:
            c = count.get(k, 0)
            string += f"\t\tT_Stage{k} = {c:5d} ({c/len(self.df):0.3f})\n"
        return string

    # -------------------------------------------------------------------
    # I/O
    # -------------------------------------------------------------------

    def _read_data(self, dd):
        """Read NIfTI image and label."""
        image_path = f"{self.image_dir}/{dd.img_name}.nii.gz"
        label_path = f"{self.mask_dir}/{dd.img_name}.nii.gz"

        image = sitk.ReadImage(image_path)
        label = sitk.ReadImage(label_path)

        assert image.GetSize() == label.GetSize()

        # Standardize direction
        identity_dir = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        image.SetDirection(identity_dir)
        label.SetDirection(identity_dir)

        return image, label

    # -------------------------------------------------------------------
    # Resampling
    # -------------------------------------------------------------------

    @staticmethod
    def _resample_xyz_axis(im_image, space=(1., 1., 1.), interp=sitk.sitkLinear):
        """Resample image along XYZ axes to target spacing."""
        identity = sitk.Transform(3, sitk.sitkIdentity)
        sp = im_image.GetSpacing()
        sz = im_image.GetSize()

        sz2 = (
            int(round(sz[0] * sp[0] / space[0])),
            int(round(sz[1] * sp[1] / space[1])),
            int(round(sz[2] * sp[2] / space[2])),
        )

        ref = sitk.Image(sz2, im_image.GetPixelIDValue())
        ref.SetSpacing(space)
        ref.SetOrigin(im_image.GetOrigin())
        ref.SetDirection(im_image.GetDirection())

        return sitk.Resample(im_image, ref, identity, interp)

    @staticmethod
    def _resample_label_to_ref(im_label, im_ref, interp=sitk.sitkNearestNeighbor):
        """Resample label to match reference image (handles multi-label correctly)."""
        identity = sitk.Transform(3, sitk.sitkIdentity)

        ref = sitk.Image(im_ref.GetSize(), im_label.GetPixelIDValue())
        ref.SetSpacing(im_ref.GetSpacing())
        ref.SetOrigin(im_ref.GetOrigin())
        ref.SetDirection(im_ref.GetDirection())

        np_label = sitk.GetArrayFromImage(im_label)
        labels = np.unique(np_label)
        resampled_list = []

        for lbl_val in labels:
            tmp = (np_label == lbl_val).astype(np.uint8)
            tmp_img = sitk.GetImageFromArray(tmp)
            tmp_img.CopyInformation(im_label)
            tmp_resampled = sitk.Resample(tmp_img, ref, identity, interp)
            resampled_list.append(sitk.GetArrayFromImage(tmp_resampled))

        one_hot = np.stack(resampled_list, axis=0)
        result = np.argmax(one_hot, axis=0).astype(np.uint8)
        out = sitk.GetImageFromArray(result)
        out.CopyInformation(im_ref)
        return out

    # -------------------------------------------------------------------
    # Padding / Cropping
    # -------------------------------------------------------------------

    def _pad_or_crop(self, image, label, target_size):
        """Pad or center-crop image and label to target_size."""
        assert image.shape == label.shape
        z, y, x = image.shape
        img, lab = image, label

        # Pad if smaller
        if z < target_size[0]:
            diff = (target_size[0] - z) // 2
            if z % 2 == 0:
                img = np.pad(img, ((diff, diff), (0, 0), (0, 0)))
                lab = np.pad(lab, ((diff, diff), (0, 0), (0, 0)))
            else:
                img = np.pad(img, ((diff, diff + 1), (0, 0), (0, 0)))
                lab = np.pad(lab, ((diff, diff + 1), (0, 0), (0, 0)))

        if y < target_size[1]:
            diff = (target_size[1] - y) // 2
            if y % 2 == 0:
                img = np.pad(img, ((0, 0), (diff, diff), (0, 0)))
                lab = np.pad(lab, ((0, 0), (diff, diff), (0, 0)))
            else:
                img = np.pad(img, ((0, 0), (diff, diff + 1), (0, 0)))
                lab = np.pad(lab, ((0, 0), (diff, diff + 1), (0, 0)))

        if x < target_size[2]:
            diff = (target_size[2] - x) // 2
            if x % 2 == 0:
                img = np.pad(img, ((0, 0), (0, 0), (diff, diff)))
                lab = np.pad(lab, ((0, 0), (0, 0), (diff, diff)))
            else:
                img = np.pad(img, ((0, 0), (0, 0), (diff, diff + 1)))
                lab = np.pad(lab, ((0, 0), (0, 0), (diff, diff + 1)))

        # Crop if larger (center crop)
        z, y, x = img.shape
        if z > target_size[0]:
            s = target_size[0]
            img = img[z // 2 - s // 2: z // 2 + s // 2, :, :]
            lab = lab[z // 2 - s // 2: z // 2 + s // 2, :, :]
        if y > target_size[1]:
            s = target_size[1]
            img = img[:, y // 2 - s // 2: y // 2 + s // 2, :]
            lab = lab[:, y // 2 - s // 2: y // 2 + s // 2, :]
        if x > target_size[2]:
            s = target_size[2]
            img = img[:, :, x // 2 - s // 2: x // 2 + s // 2]
            lab = lab[:, :, x // 2 - s // 2: x // 2 + s // 2]

        assert img.shape == lab.shape
        return img, lab

    def _get_bbox_from_mask(self, mask, outside_value=0):
        """Get bounding box coordinates from mask."""
        coords = np.where(mask != outside_value)
        return [
            [int(np.min(coords[0])), int(np.max(coords[0])) + 1],
            [int(np.min(coords[1])), int(np.max(coords[1])) + 1],
            [int(np.min(coords[2])), int(np.max(coords[2])) + 1],
        ]

    def _center_crop_around_tumor(self, image, label):
        """Crop around tumor bbox with configurable margin."""
        assert image.shape == label.shape
        bbox = self._get_bbox_from_mask(label, 0)
        z, x, y = bbox

        image = image[
            max(0, z[0] - self.distance[0]): z[1] + self.distance[0],
            max(0, x[0] - self.distance[1]): x[1] + self.distance[1],
            max(0, y[0] - self.distance[2]): y[1] + self.distance[2],
        ]
        label = label[
            max(0, z[0] - self.distance[0]): z[1] + self.distance[0],
            max(0, x[0] - self.distance[1]): x[1] + self.distance[1],
            max(0, y[0] - self.distance[2]): y[1] + self.distance[2],
        ]
        return image, label

    # -------------------------------------------------------------------
    # Main pipeline
    # -------------------------------------------------------------------

    def __getitem__(self, index):
        dd = copy.deepcopy(self.df.iloc[index])

        # Step 1: Read
        image_sitk, label_sitk = self._read_data(dd)

        # Step 2: Convert to numpy
        image = sitk.GetArrayFromImage(image_sitk)
        label = sitk.GetArrayFromImage(label_sitk)

        # Step 3: Z-score normalization
        image = (image - image.mean()) / (image.std() + 1e-8)
        image = image.astype(np.float32)

        # Step 4: Add channel dimension for transforms
        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)

        # Step 5: Apply augmentations (if any)
        sample = {"image": image, "label": label}
        if self.transforms is not None:
            sample = self.transforms(sample)

        image = sample["image"].squeeze(0)
        label = sample["label"].squeeze(0)

        # Step 6: Pad/crop to intermediate size
        image, label = self._pad_or_crop(image, label, self.padding_size)

        # Step 7: Center crop around tumor
        image, label = self._center_crop_around_tumor(image, label)

        # Step 8: Final pad/crop to target size
        image, label = self._pad_or_crop(image, label, self.final_size)

        # Build output
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)
        label_tensor = torch.from_numpy(label).long().unsqueeze(0)

        # 2-channel input: [image, mask]
        combined = torch.cat([image_tensor, label_tensor], dim=0)

        return {
            "index": index,
            "patient_id": dd.img_name,
            "image": combined,
            "T_stage": torch.FloatTensor([dd.T_Stage]),
        }


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

TENSOR_KEYS = ["image", "T_stage"]


def null_collate_3d(batch):
    """Custom collate function for 3D data."""
    d = {}
    keys = batch[0].keys()
    for k in keys:
        v = [b[k] for b in batch]
        if k in TENSOR_KEYS:
            v = torch.stack(v, dim=0)
        d[k] = v

    d["T_stage"] = d["T_stage"].reshape(-1)
    return d
