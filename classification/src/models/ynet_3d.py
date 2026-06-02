"""3D YNet architecture for joint segmentation and classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import (
    DownTransition,
    UpTransition,
    OutputTransition,
    ClassificationHead,
)


class YNet3D(nn.Module):
    """3D encoder-decoder for joint segmentation + classification.

    Architecture: 4-level 3D U-Net encoder/decoder with classification branch.

    Channel progression: 1 -> 64 -> 128 -> 256 -> 512 -> (decode back)
    """

    def __init__(self, n_class=1, act="relu", cls_classes=1):
        super().__init__()

        # Encoder
        self.down_tr64 = DownTransition(1, 0, act, ndim=3)
        self.down_tr128 = DownTransition(64, 1, act, ndim=3)
        self.down_tr256 = DownTransition(128, 2, act, ndim=3)
        self.down_tr512 = DownTransition(256, 3, act, ndim=3)

        # Decoder
        self.up_tr256 = UpTransition(512, 512, 2, act, ndim=3)
        self.up_tr128 = UpTransition(256, 256, 1, act, ndim=3)
        self.up_tr64 = UpTransition(128, 128, 0, act, ndim=3)

        # Heads
        self.out_tr = OutputTransition(64, n_class, ndim=3)
        self.clc_out = ClassificationHead(512, act, cls_classes, ndim=3)

    def forward(self, x):
        # Encoder
        out64, skip64 = self.down_tr64(x)
        out128, skip128 = self.down_tr128(out64)
        out256, skip256 = self.down_tr256(out128)
        out512, _ = self.down_tr512(out256)

        # Decoder
        up256 = self.up_tr256(out512, skip256)
        up128 = self.up_tr128(up256, skip128)
        up64 = self.up_tr64(up128, skip64)

        # Outputs
        seg_out = self.out_tr(up64)
        cls_out = self.clc_out(out512)

        return seg_out, cls_out


class Classification3D(nn.Module):
    """3D encoder-only backbone (no decoder/segmentation).

    Uses only the encoder path + classification head.
    Input: tensor [B, C, D, H, W].
    Output: raw logit tensor.
    """

    def __init__(self, act="relu", cls_classes=1, input_channels=2):
        super().__init__()

        self.down_tr64 = DownTransition(input_channels, 0, act, ndim=3)
        self.down_tr128 = DownTransition(64, 1, act, ndim=3)
        self.down_tr256 = DownTransition(128, 2, act, ndim=3)
        self.down_tr512 = DownTransition(256, 3, act, ndim=3)
        self.clc_out = ClassificationHead(512, act, cls_classes, ndim=3)

    def forward(self, x):
        out64, _ = self.down_tr64(x)
        out128, _ = self.down_tr128(out64)
        out256, _ = self.down_tr256(out128)
        out512, _ = self.down_tr512(out256)
        return self.clc_out(out512)


class YNetCls3D(nn.Module):
    """3D classifier wrapper with loss computation.

    Wraps Classification3D backbone and handles:
    - dict input (batch['image']) / dict output (losses + predictions)
    - BCE + SmoothL1 + Dice loss computation
    - Sigmoid inference output

    This matches the original YNetCls from model_3D.py.
    """

    def __init__(self, act="relu", cls_classes=1, input_channels=2):
        super().__init__()
        self.output_type = ["inference", "loss"]
        self.backbone = Classification3D(act=act, cls_classes=cls_classes,
                                         input_channels=input_channels)

    def load_pretrain(self):
        """Placeholder for loading pretrained weights."""
        pass

    def forward(self, batch):
        x = batch["image"]

        # Run backbone
        logits = self.backbone(x)

        # Flatten to 1D (original: feature = x.reshape(-1); clsout = feature)
        clsout = logits.reshape(-1)

        output = {}
        if "loss" in self.output_type:
            from ..losses.losses import DiceLoss

            loss_bce = nn.BCEWithLogitsLoss(reduction="mean")
            loss_l1s = nn.SmoothL1Loss(reduction="mean")
            loss_dice = DiceLoss()

            output["bce_loss"] = loss_bce(clsout, batch["T_stage"])
            output["l1s_loss"] = loss_l1s(clsout, batch["T_stage"].long().float())
            output["dice_loss"] = loss_dice(clsout, batch["T_stage"].long().float())

        if "inference" in self.output_type:
            prob = torch.sigmoid(clsout)
            prob = torch.nan_to_num(prob)
            output["T_stage"] = prob

        return output
