"""2D YNet architecture for joint segmentation and classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import (
    DownTransition,
    UpTransition,
    OutputTransition,
    ClassificationHead,
)


class YNet2D(nn.Module):
    """2D encoder-decoder for joint segmentation + classification.

    Architecture: 4-level U-Net encoder/decoder with classification branch
    from the bottleneck features.

    Channel progression: 1 -> 64 -> 128 -> 256 -> 512 -> (decode back)
    """

    def __init__(self, n_class=1, act="relu", cls_classes=1):
        super().__init__()

        # Encoder
        self.down_tr64 = DownTransition(1, 0, act, ndim=2)
        self.down_tr128 = DownTransition(64, 1, act, ndim=2)
        self.down_tr256 = DownTransition(128, 2, act, ndim=2)
        self.down_tr512 = DownTransition(256, 3, act, ndim=2)

        # Decoder
        self.up_tr256 = UpTransition(512, 512, 2, act, ndim=2)
        self.up_tr128 = UpTransition(256, 256, 1, act, ndim=2)
        self.up_tr64 = UpTransition(128, 128, 0, act, ndim=2)

        # Heads
        self.out_tr = OutputTransition(64, n_class, ndim=2)
        self.clc_out = ClassificationHead(512, act, cls_classes, ndim=2)

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


class Classification2D(nn.Module):
    """2D encoder-only classifier (no decoder/segmentation).

    Uses only the encoder path + classification head.
    Input: single-channel image [B, 1, H, W].
    """

    def __init__(self, act="relu", cls_classes=1):
        super().__init__()

        self.down_tr64 = DownTransition(1, 0, act, ndim=2)
        self.down_tr128 = DownTransition(64, 1, act, ndim=2)
        self.down_tr256 = DownTransition(128, 2, act, ndim=2)
        self.down_tr512 = DownTransition(256, 3, act, ndim=2)
        self.clc_out = ClassificationHead(512, act, cls_classes, ndim=2)

    def forward(self, x):
        out64, _ = self.down_tr64(x)
        out128, _ = self.down_tr128(out64)
        out256, _ = self.down_tr256(out128)
        out512, _ = self.down_tr512(out256)
        return self.clc_out(out512)
