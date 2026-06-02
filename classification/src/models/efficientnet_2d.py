"""EfficientNet-based 2D classifier for T-stage assessment."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.efficientnet import (
    efficientnet_b0,
    efficientnet_b2,
    efficientnet_b4,
)

# Feature dimensions for each EfficientNet variant
_ENCODER_DIMS = {
    "efficientnet_b0": 1280,
    "efficientnet_b2": 1408,
    "efficientnet_b4": 1792,
}

_ENCODER_FACTORY = {
    "efficientnet_b0": efficientnet_b0,
    "efficientnet_b2": efficientnet_b2,
    "efficientnet_b4": efficientnet_b4,
}


class RGB(nn.Module):
    """ImageNet normalization layer."""

    IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]
    IMAGE_RGB_STD = [0.229, 0.224, 0.225]

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(1, 3, 1, 1))
        self.register_buffer("std", torch.ones(1, 3, 1, 1))
        self.mean.data = torch.FloatTensor(self.IMAGE_RGB_MEAN).view(self.mean.shape)
        self.std.data = torch.FloatTensor(self.IMAGE_RGB_STD).view(self.std.shape)

    def forward(self, x):
        return (x - self.mean) / self.std


class EffNet(nn.Module):
    """EfficientNet encoder + classification head for binary T-stage.

    Input: batch dict with 'image' key, shape [B, C, H, W].
           C=2 (image + mask channel), expanded to 3 channels internally.
    Output: dict with 'label' (inference) or 'bce_loss', 'l1s_loss', 'dice_loss' (training).
    """

    def __init__(self, encoder_name="efficientnet_b0", pretrained=True):
        super().__init__()
        self.output_type = ["inference", "loss"]

        self.rgb = RGB()

        encoder_fn = _ENCODER_FACTORY[encoder_name]
        self.encoder = encoder_fn(pretrained=pretrained)

        feature_dim = _ENCODER_DIMS[encoder_name]
        self.classifier = nn.Linear(feature_dim, 1)

    def load_pretrain(self):
        """Placeholder for loading pretrained weights (timm models load automatically)."""
        pass

    def forward(self, batch):
        x = batch["image"]

        # Normalize with ImageNet stats
        x = self.rgb(x)

        # Extract features
        features = self.encoder.forward_features(x)

        # Global average pooling + flatten
        x = F.adaptive_avg_pool2d(features, 1)
        x = torch.flatten(x, 1, 3)

        # Classification
        logits = self.classifier(x).reshape(-1)

        output = {}

        if "loss" in self.output_type:
            from ..losses.losses import DiceLoss

            loss_bce = nn.BCEWithLogitsLoss(reduction="mean")
            loss_l1s = nn.SmoothL1Loss(reduction="mean")
            loss_dice = DiceLoss()

            output["bce_loss"] = loss_bce(logits, batch["label"])
            output["l1s_loss"] = loss_l1s(logits, batch["label"].long().float())
            output["dice_loss"] = loss_dice(logits, batch["label"].long().float())

        if "inference" in self.output_type:
            prob = torch.sigmoid(logits)
            prob = torch.nan_to_num(prob)
            output["label"] = prob

        return output
