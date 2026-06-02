"""Loss functions for rectal cancer T-stage classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Binary Dice loss for classification.

    Applies sigmoid to inputs, then computes 1 - Dice coefficient.
    """

    def __init__(self, smooth=1e-7):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            inputs.sum() + targets.sum() + self.smooth
        )
        return 1 - dice


class DiceBCELoss(nn.Module):
    """Combined Dice + Binary Cross Entropy loss."""

    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs_sigmoid = torch.sigmoid(inputs)
        inputs_flat = inputs_sigmoid.view(-1)
        targets_flat = targets.view(-1).float()

        intersection = (inputs_flat * targets_flat).sum()
        dice_loss = 1 - (2.0 * intersection + self.smooth) / (
            inputs_flat.sum() + targets_flat.sum() + self.smooth
        )
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="mean")
        return bce + dice_loss
