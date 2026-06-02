"""Shared building blocks for YNet architectures (2D and 3D)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Continuous Batch Normalization (works during both train and eval)
# ---------------------------------------------------------------------------

class ContBatchNorm2d(nn.modules.batchnorm._BatchNorm):
    """BatchNorm2d that does NOT switch to eval mode behavior."""

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError(f"Expected 4D input, got {input.dim()}D")

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var,
            self.weight, self.bias, True, self.momentum, self.eps,
        )


class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    """BatchNorm3d that does NOT switch to eval mode behavior."""

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError(f"Expected 5D input, got {input.dim()}D")

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var,
            self.weight, self.bias, True, self.momentum, self.eps,
        )


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def _get_nd_layers(ndim):
    """Return Conv, BatchNorm, MaxPool, ConvTranspose for given dimensionality."""
    if ndim == 2:
        return nn.Conv2d, ContBatchNorm2d, nn.MaxPool2d, nn.ConvTranspose2d
    elif ndim == 3:
        return nn.Conv3d, ContBatchNorm3d, nn.MaxPool3d, nn.ConvTranspose3d
    else:
        raise ValueError(f"ndim must be 2 or 3, got {ndim}")


def _get_activation(act, num_features):
    if act == "relu":
        return nn.ReLU(num_features)
    elif act == "prelu":
        return nn.PReLU(num_features)
    elif act == "elu":
        return nn.ELU(inplace=True)
    else:
        raise ValueError(f"Unknown activation: {act}")


# ---------------------------------------------------------------------------
# Conv + BN + Activation block
# ---------------------------------------------------------------------------

class LUConv(nn.Module):
    """Single convolution + batch norm + activation block."""

    def __init__(self, in_chan, out_chan, act="relu", ndim=3):
        super().__init__()
        Conv, BN, _, _ = _get_nd_layers(ndim)
        self.conv1 = Conv(in_chan, out_chan, kernel_size=3, padding=1)
        self.bn1 = BN(out_chan)
        self.activation = _get_activation(act, out_chan)

    def forward(self, x):
        return self.activation(self.bn1(self.conv1(x)))


def _make_n_conv(in_channel, depth, act="relu", ndim=3, double_channel=True):
    """Create a pair of LUConv layers following the V-Net channel pattern.

    Channel sizes: 32 * 2^(depth+1)
      depth 0 -> 64, depth 1 -> 128, depth 2 -> 256, depth 3 -> 512
    """
    out_channel = 32 * (2 ** (depth + 1))
    if double_channel:
        layer1 = LUConv(in_channel, out_channel, act, ndim)
        layer2 = LUConv(out_channel, out_channel, act, ndim)
    else:
        mid_channel = 32 * (2 ** depth)
        layer1 = LUConv(in_channel, mid_channel, act, ndim)
        layer2 = LUConv(mid_channel, out_channel, act, ndim)
    return nn.Sequential(layer1, layer2)


# ---------------------------------------------------------------------------
# Encoder (Down) transition
# ---------------------------------------------------------------------------

class DownTransition(nn.Module):
    """Downsampling block: conv pair + max pool (with skip connection)."""

    def __init__(self, in_channel, depth, act="relu", ndim=3):
        super().__init__()
        _, _, MaxPool, _ = _get_nd_layers(ndim)
        self.ops = _make_n_conv(in_channel, depth, act, ndim)
        self.maxpool = MaxPool(2)
        self.current_depth = depth

    def forward(self, x):
        if self.current_depth == 3:
            # Last encoder block: no pooling
            out = self.ops(x)
            out_before_pool = out
        else:
            out_before_pool = self.ops(x)
            out = self.maxpool(out_before_pool)
        return out, out_before_pool


# ---------------------------------------------------------------------------
# Decoder (Up) transition
# ---------------------------------------------------------------------------

class UpTransition(nn.Module):
    """Upsampling block: transposed conv + skip concat + conv pair."""

    def __init__(self, in_chans, out_chans, depth, act="relu", ndim=3):
        super().__init__()
        self.depth = depth
        _, _, _, ConvTranspose = _get_nd_layers(ndim)
        self.up_conv = ConvTranspose(in_chans, out_chans, kernel_size=2, stride=2)
        self.ops = _make_n_conv(
            in_chans + out_chans // 2, depth, act, ndim, double_channel=True,
        )

    def forward(self, x, skip_x):
        out_up = self.up_conv(x)
        concat = torch.cat((out_up, skip_x), dim=1)
        return self.ops(concat)


# ---------------------------------------------------------------------------
# Output transition (segmentation head)
# ---------------------------------------------------------------------------

class OutputTransition(nn.Module):
    """1x1 conv + sigmoid for segmentation output."""

    def __init__(self, in_chans, n_labels, ndim=3):
        super().__init__()
        Conv, _, _, _ = _get_nd_layers(ndim)
        self.final_conv = Conv(in_chans, n_labels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.final_conv(x))


# ---------------------------------------------------------------------------
# Classification head (shared by 2D and 3D)
# ---------------------------------------------------------------------------

class ClassificationHead(nn.Module):
    """Conv + BN + Act + AdaptiveAvgPool + Dropout + FC for classification."""

    def __init__(self, in_chan=1, act="prelu", cls_classes=1, ndim=3, dropout_p=0.4):
        super().__init__()
        Conv, BN, _, _ = _get_nd_layers(ndim)

        self.conv1 = Conv(in_chan, 1024, kernel_size=3, padding=1)
        self.bn1 = BN(1024)
        self.activation = _get_activation(act, 1024)

        if ndim == 2:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        else:
            self.avgpool = nn.AdaptiveAvgPool3d(1)

        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(1024, cls_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
