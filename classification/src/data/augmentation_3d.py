"""3D data augmentation transforms (nnU-Net style) for volumetric data.

All transforms operate on a sample dict with 'image' and 'label' keys.
Designed for use with torchvision.transforms.Compose.
"""

import random
import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Callable, Tuple, Union


# ---------------------------------------------------------------------------
# Crop transforms
# ---------------------------------------------------------------------------

class RandomCrop:
    """Random crop with automatic padding if needed."""

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        # Pad if necessary
        if any(s <= o for s, o in zip(label.shape, self.output_size)):
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode="constant")
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode="constant")

        w, h, d = image.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        return {
            "image": image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]],
            "label": label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]],
        }


class CenterCrop:
    """Center crop with automatic padding if needed."""

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        if any(s <= o for s, o in zip(label.shape, self.output_size)):
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode="constant")
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode="constant")

        w, h, d = image.shape
        w1 = int(round((w - self.output_size[0]) / 2.0))
        h1 = int(round((h - self.output_size[1]) / 2.0))
        d1 = int(round((d - self.output_size[2]) / 2.0))

        return {
            "image": image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]],
            "label": label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]],
        }


# ---------------------------------------------------------------------------
# Rotation & Flip
# ---------------------------------------------------------------------------

class RandomRotFlip:
    """Random 90-degree rotation + flip."""

    def __init__(self, p_per_sample=0.25):
        self.prob = p_per_sample

    def __call__(self, sample):
        image = sample["image"].squeeze(0)
        label = sample["label"].squeeze(0)

        if np.random.rand() < self.prob:
            k = np.random.randint(0, 4)
            axis = np.random.randint(0, 2)
            image = np.flip(np.rot90(image, k), axis=axis).copy()
            label = np.flip(np.rot90(label, k), axis=axis).copy()

        return {
            "image": np.expand_dims(image, axis=0),
            "label": np.expand_dims(label, axis=0),
        }


# ---------------------------------------------------------------------------
# Gaussian Noise
# ---------------------------------------------------------------------------

class GaussianNoiseTransform:
    """Add Gaussian noise to image."""

    def __init__(self, noise_variance=(0, 0.07), p_per_sample=0.25):
        self.prob = p_per_sample
        self.noise_variance = noise_variance

    def __call__(self, sample):
        image = sample["image"]
        if np.random.uniform() < self.prob:
            v0, v1 = self.noise_variance
            variance = random.uniform(v0, v1) if v0 != v1 else v0
            image = image + np.random.normal(0.0, variance, size=image.shape)
        return {"image": image, "label": sample["label"]}


# ---------------------------------------------------------------------------
# Gaussian Blur
# ---------------------------------------------------------------------------

def _get_range_val(value, rnd_type="uniform"):
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 2:
            if value[0] == value[1]:
                return value[0]
            return random.uniform(value[0], value[1])
        return value[0]
    return value


class GaussianBlurTransform:
    """Apply Gaussian blur with configurable sigma."""

    def __init__(self, blur_sigma=(1, 2.5), different_sigma_per_channel=False,
                 p_per_channel=0.25, p_per_sample=0.25):
        self.blur_sigma = blur_sigma
        self.different_sigma_per_channel = different_sigma_per_channel
        self.p_per_channel = p_per_channel
        self.p_per_sample = p_per_sample

    def __call__(self, sample):
        image = sample["image"]
        for b in range(len(image)):
            if np.random.uniform() < self.p_per_sample:
                for c in range(image[b].shape[0]) if image[b].ndim > 3 else [0]:
                    if np.random.uniform() < self.p_per_channel:
                        sigma = _get_range_val(self.blur_sigma)
                        if image[b].ndim > 3:
                            image[b][c] = gaussian_filter(image[b][c], sigma, order=0)
                        else:
                            image[b] = gaussian_filter(image[b], sigma, order=0)
        return {"image": image, "label": sample["label"]}


# ---------------------------------------------------------------------------
# Brightness
# ---------------------------------------------------------------------------

class BrightnessTransform:
    """Additive brightness adjustment."""

    def __init__(self, mu=0, sigma=1, per_channel=False,
                 p_per_channel=0.25, p_per_sample=0.25):
        self.mu = mu
        self.sigma = sigma
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel
        self.p_per_sample = p_per_sample

    def __call__(self, sample):
        image = sample["image"]
        for b in range(image.shape[0]):
            if np.random.uniform() < self.p_per_sample:
                if self.per_channel:
                    for c in range(image[b].shape[0]):
                        if np.random.uniform() <= self.p_per_channel:
                            image[b][c] += np.random.normal(self.mu, self.sigma)
                else:
                    if np.random.uniform() <= self.p_per_channel:
                        image[b] += np.random.normal(self.mu, self.sigma)
        return {"image": image, "label": sample["label"]}


# ---------------------------------------------------------------------------
# Gamma
# ---------------------------------------------------------------------------

class GammaTransform:
    """Gamma correction augmentation."""

    def __init__(self, gamma_range=(0.25, 1), per_channel=False,
                 retain_stats=False, p_per_sample=0.25):
        self.gamma_range = gamma_range
        self.per_channel = per_channel
        self.retain_stats = retain_stats
        self.p_per_sample = p_per_sample

    def __call__(self, sample):
        image = sample["image"]
        epsilon = 1e-7

        for b in range(len(image)):
            if np.random.uniform() < self.p_per_sample:
                data = image[b]
                if self.retain_stats:
                    mn, sd = data.mean(), data.std()

                g0, g1 = self.gamma_range
                if np.random.random() < 0.5 and g0 < 1:
                    gamma = np.random.uniform(g0, 1)
                else:
                    gamma = np.random.uniform(max(g0, 1), g1)

                minm = data.min()
                rnge = data.max() - minm
                data = np.power(((data - minm) / (rnge + epsilon)), gamma) * rnge + minm

                if self.retain_stats:
                    data = (data - data.mean()) / (data.std() + 1e-8) * sd + mn

                image[b] = data

        return {"image": image, "label": sample["label"]}


# ---------------------------------------------------------------------------
# Contrast
# ---------------------------------------------------------------------------

class ContrastAugmentationTransform:
    """Multiplicative contrast augmentation."""

    def __init__(self, contrast_range=(0.75, 1.25), preserve_range=True,
                 per_channel=True, p_per_channel=1.0, p_per_sample=1.0):
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel
        self.p_per_sample = p_per_sample

    def __call__(self, sample):
        image = sample["image"]
        for b in range(len(image)):
            if np.random.uniform() < self.p_per_sample:
                c0, c1 = self.contrast_range
                factor = (np.random.uniform(c0, 1) if np.random.random() < 0.5 and c0 < 1
                          else np.random.uniform(max(c0, 1), c1))
                mn = image[b].mean()
                if self.preserve_range:
                    minm, maxm = image[b].min(), image[b].max()
                image[b] = (image[b] - mn) * factor + mn
                if self.preserve_range:
                    image[b] = np.clip(image[b], minm, maxm)
        return {"image": image, "label": sample["label"]}


# ---------------------------------------------------------------------------
# Mirror
# ---------------------------------------------------------------------------

class MirrorTransform:
    """Random mirroring along spatial axes."""

    def __init__(self, axes=(0, 1, 2), p_per_sample=0.25):
        self.axes = axes
        self.p_per_sample = p_per_sample

    def __call__(self, sample):
        image = sample["image"]
        label = sample["label"]

        for b in range(len(image)):
            if np.random.uniform() < self.p_per_sample:
                if 0 in self.axes and np.random.uniform() < 0.5:
                    image[b] = image[b][:, ::-1].copy()
                    if label is not None:
                        label[b] = label[b][:, ::-1].copy()
                if 1 in self.axes and np.random.uniform() < 0.5:
                    image[b] = image[b][:, :, ::-1].copy()
                    if label is not None:
                        label[b] = label[b][:, :, ::-1].copy()

        return {"image": image, "label": label}
