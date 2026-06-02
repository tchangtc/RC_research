"""2D data augmentation functions for image + mask pairs."""

import cv2
import math
import numpy as np
from scipy.ndimage import gaussian_filter


def do_random_flip(image, mask):
    """Random horizontal, vertical, or both flips."""
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
    if np.random.rand() < 0.5:
        image = cv2.flip(image, -1)
        mask = cv2.flip(mask, -1)
    image = np.ascontiguousarray(image)
    return image, mask


def _affine_matrix(degree, scale, translate, shear):
    """Build 2x3 affine transformation matrix."""
    rotate = cv2.getRotationMatrix2D(angle=degree, center=(0, 0), scale=scale)
    shear_x = math.tan(shear[0] * math.pi / 180)
    shear_y = math.tan(shear[1] * math.pi / 180)
    matrix = np.ones([2, 3])
    matrix[0] = rotate[0] + shear_y * rotate[1]
    matrix[1] = rotate[1] + shear_x * rotate[0]
    matrix[0, 2] = translate[0]
    matrix[1, 2] = translate[1]
    return matrix


def do_random_affine(image, mask, degree=30, translate=0.1, scale=0.2, shear=10):
    """Random affine transformation."""
    h, w = image.shape[:2]
    d = np.random.uniform(-degree, degree)
    s = np.random.uniform(-scale, scale) + 1
    tx, ty = np.random.uniform(-translate, translate, 2) * [w, h]
    sx, sy = np.random.uniform(-shear, shear, 2)
    matrix = _affine_matrix(d, s, (tx, ty), (sx, sy))
    image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image, mask


def do_random_rotate(image, mask, degree=30):
    """Random rotation around image center."""
    d = np.random.uniform(-degree, degree)
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    matrix = cv2.getRotationMatrix2D((cx, cy), -d, 1.0)
    image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image, mask


def do_random_stretch(image, mask, stretch=(0.2, 0.2)):
    """Random stretch (scaling) in x and y."""
    sx, sy = stretch
    sx = np.random.uniform(-sx, sx) + 1
    sy = np.random.uniform(-sy, sy) + 1
    matrix = np.array([[sy, 0, 0], [0, sx, 1]])
    h, w = image.shape[:2]
    image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image, mask


def do_elastic_transform(image, mask, alpha=120, sigma=6.0, alpha_affine=3.6):
    """Elastic deformation (Simard et al. 2003)."""
    h, w = image.shape[:2]
    alpha = float(alpha)
    sigma = float(sigma)
    alpha_affine = float(alpha_affine)

    # Random affine
    center = np.array((h, w), dtype=np.float32) // 2
    sq_size = min(h, w) // 3
    pts1 = np.array([
        center + sq_size,
        [center[0] + sq_size, center[1] - sq_size],
        center - sq_size,
    ], dtype=np.float32)
    pts2 = pts1 + np.random.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    matrix = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Elastic displacement
    dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = np.float32(x + dx)
    map_y = np.float32(y + dy)
    image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.remap(mask, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return image, mask


def do_random_cutout(image, mask, num_block=5, block_size=(0.1, 0.3), fill="constant"):
    """Random rectangular cutout."""
    h, w = image.shape[:2]
    n = np.random.randint(1, num_block + 1)
    for _ in range(n):
        bh = int(np.random.uniform(*block_size) * h)
        bw = int(np.random.uniform(*block_size) * w)
        x = np.random.randint(0, w - bw)
        y = np.random.randint(0, h - bh)
        image[y:y + bh, x:x + bw] = 0
        mask[y:y + bh, x:x + bw] = 0
    return image, mask


def do_random_contrast(image, mask):
    """Random contrast adjustment."""
    u = np.random.choice(3)
    if u == 0:
        m = np.random.uniform(-0.3, 0.3)
        image = image * (1 + m)
    elif u == 1:
        m = np.random.uniform(-0.5, 0.5)
        image = image ** (1 + m)
    else:
        m = np.random.uniform(-0.2, 0.2)
        image = image + m
    image = np.clip(image, 0, 1)
    return image, mask


def do_random_noise(image, mask, m=0.1):
    """Additive uniform noise."""
    h, w = image.shape[:2]
    noise = np.random.uniform(-1, 1, size=(h, w)) * m
    image = image + noise
    image = np.clip(image, 0, 1)
    return image, mask


# ---------------------------------------------------------------------------
# Training augmentation pipeline
# ---------------------------------------------------------------------------

def train_augment_v00(image, mask):
    """Default training augmentation pipeline.

    Applies: flip -> geometric (70%) -> elastic (50%) -> cutout (50%) -> intensity (50%)
    """
    image, mask = do_random_flip(image, mask)

    if np.random.rand() < 0.7:
        func = np.random.choice([
            lambda i, m: do_random_affine(i, m, degree=30, translate=0.1, scale=0.3, shear=20),
            lambda i, m: do_random_rotate(i, m, degree=30),
            lambda i, m: do_random_stretch(i, m, stretch=(0.3, 0.3)),
        ], 1)[0]
        image, mask = func(image, mask)

    if np.random.rand() < 0.5:
        image, mask = do_elastic_transform(
            image, mask, alpha=image.shape[0],
            sigma=image.shape[0] * 0.05,
            alpha_affine=image.shape[0] * 0.03,
        )

    if np.random.rand() < 0.5:
        image, mask = do_random_cutout(image, mask, num_block=5, block_size=(0.1, 0.3))

    if np.random.rand() < 0.5:
        func = np.random.choice([
            lambda i, m: do_random_contrast(i, m),
            lambda i, m: do_random_noise(i, m, m=0.1),
        ], 1)[0]
        image, mask = func(image, mask)

    return image, mask
