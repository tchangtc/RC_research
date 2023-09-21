import pdb
import numpy as np
import cv2
import math
# from common import *
from scipy.ndimage import gaussian_filter

# ################################################################33
# filp  
def do_random_flip(image, mask):
    assert image.shape[:2] == mask.shape
    assert image.shape[0] == mask.shape[1]
    assert image.shape[0] == mask.shape[1]

    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)      # horizontal
        mask  = cv2.flip(mask, 1)
    
    if np.random.rand() < 0.5:          # vertical
        image = cv2.flip(image, 0)
        mask  = cv2.flip(mask, 0)

    if np.random.rand() < 0.5:          # horizontal & vertical
        image = cv2.flip(image, -1)
        mask = cv2.flip(mask, -1)
        # image = image.transpose(1, 0, 2)
        # mask  = mask.transpose(1, 0)
    
    image = np.ascontiguousarray(image)
    # mask  = np.ascontiguousarray(mask)
    return image, mask


# geometric
def affine_param_to_matrix(
    degree=10,
    scale=0.1,
    translate=(10,10),
    shear=(10,10),
):
    #h,w = image_shape
    #https://stackoverflow.com/questions/61242154/the-third-column-of-cv2-getrotationmatrix2d
    rotate = cv2.getRotationMatrix2D(angle=degree, center=(0, 0), scale=scale)

    # Shear
    shear_x = math.tan(shear[0] * math.pi / 180)
    shear_y = math.tan(shear[1] * math.pi / 180)

    matrix = np.ones([2, 3])
    matrix[0] = rotate[0] + shear_y * rotate[1]
    matrix[1] = rotate[1] + shear_x * rotate[0]
    matrix[0, 2] = translate[0]
    matrix[1, 2] = translate[1]
    return matrix


# np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
# affine
def do_random_affine(
    image,
    mask,
    degree=30,
    translate=0.1,
    scale=0.2,
    shear=10,
):
    assert image.shape[:2] == mask.shape
    assert image.shape[0] == mask.shape[1]
    assert image.shape[0] == mask.shape[1]

    h,w = image.shape[:2]
    degree = np.random.uniform(-degree, degree)
    scale  = np.random.uniform(-scale, scale)+1
    translate_x, translate_y  = np.random.uniform(-translate, translate,2)*[w,h]
    shear_x, shear_y  = np.random.uniform(-shear, shear,2)

    matrix = affine_param_to_matrix(
        degree,
        scale,
        (translate_x, translate_y),
        (shear_x, shear_y),
    )

    image = cv2.warpAffine( image, matrix, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.warpAffine( mask, matrix, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return image, mask


# rotate
def do_random_rotate(image, mask, degree=30, ):
    assert image.shape[:2] == mask.shape
    assert image.shape[0] == mask.shape[1]
    assert image.shape[0] == mask.shape[1]

    degree = np.random.uniform(-degree, degree)

    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    matrix = cv2.getRotationMatrix2D((cx, cy), -degree, 1.0)
    image = cv2.warpAffine( image, matrix, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.warpAffine( mask, matrix, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image, mask


# stretch
def do_random_stretch(image, mask, stretch=(0.2,0.4) ):
    assert image.shape[:2] == mask.shape
    assert image.shape[0] == mask.shape[1]
    assert image.shape[0] == mask.shape[1]

    stretchx, stretchy = stretch
    stretchx = np.random.uniform(-stretchx, stretchx) + 1
    stretchy = np.random.uniform(-stretchy, stretchy) + 1

    matrix = np.array([
        [stretchy,0,0],
        [0,stretchx,1],
    ])
    h, w = image.shape[:2]
    image = cv2.warpAffine( image, matrix, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.warpAffine( mask, matrix, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return image, mask


# contrast
def do_random_contrast(image, mask):
    assert image.shape[:2] == mask.shape
    assert image.shape[0] == mask.shape[1]
    assert image.shape[0] == mask.shape[1]

    #image = image.astype(np.float32)/255
    u = np.random.choice(3)
    if u==0:
        m = np.random.uniform(-0.3,0.3)
        image = image*(1+m)
    if u==1:
        m = np.random.uniform(-0.5,0.5)
        image = image**(1+m)
    if u==2:
        m = np.random.uniform(-0.2,0.2)
        image = image + m

    image = np.clip(image,0,1)
    #image = (image*255).astype(np.uint8)
    return image, mask


# noise
def do_random_noise(image, mask, m=0.07):
    assert image.shape[:2] == mask.shape
    assert image.shape[0] == mask.shape[1]
    assert image.shape[0] == mask.shape[1]
    # pdb.set_trace()    
    height, width = image.shape

    #image = (image).astype(np.float32)/255
    noise = np.random.uniform(-1,1,size=(height,width))*m
    image = image+noise

    image = np.clip(image,0,1)
    #image = (image*255).astype(np.uint8)
    return image, mask


def do_random_cutout(image, mask, num_block=5, block_size=[0.1,0.3], fill='constant'):
    assert image.shape[:2] == mask.shape
    assert image.shape[0] == mask.shape[1]
    assert image.shape[0] == mask.shape[1]

    height, width = image.shape[:2]

    num_block = np.random.randint(1,num_block+1)
    for n in range(num_block):
        h = np.random.uniform(*block_size)
        w = np.random.uniform(*block_size)
        h = int(h*height)
        w = int(w*width)
        x = np.random.randint(0,width-w)
        y = np.random.randint(0,height-h)
        if fill=='constant':
            image[y:y+h,x:x+w]=0
            mask[y:y+h,x:x+w]=0
        else:
            raise NotImplementedError
    return image, mask


############################################################################33

#https://github.com/albumentations-team/albumentations/blob/b7877b7dcae4d0f7534f2f4d8708b381ca0800e6/albumentations/augmentations/geometric/functional.py#L303
#https://medium.com/@ArjunThoughts/albumentations-package-is-a-fast-and-%EF%AC%82exible-library-for-image-augmentations-with-many-various-207422f55a24
#https://github.com/Project-MONAI/MONAI/issues/2186
#https://albumentations.ai/docs/examples/example_kaggle_salt/#non-rigid-transformations-elastictransform-griddistortion-opticaldistortion

def do_elastic_transform(
    image,
    mask,
    alpha=120,
    sigma=120* 0.05,
    alpha_affine=120* 0.03

):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    Based on https://gist.github.com/ernestum/601cdf56d2b424757de5
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
    """
    assert image.shape[:2] == mask.shape
    assert image.shape[0] == mask.shape[1]
    assert image.shape[0] == mask.shape[1]
    height, width = image.shape[:2]

    # Random affine
    center_square = np.array((height, width), dtype=np.float32) // 2
    square_size = min((height, width)) // 3
    alpha = float(alpha)
    sigma = float(sigma)
    alpha_affine = float(alpha_affine)

    pts1 = np.array(
        [
            center_square + square_size,
            [center_square[0] + square_size, center_square[1] - square_size],
            center_square - square_size,
        ],
        dtype=np.float32,
    )
    pts2 = pts1 + np.random.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(
        np.float32
    )
    matrix = cv2.getAffineTransform(pts1, pts2)

    image = cv2.warpAffine(image, M=matrix, dsize=(width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.warpAffine(mask, M=matrix, dsize=(width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


    if 1:
        dx = gaussian_filter((np.random.rand(height, width) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(height, width) * 2 - 1), sigma) * alpha


    x, y = np.meshgrid(np.arange(width), np.arange(height))
    map_x = np.float32(x + dx)
    map_y = np.float32(y + dy)
    image = cv2.remap( image, map1=map_x, map2=map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    mask = cv2.remap( mask, map1=map_x, map2=map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return image, mask


