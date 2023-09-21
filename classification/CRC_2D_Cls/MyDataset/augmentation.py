import numpy as np
import cv2
import math
# from common import *
from scipy.ndimage import gaussian_filter

# ################################################################33
# #basic
def do_random_hflip(image, ):
    if np.random.rand()<0.5:
        image = cv2.flip(image,1)
    return image


# #geometric
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
def do_random_affine(
    image,
    degree=30,
    translate=0.1,
    scale=0.2,
    shear=10,
):
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

    return image


def do_random_rotate(image, degree=15, ):
    degree = np.random.uniform(-degree, degree)

    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    matrix = cv2.getRotationMatrix2D((cx, cy), -degree, 1.0)
    image = cv2.warpAffine( image, matrix, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image


def do_random_stretch(image, stretch=(0.1,0.2) ):
    stretchx, stretchy = stretch
    stretchx = np.random.uniform(-stretchx, stretchx) + 1
    stretchy = np.random.uniform(-stretchy, stretchy) + 1

    matrix = np.array([
        [stretchy,0,0],
        [0,stretchx,1],
    ])
    h, w = image.shape[:2]
    image = cv2.warpAffine( image, matrix, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return image


##intensity ##################


def do_random_contrast(image):
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
    return image



#noise
def do_random_noise(image, m=0.08):
    height, width = image.shape[:2]

    #image = (image).astype(np.float32)/255
    noise = np.random.uniform(-1,1,size=(height,width))*m
    image = image+noise

    image = np.clip(image,0,1)
    #image = (image*255).astype(np.uint8)
    return image

def do_random_cutout(image, num_block=5, block_size=[0.1,0.3], fill='constant'):
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
        else:
            raise NotImplementedError
    return image


############################################################################33

#https://github.com/albumentations-team/albumentations/blob/b7877b7dcae4d0f7534f2f4d8708b381ca0800e6/albumentations/augmentations/geometric/functional.py#L303
#https://medium.com/@ArjunThoughts/albumentations-package-is-a-fast-and-%EF%AC%82exible-library-for-image-augmentations-with-many-various-207422f55a24
#https://github.com/Project-MONAI/MONAI/issues/2186
#https://albumentations.ai/docs/examples/example_kaggle_salt/#non-rigid-transformations-elastictransform-griddistortion-opticaldistortion

def do_elastic_transform(
    image,
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


    if 1:
        dx = gaussian_filter((np.random.rand(height, width) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(height, width) * 2 - 1), sigma) * alpha


    x, y = np.meshgrid(np.arange(width), np.arange(height))
    map_x = np.float32(x + dx)
    map_y = np.float32(y + dy)
    image = cv2.remap( image, map1=map_x, map2=map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return image


# image = np.zeros((16,16),np.uint8)
# for i in range(16):
#     image[i,i%2::2]=255
# image =cv2.resize(image,dsize=(768,768),interpolation=cv2.INTER_NEAREST)
# for t in range(100):
#     m = do_elastic_transform(
#         image.copy(),
#         alpha=768,
#         sigma=768* 0.05,
#         alpha_affine=768* 0.03
#
#     )
#     m1 = do_random_cutout(
#         image.copy(),
#     )
#     image_show('image',image, resize=1)
#     image_show('m',m, resize=1)
#     image_show('m1',m1, resize=1)
#     cv2.waitKey(0)
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import math


# def gaussian_noise(tensor_img, std, mean=0):
    
#     return tensor_img + torch.rand(tensor_img.shape).to(tensor_img.device) * std + mean


# def brightness_additive(tensor_img, std, mean=0, per_channel=False):

#     if per_channel:
#         C = tensor_img.shape[1]
    
#     else:
#         C = 1

#     if len(tensor_img.shape) == 5:
#         rand_brightness = torch.normal(mean, std, size=(1, C, 1, 1, 1)).to(tensor_img.device)
#     elif len(tensor_img.shape) == 4:
#         rand_brightness = torch.normal(mean, std, size=(1, C, 1, 1)).to(tensor_img.device)
#     else:
#         raise ValueError('Invalid input tensor dimension, should be 5d for volume image or 4d for 2d image')

#     return tensor_img + rand_brightness


# def gamma(tensor_img, gamma_range=(0.5, 2), per_channel=False, retain_stats=False):
    
#     if len(tensor_img.shape) == 5:
#         dim = '3d'
#         _, C, D, H, W = tensor_img.shape
#     elif len(tensor_img.shape) == 4:
#         dim = '2d'
#         _, C, H, W = tensor_img.shape
#     else:
#         raise ValueError('Invalid input tensor dimension, should be 5d for volume image or 4d for 2d image')
    
#     tmp_C = C if per_channel else 1
    
#     tensor_img = tensor_img.view(tmp_C, -1)
#     minm, _ = tensor_img.min(dim=1)
#     maxm, _ = tensor_img.max(dim=1)
#     minm, maxm = minm.unsqueeze(1), maxm.unsqueeze(1) # unsqueeze for broadcast machanism

#     rng = maxm - minm

#     mean = tensor_img.mean(dim=1).unsqueeze(1)
#     std = tensor_img.std(dim=1).unsqueeze(1)
#     gamma = torch.rand(C, 1) * (gamma_range[1] - gamma_range[0]) + gamma_range[0]

#     tensor_img = torch.pow((tensor_img - minm) / rng, gamma) * rng + minm

#     if retain_stats:
#         tensor_img -= tensor_img.mean(dim=1).unsqueeze(1)
#         tensor_img = tensor_img / tensor_img.std(dim=1).unsqueeze(1) * std + mean

#     if dim == '3d':
#         return tensor_img.view(1, C, D, H, W)
#     else:
#         return tensor_img.view(1, C, H, W)
    

# def crop_3d(tensor_img, crop_size, mode):
#     assert mode in ['random', 'center'], "Invalid Mode, should be \'random\' or \'center\'"
#     if isinstance(crop_size, int):
#         crop_size = [crop_size] * 3

#     _, _, D, H, W = tensor_img.shape

#     diff_D = D - crop_size[0]
#     diff_H = H - crop_size[1]
#     diff_W = W - crop_size[2]
    
#     if mode == 'random':
#         rand_z = np.random.randint(0, max(diff_D, 1))
#         rand_x = np.random.randint(0, max(diff_H, 1))
#         rand_y = np.random.randint(0, max(diff_W, 1))
#     else:
#         rand_z = diff_D // 2
#         rand_x = diff_H // 2
#         rand_y = diff_W // 2

#     cropped_img = tensor_img[:, :, rand_z:rand_z+crop_size[0], rand_x:rand_x+crop_size[1], rand_y:rand_y+crop_size[2]]

#     return cropped_img


# # def random_scale_rotate_translate_3d(tensor_img, tensor_lab, scale, rotate, translate, noshear=True):
# def random_scale_rotate_translate_3d(tensor_img, scale, rotate, translate, noshear=True):
    
#     if isinstance(scale, float) or isinstance(scale, int):
#         scale = [scale] * 3
#     if isinstance(translate, float) or isinstance(translate, int):
#         translate = [translate] * 3
#     if isinstance(rotate, float) or isinstance(rotate, int):
#         rotate = [rotate] * 3

#     scale_z = 1 - scale[0] + np.random.random() * 2*scale[0]
#     scale_x = 1 - scale[1] + np.random.random() * 2*scale[1]
#     scale_y = 1 - scale[2] + np.random.random() * 2*scale[2]
#     shear_xz = 0 if noshear else np.random.random() * 2*scale[0] - scale[0]
#     shear_yz = 0 if noshear else np.random.random() * 2*scale[0] - scale[0]
#     shear_zx = 0 if noshear else np.random.random() * 2*scale[1] - scale[1]
#     shear_yx = 0 if noshear else np.random.random() * 2*scale[1] - scale[1]
#     shear_zy = 0 if noshear else np.random.random() * 2*scale[2] - scale[2]
#     shear_xy = 0 if noshear else np.random.random() * 2*scale[2] - scale[2]
#     translate_z = np.random.random() * 2*translate[0] - translate[0]
#     translate_x = np.random.random() * 2*translate[1] - translate[1]
#     translate_y = np.random.random() * 2*translate[2] - translate[2]


#     theta_scale = torch.tensor([[scale_y, shear_xy, shear_zy, translate_y],
#                                 [shear_yx, scale_x, shear_zx, translate_x],
#                                 [shear_yz, shear_xz, scale_z, translate_z], 
#                                 [0, 0, 0, 1]]).float()
#     angle_xy = (float(np.random.randint(-rotate[0], max(rotate[0], 1))) / 180.) * math.pi
#     angle_xz = (float(np.random.randint(-rotate[1], max(rotate[1], 1))) / 180.) * math.pi
#     angle_yz = (float(np.random.randint(-rotate[2], max(rotate[2], 1))) / 180.) * math.pi
    
#     theta_rotate_xz = torch.tensor([[1, 0, 0, 0],
#                                     [0, math.cos(angle_xz), -math.sin(angle_xz), 0],
#                                     [0, math.sin(angle_xz), math.cos(angle_xz), 0],
#                                     [0, 0, 0, 1]]).float()
#     theta_rotate_xy = torch.tensor([[math.cos(angle_xy), -math.sin(angle_xy), 0, 0],
#                                     [math.sin(angle_xy), math.cos(angle_xy), 0, 0],
#                                     [0, 0, 1, 0],
#                                     [0, 0, 0, 1]]).float()
#     theta_rotate_yz = torch.tensor([[math.cos(angle_yz), 0, -math.sin(angle_yz), 0],
#                                     [0, 1, 0, 0],
#                                     [math.sin(angle_yz), 0, math.cos(angle_yz), 0],
#                                     [0, 0, 0, 1]]).float()

#     theta = torch.mm(theta_rotate_xy, theta_rotate_xz)
#     theta = torch.mm(theta, theta_rotate_yz)
#     theta = torch.mm(theta, theta_scale)[0:3, :].unsqueeze(0)
    
#     grid = F.affine_grid(theta, tensor_img.size(), align_corners=True)
#     tensor_img = F.grid_sample(tensor_img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
#     # tensor_lab = F.grid_sample(tensor_lab.float(), grid, mode='nearest', padding_mode='zeros', align_corners=True).long()
#     # return tensor_img, tensor_lab

#     return tensor_img
    