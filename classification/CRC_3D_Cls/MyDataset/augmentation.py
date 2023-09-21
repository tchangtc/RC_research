import pdb
import random
import torch
import numpy as np
import torch.nn as nn
from scipy.ndimage import gaussian_filter
from typing import Callable, Tuple, Union


############## Crop #####################
class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {'image': image, 'label': label}


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


################# Rotate & Filp #####################
class RandomRotFlip(object):
    '''
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    '''
    def __init__(self, p_per_sample):
        self.prob = p_per_sample

    def __call__(self, sample):

        sample['image'], sample['label'] = sample['image'].squeeze(0), sample['label'].squeeze(0)

        k = np.random.randint(0, 4)
        axis = np.random.randint(0, 2)
        
        ret_dict = {}
        for key in sample.keys():
            if np.random.rand() < self.prob:
                item = sample[key]
                item = np.rot90(item, k)
                item = np.flip(item, axis=axis).copy()
                ret_dict[key] = item
        
        sample['image'], sample['label'] = np.expand_dims(sample['image'], axis=0), np.expand_dims(sample['label'], axis=0)

        return {'image': sample['image'], 'label': sample['label']}


################# Gaussian Noise #####################
def augment_gaussian_noise(data_sample, noise_variance=(0, 0.1)):
    if noise_variance[0] == noise_variance[1]:
        variance = noise_variance[0]
    else:
        variance = random.uniform(noise_variance[0], noise_variance[1])
    data_sample = data_sample + np.random.normal(0.0, variance, size=data_sample.shape)
    return data_sample

class GaussianNoiseTransform(object):
    def __init__(self, noise_variance=(0, 0.1), p_per_sample=0.5):
        self.prob = p_per_sample
        self.noise_variance = noise_variance
    
    def __call__(self, sample):

        # pdb.set_trace()

        image = sample['image']
        label = sample['label']
        if np.random.uniform() < self.prob:
            image = augment_gaussian_noise(image, self.noise_variance)

        return {'image': image, 'label': label}


################# Gaussian Blur #####################
def get_range_val(value, rnd_type="uniform"):
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 2:
            if value[0] == value[1]:
                n_val = value[0]
            else:
                orig_type = type(value[0])
                if rnd_type == "uniform":
                    n_val = random.uniform(value[0], value[1])
                elif rnd_type == "normal":
                    n_val = random.normalvariate(value[0], value[1])
                n_val = orig_type(n_val)
        elif len(value) == 1:
            n_val = value[0]
        else:
            raise RuntimeError("value must be either a single value or a list/tuple of len 2")
        return n_val
    else:
        return value

def augment_gaussian_blur(data_sample: np.ndarray, sigma_range: Tuple[float, float], per_channel: bool = True,
                          p_per_channel: float = 1, different_sigma_per_axis: bool = False,
                          p_isotropic: float = 0) -> np.ndarray:
    if not per_channel:
        # Godzilla Had a Stroke Trying to Read This and F***ing Died
        # https://i.kym-cdn.com/entries/icons/original/000/034/623/Untitled-3.png
        sigma = get_range_val(sigma_range) if ((not different_sigma_per_axis) or
                                               ((np.random.uniform() < p_isotropic) and
                                                different_sigma_per_axis)) \
            else [get_range_val(sigma_range) for _ in data_sample.shape[1:]]
    else:
        sigma = None
    for c in range(data_sample.shape[0]):
        if np.random.uniform() <= p_per_channel:
            if per_channel:
                sigma = get_range_val(sigma_range) if ((not different_sigma_per_axis) or
                                                       ((np.random.uniform() < p_isotropic) and
                                                        different_sigma_per_axis)) \
                    else [get_range_val(sigma_range) for _ in data_sample.shape[1:]]
            data_sample[c] = gaussian_filter(data_sample[c], sigma, order=0)
    return data_sample

class GaussianBlurTransform(object):
    def __init__(self, blur_sigma: Tuple[float, float] = (1, 5), different_sigma_per_channel: bool = True,
                 different_sigma_per_axis: bool = False, p_isotropic: float = 0, p_per_channel: float = 1,
                 p_per_sample: float = 1,):
        """

        :param blur_sigma:
        :param different_sigma_per_axis: if True, anisotropic kernels are possible
        :param p_isotropic: only applies if different_sigma_per_axis=True, p_isotropic is the proportion of isotropic
        kernels, the rest gets random sigma per axis
        :param different_sigma_per_channel: whether to sample a sigma for each channel or all channels at once
        :param p_per_channel: probability of applying gaussian blur for each channel. Default = 1 (all channels are
        blurred with prob 1)
        """
        self.p_per_sample = p_per_sample
        self.different_sigma_per_channel = different_sigma_per_channel
        self.p_per_channel = p_per_channel
        self.blur_sigma = blur_sigma
        self.different_sigma_per_axis = different_sigma_per_axis
        self.p_isotropic = p_isotropic

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        for b in range(len(image)):
            if np.random.uniform() < self.p_per_sample:
                image[b] = augment_gaussian_blur(image[b], self.blur_sigma,
                                                self.different_sigma_per_channel,
                                                self.p_per_channel,
                                                different_sigma_per_axis=self.different_sigma_per_axis,
                                                p_isotropic=self.p_isotropic)
        return {'image': image, 'label': label}


################# Contrast Augmentation Transform #####################
def augment_contrast(data_sample: np.ndarray,
                     contrast_range: Union[Tuple[float, float], Callable[[], float]] = (0.75, 1.25),
                     preserve_range: bool = True,
                     per_channel: bool = True,
                     p_per_channel: float = 1) -> np.ndarray:
    if not per_channel:
        if callable(contrast_range):
            factor = contrast_range()
        else:
            if np.random.random() < 0.5 and contrast_range[0] < 1:
                factor = np.random.uniform(contrast_range[0], 1)
            else:
                factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])

        for c in range(data_sample.shape[0]):
            if np.random.uniform() < p_per_channel:
                mn = data_sample[c].mean()
                if preserve_range:
                    minm = data_sample[c].min()
                    maxm = data_sample[c].max()

                data_sample[c] = (data_sample[c] - mn) * factor + mn

                if preserve_range:
                    data_sample[c][data_sample[c] < minm] = minm
                    data_sample[c][data_sample[c] > maxm] = maxm
    else:
        for c in range(data_sample.shape[0]):
            if np.random.uniform() < p_per_channel:
                if callable(contrast_range):
                    factor = contrast_range()
                else:
                    if np.random.random() < 0.5 and contrast_range[0] < 1:
                        factor = np.random.uniform(contrast_range[0], 1)
                    else:
                        factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])

                mn = data_sample[c].mean()
                if preserve_range:
                    minm = data_sample[c].min()
                    maxm = data_sample[c].max()

                data_sample[c] = (data_sample[c] - mn) * factor + mn

                if preserve_range:
                    data_sample[c][data_sample[c] < minm] = minm
                    data_sample[c][data_sample[c] > maxm] = maxm
    return data_sample


class ContrastAugmentationTransform(object):
    def __init__(self, contrast_range=(0.75, 1.25), preserve_range=True, per_channel=True, p_per_sample=1., p_per_channel=1.):
        self.p_per_sample = p_per_sample
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        for b in range(len(image)):
            if np.random.uniform() < self.p_per_sample:
                image[b] = augment_contrast(image[b], 
                                            contrast_range=self.contrast_range,
                                            preserve_range=self.preserve_range, 
                                            per_channel=self.per_channel,
                                            p_per_channel=self.p_per_channel)
        return {'image': image, 'label': label}

################# Brightness Additive #####################
def augment_brightness_additive(data_sample, mu: float, sigma: float, per_channel: bool=True, p_per_channel: float=1.):
    if not per_channel:
        rnd_nb = np.random.normal(mu, sigma)
        for c in range(data_sample.shape[0]):
            if np.random.uniform() <= p_per_channel:
                data_sample[c] = data_sample[c] + rnd_nb
    
    else:
        for c in range(data_sample.shape[0]):
            if np.random.uniform() <= p_per_channel:
                rnd_nb = np.random.normal(mu, sigma)
                data_sample[c] = data_sample[c] + rnd_nb
    return data_sample

class BrightnessTransform(object):
    def __init__(self, mu, sigma, per_channel=True, p_per_sample=1., p_per_channel=1.):
        self.p_per_sample = p_per_sample
        self.mu = mu
        self.sigma = sigma
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel
    
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        for b in range(image.shape[0]):
            if np.random.uniform() < self.p_per_sample:
                image[b] = augment_brightness_additive(image[b], self.mu, self.sigma, self.per_channel,
                                                      p_per_channel=self.p_per_channel)

        return {'image': image, 'label': label}

################# Gamma  #####################
def augment_gamma(data_sample, gamma_range=(0.5, 2), invert_image=False, epsilon=1e-7, per_channel=False,
                  retain_stats: Union[bool, Callable[[], bool]] = False):
    if invert_image:
        data_sample = - data_sample

    if not per_channel:
        retain_stats_here = retain_stats() if callable(retain_stats) else retain_stats
        if retain_stats_here:
            mn = data_sample.mean()
            sd = data_sample.std()
        if np.random.random() < 0.5 and gamma_range[0] < 1:
            gamma = np.random.uniform(gamma_range[0], 1)
        else:
            gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
        minm = data_sample.min()
        rnge = data_sample.max() - minm
        data_sample = np.power(((data_sample - minm) / float(rnge + epsilon)), gamma) * rnge + minm
        if retain_stats_here:
            data_sample = data_sample - data_sample.mean()
            data_sample = data_sample / (data_sample.std() + 1e-8) * sd
            data_sample = data_sample + mn
    else:
        for c in range(data_sample.shape[0]):
            retain_stats_here = retain_stats() if callable(retain_stats) else retain_stats
            if retain_stats_here:
                mn = data_sample[c].mean()
                sd = data_sample[c].std()
            if np.random.random() < 0.5 and gamma_range[0] < 1:
                gamma = np.random.uniform(gamma_range[0], 1)
            else:
                gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
            minm = data_sample[c].min()
            rnge = data_sample[c].max() - minm
            data_sample[c] = np.power(((data_sample[c] - minm) / float(rnge + epsilon)), gamma) * float(rnge + epsilon) + minm
            if retain_stats_here:
                data_sample[c] = data_sample[c] - data_sample[c].mean()
                data_sample[c] = data_sample[c] / (data_sample[c].std() + 1e-8) * sd
                data_sample[c] = data_sample[c] + mn
    if invert_image:
        data_sample = - data_sample
    return data_sample

class GammaTransform(object):
    def __init__(self, gamma_range=(0.5, 2), invert_image=False, per_channel=False,
                 retain_stats: Union[bool, Callable[[], bool]] = False, p_per_sample=1):
        """
        Augments by changing 'gamma' of the image (same as gamma correction in photos or computer monitors

        :param gamma_range: range to sample gamma from. If one value is smaller than 1 and the other one is
        larger then half the samples will have gamma <1 and the other >1 (in the inverval that was specified).
        Tuple of float. If one value is < 1 and the other > 1 then half the images will be augmented with gamma values
        smaller than 1 and the other half with > 1
        :param invert_image: whether to invert the image before applying gamma augmentation
        :param per_channel:
        :param data_key:
        :param retain_stats: Gamma transformation will alter the mean and std of the data in the patch. If retain_stats=True,
        the data will be transformed to match the mean and standard deviation before gamma augmentation. retain_stats
        can also be callable (signature retain_stats() -> bool)
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.retain_stats = retain_stats
        self.per_channel = per_channel
        self.gamma_range = gamma_range
        self.invert_image = invert_image

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        for b in range(len(image)):
            if np.random.uniform() < self.p_per_sample:
                image[b] = augment_gamma(image[b], self.gamma_range,
                                                            self.invert_image,
                                                            per_channel=self.per_channel,
                                                            retain_stats=self.retain_stats)
        return sample

################# Mirror #####################
def augment_mirroring(sample_data, sample_seg=None, axes=(0, 1, 2)):

    # pdb.set_trace()
    if (len(sample_data.shape) != 3) and (len(sample_data.shape) != 4):
        raise Exception(
            "Invalid dimension for sample_data and sample_seg. sample_data and sample_seg should be either "
            "[channels, x, y] or [channels, x, y, z]")
    if 0 in axes and np.random.uniform() < 0.5:
        sample_data[:, :] = sample_data[:, ::-1]
        if sample_seg is not None:
            sample_seg[:, :] = sample_seg[:, ::-1]
    if 1 in axes and np.random.uniform() < 0.5:
        sample_data[:, :, :] = sample_data[:, :, ::-1]
        if sample_seg is not None:
            sample_seg[:, :, :] = sample_seg[:, :, ::-1]
    if 2 in axes and len(sample_data.shape) == 4:
        if np.random.uniform() < 0.5:
            sample_data[:, :, :, :] = sample_data[:, :, :, ::-1]
            if sample_seg is not None:
                sample_seg[:, :, :, :] = sample_seg[:, :, :, ::-1]
    return sample_data, sample_seg

class MirrorTransform(object):
    """ Randomly mirrors data along specified axes. Mirroring is evenly distributed. Probability of mirroring along
    each axis is 0.5

    Args:
        axes (tuple of int): axes along which to mirror

    """

    def __init__(self, axes=(0, 1, 2), p_per_sample=1):
        self.p_per_sample = p_per_sample
        self.axes = axes
        if max(axes) > 2:
            raise ValueError("MirrorTransform now takes the axes as the spatial dimensions. What previously was "
                             "axes=(2, 3, 4) to mirror along all spatial dimensions of a 5d tensor (b, c, x, y, z) "
                             "is now axes=(0, 1, 2). Please adapt your scripts accordingly.")

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        # data = data_dict.get(self.data_key)
        # seg = data_dict.get(self.label_key)

        for b in range(len(image)):
            if np.random.uniform() < self.p_per_sample:
                sample_seg = None
                if label is not None:
                    sample_seg = label[b]
                ret_val = augment_mirroring(image[b], sample_seg, axes=self.axes)
                image[b] = ret_val[0]
                if label is not None:
                    label[b] = ret_val[1]

        sample['image'] = image

        if label is not None:
            sample['label'] = label

        return {'image': image, 'label': label}
