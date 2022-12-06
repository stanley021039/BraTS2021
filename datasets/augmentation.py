# from imgaug import augmenters as iaa
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from PIL import Image
from scipy import ndimage


def get_composed_augmentations(aug_dict):
    augmentations = []
    for aug_key, aug_param in aug_dict.items():
        augmentations.append(key2aug[aug_key](**aug_param))
    return Compose(augmentations)


class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, img, label):
        #if isinstance(img, np.ndarray):
        #    img = Image.fromarray(img.astype('uint8'), mode='L')
        #    label = Image.fromarray(label.astype('uint8'), mode='L')
        for a in self.augmentations:
            img, label = a(img, label)
        img, label = np.array(img), np.array(label, dtype=np.uint8)
        return img, label


class CenterCrop3D(object):
    def __init__(self, crop_sz):
        self.crop_sz  = crop_sz

    def __call__(self, image, label):
        x,  y,  z  = image.shape
        cx, cy = self.crop_sz
        startx = x // 2 - (cx // 2)
        starty = y // 2 - (cy // 2)
        return image[startx:startx+cx, starty:starty+cy, :], label[startx:startx+cx, starty:starty+cy, :]


class RandomHorizontalFlip(object):
    def __init__(self, probability):
        self.probability = probability

    def __call__(self, img, label):
        if random.random() < self.probability:
            img = TF.hflip(img)
            label = TF.hflip(label)
        return img, label

"""
class RandomCrop(object):
    def __init__(self, img_size, padding):
        self.img_size = img_size
        self.padding = padding

    def __call__(self, img, label):
        if self.padding > 0:
            img = TF.pad(img, self.padding)
            label = TF.pad(label, self.padding, fill=0)

        w, h, c = img.size
        th, tw = self.img_size
        if w == tw and h == th:
            return img, label

        new_x = random.randint(0, h - th)
        new_y = random.randint(0, w - tw)
        return (
            TF.crop(img, new_x, new_y, th, tw),
            TF.crop(label,  new_x, new_y, th, tw),
        )


class RandomCrop3D(object):
    def __init__(self, img_size, crop_sz):
        h, w, d = img_size  # c,
        assert [h, w, d] > crop_sz
        self.img_sz = tuple((h, w, d))
        self.crop_sz = tuple(crop_sz)

    def __call__(self, image, label):
        slice_hwd = [self._get_slice(i, k) for i, k in zip(self.img_sz, self.crop_sz)]
        return self._crop(image, *slice_hwd), self._crop(label, *slice_hwd)

    @staticmethod
    def _get_slice(sz, crop_sz):
        try:
            lower_bound = torch.randint(sz - crop_sz, (1,)).item()
            return lower_bound, lower_bound + crop_sz
        except:
            return (None, None)

    @staticmethod
    def _crop(x, slice_h, slice_w, slice_d):
        return x[slice_h[0]:slice_h[1], slice_w[0]:slice_w[1], slice_d[0]:slice_d[1]]
"""

class RandomCrop3D(object):
    def __init__(self, crop_sz):
        self.crop_sz = crop_sz

    def __call__(self, image, label):
        h , w , d  = image.shape
        ch, cw, cd = self.crop_sz
        starth = np.random.randint(0, h-self.crop_sz[0]+1)
        startw = np.random.randint(0, w-self.crop_sz[1]+1)
        startd = np.random.randint(0, d-self.crop_sz[2]+1)
        return image[starth:starth+ch, startw:startw+cw, startd:startd+cd], label[starth:starth+ch, startw:startw+cw, startd:startd+cd]

class GaussianBlur(object):
    def __init__(self, sigma=0.5, probability=0.5):
        self.sigma = sigma
        self.probability = probability

    def __call__(self, img, label):
        if random.random() < self.probability:
            img = ndimage.gaussian_filter(img, self.sigma)
        return img, label

key2aug = {
            'GaussianBlur': GaussianBlur
            }
