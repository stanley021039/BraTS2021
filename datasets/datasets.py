import numpy as np
import os
import nibabel as nib
import cv2
import importlib
import torch
from torch.utils.data import Dataset


def pad_image3D(image, proportion, set='image'):
    h, w, d = image.shape
    if proportion == 1:
        return image
    else:
        img_stackz = np.zeros([round(h * proportion), round(w * proportion), image.shape[2]])
        for i in range(image.shape[2]):
            imgz = cv2.resize(image[:, :, i], dsize=None, fx=proportion, fy=proportion, interpolation=cv2.INTER_NEAREST)
            img_stackz[:, :, i] = imgz
        if set != 'image':
            img_stackz = np.array(img_stackz > 0.5, dtype='uint8')

    return img_stackz


def CenterCrop3Dextrain(image, label, label256, crop_sz):
    x, y, z = image.shape
    cx, cy = crop_sz
    startx = x // 2 - (cx // 2)
    starty = y // 2 - (cy // 2)
    return image[startx:startx + cx, starty:starty + cy, :], label[startx:startx + cx, starty:starty + cy, :], label256[
                                                                                                               startx:startx + cx,
                                                                                                               starty:starty + cy,
                                                                                                               :]


def RandomCrop3Dextrain(image, label, label256, crop_sz):
    x, y, z = image.shape
    cx, cy, cz = crop_sz
    while True:
        startx = np.random.randint(0, 145)
        starty = np.random.randint(0, y - cy + 1)
        startz = np.random.randint(0, z - cz + 1)
        x = label[startx:startx + cx, starty:starty + cy, startz:startz + cz]
        if np.max(x) == 1:
            return image[startx:startx + cx, starty:starty + cy, startz:startz + cz], x, label256[startx:startx + cx,
                                                                                         starty:starty + cy,
                                                                                         startz:startz + cz]


def CenterCrop3Dtrain(image, label, crop_sz):
    x, y, z = image.shape
    cx, cy, cz = crop_sz
    startx = x // 2 - (cx // 2)
    starty = y // 2 - (cy // 2)
    image = image[startx:startx + cx, starty:starty + cy, :]
    label = label[startx:startx + cx, starty:starty + cy, :]
    if z == cz:
        return image, label
    while True:
        startz = np.random.randint(0, z - cz + 1)
        if np.max(label[:, :, startz:startz + cz]) == 1:
            break
    return image[:, :, startz:startz + cz], label[:, :, startz:startz + cz]


def CenterCrop3Dvalid(image, label, crop_sz):
    x, y, z = image.shape
    cx, cy = crop_sz
    startx = x // 2 - (cx // 2)  # 56
    starty = y // 2 - (cy // 2)  # 56
    return image[startx:startx + cx, starty:starty + cy, :], label[startx:startx + cx, starty:starty + cy, :]


def CenterCrop3Dtest(image, crop_sz):
    x, y, z = image.shape
    cx, cy = crop_sz
    startx = x // 2 - (cx // 2)  # 56
    starty = y // 2 - (cy // 2)  # 56
    return image[startx:startx + cx, starty:starty + cy, :], startx, starty


class BrainMriDataset3D(Dataset):
    def __init__(self, root, split, img_proportion, set='train', augmentation=None):
        self.root = root
        self.split = split
        self.img_proportion = img_proportion
        self.set = set
        self.augmentation = augmentation
        self.path = os.path.join(self.root, self.split)
        self.filename = []
        self.LABELS = importlib.import_module('utils.labels_all')
        self.all_image = []
        self.all_label = []
        self.img = []
        self.label = []
        self.pt = ''

        all_filename = os.listdir(self.image_dir)
        for i, file in enumerate(all_filename):
            if nib.load(os.path.join(self.image_dir, file)).shape[0] == 512:
                self.filename.append(file)

    @property
    def image_dir(self):
        return os.path.join(self.path, 'image')

    @property
    def label_dir(self):
        return os.path.join(self.path, 'label')

    @property
    def pred_dir(self):
        return os.path.join(self.path, 'pred')

    def __getitem__(self, index):
        image = nib.load(os.path.join(self.image_dir, self.filename[index])).get_fdata()
        image = image / np.max(image)  # normalize vary by patient
        original_shape = image.shape

        if self.set == 'train':
            label = nib.load(os.path.join(self.label_dir, self.filename[index])).get_fdata()
            image, label = CenterCrop3Dtrain(image, label, (400, 400, 46))
            if self.augmentation is not None:
                image, label = self.augmentation(image, label)
            image = pad_image3D(image, self.img_proportion)
            label = pad_image3D(label, self.img_proportion, set='label')
            image = np.expand_dims(image, axis=0)
            return torch.from_numpy(image).float(), torch.from_numpy(label).long()

        elif self.set == 'valid':
            label = nib.load(os.path.join(self.label_dir, self.filename[index])).get_fdata()
            image, label = CenterCrop3Dvalid(image, label, (400, 400))
            image = pad_image3D(image, self.img_proportion)
            label = pad_image3D(label, self.img_proportion, set='label')
            image = np.expand_dims(image, axis=0)
            return torch.from_numpy(image).float(), torch.from_numpy(label).long()

        elif self.set == 'extrain':
            label256 = nib.load(os.path.join(self.pred_dir, self.filename[index])).get_fdata()
            label = nib.load(os.path.join(self.label_dir, self.filename[index])).get_fdata()
            image, label, label256 = CenterCrop3Dextrain(image, label, label256, (400, 400))
            image, label, label256 = RandomCrop3Dextrain(image, label, label256, (256, 256, 46))
            image = np.expand_dims(image, axis=0)
            label256 = np.expand_dims(label256, axis=0)
            return torch.from_numpy(image).float(), torch.from_numpy(label256).float(), torch.from_numpy(label).long()

        elif self.set == 'exvalid':
            label256 = nib.load(os.path.join(self.pred_dir, self.filename[index])).get_fdata()
            label = nib.load(os.path.join(self.label_dir, self.filename[index])).get_fdata()
            image = np.expand_dims(image, axis=0)
            label256 = np.expand_dims(label256, axis=0)
            return torch.from_numpy(image).float(), torch.from_numpy(label256).float(), torch.from_numpy(label).long()

        elif self.set == 'extest':
            label256 = nib.load(os.path.join(self.pred_dir, self.filename[index])).get_fdata()
            image = np.expand_dims(image, axis=0)
            label256 = np.expand_dims(label256, axis=0)
            print(label256.shape)
            return torch.from_numpy(image).float(), torch.from_numpy(label256).float(), self.filename[index]

        else:
            image, padx, pady = CenterCrop3Dtest(image, (400, 400))
            image = pad_image3D(image, self.img_proportion)
            padz = 0
            if image.shape[2] > 200:
                for j in range(image.shape[2]):
                    maxvalue = np.max(image[:, :, j])
                    if maxvalue == 0:
                        padz += 1
                    else:
                        break
                image = image[:, :, padz:min(padz + 190, image.shape[2])]
                print("ori_z =", original_shape[2])
                print("pad_z =", image.shape[2])
                print("start_z =", padz)
            image = np.expand_dims(image, axis=0)
            return torch.from_numpy(image).float(), self.filename[index], original_shape, (padx, pady, padz)

    def __len__(self):
        return len(self.filename)
