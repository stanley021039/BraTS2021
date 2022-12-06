import nibabel as nib
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--nii_path", default='datasets/example/image/2JA4PBWY.nii.gz', type=str, help="data path")
parser.add_argument("--view", default='z', type=str, help="x/y/z")
args = parser.parse_args()

def pad_image3D(image, img_sz, set='image'):
    h, w, d = image.shape
    if img_sz == image.shape:
        return image
    else:
        img_stackz = np.zeros([img_sz[0], img_sz[1],  image.shape[2]])
        for i in range(image.shape[2]):
            imgz = cv2.resize(image[:, :, i], dsize=img_sz, interpolation=cv2.INTER_NEAREST)
            img_stackz[:, :, i] = imgz
        if set != 'image':
            img_stackz = np.array(img_stackz > 0.5, dtype='uint8')

    return img_stackz


image = nib.load(args.nii_path).get_fdata()
if args.view == 'x':
    for i in range(image.shape[0]):
        plt.imshow(image[i, :, :], 'gray')
        plt.show()
elif args.view == 'y':
    for i in range(image.shape[1]):
        plt.imshow(image[:, i, :], 'gray')
        plt.show()
else:
    for i in range(image.shape[2]):
        plt.imshow(image[:, i, i], 'gray')
        plt.show()


