import os
from tqdm import tqdm
import cv2
import numpy as np
import nibabel as nib

def pad_image3D(image, targetsize, set='image'):
    w, h, c = targetsize
    if image.shape == targetsize:
        return image
    else:
        img_stackz = np.zeros([w, h, image.shape[2]])
        for i in range(image.shape[2]):
            imgz = cv2.resize(image[:, :, i], (w, h), cv2.INTER_NEAREST)
            img_stackz[:, :, i] = imgz
        img_stacky = np.zeros([w, h, c])
        for i in range(h):
            imgy = cv2.resize(img_stackz[:, i, :], (c, w), cv2.INTER_NEAREST)
            imgy = np.expand_dims(imgy, axis=1)
            img_stacky[:, i, :] = imgy[:, 0, :]
        if set != 'image':
            img_stacky = np.array(img_stacky > 0.5, dtype='uint8')

    return img_stacky

data_path1 = r'0717 Release of training data/image'
data_path2 = r'0717 Release of training data/label'
data_path3 = r'0724 Release of validation/image'
data_path4 = r'validation/image'
data_path5 = r'validation/label'
img_size = (200, 200, 92)
'''
path = data_path1
all_filename = os.listdir(path)

all_image = np.zeros([img_size[0], img_size[1], img_size[2], len(all_filename)], dtype='int16')
for i, file in enumerate(tqdm(all_filename)):
    image = nib.load(os.path.join(path, file)).get_fdata()
    reimage = pad_image3D(image, img_size, set='image')
    all_image[:, :, :, i] = reimage
print(all_image.shape)
print(type(all_image[1, 1, 1, 1]))

np.save('0717 Release of training data/all_image_for3D', all_image)
'''
load_image = np.load(r'0717 Release of training data/all_image_for3D.npy')
print(load_image.shape)
print(np.max(load_image))

'''
path = data_path2
all_filename = os.listdir(path)
all_image = np.zeros([img_size[0], img_size[1], img_size[2], len(all_filename)], dtype='int8')
for i, file in enumerate(tqdm(all_filename)):
    image = nib.load(os.path.join(path, file)).get_fdata()
    reimage = pad_image3D(image, img_size, set='label')
    all_image[:, :, :, i] = reimage
print(all_image.shape)
print(type(all_image[1, 1, 1, 1]))

np.save('0717 Release of training data/all_label_for3D', all_image)

load_image = np.load(r'0717 Release of training data/all_label_for3D.npy')
print(load_image.shape)
for i in range(load_image.shape[3]):
    print(np.max(load_image[:, :, :, i]))
    if i > 10:
        break
'''
'''
path = data_path4
all_filename = os.listdir(path)
all_image = np.zeros([img_size[0], img_size[1], img_size[2], len(all_filename)], dtype='int16')
for i, file in enumerate(tqdm(all_filename)):
    image = nib.load(os.path.join(path, file)).get_fdata()
    reimage = pad_image3D(image, img_size, set='image')
    all_image[:, :, :, i] = reimage
print(all_image.shape)
print(type(all_image[1, 1, 1, 1]))

np.save('validation/all_image_for3D', all_image)
'''
load_image = np.load(r'validation/all_image_for3D.npy')
print(load_image.shape)
print(np.max(load_image))

'''
path = data_path5
all_filename = os.listdir(path)
all_image = np.zeros([img_size[0], img_size[1], img_size[2], len(all_filename)], dtype='int8')
for i, file in enumerate(tqdm(all_filename)):
    image = nib.load(os.path.join(path, file)).get_fdata()
    reimage = pad_image3D(image, img_size, set='label')
    all_image[:, :, :, i] = reimage
print(all_image.shape)
print(type(all_image[1, 1, 1, 1]))

np.save('validation/all_label_for3D', all_image)

load_image = np.load(r'validation/all_label_for3D.npy')
print(load_image.shape)
for i in range(load_image.shape[3]):
    print(np.max(load_image[:, :, :, i]))
    if i > 10:
        break
'''



