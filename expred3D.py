import sys
import time
import cv2
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils.core_utils import *
from torch import nn
from tqdm import tqdm
from datasets.datasets import BrainMriDataset3D
from torch.utils.data import DataLoader

model_path = r'savemodel3D0924'
model_epoch = 63
sys.path.append(model_path)
from savemodel3D0924 import *

load_path = os.path.join(model_path, 'unet_'+str(model_epoch)+'.pth')
savenii_path = os.path.join('datasets', 'test_image', 'pred')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(1)

def pad_image3D(image, img_size, set='image'):
    h, w, d = image.shape
    if [img_size[0], img_size[1]] == img_size:
        return image
    else:
        img_stackz = np.zeros([img_size[0], img_size[1], image.shape[2]])
        for i in range(image.shape[2]):
            imgz = cv2.resize(image[:, :, i], dsize=img_size, interpolation=cv2.INTER_NEAREST)
            img_stackz[:, :, i] = imgz
        if set != 'image':
            img_stackz = np.array(img_stackz > 0.5, dtype='uint8')

    return img_stackz

def save_nii(img, path):
    img = nib.Nifti1Image(img, np.eye(4))
    img.set_data_dtype(np.uint8)
    nib.save(img, path)

def test(TestingLoader, Net):
    Net.eval()
    timeStart = time.time()
    with torch.no_grad():
        for i, (data, filename, original_sz, pad_pos) in enumerate(tqdm(TestingLoader)):
            print(i, filename[0])
            data = data.to(device)
            pred = Net(data)
            m    = nn.ReLU()
            pred = m(pred)
            pred = pred.data.cpu().numpy().astype('float64')
            pred = pad_image3D(pred[0, 0, :, :, :], (400, 400))
            if pad_pos != 0:
                base = np.zeros(original_sz, dtype='float64')
                base[pad_pos[0]:pad_pos[0]+pred.shape[0], pad_pos[1]:pad_pos[1]+pred.shape[1], pad_pos[2]:pad_pos[2]+pred.shape[2]] = pred
                pred = base
            save_nii(pred, os.path.join(savenii_path, filename[0]))

    timeEnd = time.time()
    print("Pack", i+1, "data within time_used:", timeEnd-timeStart)

TestingDataset = BrainMriDataset3D('datasets', 'test_image', img_proportion=0.5, set='test')
TestingLoader  = DataLoader(TestingDataset, batch_size=1, shuffle=False, num_workers=4)

Net = U_net3D_Resnet(num_classes=1)
# Net.load_state_dict(torch.load(load_path, map_location='cuda:0')['model_state_dict'])
Net.load_state_dict(torch.load(load_path)['model_state_dict'])
Net = Net.to(device)

test(TestingLoader, Net)
