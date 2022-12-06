import time
import cv2
import numpy as np
import nibabel as nib
from utils.core_utils import *
from torch import nn
from tqdm import tqdm
from datasets.datasets import BrainMriDataset3D
from torch.utils.data import DataLoader
import argparse
from savemodel3D0924.U_net3D_Resnet import U_net3D_Resnet

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default='example', type=str, help="data path")
parser.add_argument("--load_epoch", default=64, type=int, help="data path")
parser.add_argument("--save_dir", default='savemodel3D0924', type=str, help="data path")
args = parser.parse_args()

load_path = os.path.join(args.save_dir, 'model_state_dict', 'unet_' + str(args.load_epoch) + '.pth')
savenii_path = os.path.join(args.save_dir, 'pred')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)


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
            m = nn.Sigmoid()
            pred = m(pred)
            pred = pred.data.cpu().numpy().astype('float64')
            pred = pad_image3D(pred[0, 0, :, :, :], (400, 400), set='label')
            if pad_pos != 0:
                base = np.zeros(original_sz, dtype='uint8')
                base[pad_pos[0]:pad_pos[0] + pred.shape[0], pad_pos[1]:pad_pos[1] + pred.shape[1],
                pad_pos[2]:pad_pos[2] + pred.shape[2]] = pred
                pred = base
            # print(np.bincount(pred.flatten()))
            save_nii(pred, os.path.join(savenii_path, filename[0]))

    timeEnd = time.time()
    print("Pack", i + 1, "data within time_used:", timeEnd - timeStart)


TestingDataset = BrainMriDataset3D('datasets', args.data_dir, img_proportion=0.5, set='example')
TestingLoader = DataLoader(TestingDataset, batch_size=1, shuffle=False, num_workers=0)

Net = U_net3D_Resnet(num_classes=1)
Net.load_state_dict(torch.load(load_path, map_location='cuda:0')['model_state_dict'])
# Net.load_state_dict(torch.load(load_path)['model_state_dict'])
Net = Net.to(device)

test(TestingLoader, Net)
