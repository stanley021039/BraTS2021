import sys
import time
import json
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from math import cos, pi
from tqdm import tqdm
from utils.core_utils import *
from torch.nn.modules.loss import _Loss
from datasets.datasets import BrainMriDataset3D
from utils.metric import runningScore, averageMeter
from torch.utils.data import DataLoader
sys.path.append(r'savemodel3D0924')
from savemodel3D0924 import *

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(1)

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N      = target.size(0)
        smooth = 1e-8

        input_flat   = input.view(N, -1)
        target_flat  = target.view(N, -1)
        intersection = input_flat * target_flat
        loss = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N
        return loss

def count_parameters(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("total parameters =", total_num)
    print("trainable parameters =", trainable_num)

# define Decay policy and adjust learning rate
def cosine_decay(ini_lr, global_step, decay_steps, optim, alpha=0.0):
    global_step = min(global_step, decay_steps)
    cosine_decay = 0.5 * (1 + cos(pi * global_step / decay_steps))
    decayed_coff = (1 - alpha) * cosine_decay + alpha
    decayed_learning_rate = ini_lr * decayed_coff
    for param_group in optim.param_groups:
        param_group['lr'] = decayed_learning_rate

def auto_decay(init_lr, train_loss, optim):
    decrease_count = 0
    min_loss = train_loss[0]
    for i in range(1, 19):  # 1~19
        if train_loss[i] < min_loss:
            decrease_count += i
            min_loss = train_loss[i]
    decay_proportion = 1.02 - (190 - decrease_count) * (0.12 / 190.)  # 0.9~1.05
    lr = init_lr * decay_proportion
    for param_group in optim.param_groups:
        param_group['lr'] = lr


# train
def train(epoch, trainloader, Net, optimizer, loss_fn, Meter, log_file):  # writer
    Net.train()
    l, m = 0, 0
    for i, (image, label256, target) in enumerate(tqdm(trainloader)):
        image, label256, target = image.to(device), label256.to(device), target.to(device)
        target = torch.unsqueeze(target, 1).float()
        optimizer.zero_grad()
        pred = Net(image, label256)
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()
        training_loss = loss.item()
        # pred = torch.max(pred, 1)[1]  # for class=2
        pred = torch.ge(pred, 0.5)      # for class=1
        Meter['metric'].update(target.data.cpu().numpy(), pred.data.cpu().numpy())
        Meter['loss'].update(training_loss, image.size()[0])
        score, class_iou = Meter['metric'].get_scores()
        loss_avg = Meter['loss'].avg
        l, m = loss_avg, score['MeanDice']

        verbose_step = len(trainloader) // 10
        if (i+1) % verbose_step == 0 and (i+1) != len(trainloader):
            print('Epoch %3d : %10s loss_avg: %f OverallAcc: %f mIoU: %f MeanDice: %f'
                  % (epoch, 'training', loss_avg, score['OverallAcc'], score['mIoU'], score['MeanDice']))
        elif (i+1) == len(trainloader):
            print_with_write(log_file, 'Epoch %3d : %10s loss_avg: %f OverallAcc: %f mIoU: %f MeanDice: %f'
                             % (epoch, 'training', loss_avg, score['OverallAcc'], score['mIoU'], score['MeanDice']))
            l, m = loss_avg, score['MeanDice']
    return l, m


# valid
def val(epoch, validationloader, Net, loss_fn, Meter, log_file):
    Net.eval()
    with torch.no_grad():
        for i, (image, label256, target) in enumerate(tqdm(validationloader)):
            timeStart = time.time()
            image, label256, target = image.to(device), label256.to(device), target.to(device)
            pred = Net(image, label256)
            timeEnd = time.time()
            validation_loss = loss_fn(pred,target)
            # pred = torch.max(pred, 1)[1]
            pred = torch.ge(pred, 0.5)
            Meter['metric'].update(target.data.cpu().numpy(), pred.data.cpu().numpy())
            Meter['loss'].update(validation_loss, image.size()[0])
            Meter['time'].update(timeEnd-timeStart,1)

    score, class_iou = Meter['metric'].get_scores()
    loss_avg = Meter['loss'].avg
    time_avg = Meter['time'].avg
    print_with_write(log_file, 'Epoch %3d : %10s loss_avg: %f OverallAcc: %f mIoU: %f MeanDice: %f time: %f \n'
                     % (epoch, 'validation', loss_avg, score['OverallAcc'], score['mIoU'], score['MeanDice'], time_avg))
    return loss_avg, score['MeanDice']

# Scores(/utils/metric.py)
training_meter   = {'metric':runningScore(n_classes=2),'loss':averageMeter(),'time':averageMeter()}
validation_meter = {'metric':runningScore(n_classes=2),'loss':averageMeter(),'time':averageMeter()}

# Set DataLoader(/datasets/datasets.py)
train_batch = 8
valid_batch = 1

TrainingDataset   = BrainMriDataset3D('datasets', '0717 Release of training data', img_proportion=1, set='extrain')
ValidationDataset = BrainMriDataset3D('datasets', 'validation', img_proportion=1, set='exvalid')
TrainingLoader    = DataLoader(TrainingDataset, batch_size=train_batch, shuffle=True, num_workers=16)
ValidationLoader  = DataLoader(ValidationDataset, batch_size=valid_batch, shuffle=True, num_workers=4)

# Super variables
Epoch = 200
learning_rate = 0.003  # default = 0.001

# Net construct
Net = ExNet(num_classes=1).to(device)  # class=1 for BCE or Dice, 2 for Focal
# Net = nn.DataParallel(UNet(num_classes=1), device_ids=[0, 1])
Net.train()
count_parameters(ExNet(num_classes=1))

# Optimizer
optimizer = optim.SGD(Net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
# optimizer = optim.Adam(Net.parameters(), lr=learning_rate)

# Loss function
# loss_fn = FocalLoss()
# loss_fn = nn.BCEWithLogitsLoss()
loss_fn = DiceLoss()

valid_loss  = np.ones([1])
valid_score = np.ones([1])
train_loss  = np.ones([1])
train_score = np.ones([1])
lr          = np.ones([1])

log_file = open(r'savemodel3D0924/scorex.txt', 'w')
model_name = 'exnet'
Max_score = 0.
Max_epoch = 0

# Main
for epoch in range(1, Epoch):
    print("Epoch: %d learning_rate: %f Max_score: %d, %f" % (epoch, optimizer.state_dict()['param_groups'][0]['lr'], Max_epoch, Max_score))
    # reset Meters
    for _, v in training_meter.items():
        v.reset()
    for _, v in validation_meter.items():
        v.reset()
    # train
    t_loss, t_score = train(epoch, TrainingLoader, Net, optimizer, loss_fn, training_meter, log_file)
    # valid
    v_loss, v_score = val(epoch, ValidationLoader, Net, loss_fn, validation_meter, log_file)
    # save model
    if v_score > Max_score:
        Max_epoch = epoch
        Max_score = v_score
    if v_score > Max_score * 0.98:
        save_model({'epoch': epoch-1,
                    'model_state_dict': Net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    },
                   'savemodel3D0924', model_name)

    # Variable visualize
    valid_loss = np.append(valid_loss, v_loss.cpu())
    valid_score = np.append(valid_score, v_score)
    train_loss = np.append(train_loss, t_loss)
    train_score = np.append(train_score, t_score)
    lr = np.append(lr, optimizer.state_dict()['param_groups'][0]['lr'])
    # lr adjust
    if epoch > 20:
        cosine_decay(optimizer.state_dict()['param_groups'][0]['lr'], epoch, Epoch, optimizer)

    np.savez(r'savemodel3D0924/super_v/exsuper_v.npz', vl=valid_loss, vs=valid_score, tl=train_loss, ts=train_score, lr=lr)

log_file.close()
