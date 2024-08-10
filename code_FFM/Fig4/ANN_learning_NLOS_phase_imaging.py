from PIL import Image
import numpy as np
from ctypes import *
import ctypes
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
import time
import pdb

import scipy
import scipy.ndimage
import matplotlib.pyplot as plt
from Datasets import *
from tensorboardX import SummaryWriter
from torchvision import models
from ptflops import get_model_complexity_info
from unet_model import UNet
from torchstat import stat

##### ANN network #####
class Classifer_net(nn.Module):
    def __init__(self,feature_extract=True,num_classes=10):
        super(Classifer_net, self).__init__()

        self.features = models.resnet18(pretrained=True)
        self.features.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.features.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)

    def forward(self, x):
        out = self.features(x)
        return out

##### ---------initialize---------- #####
## base params
MASK_SIZE = 400
LR = 0.002
MAX_EPOCH = 500
DECAY = 0.99
BATCH_SIZE = 10
BATCH_SIZE_TEST = 10

exp_name = 'debug'
writer = SummaryWriter(os.path.join('./runs', exp_name))
out_put_dir = os.path.join('./output', exp_name)
debugidx = 0
debug_dir = os.path.join('./output', 'debug' + str(debugidx))
if not os.path.exists(debug_dir):
    os.makedirs(debug_dir)
if not os.path.exists(out_put_dir):
    os.makedirs(out_put_dir)

use_gpu = True
device = torch.device(f"cuda:{0}" if use_gpu else "cpu")

## dataset
use_gpu = True
gpu_args = {'num_workers': 12, 'pin_memory': True} if use_gpu else {}

train_dataset = NLOSdata2i2_cap2(train=True)
test_dataset = NLOSdata2i2_cap2(train=False)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=BATCH_SIZE, shuffle=True, **gpu_args)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=BATCH_SIZE_TEST, shuffle=False, **gpu_args)

model = Classifer_net().to(device)

## optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

rate_NOISE = 0.5
##### ---------train--------- #####
for epoch in range(MAX_EPOCH):
    train_loss_all = 0
    train_correct_all = 0
    model.train()
    for b, (data, target) in enumerate(train_loader):
        data = data.to(torch.float32).to(device)
        target = target.to(device)

        output = model(data)

        idx = np.random.randint(0,data.shape[0])
        x_abs = output[idx,0].detach().cpu()

        loss = F.cross_entropy(output, target)

        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()

        train_loss_all += loss
        train_correct_all += correct

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Train---batch {b}: loss {loss.item():.6f}.", end='\r')

    writer.add_scalar('train_loss', train_loss_all/len(train_loader), global_step=epoch)
    writer.add_scalar('correct', train_correct_all/len(train_loader)/BATCH_SIZE, global_step=epoch)
    print(f"Train---Epoch {epoch}: loss {train_loss_all/len(train_loader):.6f}.accuracy {train_correct_all/len(train_loader)/BATCH_SIZE}.")

##### ---------test--------- #####
    if epoch % 1 == 0:
        test_loss_all = 0
        test_correct_all = 0
        model.eval()
        with torch.no_grad():
            for b, (data, target) in enumerate(test_loader):
                data = data.to(torch.float32).to(device)
                target = target.to(device)

                ref_value = data.mean().detach().cpu()
                noise = torch.from_numpy(np.random.normal(0,ref_value*rate_NOISE,data.shape)).to(torch.float32).to(device)
                data = data + noise
                idx = torch.where(data<0)
                data[idx] = 0

                output = model(data)
                
                loss = F.cross_entropy(output, target)
                
                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred)).sum().item()

                test_loss_all += loss
                test_correct_all += correct

            writer.add_scalar('test_loss', test_loss_all/len(test_loader), global_step=epoch)
            writer.add_scalar('test_correct', test_correct_all/len(test_loader)/BATCH_SIZE_TEST, global_step=epoch)
            print(f"Test---Epoch {epoch}: loss {test_loss_all/len(test_loader):.6f}, accuracy {test_correct_all/len(test_loader)/BATCH_SIZE_TEST}.")