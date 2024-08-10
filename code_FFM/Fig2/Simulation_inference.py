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
import torchvision
import scipy
import scipy.ndimage
import PySpin
import matplotlib.pyplot as plt
from Datasets import *
from utils import showOn2ndDisplay
from utils import prop_simu
from func import *
from tensorboardX import SummaryWriter
from optics import *
import matlab.engine 
from ALP4 import *

##### detection region of output #####
def detector_region_10(x):
    return torch.cat((
        x[:, 92 : 132, 92 : 132].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 92 : 132, 186 : 226].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 92 : 132, 280 : 320].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 170 : 210, 92 : 132].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 170 : 210, 156 : 196].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 170 : 210, 218 : 258].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 170 : 210, 280 : 320].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 250 : 290, 92 : 132].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 250 : 290, 186 : 226].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 250 : 290, 280 : 320].mean(dim=(1, 2)).unsqueeze(-1)), dim=-1)

##### ---------initialize---------- #####
## base params
MASK_SIZE = 400
LR = 0.05
MAX_EPOCH = 1
DECAY = 0.99
BATCH_SIZE = 10

OBJECT_SIZE = 28
UP_SAMPLE = 14
MASK_SIZE = 400
RESIZE = int(OBJECT_SIZE * UP_SAMPLE)
PADDING_SZIE = (MASK_SIZE - RESIZE) // 2
img_transforms = transforms.Compose([
        transforms.Resize((RESIZE, RESIZE), interpolation=Image.BICUBIC),
        transforms.Pad((PADDING_SZIE,PADDING_SZIE)),
        transforms.ToTensor(),
    ])

exp_name = 'debug'
writer = SummaryWriter(os.path.join('./runs', exp_name))
out_put_dir = os.path.join('./output', exp_name)
debugidx = 0
debug_dir = os.path.join('./output', 'debug' + str(debugidx))
if not os.path.exists(debug_dir):
    os.makedirs(debug_dir)
if not os.path.exists(out_put_dir):
    os.makedirs(out_put_dir)

device = torch.device("cpu")
prop_layer = ASM_propagate(device, input_size=[800,800], lamda=532, pixelsize=8, z=388000)

## SLM phase
layer_num = 8
SLM_phase_list = []

load_dir = 'XXX'
load_idx = 9
for i in range(layer_num):
    SLM_phase_init = np.load(os.path.join(load_dir, 'SLM_phase'+str(i)+'_epoch_%03d.npy' % load_idx))
    SLM_phase_list.append(torch.nn.Parameter(torch.from_numpy(SLM_phase_init)))

SLM_shape = SLM_phase_list[0].shape

shift_phase_pi2 = 0.5*np.pi*torch.ones(SLM_shape)
shift_phase_pi = np.pi*torch.ones(SLM_shape)
repeat_num = 1

## dataset
use_gpu = False
gpu_args = {'num_workers': 12, 'pin_memory': True} if use_gpu else {}

test_dataset = NLOSdataFM(train=False)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=1, shuffle=False, **gpu_args)

## optimizer
optimizer = torch.optim.Adam(SLM_phase_list, lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

##### ---------inference--------- #####
epoch = 0

test_loss_all = 0
test_correct_all = 0

for b, (data, target) in enumerate(test_loader):
    SLM_outinten = data[0,0]

    for ii in range(layer_num):
        if ii == 0:
            input_amp = SLM_outinten
            input_phase = torch.zeros(SLM_outinten.shape) + SLM_phase_list[0]
            Pkf_temp = input_amp * torch.exp(1j * input_phase)
        else:
            input_amp = out_amp
            input_phase = out_phase + SLM_phase_list[ii]
            Pkf_temp = input_amp * torch.exp(1j * input_phase)
        
        out_gt = prop_simu(Pkf_temp.detach(), prop_layer)
        out_amp = torch.abs(out_gt)
        out_phase = out_gt.angle()
        
        if ii != layer_num - 1:
            out_amp = out_amp / out_amp.max()

    U_No = out_amp * torch.exp(1j * out_phase)
    x_abs = torch.square(torch.abs(U_No))
    
    rate = torch.sum(torch.mul(x_abs,target)) / torch.sum(torch.mul(x_abs,x_abs))
    x_abs = rate.detach()*x_abs

    loss = F.mse_loss(x_abs.unsqueeze(0), target)
    target_out = detector_region_10(target)
    target_label = target_out.argmax(dim=1, keepdim=True)
    output = detector_region_10(x_abs.unsqueeze(0))
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(target_label.view_as(pred)).sum().item()

    test_loss_all += loss.item()
    test_correct_all += correct

writer.add_scalar('test_loss', test_loss_all/len(test_loader), global_step=epoch)
writer.add_scalar('test_correct', test_correct_all/len(test_loader), global_step=epoch)
print(f"Test---Epoch {epoch}: loss {test_loss_all/len(test_loader):.6f}, accuracy {test_correct_all/len(test_loader)}.")
