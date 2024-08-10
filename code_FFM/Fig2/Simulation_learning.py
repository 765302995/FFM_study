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
from Datasets import NLOSdata2pSingle, NLOSdata2p, NLOSdata2iSingle, NLOSdata2i, NLOSdata2i_4, NLOSdata2iSingle2, NLOSdatacifar2iSingle, NLOSdata2i2, NLOSdatacifar2, NLOSdatacifar3
from utils import showOn2ndDisplay
from utils import prop_simu
from func import *
from tensorboardX import SummaryWriter
from optics import *
import matlab.engine 
from ALP4 import *

##### GT create #####
def create_shift(idx = 1, mask_size = 400):
    range_list = np.array([[92,132,92,132],
                [92,132, 186,226],
            [92,132, 280,320],
            [170,210, 92,132],
            [170,210, 156,196],
            [170,210, 218,258],
            [170,210, 280,320],
            [250,290, 92,132],
            [250,290, 186,226],
            [250,290, 280,320]])

    mask2 = np.zeros((mask_size, mask_size), dtype=np.float32)
    mask2[range_list[idx,0]:range_list[idx,1], range_list[idx,2]:range_list[idx,3]] = 1.0

    return mask2

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
MAX_EPOCH = 500
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
debugidx = 12
debug_dir = os.path.join('./output', 'debug' + str(debugidx))
if not os.path.exists(debug_dir):
    os.makedirs(debug_dir)
if not os.path.exists(out_put_dir):
    os.makedirs(out_put_dir)

device = torch.device("cpu")
prop_layer = ASM_propagate(device, input_size=[800,800], lamda=532, pixelsize=8, z=388000)

## SLM phase
layer_num = 4
SLM_phase_list = []
for i in range(layer_num):
    phase_init = 2 * np.pi * np.random.random(size=(MASK_SIZE, MASK_SIZE)).astype('float32')
    grainSize = 4
    phase_init_blur = scipy.ndimage.filters.gaussian_filter(phase_init, grainSize/2)
    SLM_phase_list.append(torch.nn.Parameter(torch.from_numpy(phase_init_blur)))

SLM_shape = SLM_phase_list[0].shape

shift_phase_pi2 = 0.5*np.pi*torch.ones(SLM_shape)
shift_phase_pi = np.pi*torch.ones(SLM_shape)
repeat_num = 1

## dataset
use_gpu = False
gpu_args = {'num_workers': 12, 'pin_memory': True} if use_gpu else {}

train_dataset = torchvision.datasets.MNIST(root='./dataset_method_1', train=True, transform=img_transforms, download=True)
test_dataset = NLOSdata2i2(train=False)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=BATCH_SIZE, shuffle=True, **gpu_args)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=1, shuffle=False, **gpu_args)

## optimizer
optimizer = torch.optim.Adam(SLM_phase_list, lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

##### ---------train--------- #####
for epoch in range(MAX_EPOCH):
    train_loss_all = 0
    train_correct_all = 0

    for b, (data, target) in enumerate(train_loader):
        grad_all_list = []
        for ii in range(layer_num):
            grad_all_list.append(0)
        
        train_loss_all_batch = 0 
        data_batch, target_batch = data.to(torch.float32), target.to(torch.float32)

        for i in range(data_batch.shape[0]):
            optimizer.zero_grad()

            target_temp = target_batch[i]
            target_temp = torch.tensor(create_shift(idx=target_temp.to(torch.long)))

            SLM_outinten = data_batch[i,0]

            for ii in range(layer_num):
                if ii == 0:
                    input_amp = SLM_outinten
                    input_phase = torch.zeros(SLM_outinten.shape) + SLM_phase_list[0]
                    Pkf_temp = input_amp * torch.exp(1j * input_phase)
                else:
                    input_amp = out_amp
                    input_phase = out_phase + SLM_phase_list[ii]
                    Pkf_temp = input_amp * torch.exp(1j * input_phase)
                
                out_gt = prop_simu(Pkf_temp, prop_layer)
                out_amp = torch.abs(out_gt)
                out_phase = out_gt.angle()
            
            U_No = out_amp * torch.exp(1j * out_phase)
            x_abs = torch.square(torch.abs(U_No))
            
            loss = F.mse_loss(x_abs.unsqueeze(0), target_temp.unsqueeze(0))
            target_out = detector_region_10(target_temp.unsqueeze(0))
            target_label = target_out.argmax(dim=1, keepdim=True)
            output = detector_region_10(x_abs.unsqueeze(0))
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target_label.view_as(pred)).sum().item()

            train_loss_all += loss.item()
            train_loss_all_batch += loss.item()
            train_correct_all += correct
            loss.backward()

            for ii in range(layer_num):
                grad_all_list[ii] += SLM_phase_list[ii].grad
        
        grad_1_mean_list = []
        for ii in range(layer_num):
            grad_1_mean_list.append(grad_all_list[ii] / target_batch.shape[0])
        
        optimizer.zero_grad()
        for ii in range(layer_num):
            SLM_phase_list[ii].grad = grad_1_mean_list[ii]
        optimizer.step()
        print(f"Train---batch {b}: loss {train_loss_all_batch/target_batch.shape[0]:.6f}.")

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

    loss_epoch = train_loss_all/len(train_loader)/BATCH_SIZE
    correct_epoch = train_correct_all/len(train_loader)/BATCH_SIZE

    writer.add_scalar('train_loss', loss_epoch, global_step=epoch)
    writer.add_scalar('correct', correct_epoch, global_step=epoch)
    print(f"Train---Epoch {epoch}: loss {loss_epoch:.6f}, accuracy {correct_epoch}.")

    if epoch % 1 == 0:
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
