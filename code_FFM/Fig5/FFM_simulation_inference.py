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
import socket
import pickle
import scipy
import scipy.ndimage
import matplotlib.pyplot as plt
from dataset import OCdata, OCFlowerdata
import socket
import struct
from tqdm import *
import matplotlib.pyplot as plt
import time
import random

def dB2inten(dB):
    inten = 1000*(10 ** (dB / 10))
    return inten

def calculate_optical_2_2_simu(x_0, awg, op_reader, input_curve, ref_inten_list, v_list_input):
    x1 = x_0[0]
    x2 = x_0[1]
    
    result_list = torch.zeros([2,])
    result_list[0] = x1 + x2
    result_list[1] = x1 + x2
    return result_list

def calculate_optical_2_add_simu(x_0, awg, op_reader, input_curve, ref_inten_list, v_list_input):
    return x_0[0] + x_0[1]

def forward_data_network(M_I, modu_tensor_list, awg, op_reader, input_curve, ref_inten_list, v_list_input):
    mini_size = 2
    M_I = M_I.view(-1)
    ### layer_1 ###
    PKf_1 = M_I
    M_X = torch.mul(M_I, modu_tensor_list[0].detach())
    M_X_out = torch.zeros((M_X.shape))
    for ii in range(M_X_out.shape[0] // mini_size):
        temp_x = M_X[ii*mini_size:(ii+1)*mini_size]
        temp_x_out = calculate_optical_2_2_simu(temp_x, awg, op_reader, input_curve, ref_inten_list, v_list_input)
        M_X_out[ii*mini_size:(ii+1)*mini_size] = temp_x_out
    
    ### layer_2 ###
    set_list = [0,2,1,3, 4,6,5,7, 8,10,9,11, 12,14,13,15]
    M_X_out = M_X_out[set_list]
    
    PKf_2 = M_X_out
    M_X = torch.mul(M_X_out, modu_tensor_list[1].detach())
    
    M_X_out = torch.zeros((M_X.shape))
    for ii in range(M_X_out.shape[0] // mini_size):
        temp_x = M_X[ii*mini_size:(ii+1)*mini_size]
        temp_x_out = calculate_optical_2_2_simu(temp_x, awg, op_reader, input_curve, ref_inten_list, v_list_input)
        M_X_out[ii*mini_size:(ii+1)*mini_size] = temp_x_out
    
    ### layer_3 ###
    set_list = [0,4,8,12, 1,5,9,13, 2,6,10,14, 3,7,11,15]
    M_X_out = M_X_out[set_list]
    
    PKf_3 = M_X_out
    M_X = torch.mul(M_X_out, modu_tensor_list[2].detach())
    M_X_out = torch.zeros((M_X.shape[0] // mini_size, ))
    for ii in range(M_X_out.shape[0]):
        temp_x = M_X[ii*mini_size:(ii+1)*mini_size]
        temp_x_out = calculate_optical_2_add_simu(temp_x, awg, op_reader, input_curve, ref_inten_list, v_list_input)
        M_X_out[ii] = temp_x_out
    
    ### layer_4 ###
    set_list = [0,2,1,3, 4,6,5,7]
    M_X_out = M_X_out[set_list]
    PKf_4 = M_X_out

    M_X = torch.mul(M_X_out, modu_tensor_list[3].detach())
    M_X_out = torch.zeros((M_X.shape))
    for ii in range(M_X_out.shape[0] // mini_size):
        temp_x = M_X[ii*mini_size:(ii+1)*mini_size]
        temp_x_out = calculate_optical_2_2_simu(temp_x, awg, op_reader, input_curve, ref_inten_list, v_list_input)
        M_X_out[ii*mini_size:(ii+1)*mini_size] = temp_x_out

    ### layer_5 ###
    set_list = [0,4, 1,5, 2,6, 3,7]
    M_X_out = M_X_out[set_list]
    PKf_5 = M_X_out

    M_X = torch.mul(M_X_out, modu_tensor_list[4].detach())
    M_X_out = torch.zeros((M_X.shape[0] // mini_size, ))
    for ii in range(M_X_out.shape[0]):
        temp_x = M_X[ii*mini_size:(ii+1)*mini_size]
        temp_x_out = calculate_optical_2_add_simu(temp_x, awg, op_reader, input_curve, ref_inten_list, v_list_input)
        M_X_out[ii] = temp_x_out

    Pkf_list = [PKf_1, PKf_2, PKf_3, PKf_4, PKf_5]
    return M_X_out, Pkf_list

##### ---------initialize---------- #####
## system
awg = 0
op_reader = 0

## base params
MASK_SIZE = 4
LR = 0.01
MAX_EPOCH = 300
DECAY = 0.99
BATCH_SIZE = 1

exp_name = 'debug'
out_put_dir = os.path.join('./output', exp_name)
debugidx = 0
debug_dir = os.path.join('./output', 'debug' + str(debugidx))
if not os.path.exists(debug_dir):
    os.makedirs(debug_dir)
if not os.path.exists(out_put_dir):
    os.makedirs(out_put_dir)

## modulation region
layer_size_list = [16,16,16,8,8]
modu_tensor_list = []
for ii in range(len(layer_size_list)):
    modu_tensor_init = np.load('SLM_phase_'+str(ii)+'_epoch_XXX.npy')
    modu_tensor = torch.nn.Parameter(torch.from_numpy(modu_tensor_init))
    modu_tensor_list.append(modu_tensor)

input_curve = []

v_list_input = np.linspace(0,5,101)
v_list_weight = np.linspace(0.5,1.8,66)

## dataset
use_gpu = False
gpu_args = {'num_workers': 12, 'pin_memory': True} if use_gpu else {}

train_dataset = OCFlowerdata(train=True)
test_dataset = OCFlowerdata(train=False)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=BATCH_SIZE, shuffle=False, **gpu_args)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=1, shuffle=False, **gpu_args)

test_loss_all = 0
test_correct_all = 0
ref_inten_list = 0

##### ---------inference--------- #####
for b, (data, target) in enumerate(train_loader):
    SLM_outinten = data[0,0]

    ###### forward data ######
    U_No, Pkf_list = forward_data_network(SLM_outinten, modu_tensor_list, awg, op_reader, input_curve, ref_inten_list, v_list_input)
    
    x_abs = torch.square(torch.abs(U_No.detach()))
    loss = F.cross_entropy(x_abs.unsqueeze(0), target)
    pred = x_abs.unsqueeze(0).argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    
    test_loss_all += loss.item()
    test_correct_all += correct

print(f"Test---Epoch {0}: loss {test_loss_all/len(train_loader):.6f} accuracy {test_correct_all/len(train_loader)}.")
