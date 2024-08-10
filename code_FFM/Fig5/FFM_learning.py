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
from dataset import *
import socket
import struct
from tqdm import *
import matplotlib.pyplot as plt
from power_readout import *
from set_voltage_512 import *
from tensorboardX import SummaryWriter
import matlab.engine 
from ALP4 import *
import time
from sklearn.metrics import confusion_matrix
from utils import *
from awg import *
from channnel_shift import ChannelShifter

##### power readout #####
def dB2inten(dB):
    inten = 1000*(10 ** (dB / 10))
    return inten

##### add #####
def calculate_optical_2_add(x_0, awg, op_reader, input_curve, ref_inten_list, v_list_input):
    return x_0[0] + x_0[1]

##### forward data propagation #####
def forward_data_network(M_I, modu_tensor_list, awg, op_reader, input_curve, ref_inten_list, v_list_input):
    mini_size = 2
    M_I = M_I.view(-1)
    ### layer_1 ###
    Pkd_1 = M_I
    M_X = torch.mul(M_I, modu_tensor_list[0].detach())
    M_X_out = torch.zeros((M_X.shape))
    for ii in range(M_X_out.shape[0] // mini_size):
        temp_x = M_X[ii*mini_size:(ii+1)*mini_size]
        temp_x_out = calculate_optical_2_2(temp_x, awg, op_reader, input_curve, ref_inten_list, v_list_input)
        M_X_out[ii*mini_size:(ii+1)*mini_size] = temp_x_out

    ### layer_2 ###
    set_list = [0,2,1,3, 4,6,5,7, 8,10,9,11, 12,14,13,15]
    M_X_out = M_X_out[set_list]
    
    Pkd_2 = M_X_out
    M_X = torch.mul(M_X_out, modu_tensor_list[1].detach())
    
    M_X_out = torch.zeros((M_X.shape))
    for ii in range(M_X_out.shape[0] // mini_size):
        temp_x = M_X[ii*mini_size:(ii+1)*mini_size]
        temp_x_out = calculate_optical_2_2(temp_x, awg, op_reader, input_curve, ref_inten_list, v_list_input)
        M_X_out[ii*mini_size:(ii+1)*mini_size] = temp_x_out
    
    ### layer_3 ###
    set_list = [0,4,8,12, 1,5,9,13, 2,6,10,14, 3,7,11,15]
    M_X_out = M_X_out[set_list]
    
    Pkd_3 = M_X_out
    M_X = torch.mul(M_X_out, modu_tensor_list[2].detach())
    M_X_out = torch.zeros((M_X.shape[0] // mini_size, ))
    for ii in range(M_X_out.shape[0]):
        temp_x = M_X[ii*mini_size:(ii+1)*mini_size]
        temp_x_out = calculate_optical_2_add(temp_x, awg, op_reader, input_curve, ref_inten_list, v_list_input)
        M_X_out[ii] = temp_x_out

    ### layer_4 ###
    set_list = [0,2,1,3, 4,6,5,7]
    M_X_out = M_X_out[set_list]
    Pkd_4 = M_X_out

    M_X = torch.mul(M_X_out, modu_tensor_list[3].detach())
    M_X_out = torch.zeros((M_X.shape))
    for ii in range(M_X_out.shape[0] // mini_size):
        temp_x = M_X[ii*mini_size:(ii+1)*mini_size]
        temp_x_out = calculate_optical_2_2(temp_x, awg, op_reader, input_curve, ref_inten_list, v_list_input)
        M_X_out[ii*mini_size:(ii+1)*mini_size] = temp_x_out

    ### layer_5 ###
    set_list = [0,4, 1,5, 2,6, 3,7]
    M_X_out = M_X_out[set_list]
    Pkd_5 = M_X_out

    M_X = torch.mul(M_X_out, modu_tensor_list[4].detach())
    M_X_out = torch.zeros((M_X.shape[0] // mini_size, ))
    for ii in range(M_X_out.shape[0]):
        temp_x = M_X[ii*mini_size:(ii+1)*mini_size]
        temp_x_out = calculate_optical_2_add(temp_x, awg, op_reader, input_curve, ref_inten_list, v_list_input)
        M_X_out[ii] = temp_x_out

    Pkd_list = [Pkd_1, Pkd_2, Pkd_3, Pkd_4, Pkd_5]
    return M_X_out, Pkd_list

##### forward error propagation #####
def forward_error_network(M_E, modu_tensor_list, awg, op_reader, input_curve, ref_inten_list, v_list_input): 
    mini_size = 2
    ### layer_5 ###
    M_b_add = torch.ones((2,))
    M_E_out = torch.zeros(M_E.shape[0]*mini_size,)
    for ii in range(M_E.shape[0]):
        M_E_out[ii*mini_size:(ii+1)*mini_size] = M_b_add * M_E[ii]

    Pke_5 = M_E_out
    M_E_out = torch.mul(M_E_out, modu_tensor_list[4].detach())

    set_list = [0,2, 4,6, 1,3, 5,7]
    M_E_out = M_E_out[set_list]

    ### layer_4 ###
    M_E = M_E_out
    M_E_out = torch.zeros((M_E.shape))
    for ii in range(M_E_out.shape[0] // mini_size):
        temp_x = M_E[ii*mini_size:(ii+1)*mini_size]
        temp_x_out = calculate_optical_2_2(temp_x, awg, op_reader, input_curve, ref_inten_list, v_list_input)
        M_E_out[ii*mini_size:(ii+1)*mini_size] = temp_x_out
    
    Pke_4 = M_E_out
    M_E_out = torch.mul(M_E_out, modu_tensor_list[3].detach())

    set_list = [0,2,1,3, 4,6,5,7]
    M_E_out = M_E_out[set_list]

    ### layer_3 ###
    M_E = M_E_out
    M_E_out = torch.zeros(M_E.shape[0]*mini_size,)
    for ii in range(M_E.shape[0]):
        M_E_out[ii*mini_size:(ii+1)*mini_size] = M_b_add * M_E[ii]
    
    Pke_3 = M_E_out
    M_E_out = torch.mul(M_E_out, modu_tensor_list[2].detach())

    set_list = [0,4,8,12, 1,5,9,13, 2,6,10,14, 3,7,11,15]
    M_E_out = M_E_out[set_list]

    ### layer_2 ###
    M_E = M_E_out
    M_E_out = torch.zeros((M_E.shape))
    for ii in range(M_E_out.shape[0] // mini_size):
        temp_x = M_E[ii*mini_size:(ii+1)*mini_size]
        temp_x_out = calculate_optical_2_2(temp_x, awg, op_reader, input_curve, ref_inten_list, v_list_input)
        M_E_out[ii*mini_size:(ii+1)*mini_size] = temp_x_out

    Pke_2 = M_E_out
    M_E_out = torch.mul(M_E_out, modu_tensor_list[1].detach())

    set_list = [0,2,1,3, 4,6,5,7, 8,10,9,11, 12,14,13,15]
    M_E_out = M_E_out[set_list]

    ### layer_1 ###
    M_E = M_E_out
    M_E_out = torch.zeros((M_E.shape))
    for ii in range(M_E_out.shape[0] // mini_size):
        temp_x = M_E[ii*mini_size:(ii+1)*mini_size]
        temp_x_out = calculate_optical_2_2(temp_x, awg, op_reader, input_curve, ref_inten_list, v_list_input)
        M_E_out[ii*mini_size:(ii+1)*mini_size] = temp_x_out

    Pke_1 = M_E_out
    Pke_list = [Pke_1, Pke_2, Pke_3, Pke_4, Pke_5]
    return Pke_list

##### ---------initialize---------- #####
## system
rm = pyvisa.ResourceManager()
awg = AWG4(rm, 'XXX')

op = VoltageOperator()
op.init("XXX", 7, 512)

CS = ChannelShifter(port='COM9', flag=True)

awg.set_state(state=[1,1], ch=[1,2])

op_reader = OpticalPower(port='COM8', flag=True)
print(f"OPM lambda: {op_reader.get(mode='l')}")
print(f"OPM power: {op_reader.get(mode='p')}")

## base params
MASK_SIZE = 4
LR = 0.01
MAX_EPOCH = 300
DECAY = 0.99
BATCH_SIZE = 10

exp_name = 'debug'
writer = SummaryWriter(os.path.join('./runs', exp_name))
out_put_dir = os.path.join('./output', exp_name)
debugidx = 9
debug_dir = os.path.join('./output', 'debug' + str(debugidx))
if not os.path.exists(debug_dir):
    os.makedirs(debug_dir)
if not os.path.exists(out_put_dir):
    os.makedirs(out_put_dir)

layer_size_list = [16,16,16,8,8]
modu_tensor_list = []
for ii in range(len(layer_size_list)):
    modu_tensor_init = 2*np.random.random(size=(layer_size_list[ii],)).astype('float32')-1
    modu_tensor = torch.nn.Parameter(torch.from_numpy(modu_tensor_init))
    modu_tensor_list.append(modu_tensor)

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

## optimizer
optimizer = torch.optim.Adam(modu_tensor_list, lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

##### ---------train--------- #####
for epoch in range(MAX_EPOCH):
    train_loss_all = 0
    train_correct_all = 0
    y_pred_all = []
    y_gt_all = []

    awg.dc(ch=1, offset=0.0)
    awg.dc(ch=2, offset=0.0)
    
    ## matrix symmetry calibration
    t_v_1_1, t_v_1_2, t_v_2_1, t_v_2_2 = find_weight(CS, op, op_reader)
    weight_modu_list = [391, 385, 431, 437]
    voltage_set_list = [t_v_2_2,t_v_2_1,t_v_1_1, t_v_1_2]
    op.set_voltage(voltage_set_list, weight_modu_list)
    time.sleep(1.0)
    
    ## input calibration
    calibrate_input_curve(awg, v_list_input, CS, op_reader)
    input_curve = [np.load('input_1.npy'), np.load('input_2.npy')]
    ref_inten_list = get_ref_inten(awg, op_reader, CS)

    for b, (data, target) in enumerate(train_loader):
        grad_1_all_list = [0,0,0,0,0]
        grad_1_all_list_gt = [0,0,0,0,0]
        train_loss_all_batch = 0 
        train_correct_all_batch = 0
        data_batch, target_batch = data, target
        
        for ii in range(data_batch.shape[0]):
            target_temp = target_batch[ii]
            SLM_outinten = data_batch[ii,0]
            
            ##### forward data #####
            Pkd_1 = SLM_outinten.view(-1)
            U_No, Pkd_list = forward_data_network(SLM_outinten, modu_tensor_list, awg, op_reader, input_curve, ref_inten_list, v_list_input)
            
            ##### Error field generation #####
            grad_record = {}
            def save_grad(name):
                def hook(grad):
                    grad_record[name] = grad
                return hook

            x_abs = torch.square(U_No)
            
            x_abs.requires_grad = True
            U_No_av = x_abs
            U_No_av.register_hook(save_grad('U_No_av'))
            
            loss = F.cross_entropy(x_abs.unsqueeze(0), target_temp.unsqueeze(0))
            pred = x_abs.unsqueeze(0).argmax(dim=1, keepdim=True)
            correct = pred.eq(target_temp.view_as(pred)).sum().item()

            y_pred_all.append(pred[0,0])
            y_gt_all.append(target_temp)
            
            train_loss_all += loss.item()
            train_loss_all_batch += loss.item()
            train_correct_all += correct
            train_correct_all_batch += correct

            optimizer.zero_grad()
            loss.backward()

            grad_UNo = grad_record['U_No_av']
            E_field = 2*torch.mul(grad_UNo, U_No)

            ##### forward error #####
            Pke_list = forward_error_network(E_field, modu_tensor_list, awg, op_reader, input_curve, ref_inten_list, v_list_input) 

            for i in range(len(layer_size_list)):
                grad_1 = torch.mul(Pkd_list[i].detach(), Pke_list[i])
                grad_1 = grad_1.to(torch.float32)
                grad_1_all_list[i] += grad_1
        
        grad_1_mean_list = []
        for i in range(len(layer_size_list)):
            grad_1_mean_list.append(grad_1_all_list[i] / target_batch.shape[0])
        
        ##### update #####
        optimizer.zero_grad()
        for i in range(len(layer_size_list)):
            modu_tensor_list[i].grad = grad_1_mean_list[i]
        optimizer.step()
        print(f"Train---batch {b}: loss {train_loss_all_batch/target_batch.shape[0]:.6f} correct {train_correct_all_batch/target_batch.shape[0]:.2f}.", end='\r')

    loss_epoch = train_loss_all/len(train_loader)/BATCH_SIZE
    correct_epoch = train_correct_all/len(train_loader)/BATCH_SIZE
    writer.add_scalar('train_loss', loss_epoch, global_step=epoch)
    writer.add_scalar('correct', correct_epoch, global_step=epoch)
    print(f"Train---Epoch {epoch}: loss {loss_epoch:.6f}, accuracy {correct_epoch}.")

##### ---------test--------- #####
    if epoch % 10 == 0:
        test_loss_all = 0
        test_correct_all = 0

        for b, (data, target) in enumerate(test_loader):
            SLM_outinten = data[0,0]

            ##### forward data #####
            U_No, Pkd_list = forward_data_network(SLM_outinten, modu_tensor_list, awg, op_reader, input_curve, ref_inten_list, v_list_input)
            
            x_abs = torch.square(torch.abs(U_No.detach()))
            
            loss = F.cross_entropy(x_abs.unsqueeze(0), target)
            pred = x_abs.unsqueeze(0).argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            
            test_loss_all += loss.item()
            test_correct_all += correct

        writer.add_scalar('test_loss', test_loss_all/len(test_loader), global_step=epoch)
        writer.add_scalar('test_correct', test_correct_all/len(test_loader), global_step=epoch)
        print(f"Test---Epoch {epoch}: loss {test_loss_all/len(test_loader):.6f} accuracy {test_correct_all/len(test_loader)}.")
