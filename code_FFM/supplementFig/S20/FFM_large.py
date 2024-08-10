from PIL import Image
import numpy as np
from ctypes import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from dataset import OCFlowerdata, NLOSdataMNIST_PCA, NLOSdataMNIST_PCA_C
from tqdm import *
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

## network initialize parameter
layer_size_list = [512,512,128,128,32,32,8,8]
layer_begin_list = []
temp = 0
for ii in range(len(layer_size_list)):
    layer_begin_list.append(temp)
    temp += layer_size_list[ii]

## forward data evaluation
def forward_data_network_eval(M_I, modu_tensor_list, SYYM_matrix_list, nonlinear_flag, model_bn, SYYM_matrix_add_list):
    M_X_out = M_I
    for ii in range(len(layer_size_list)):
        M_X_layer_out = 0
        for jj in range(layer_size_list[ii]):
            M_X = torch.mul(M_X_out, modu_tensor_list[layer_begin_list[ii]+jj].detach())
            M_X_layer_out += torch.matmul(M_X, SYYM_matrix_list[layer_begin_list[ii]+jj])
        
        M_X_out = M_X_layer_out
        if ii == 1:
            M_X_out = torch.matmul(M_X_out, SYYM_matrix_add_list[0])
        if ii == 3:
            M_X_out = torch.matmul(M_X_out, SYYM_matrix_add_list[1])
        if ii == 5:
            M_X_out = torch.matmul(M_X_out, SYYM_matrix_add_list[2])

        if nonlinear_flag:
            M_X_out = model_bn.bnlayer[ii](M_X_out)
            M_X_out = F.relu(M_X_out)
    return M_X_out

## forward data train
def forward_data_network(M_I, modu_tensor_list, SYYM_matrix_list, nonlinear_flag, model_bn, SYYM_matrix_add_list):
    M_X_out = M_I
    for ii in range(len(layer_size_list)):
        M_X_layer_out = 0
        for jj in range(layer_size_list[ii]):
            M_X = torch.mul(M_X_out, modu_tensor_list[layer_begin_list[ii]+jj])
            M_X_layer_out += torch.matmul(M_X, SYYM_matrix_list[layer_begin_list[ii]+jj])
        
        M_X_out = M_X_layer_out
        if ii == 1:
            M_X_out = torch.matmul(M_X_out, SYYM_matrix_add_list[0])
        if ii == 3:
            M_X_out = torch.matmul(M_X_out, SYYM_matrix_add_list[1])
        if ii == 5:
            M_X_out = torch.matmul(M_X_out, SYYM_matrix_add_list[2])

        if nonlinear_flag:
            M_X_out = model_bn.bnlayer[ii](M_X_out)
            M_X_out = F.relu(M_X_out)
    return M_X_out

## batch_norm layer
class BN_net(nn.Module):
    def __init__(self):
        super(BN_net, self).__init__()
        self.bnlayer = nn.ModuleList([nn.BatchNorm1d(512),nn.BatchNorm1d(128),
                                      nn.BatchNorm1d(128),nn.BatchNorm1d(32),
                                      nn.BatchNorm1d(32),nn.BatchNorm1d(8),
                                      nn.BatchNorm1d(8),nn.BatchNorm1d(8)])
 
    def forward(self, x):
        x = self.outlayer(x)

        return x

##### ---------initialize---------- #####
## system
nonlinear_flag = True

## base params
MASK_SIZE = 4
LR = 0.01
MAX_EPOCH = 300
DECAY = 0.99
BATCH_SIZE_TRAIN = 10
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

## modulation region
layer_size_list = [512,512,128,128,32,32,8,8]
modu_tensor_list = []
for ii in range(len(layer_size_list)):
    for jj in range(layer_size_list[ii]):
        modu_tensor_init = 2*np.random.random(size=(layer_size_list[ii],)).astype('float32')-1
        modu_tensor = torch.nn.Parameter(torch.from_numpy(modu_tensor_init).to(device))
        modu_tensor_list.append(modu_tensor)

## propagation matrix
SYYM_matrix_add_1 = torch.zeros((512,128)).to(torch.float32).to(device)
for ii in range(128):
    SYYM_matrix_add_1[ii*4:(ii+1)*4,ii] = 1

SYYM_matrix_add_2 = torch.zeros((128,32)).to(torch.float32).to(device)
for ii in range(32):
    SYYM_matrix_add_2[ii*4:(ii+1)*4,ii] = 1

SYYM_matrix_add_3 = torch.zeros((32,8)).to(torch.float32).to(device)
for ii in range(8):
    SYYM_matrix_add_3[ii*4:(ii+1)*4,ii] = 1

SYYM_matrix_add_list = [SYYM_matrix_add_1, SYYM_matrix_add_2, SYYM_matrix_add_3]

SYYM_matrix_save = []
for ii in range(len(layer_size_list)):
    if ii == 1 or ii == 3 or ii == 5:
        for jj in range(layer_size_list[ii]):
            matrix_init = 2*np.random.random(size=(layer_size_list[ii],layer_size_list[ii])).astype('float32')-1
            SYYM_matrix_save.append(np.matmul(matrix_init, matrix_init.T))
    else:
        for jj in range(layer_size_list[ii]):
            matrix_init = np.zeros((layer_size_list[ii],layer_size_list[ii]))
            matrix_init[jj,:] = 1
            matrix_init[:,jj] = 1
            SYYM_matrix_save.append(matrix_init)

SYYM_matrix_init = SYYM_matrix_save
SYYM_matrix_list = []
for ii in range(len(SYYM_matrix_init)):
    SYYM_matrix = torch.from_numpy(SYYM_matrix_init[ii]).to(torch.float32).to(device)
    SYYM_matrix_list.append(SYYM_matrix)

## batch_norm layer
model_bn = BN_net().to(device)

## dataset
use_gpu = False
gpu_args = {'num_workers': 12, 'pin_memory': True} if use_gpu else {}

train_dataset = NLOSdataMNIST_PCA(train=True)
test_dataset = NLOSdataMNIST_PCA(train=False)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=BATCH_SIZE_TRAIN, shuffle=True, **gpu_args)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=BATCH_SIZE_TEST, shuffle=False, **gpu_args)

## optimizer
optimizer = torch.optim.Adam(modu_tensor_list+list(model_bn.parameters()), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

for epoch in range(MAX_EPOCH):
    train_loss_all = 0
    train_correct_all = 0

    ##### ---------train--------- #####
    for b, (data, target) in enumerate(train_loader):
        data_batch, target_batch = data.to(torch.float32).to(device), target.to(device)
        data_batch = data_batch.view(data_batch.shape[0],-1)

        x_abs = forward_data_network(data_batch, modu_tensor_list, SYYM_matrix_list, nonlinear_flag, model_bn, SYYM_matrix_add_list)

        loss = F.cross_entropy(x_abs, target_batch)
        pred = x_abs.argmax(dim=1, keepdim=True)
        correct = pred.eq(target_batch.view_as(pred)).sum().item()

        train_loss_all += loss.item()
        train_correct_all += correct

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_epoch = train_loss_all/len(train_loader)/BATCH_SIZE_TRAIN
    correct_epoch = train_correct_all/len(train_loader)/BATCH_SIZE_TRAIN
    writer.add_scalar('train_loss', loss_epoch, global_step=epoch)
    writer.add_scalar('correct', correct_epoch, global_step=epoch)
    print(f"Train---Epoch {epoch}: loss {loss_epoch:.6f}, accuracy {correct_epoch}.")

    ##### ---------test--------- #####
    if epoch % 1 == 0:
        test_loss_all = 0
        test_correct_all = 0

        for b, (data, target) in enumerate(test_loader):
            data_batch = data.view(data.shape[0],-1).to(torch.float32).to(device)
            target = target.to(device)

            x_abs = forward_data_network_eval(data_batch, modu_tensor_list, SYYM_matrix_list, nonlinear_flag, model_bn, SYYM_matrix_add_list)

            loss = F.cross_entropy(x_abs, target)
            pred = x_abs.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            
            test_loss_all += loss.item()
            test_correct_all += correct
        
        writer.add_scalar('test_loss', test_loss_all/len(test_loader)/BATCH_SIZE_TEST, global_step=epoch)
        writer.add_scalar('test_correct', test_correct_all/len(test_loader)/BATCH_SIZE_TEST, global_step=epoch) 
        print(f"Test---Epoch {epoch}: loss {test_loss_all/len(test_loader)/BATCH_SIZE_TEST:.6f} accuracy {test_correct_all/len(test_loader)/BATCH_SIZE_TEST}.")
