from PIL import Image
import numpy as np
from ctypes import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from dataset import OCFlowerdata, NLOSdataMNIST_PCA
from tqdm import *
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

class FC_net(nn.Module):
    def __init__(self):
        super(FC_net, self).__init__()

        device = torch.device(f"cuda:{0}")
        self.SYYM_matrix_add = torch.zeros((16,8)).to(torch.float32).to(device)
        for ii in range(8):
            self.SYYM_matrix_add[ii*2:(ii+1)*2,ii] = 1

        self.layer1 = nn.Linear(16, 16, bias=False)
        self.layer2 = nn.Linear(16, 16, bias=False)
        self.layer3 = nn.Linear(16, 16, bias=False)
        self.layer4 = nn.Linear(8, 8, bias=False)
        self.layer5 = nn.Linear(8, 8, bias=False)

        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(8)
        self.bn5 = nn.BatchNorm1d(8)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.layer2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.layer3(x)
        x = torch.matmul(x, self.SYYM_matrix_add)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.layer4(x)
        x = self.bn4(x)
        x = F.relu(x)

        x = self.layer5(x)
        x = self.bn5(x)
        x = F.relu(x)

        return x

##### ---------initialize---------- #####
## system
awg = 0
op_reader = 0

## base params
MASK_SIZE = 4
LR = 0.01
MAX_EPOCH = 300
DECAY = 0.99
BATCH_SIZE = 10

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
model = FC_net().to(device)

## dataset
use_gpu = False
gpu_args = {'num_workers': 12, 'pin_memory': True} if use_gpu else {}

train_dataset = NLOSdataMNIST_PCA(train=True)
test_dataset = NLOSdataMNIST_PCA(train=False)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=BATCH_SIZE, shuffle=True, **gpu_args)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=BATCH_SIZE, shuffle=False, **gpu_args)

## optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

for epoch in range(MAX_EPOCH):
    train_loss_all = 0
    train_correct_all = 0

    ##### ---------train--------- #####
    for b, (data, target) in enumerate(train_loader):
        data_batch, target_batch = data.to(torch.float32).to(device), target.to(device)
        data_batch = data_batch.view(data_batch.shape[0],-1)

        x_abs = model(data_batch)

        loss = F.cross_entropy(x_abs, target_batch)
        pred = x_abs.argmax(dim=1, keepdim=True)
        correct = pred.eq(target_batch.view_as(pred)).sum().item()

        train_loss_all += loss.item()
        train_correct_all += correct

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_epoch = train_loss_all/len(train_loader)/BATCH_SIZE
    correct_epoch = train_correct_all/len(train_loader)/BATCH_SIZE
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

            x_abs = model(data_batch)
            
            loss = F.cross_entropy(x_abs, target)
            pred = x_abs.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            
            test_loss_all += loss.item()
            test_correct_all += correct
        
        writer.add_scalar('test_loss', test_loss_all/len(test_loader)/BATCH_SIZE, global_step=epoch)
        writer.add_scalar('test_correct', test_correct_all/len(test_loader)/BATCH_SIZE, global_step=epoch) 
        print(f"Test---Epoch {epoch}: loss {test_loss_all/len(test_loader)/BATCH_SIZE:.6f} accuracy {test_correct_all/len(test_loader)/BATCH_SIZE}.")
