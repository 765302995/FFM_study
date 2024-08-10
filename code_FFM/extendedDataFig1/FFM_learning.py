import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import os
simu_step = 1e-1

##### forward data propagation #####
def propagation_forward_data(E1,E2,gamma,kappa):
    Pkd = [E1,E2]
    dE1 = simu_step*(1j*0.5*gamma*E1-kappa*E2)/(1j)
    dE2 = simu_step*(-1j*0.5*gamma*E2-kappa*E1)/(1j)
    E1_new = E1 + dE1
    E2_new = E2 + dE2
    return E1_new, E2_new, Pkd

##### forward error propagation #####
def propagation_forward_error(E1,E2,gamma,kappa):
    dE1 = simu_step*(1j*0.5*gamma*E1-kappa*E2)/(1j)
    dE2 = simu_step*(-1j*0.5*gamma*E2-kappa*E1)/(1j)
    E1_new = E1 + dE1
    E2_new = E2 + dE2
    Pke = [E1_new,E2_new]
    return E1_new, E2_new, Pke

##### ---------initialize---------- #####
## base params
MAX_EPOCH = 300
LR = 0.03
STEP_NUM = 15

exp_name = 'debug'
device = torch.device('cuda')

gamma_init = 4.5
gamma = torch.nn.Parameter(torch.tensor(gamma_init).to(device))

kappa = 1.9 #cm^-1

## optimizer
optimizer = torch.optim.Adam([gamma], lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

##### ---------train--------- #####
for epoch in range(MAX_EPOCH):
    Pke_list_1 = []
    Pke_list_2 = []
    Pke_list_3 = []
    Pke_list_4 = []
    Pkd_list_1 = []
    Pkd_list_2 = []
    Pkd_list_3 = []
    Pkd_list_4 = []

    ##### forward data #####
    optimizer.zero_grad()
    E1 = 1
    E2 = 0
    for x in range(STEP_NUM):
        E1,E2,Pkd = propagation_forward_data(E1,E2,gamma.detach(),kappa)
        Pkd_list_1.append(Pkd)
    x1 = [E1,E2]

    E1 = 0
    E2 = 1
    for x in range(STEP_NUM):
        E1,E2,Pkd = propagation_forward_data(E1,E2,gamma.detach(),kappa)
        Pkd_list_2.append(Pkd)
    x2 = [E1,E2]

    E1 = 1j
    E2 = 0
    for x in range(STEP_NUM):
        E1,E2,Pkd = propagation_forward_data(E1,E2,gamma.detach(),kappa)
        Pkd_list_3.append(Pkd)
    x3 = [E1,E2]

    E1 = 0
    E2 = 1j
    for x in range(STEP_NUM):
        E1,E2,Pkd = propagation_forward_data(E1,E2,gamma.detach(),kappa)
        Pkd_list_4.append(Pkd)
    x4 = [E1,E2]

    ##### Error field generation #####
    grad_record_11 = {}
    grad_record_12 = {}
    grad_record_21 = {}
    grad_record_22 = {}
    grad_record_31 = {}
    grad_record_32 = {}
    grad_record_41 = {}
    grad_record_42 = {}
    def save_grad(name, grad_record):
        def hook(grad):
            grad_record[name] = grad
        return hook
    
    x11_abs = torch.square(torch.abs(x1[0]))
    x12_abs = torch.square(torch.abs(x1[1]))
    x21_abs = torch.square(torch.abs(x2[0]))
    x22_abs = torch.square(torch.abs(x2[1]))
    x31_abs = torch.square(torch.abs(x3[0]))
    x32_abs = torch.square(torch.abs(x3[1]))
    x41_abs = torch.square(torch.abs(x4[0]))
    x42_abs = torch.square(torch.abs(x4[1]))

    x11_abs.requires_grad = True
    x12_abs.requires_grad = True
    x21_abs.requires_grad = True
    x22_abs.requires_grad = True
    x31_abs.requires_grad = True
    x32_abs.requires_grad = True
    x41_abs.requires_grad = True
    x42_abs.requires_grad = True

    x11_abs.register_hook(save_grad('U_No_av', grad_record_11))
    x12_abs.register_hook(save_grad('U_No_av', grad_record_12))
    x21_abs.register_hook(save_grad('U_No_av', grad_record_21))
    x22_abs.register_hook(save_grad('U_No_av', grad_record_22))
    x31_abs.register_hook(save_grad('U_No_av', grad_record_31))
    x32_abs.register_hook(save_grad('U_No_av', grad_record_32))
    x41_abs.register_hook(save_grad('U_No_av', grad_record_41))
    x42_abs.register_hook(save_grad('U_No_av', grad_record_42))

    loss1 = abs((x11_abs + x12_abs + x21_abs + x22_abs + x31_abs + x32_abs + x41_abs + x42_abs) - 8)
    optimizer.zero_grad()
    loss1.backward()

    E_e_11_ori = 2*torch.mul(grad_record_11['U_No_av'], x1[0].conj().detach())
    E_e_12_ori = 2*torch.mul(grad_record_12['U_No_av'], x1[1].conj().detach())
    E_e_21_ori = 2*torch.mul(grad_record_21['U_No_av'], x2[0].conj().detach())
    E_e_22_ori = 2*torch.mul(grad_record_22['U_No_av'], x2[1].conj().detach())
    E_e_31_ori = 2*torch.mul(grad_record_31['U_No_av'], x3[0].conj().detach())
    E_e_32_ori = 2*torch.mul(grad_record_32['U_No_av'], x3[1].conj().detach())
    E_e_41_ori = 2*torch.mul(grad_record_41['U_No_av'], x4[0].conj().detach())
    E_e_42_ori = 2*torch.mul(grad_record_42['U_No_av'], x4[1].conj().detach())

    ##### forward error #####
    for x in range(STEP_NUM):
        E_e_11_ori,E_e_12_ori,Pke = propagation_forward_error(E_e_11_ori,E_e_12_ori,gamma.detach(),kappa)
        Pke_list_1.append(Pke)
    
    for x in range(STEP_NUM):
        E_e_21_ori,E_e_22_ori,Pke = propagation_forward_error(E_e_21_ori,E_e_22_ori,gamma.detach(),kappa)
        Pke_list_2.append(Pke)

    for x in range(STEP_NUM):
        E_e_31_ori,E_e_32_ori,Pke = propagation_forward_error(E_e_31_ori,E_e_32_ori,gamma.detach(),kappa)
        Pke_list_3.append(Pke)

    for x in range(STEP_NUM):
        E_e_41_ori,E_e_42_ori,Pke = propagation_forward_error(E_e_41_ori,E_e_42_ori,gamma.detach(),kappa)
        Pke_list_4.append(Pke)
    
    Pke_list_1.reverse()
    Pke_list_2.reverse()
    Pke_list_3.reverse()
    Pke_list_4.reverse()

    grad_all = 0
    for x in range(STEP_NUM):
        grad_11 = 2*torch.mul(Pkd_list_1[x][0], Pke_list_1[x][0]).real
        grad_11 = grad_11.to(torch.float32)
        grad_12 = 2*torch.mul(Pkd_list_1[x][1], Pke_list_1[x][1]).real
        grad_12 = grad_12.to(torch.float32)

        grad_21 = 2*torch.mul(Pkd_list_2[x][0], Pke_list_2[x][0]).real
        grad_21 = grad_21.to(torch.float32)
        grad_22 = 2*torch.mul(Pkd_list_2[x][1], Pke_list_2[x][1]).real
        grad_22 = grad_22.to(torch.float32)

        grad_31 = 2*torch.mul(Pkd_list_3[x][0], Pke_list_3[x][0]).real
        grad_31 = grad_31.to(torch.float32)
        grad_32 = 2*torch.mul(Pkd_list_3[x][1], Pke_list_3[x][1]).real
        grad_32 = grad_32.to(torch.float32)

        grad_41 = 2*torch.mul(Pkd_list_4[x][0], Pke_list_4[x][0]).real
        grad_41 = grad_41.to(torch.float32)
        grad_42 = 2*torch.mul(Pkd_list_4[x][1], Pke_list_4[x][1]).real
        grad_42 = grad_42.to(torch.float32)

        if x == STEP_NUM-1:
            grad_all += (grad_11+grad_12+grad_21+grad_22+grad_31+grad_32+grad_41+grad_42)/8
    
    grad_mean = grad_all / 1
    print(f"Train---epoch {epoch}: loss {loss1.item():.6f}.")
    print(gamma.item())
    grad_mean_final = grad_mean.to(torch.float32)

    ##### update #####
    optimizer.zero_grad()
    gamma.grad = grad_mean_final
    optimizer.step()