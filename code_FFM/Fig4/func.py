from tkinter import Y
import numpy as np
import matplotlib.pyplot as plt
import pdb
import torch
import torch.nn.functional as F
import cv2
import scipy
from Datasets import MASK_SIZE
from utils import * 
import torchvision.transforms as transforms

def get_crop_new(input_map, cali_M, flip = False, msize=400):
    M = cali_M

    img = input_map

    temp_result = cv2.warpPerspective(img, M, (msize, msize))
    #temp_result = temp_result / 255.0

    temp_result = np.rot90(temp_result, k=1)

    imgae_ps = temp_result
    return imgae_ps

def get_crop(input_map, cali_M, flip = False, msize=400):
    M = cali_M

    img = input_map

    temp_result = cv2.warpPerspective(img, M, (msize, msize))
    #temp_result = temp_result / 255.0

    if flip == True:
        temp_result = np.rot90(temp_result, k=2)
        temp_result = temp_result[::-1]

    imgae_ps = temp_result
    return imgae_ps

def SLM_create_object_phase(phase_input, over_phase):
    phase_input = torch.flip(phase_input,[1])
    mode_type = 'reflect' #replicate
    MASK_SIZE = 400
    phase_input = -phase_input
    phase_input = torch.remainder(phase_input, torch.tensor([2*np.pi]))

    phase_input = phase_input / (2*np.pi)
    phase_cali = phase_input * 255

    pad_size = 800
    phase_cali = F.pad(phase_cali.unsqueeze(0).unsqueeze(0), ((pad_size-MASK_SIZE)//2, (pad_size-MASK_SIZE)//2, (pad_size-MASK_SIZE)//2, (pad_size-MASK_SIZE)//2), mode=mode_type).squeeze() #replicate
    
    out_size = [1920,1200]
    SLM_input = F.pad(phase_cali.unsqueeze(0).unsqueeze(0), ((out_size[0]-pad_size)//2-230, (out_size[0]-pad_size)//2+230, (out_size[1]-pad_size)//2+130, (out_size[1]-pad_size)//2-130), mode=mode_type).squeeze()

    SLM_input = SLM_input.to(torch.uint8)
    return SLM_input

def SLM_create_amp(amp_input):
    idx = torch.where(amp_input > 1)
    amp_input[idx] = 1
    
    mode_type = 'replicate' #replicate constant reflect
    min_map = torch.from_numpy(np.load('./cali/min_map.npy'))
    max_map = torch.from_numpy(np.load('./cali/max_map.npy'))
    min_value = torch.from_numpy(np.load('./cali/min_map_value.npy'))
    max_value = torch.from_numpy(np.load('./cali/max_map_value.npy'))

    MASK_SIZE = 400
    RESIZE_SIZE = 256
    amp_input = F.interpolate(amp_input.unsqueeze(0).unsqueeze(0), (RESIZE_SIZE, RESIZE_SIZE)).squeeze()
    amp_input = F.pad(amp_input.unsqueeze(0).unsqueeze(0), ((MASK_SIZE-RESIZE_SIZE)//2, (MASK_SIZE-RESIZE_SIZE)//2, (MASK_SIZE-RESIZE_SIZE)//2, (MASK_SIZE-RESIZE_SIZE)//2), mode=mode_type).squeeze()
    
    min_map = F.pad(min_map.unsqueeze(0).unsqueeze(0), ((MASK_SIZE-RESIZE_SIZE)//2, (MASK_SIZE-RESIZE_SIZE)//2, (MASK_SIZE-RESIZE_SIZE)//2, (MASK_SIZE-RESIZE_SIZE)//2), mode=mode_type).squeeze()
    max_map = F.pad(max_map.unsqueeze(0).unsqueeze(0), ((MASK_SIZE-RESIZE_SIZE)//2, (MASK_SIZE-RESIZE_SIZE)//2, (MASK_SIZE-RESIZE_SIZE)//2, (MASK_SIZE-RESIZE_SIZE)//2), mode=mode_type).squeeze()
    min_value = F.pad(min_value.unsqueeze(0).unsqueeze(0), ((MASK_SIZE-RESIZE_SIZE)//2, (MASK_SIZE-RESIZE_SIZE)//2, (MASK_SIZE-RESIZE_SIZE)//2, (MASK_SIZE-RESIZE_SIZE)//2), mode=mode_type).squeeze()
    max_value = F.pad(max_value.unsqueeze(0).unsqueeze(0), ((MASK_SIZE-RESIZE_SIZE)//2, (MASK_SIZE-RESIZE_SIZE)//2, (MASK_SIZE-RESIZE_SIZE)//2, (MASK_SIZE-RESIZE_SIZE)//2), mode=mode_type).squeeze()
    
    phase_input_ori = 2 * torch.arcsin(amp_input)

    phase_input = phase_input_ori
    phase_input = torch.remainder(phase_input, torch.tensor([2*np.pi]))

    phase_input = phase_input / np.pi
    phase_cali = min_map + phase_input * (max_map - min_map)
    input_amp = min_value + phase_input *(max_value - min_value)

    rate = 213/255
    phase_cali = phase_cali * rate

    over_phase = 2*np.pi * phase_cali / 213
    over_phase = 0.5*over_phase

    over_phase = F.interpolate(over_phase[72:328, 72:328].unsqueeze(0).unsqueeze(0), (MASK_SIZE, MASK_SIZE)).squeeze()

    pad_size = 800
    phase_cali = F.pad(phase_cali.unsqueeze(0).unsqueeze(0), ((pad_size-MASK_SIZE)//2, (pad_size-MASK_SIZE)//2, (pad_size-MASK_SIZE)//2, (pad_size-MASK_SIZE)//2), mode=mode_type).squeeze()
    
    out_size = [1280,1024]#[1920,1200]
    SLM_input = F.pad(phase_cali.unsqueeze(0).unsqueeze(0), ((out_size[0]-pad_size)//2, (out_size[0]-pad_size)//2, (out_size[1]-pad_size)//2, (out_size[1]-pad_size)//2), mode=mode_type).squeeze()

    SLM_input = SLM_input.to(torch.uint8)

    return SLM_input, over_phase, input_amp

def SLM_create_phase_1024(phase_input, over_phase):
    mode_type = 'replicate'
    MASK_SIZE = 200
    phase_input = -phase_input
    #phase_input = phase_input - over_phase
    phase_input = torch.remainder(phase_input, torch.tensor([2*np.pi]))

    phase_input = phase_input / (2*np.pi)
    phase_cali = phase_input * 255
    
    pad_size = 400
    phase_cali = F.pad(phase_cali.unsqueeze(0).unsqueeze(0), ((pad_size-MASK_SIZE)//2, (pad_size-MASK_SIZE)//2, (pad_size-MASK_SIZE)//2, (pad_size-MASK_SIZE)//2), mode=mode_type).squeeze() #replicate
    
    pad_size = 800
    phase_cali = F.pad(phase_cali.unsqueeze(0).unsqueeze(0), ((pad_size-400)//2, (pad_size-400)//2, (pad_size-400)//2, (pad_size-400)//2), mode=mode_type).squeeze() #replicate
    
    out_size = [1024,1024]
    SLM_input = F.pad(phase_cali.unsqueeze(0).unsqueeze(0), ((out_size[0]-pad_size)//2, (out_size[0]-pad_size)//2, (out_size[1]-pad_size)//2, (out_size[1]-pad_size)//2), mode=mode_type).squeeze()
    
    SLM_input = SLM_input.to(torch.uint8)
    return SLM_input


def SLM_create_phase(phase_input, over_phase):
    mode_type = 'replicate' #replicate constant
    MASK_SIZE = 400
    phase_input = -phase_input
    phase_input = torch.remainder(phase_input, torch.tensor([2*np.pi]))

    phase_input = phase_input / (2*np.pi)
    phase_cali = phase_input * 255
    
    pad_size = 800
    phase_cali = F.pad(phase_cali.unsqueeze(0).unsqueeze(0), ((pad_size-MASK_SIZE)//2, (pad_size-MASK_SIZE)//2, (pad_size-MASK_SIZE)//2, (pad_size-MASK_SIZE)//2), mode=mode_type).squeeze() #replicate
    
    out_size = [1920,1200]
    SLM_input = F.pad(phase_cali.unsqueeze(0).unsqueeze(0), ((out_size[0]-pad_size)//2, (out_size[0]-pad_size)//2, (out_size[1]-pad_size)//2, (out_size[1]-pad_size)//2), mode=mode_type).squeeze()
    
    SLM_input = SLM_input.to(torch.uint8)
    return SLM_input