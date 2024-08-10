import os
import h5py
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import pdb
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import scipy.io as scio
from torch.utils.data import Dataset
from PIL import Image
from skimage.morphology import dilation
import skimage.morphology as smo
from utils import prop_simu
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

class NLOSdatafocus(Dataset):
    def __init__(self, train):
        super().__init__()
        
        plase_input = np.zeros((1,400,400))
        gt = np.zeros((400,400))

        gt[20-3:20+3,20-3:20+3] = 1

        self.data = []
        self.label = []

        self.data.append(plase_input)
        self.label.append(gt)


    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        
        img = self.data[idx]
        gt = self.label[idx]
        
        return img, gt