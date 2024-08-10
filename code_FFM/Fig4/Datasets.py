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

OBJECT_SIZE = 28
UP_SAMPLE = 14
MASK_SIZE = 400
RESIZE = int(OBJECT_SIZE * UP_SAMPLE)
PADDING_SZIE = (MASK_SIZE - RESIZE) // 2

UP_SAMPLE_1024 = 4
MASK_SIZE_1024 = 200
RESIZE_1024 = int(OBJECT_SIZE * UP_SAMPLE_1024)
PADDING_SZIE_1024 = (MASK_SIZE_1024 - RESIZE_1024) // 2

class NLOSdata2i2(Dataset):
    def __init__(self, train):
        super().__init__()
        
        self.img_transforms = transforms.Compose([
            transforms.Resize((RESIZE, RESIZE), interpolation=Image.BICUBIC),
            transforms.Pad((PADDING_SZIE,PADDING_SZIE)),
            transforms.ToTensor(),
        ])

        mnist_train = torchvision.datasets.MNIST(root='./dataset_method_1', train=True, transform=self.img_transforms, download=True)
        mnist_test = torchvision.datasets.MNIST(root='./dataset_method_1', train=False, transform=self.img_transforms, download=True)

        self.data = []
        self.label = []
        self.real_label = []
        label_list = [0,1,2,3,4,5,6,7,8,9]
        num_num = 100
        num_num_t = 100

        if train:
            for i, data in enumerate(mnist_test):
                img, label = data[0], data[1]
                if label in label_list and sum(self.label == np.ones(len(self.label)) * label) < num_num:
                    img_com = img

                    self.data.append(img_com)
                    self.label.append(label)
                if len(self.label) == num_num*len(label_list):
                    break
        
        else:
            for i, data in enumerate(mnist_test):
                img, label = data[0], data[1]
                if label in label_list and sum(self.label == np.ones(len(self.label)) * label) < num_num_t:
                    img_com = img

                    self.data.append(img_com)
                    self.label.append(label)
                if len(self.label) == num_num_t*len(label_list):
                    break

        self.data = [self.data[5],self.data[8]]
        self.label = [self.label[5],self.label[8]]

    def create_shift(self, idx = 1, mask_size = 400):
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

        mask2 = np.zeros((mask_size, mask_size), dtype=np.float64)
        mask2[range_list[idx,0]:range_list[idx,1], range_list[idx,2]:range_list[idx,3]] = 0.5

        return mask2

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        
        img = self.data[idx]
        gt = self.create_shift(idx=self.label[idx])
        
        return img, gt
    
class NLOSdata2i_IMG(Dataset):
    def __init__(self, train):
        super().__init__()
        
        self.img_transforms = transforms.Compose([
            transforms.Resize((RESIZE, RESIZE), interpolation=Image.BICUBIC),
            transforms.Pad((PADDING_SZIE,PADDING_SZIE)),
            transforms.ToTensor(),
        ])

        mnist_train = torchvision.datasets.MNIST(root='./dataset_method_1', train=True, transform=self.img_transforms, download=True)
        mnist_test = torchvision.datasets.MNIST(root='./dataset_method_1', train=False, transform=self.img_transforms, download=True)

        self.data = []
        self.label = []
        self.real_label = []
        label_list = [7]
        num_num = 100
        num_num_t = 100

        if train:
            for i, data in enumerate(mnist_train):
                img, label = data[0], data[1]
                if label in label_list and sum(self.label == np.ones(len(self.label)) * label) < num_num:
                    img_com = img
                    
                    idx = np.where(img_com > 0.5)
                    img_com[idx] = 1
                    idx = np.where(img_com < 0.5)
                    img_com[idx] = 0
                    
                    self.data.append(img_com)
                    self.label.append(label)

                    self.real_label.append(img_com)
                if len(self.label) == num_num*len(label_list):
                    break
        
        else:
            for i, data in enumerate(mnist_test):
                img, label = data[0], data[1]
                if label in label_list and sum(self.label == np.ones(len(self.label)) * label) < num_num_t:
                    img_com = img

                    idx = np.where(img_com > 0.5)
                    img_com[idx] = 1
                    idx = np.where(img_com < 0.5)
                    img_com[idx] = 0

                    self.data.append(img_com)
                    self.label.append(label)

                    self.real_label.append(img_com)
                if len(self.label) == num_num_t*len(label_list):
                    break
    
    def create_shift(self, idx = 1, mask_size = 400):
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
        
        mask2 = np.zeros((mask_size, mask_size), dtype=np.float64)
        mask2[range_list[idx,0]:range_list[idx,1], range_list[idx,2]:range_list[idx,3]] = 1.0

        return mask2
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        gt = self.real_label[idx]
        
        return img, gt

class NLOS_IMG(Dataset):
    def __init__(self, train):
        super().__init__()

        self.data = []
        self.label = []
        
        num_list = [0,1,2,5,4,3,6,7,8]#[2,1,0,3,4,5,8,7,6]
        if train:
            for num_ in num_list:
                img = torch.from_numpy(self.create_shift(num_)).unsqueeze(0)
                self.data.append(img)
                self.label.append(img)

        else:
            img = torch.from_numpy(self.create_shift(0)).unsqueeze(0) + \
                    torch.from_numpy(self.create_shift(1)).unsqueeze(0) + \
                    torch.from_numpy(self.create_shift(2)).unsqueeze(0) + \
                    torch.from_numpy(self.create_shift(4)).unsqueeze(0) + \
                    torch.from_numpy(self.create_shift(7)).unsqueeze(0)
            
            self.data.append(img)
            self.label.append(img)

            img = torch.from_numpy(self.create_shift(0)).unsqueeze(0) + \
                    torch.from_numpy(self.create_shift(2)).unsqueeze(0) + \
                    torch.from_numpy(self.create_shift(3)).unsqueeze(0) + \
                    torch.from_numpy(self.create_shift(4)).unsqueeze(0) + \
                    torch.from_numpy(self.create_shift(5)).unsqueeze(0) + \
                    torch.from_numpy(self.create_shift(6)).unsqueeze(0) + \
                    torch.from_numpy(self.create_shift(8)).unsqueeze(0)

            self.data.append(img)
            self.label.append(img)
    
    def create_shift(self, idx = 1, mask_size = 400):
        range_list = np.array([[40,130,40,130],
            [40,130, 155,245],
        [40,130, 270,360],
        [155,245, 40,130],
        [155,245, 155,245],
        [155,245, 270,360],
        [270,360, 40,130],
        [270,360, 155,245],
        [270,360, 270,360]])   
        
        mask2 = np.zeros((mask_size, mask_size), dtype=np.float64)
        mask2[range_list[idx,0]:range_list[idx,1], range_list[idx,2]:range_list[idx,3]] = 1.0

        return mask2
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        
        img = self.data[idx]
        gt = self.label[idx]
        
        return img, gt
    

class NLOSdata2i2_cap2(Dataset):
    def __init__(self, train):
        super().__init__()
        
        device = torch.device("cpu")

        root_path = './capture_flir_data/capture_data'
        root_path2 = './capture_flir_data/capture_data'
        self.train_data_path = os.path.join(root_path, 'train/data')
        self.train_label_path = os.path.join(root_path, 'train/label')
        self.test_data_path = os.path.join(root_path2, 'test/data')
        self.test_label_path = os.path.join(root_path2, 'test/label')

        self.data = []
        self.label = []
        self.real_label = []

        if train:
            data_list = os.listdir(self.train_data_path)
            for i in range(len(data_list)):
                self.data.append(os.path.join(self.train_data_path, '%03d.npy'%i))
                self.label.append(os.path.join(self.train_label_path, '%03d.npy'%i))
        else:
            data_list = os.listdir(self.test_data_path)
            for i in range(len(data_list)):

                self.data.append(os.path.join(self.test_data_path, '%03d.npy'%i))
                self.label.append(os.path.join(self.test_label_path, '%03d.npy'%i))

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        img = np.expand_dims(np.load(self.data[idx]), 0)
        img = img / img.max()
        
        gt_map = torch.from_numpy(np.load(self.label[idx])).unsqueeze(0)
        target_out = detector_region_10(gt_map)
        gt = target_out.argmax(dim=1, keepdim=True)[0,0].item()

        return img, gt