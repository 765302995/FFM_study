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

OBJECT_SIZE = 28
UP_SAMPLE = 14
MASK_SIZE = 400
RESIZE = int(OBJECT_SIZE * UP_SAMPLE)
PADDING_SZIE = (MASK_SIZE - RESIZE) // 2

UP_SAMPLE_ODP = 7
MASK_SIZE_ODP = 200
RESIZE_ODP = int(OBJECT_SIZE * UP_SAMPLE_ODP)
PADDING_SZIE_ODP = (MASK_SIZE_ODP - RESIZE_ODP) // 2

UP_SAMPLE_1024 = 4
MASK_SIZE_1024 = 200
RESIZE_1024 = int(OBJECT_SIZE * UP_SAMPLE_1024)
PADDING_SZIE_1024 = (MASK_SIZE_1024 - RESIZE_1024) // 2

## MNIST
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

## CIFAR
class NLOSdatacifar(Dataset):
    def __init__(self, train):
        super().__init__()
        
        self.img_transforms = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((400, 400), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])

        mnist_train = torchvision.datasets.CIFAR10(root='./data_cifar_1', train=True, download=True, transform=self.img_transforms)
        mnist_test = torchvision.datasets.CIFAR10(root='./data_cifar_1', train=False, download=True, transform=self.img_transforms)

        self.data = []
        self.label = []
        num_num = 1

        if train:
            for i, data in enumerate(mnist_train):
                img, label = data[0], data[1]
                if sum(self.label == np.ones(len(self.label)) * label) < num_num:
                    img_com = img

                    self.data.append(img_com)
                    self.label.append(label)
                if len(self.label) == num_num*10:
                    break

        else:
            for i, data in enumerate(mnist_train):
                img, label = data[0], data[1]
                if sum(self.label == np.ones(len(self.label)) * label) < num_num:
                    img_com = img

                    self.data.append(img_com)
                    self.label.append(label)
                if len(self.label) == num_num*10:
                    break

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        
        img = self.data[idx]
        gt = self.label[idx]
        
        return img, gt

## Fashion-MNIST
class NLOSdataFM(Dataset):
    def __init__(self, train):
        super().__init__()
        
        self.img_transforms = transforms.Compose([
            transforms.Resize((RESIZE, RESIZE), interpolation=Image.BICUBIC),
            transforms.Pad((PADDING_SZIE,PADDING_SZIE)),
            transforms.ToTensor(),
        ])

        # MNIST
        mnist_train = torchvision.datasets.FashionMNIST(root='./data_FM', train=True, transform=self.img_transforms, download=True)
        mnist_test = torchvision.datasets.FashionMNIST(root='./data_FM', train=False, transform=self.img_transforms, download=True)

        self.data = []
        self.label = []
        self.real_label = []
        label_list = [0,1,2,3,4,5,6,7,8,9]
        num_num = 1000
        num_num_t = 10

        if train:
            for i, data in enumerate(mnist_train):
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
        gt = self.create_shift(idx=self.label[idx])
        
        return img, gt

# MWD
class NLOSdataWeather(Dataset):
    def __init__(self, train):
        super().__init__()

        self.total_data_train = []
        self.total_label_train = []
        self.total_data_test = []
        self.total_label_test = []
        self.data = []
        self.label = []

        data_root = './data_weather'
        label_list = ['cloudy', 'rain', 'shine']
        name_list = os.listdir(data_root)

        for label_idx in range(len(label_list)):
            self.temp_data = []
            self.temp_label = []
            for i in range(215):
                image_dir = os.path.join(data_root, label_list[label_idx]+str(i+1)+'.jpg')
                if label_list[label_idx]+str(i+1)+'.jpg' not in name_list:
                    image_dir = os.path.join(data_root, label_list[label_idx]+str(i+1)+'.jpeg')
                imgae_temp = cv2.imread(image_dir,0)
                imgae_temp = F.interpolate(torch.from_numpy(imgae_temp).unsqueeze(0).unsqueeze(0), (MASK_SIZE, MASK_SIZE)).squeeze()
                self.temp_data.append(imgae_temp.unsqueeze(0)/255)
                self.temp_label.append(label_idx)
            
            self.total_data_train += self.temp_data[:200]
            self.total_label_train += self.temp_label[:200]
            self.total_data_test += self.temp_data[200:]
            self.total_label_test += self.temp_label[200:]

        if train:
            self.data = self.total_data_train
            self.label = self.total_label_train
        else:
            self.data = self.total_data_test
            self.label = self.total_label_test

    def create_shift_4(self, idx = 1, mask_size = 400):
        range_list = np.array([[80,170,80,170],
                    [80,170, 230,320],
                [230,320, 80,170],
                [230,320, 230,320]])

        mask2 = np.zeros((mask_size, mask_size), dtype=np.float64)
        mask2[range_list[idx,0]:range_list[idx,1], range_list[idx,2]:range_list[idx,3]] = 1.0

        return mask2
    def create_shift_3(self, idx = 1, mask_size = 400):
        range_list = np.array([[80,140,80,140],
                    [80,140, 260,320],
                [260,320, 170,230]])

        mask2 = np.zeros((mask_size, mask_size), dtype=np.float64)
        mask2[range_list[idx,0]:range_list[idx,1], range_list[idx,2]:range_list[idx,3]] = 1.0

        return mask2

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        
        img = self.data[idx]
        gt = self.create_shift_3(idx=self.label[idx])
        
        return img, gt
    

class NLOSdatacifar3_ODP(Dataset):
    def __init__(self, train):
        super().__init__()
        
        self.img_transforms = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((200, 200), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])

        mnist_train = torchvision.datasets.CIFAR10(root='./data_cifar_1', train=True, download=True, transform=self.img_transforms)
        mnist_test = torchvision.datasets.CIFAR10(root='./data_cifar_1', train=False, download=True, transform=self.img_transforms)

        self.data = []
        self.label = []
        self.real_label = []
        label_list = [1,5,6,7]
        label_set_list = np.array([2,5,7,9])
        num_num = 2000
        num_num_t = 50

        if train:
            for i, data in enumerate(mnist_train):
                img, label = data[0], data[1]
                if label in label_list and sum(self.real_label == np.ones(len(self.real_label)) * label) < num_num:
                    img_com = img

                    idx = np.where(np.array(label_list) == label)
                    labelset = label_set_list[idx][0]

                    self.data.append(img_com)
                    self.real_label.append(label)
                    self.label.append(labelset)
                if len(self.real_label) == num_num*len(label_list):
                    break
        
        else:
            for i, data in enumerate(mnist_test):
                img, label = data[0], data[1]
                if label in label_list and sum(self.real_label == np.ones(len(self.real_label)) * label) < num_num_t:
                    img_com = img

                    idx = np.where(np.array(label_list) == label)
                    labelset = label_set_list[idx][0]

                    self.data.append(img_com)
                    self.real_label.append(label)
                    self.label.append(labelset)
                if len(self.real_label) == num_num_t*len(label_list):
                    break

    def create_shift(self, idx = 1, mask_size = 200):
        range_list = np.array([[15,75,15,75],
                                [15,75, 70,130],
                                [15,75, 125,185],
                                [70,130, 15,75],
                                [70,130, 47,107],
                                [70,130, 93,153],
                                [70,130, 125,185],
                                [125,185, 15,75],
                                [125,185, 70,130],
                                [125,185, 125,185]])   
        
        mask2 = np.zeros((mask_size, mask_size), dtype=np.float64)
        mask2[range_list[idx,0]:range_list[idx,1], range_list[idx,2]:range_list[idx,3]] = 1.0

        return mask2

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        
        img = self.data[idx]
        gt = self.create_shift(idx=self.label[idx])
        
        return img, gt
    

class NLOSdataImagenet_ODP(Dataset):
    def __init__(self, train):
        super().__init__()
        
        self.img_transforms = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((200, 200), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])

        train_dir = './data_IN/train'
        val_dir = './data_IN/val'
        
        self.data = []
        self.label = []
        self.real_label = []
        name_list = ['n02123045','n02802426', 'n02980441', 'n03291819']
        label_list = [0,2,5,9]
        num_num = 1000
        num_num_t = 50
        
        if train:
            for ii in range(len(name_list)):
                img_dir = os.path.join(train_dir, name_list[ii])
                img_name_list = os.listdir(img_dir)
                for jj in range(num_num):
                    temp_dir = os.path.join(img_dir, img_name_list[jj])
                    img = Image.open(temp_dir)
                    img = self.img_transforms(img)

                    self.data.append(img)
                    self.label.append(label_list[ii])
        
        else:
            for ii in range(len(name_list)):
                img_dir = os.path.join(val_dir, name_list[ii])
                img_name_list = os.listdir(img_dir)
                for jj in range(num_num_t):
                    temp_dir = os.path.join(img_dir, img_name_list[jj])
                    img = Image.open(temp_dir)
                    img = self.img_transforms(img)

                    self.data.append(img)
                    self.label.append(label_list[ii])

    def create_shift(self, idx = 1, mask_size = 200):
        range_list = np.array([[15,75,15,75],
                                [15,75, 70,130],
                                [15,75, 125,185],
                                [70,130, 15,75],
                                [70,130, 47,107],
                                [70,130, 93,153],
                                [70,130, 125,185],
                                [125,185, 15,75],
                                [125,185, 70,130],
                                [125,185, 125,185]])   
        
        mask2 = np.zeros((mask_size, mask_size), dtype=np.float64)
        mask2[range_list[idx,0]:range_list[idx,1], range_list[idx,2]:range_list[idx,3]] = 1.0

        return mask2

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        
        img = self.data[idx]
        gt = self.create_shift(idx=self.label[idx])
        
        return img, gt


class NLOSdata2i2_ODP(Dataset):
    def __init__(self, train):
        super().__init__()

        self.img_transforms = transforms.Compose([
            transforms.Resize((RESIZE_ODP, RESIZE_ODP), interpolation=Image.BICUBIC),
            transforms.Pad((PADDING_SZIE_ODP,PADDING_SZIE_ODP)),
            transforms.ToTensor(),
        ])

        mnist_train = torchvision.datasets.MNIST(root='./dataset_method_1', train=True, transform=self.img_transforms, download=True)
        mnist_test = torchvision.datasets.MNIST(root='./dataset_method_1', train=False, transform=self.img_transforms, download=True)

        self.data = []
        self.label = []
        self.real_label = []
        label_list = [0,2,4,6,9]
        num_num = 2000
        num_num_t = 20

        if train:
            for i, data in enumerate(mnist_train):
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

    def create_shift(self, idx = 1, mask_size = 200):
        range_list = np.array([[15,75,15,75],
                                [15,75, 70,130],
                                [15,75, 125,185],
                                [70,130, 15,75],
                                [70,130, 47,107],
                                [70,130, 93,153],
                                [70,130, 125,185],
                                [125,185, 15,75],
                                [125,185, 70,130],
                                [125,185, 125,185]])       

        mask2 = np.zeros((mask_size, mask_size), dtype=np.float64)
        mask2[range_list[idx,0]:range_list[idx,1], range_list[idx,2]:range_list[idx,3]] = 1.0

        return mask2

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        
        img = self.data[idx]
        gt = self.create_shift(idx=self.label[idx])
        
        return img, gt