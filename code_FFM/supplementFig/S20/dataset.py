from posixpath import split
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import scipy.io as scio
from torch.utils.data import Dataset
from PIL import Image
from optics import ASM_propagate, fft_conv2d
from model import identity_net, identity_net2

class NLOSdataMNIST_PCA_16(Dataset):
    def __init__(self, train):
        super().__init__()

        self.pca = joblib.load('pca.m')

        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

        # MNIST
        mnist_train = torchvision.datasets.MNIST(root='./dataset_method_1', train=True, transform=self.img_transforms, download=False)
        mnist_test = torchvision.datasets.MNIST(root='./dataset_method_1', train=False, transform=self.img_transforms, download=False)

        self.data = []
        self.label = []
        self.real_label = []
        label_list = [0,1,2,3,4,5,6,7]
        num_num = 1000
        num_num_t = 100

        if train:
            for i, data in enumerate(mnist_train):
                img, label = data[0], data[1]
                if label in label_list and sum(self.label == np.ones(len(self.label)) * label) < num_num:
                    img_com = img.view(-1).unsqueeze(0)
                    img_com = self.pca.transform(img_com)

                    img_com = img_com / abs(img_com).max()

                    self.data.append(img_com)
                    self.label.append(label)
                if len(self.label) == num_num*len(label_list):
                    break
        
        else:
            for i, data in enumerate(mnist_test):
                img, label = data[0], data[1]
                if label in label_list and sum(self.label == np.ones(len(self.label)) * label) < num_num_t:
                    img_com = img.view(-1).unsqueeze(0)
                    img_com = self.pca.transform(img_com)

                    img_com = img_com / abs(img_com).max()

                    self.data.append(img_com)
                    self.label.append(label)
                if len(self.label) == num_num_t*len(label_list):
                    break

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        
        img = self.data[idx]
        gt = self.label[idx]
        
        return img, gt
    

class NLOSdataMNIST_PCA_256(Dataset):
    def __init__(self, train):
        super().__init__()

        self.pca = joblib.load('pca.m')

        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

        # MNIST
        mnist_train = torchvision.datasets.MNIST(root='./dataset_method_1', train=True, transform=self.img_transforms, download=False)
        mnist_test = torchvision.datasets.MNIST(root='./dataset_method_1', train=False, transform=self.img_transforms, download=False)

        self.data = []
        self.label = []
        self.real_label = []
        label_list = [0,1,2,3,4,5,6,7]
        num_num = 1000
        num_num_t = 100

        if train:
            for i, data in enumerate(mnist_train):
                img, label = data[0], data[1]
                if label in label_list and sum(self.label == np.ones(len(self.label)) * label) < num_num:
                    img_com = img.view(-1).unsqueeze(0)
                    img_com = self.pca.transform(img_com)
                    img_com = np.reshape(np.repeat(np.expand_dims(img_com,-1),16,-1), (img_com.shape[0],16*16))

                    img_com = img_com / abs(img_com).max()

                    self.data.append(img_com)
                    self.label.append(label)
                if len(self.label) == num_num*len(label_list):
                    break
        
        else:
            for i, data in enumerate(mnist_test):
                img, label = data[0], data[1]
                if label in label_list and sum(self.label == np.ones(len(self.label)) * label) < num_num_t:
                    img_com = img.view(-1).unsqueeze(0)
                    img_com = self.pca.transform(img_com)
                    img_com = np.reshape(np.repeat(np.expand_dims(img_com,-1),16,-1), (img_com.shape[0],16*16))

                    img_com = img_com / abs(img_com).max()

                    self.data.append(img_com)
                    self.label.append(label)
                if len(self.label) == num_num_t*len(label_list):
                    break

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        
        img = self.data[idx]
        gt = self.label[idx]
        
        return img, gt


class NLOSdataMNIST_PCA_512(Dataset):
    def __init__(self, train):
        super().__init__()

        self.pca = joblib.load('pca_512.m')

        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

        # MNIST
        mnist_train = torchvision.datasets.MNIST(root='./dataset_method_1', train=True, transform=self.img_transforms, download=False)
        mnist_test = torchvision.datasets.MNIST(root='./dataset_method_1', train=False, transform=self.img_transforms, download=False)

        self.data = []
        self.label = []
        self.real_label = []
        label_list = [0,1,2,3,4,5,6,7]
        num_num = 1000
        num_num_t = 100

        if train:
            for i, data in enumerate(mnist_train):
                img, label = data[0], data[1]
                if label in label_list and sum(self.label == np.ones(len(self.label)) * label) < num_num:
                    img_com = img.view(-1).unsqueeze(0)
                    img_com = self.pca.transform(img_com)

                    img_com = img_com / abs(img_com).max()

                    self.data.append(img_com)
                    self.label.append(label)
                if len(self.label) == num_num*len(label_list):
                    break
        
        else:
            for i, data in enumerate(mnist_test):
                img, label = data[0], data[1]
                if label in label_list and sum(self.label == np.ones(len(self.label)) * label) < num_num_t:
                    img_com = img.view(-1).unsqueeze(0)
                    img_com = self.pca.transform(img_com)

                    img_com = img_com / abs(img_com).max()

                    self.data.append(img_com)
                    self.label.append(label)
                if len(self.label) == num_num_t*len(label_list):
                    break

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        
        img = self.data[idx]
        gt = self.label[idx]
        
        return img, gt

