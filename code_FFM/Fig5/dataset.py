import os
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
import torchvision.models as models
from sklearn.datasets import load_iris

RESIZE = 4
class OCFlowerdata(Dataset):
    def __init__(self, train):
        super().__init__()
        
        iris = load_iris()
        
        self.ori_data = iris['data']
        self.ori_label = iris['target']
        for ii in range(self.ori_data.shape[1]):
            self.ori_data[:,ii] = self.ori_data[:,ii] / self.ori_data[:,ii].max()

        self.train_data = []
        self.train_label = []
        self.test_data = []
        self.test_label = []
        
        for ii in range(self.ori_data.shape[0]):
            temp_data = np.expand_dims(self.ori_data[ii], 1)
            temp_label = self.ori_label[ii]

            temp_data = np.expand_dims(np.concatenate((temp_data,temp_data,temp_data,temp_data),1),0)
            if ii %50 < 40:
                self.train_data.append(temp_data)
                self.train_label.append(temp_label.astype(np.int64))
            else:
                self.test_data.append(temp_data)
                self.test_label.append(temp_label.astype(np.int64))
        
        if train:
            self.data = self.train_data
            self.label = self.train_label
        else:
            self.data = self.test_data
            self.label = self.test_label
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):

        img = self.data[idx]
        gt = self.label[idx]

        return img, gt


class OCdata(Dataset):
    def __init__(self, train):
        super().__init__()
        
        self.img_transforms = transforms.Compose([
            transforms.Resize((RESIZE, RESIZE), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])
        
        # MNIST
        mnist_train = torchvision.datasets.MNIST(root='./dataset_method_1', train=True, transform=self.img_transforms, download=True)
        mnist_test = torchvision.datasets.MNIST(root='./dataset_method_1', train=False, transform=self.img_transforms, download=True)

        self.data = []
        self.label = []
        self.real_label = []
        label_list = [0,1,2,3]
        num_num = 100
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

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        
        img = self.data[idx]
        gt = self.label[idx]
        
        return img, gt
