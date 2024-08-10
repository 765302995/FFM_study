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
import PySpin
import matplotlib.pyplot as plt
from Datasets import NLOSdata2pSingle, NLOSdata2p, NLOSdata2iSingle, NLOSdata2i, NLOSdata2i_4, NLOSdata2iSingle2, NLOSdatacifar2iSingle, NLOSdatacifar2, NLOSdata2i2, NLOSdata2i2_plane, NLOSdata2i_IMG, NLOS_IMG
from utils import showOn2ndDisplay
from utils import prop_simu
from func import *
from tensorboardX import SummaryWriter
from optics import *
import matlab.engine 
from ALP4 import *
import pyautogui
import time
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from zernike import RZern
from gradient_free_optimizers import HillClimbingOptimizer, ParticleSwarmOptimizer, SimulatedAnnealingOptimizer, SimulatedAnnealingOptimizer

##### hardware control #####
x_idx = 1585
move_1_up = (x_idx,440)
move_1_down = (x_idx,460)
move_2_up = (x_idx,650)
move_2_down = (x_idx,670)

def get_position(times):
    for i in range(times):
        print("current position is ",pyautogui.position())
        time.sleep(5)

def Scene_change_region_left1():
    position = [move_2_down]  
    for i in position:
        pyautogui.moveTo(i)
        pyautogui.click(clicks=1)
        time.sleep(2.0)

def Scene_change_region_right1():
    position = [move_2_up]  
    for i in position:
        pyautogui.moveTo(i)
        pyautogui.click(clicks=1)
        time.sleep(2.0)

def Scene_change_region_up():
    position = [move_1_down,move_1_down,move_1_down,move_1_down]  
    for i in position:
        pyautogui.moveTo(i)
        pyautogui.click(clicks=1)
        time.sleep(2.0)

def Scene_change_region_up1():
    position = [move_1_down]  
    for i in position:
        pyautogui.moveTo(i)
        pyautogui.click(clicks=1)
        time.sleep(2.0)

def Scene_change_region_up2():
    position = [move_1_down,move_1_down,move_1_down]  
    for i in position:
        pyautogui.moveTo(i)
        pyautogui.click(clicks=1)
        time.sleep(2.0)

def Scene_change_region_down():
    position = [move_1_up,move_1_up,move_1_up,move_1_up]  
    for i in position:
        pyautogui.moveTo(i)
        pyautogui.click(clicks=1)
        time.sleep(2.0) 

def Scene_change_region_down1():
    position = [move_1_up]  
    for i in position:
        pyautogui.moveTo(i)
        pyautogui.click(clicks=1)
        time.sleep(2.0) 

def Scene_change_region_down2():
    position = [move_1_up,move_1_up,move_1_up]  
    for i in position:
        pyautogui.moveTo(i)
        pyautogui.click(clicks=1)
        time.sleep(2.0) 

##### ----------initialize the SLM and camera---------- #####
##### camera preparation #####
target_fps = 30
max_exposure = 1000000.0 / target_fps
exposure_time = 1000
system = PySpin.System.GetInstance()
version = system.GetLibraryVersion()
print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))
cam_list = system.GetCameras()
num_cameras = cam_list.GetSize()
print('Number of cameras detected: %d' % num_cameras)

cam1 = cam_list[0]
result = True
cam1.Init()
nodemap_tldevice = cam1.GetTLDeviceNodeMap()
nodemap_tlstream = cam1.GetTLStreamNodeMap()
nodemap = cam1.GetNodeMap()
StreamModeNode = PySpin.CEnumerationPtr(nodemap_tlstream.GetNode("StreamBufferHandlingMode"))
StreamModeNode.SetIntValue(StreamModeNode.GetEntryByName("NewestOnly").GetValue())
exposureUpperLimitPtr = PySpin.CFloatPtr(nodemap.GetNode("AutoExposureExposureTimeUpperLimit"))
exposureUpperLimitPtr.SetValue(max_exposure)
fpsEnablePtr = PySpin.CBooleanPtr(nodemap.GetNode("AcquisitionFrameRateEnable"))
fpsEnablePtr.SetValue(True)
fpsPtr = PySpin.CFloatPtr(nodemap.GetNode("AcquisitionFrameRate"))
fpsPtr.SetValue(target_fps)
exposureModePtr = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureMode"))
exposureModePtr.SetIntValue(exposureModePtr.GetEntryByName("Timed").GetValue())
exposureTimePtr = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime"))
exposureTimePtr.SetValue(exposure_time)
gainAutoPtr = PySpin.CEnumerationPtr(nodemap.GetNode("GainAuto"))
gainAutoPtr.SetIntValue(gainAutoPtr.GetEntryByName("Off").GetValue())
gainPtr = PySpin.CFloatPtr(nodemap.GetNode("Gain"))
gainPtr.SetValue(0)
cam1.BeginAcquisition()
_ = cam1.GetNextImage(1000)
device_serial_number = ''
node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
if PySpin.IsAvailable(node_device_serial_number) and PySpin.IsReadable(node_device_serial_number):
    device_serial_number = node_device_serial_number.GetValue()
    print('Device serial number retrieved as %s...' % device_serial_number)


##### SLM preparation #####
eng = matlab.engine.start_matlab()  
success = eng.setup_SLM()
temp_stack_path = 'temp.bmp'


pitch = 1
x = 1280
y = 1024
x2 = 1920
y2 = 1200
monitorNo = 1
windowNo = 0
monitorNo2 = 3
windowNo2 = 1
xShift = 0
yShift = 0
array_size = x * y
array_size2 = x2 * y2
FARRAY = c_uint8 * array_size
FARRAY2 = c_uint8 * array_size2
farray = FARRAY(0)
farray2 = FARRAY2(0)


##### ---------initialize---------- #####
## base params
MASK_SIZE = 400
LR = 0.01
MAX_EPOCH = 300
DECAY = 0.99
BATCH_SIZE = 9

cali_1_M = np.load('./basedata/cali_M_1.npy')

exp_name = 'debug'
writer = SummaryWriter(os.path.join('./runs', exp_name))
out_put_dir = os.path.join('./output', exp_name)
debugidx = 4
debug_dir = os.path.join('./output', 'debug_phase' + str(debugidx))
if not os.path.exists(debug_dir):
    os.makedirs(debug_dir)
if not os.path.exists(out_put_dir):
    os.makedirs(out_put_dir)

device = torch.device("cpu")

## SLM_phase
phase_init = 2 * np.pi * np.random.random(size=(MASK_SIZE, MASK_SIZE)).astype('float32')
grainSize = 4
phase_init_blur = scipy.ndimage.filters.gaussian_filter(phase_init, grainSize/2)
SLM_phase = torch.nn.Parameter(torch.from_numpy(phase_init_blur))

## dataset
use_gpu = False
gpu_args = {'num_workers': 12, 'pin_memory': True} if use_gpu else {}

train_dataset = NLOS_IMG(train=True)
test_dataset = NLOS_IMG(train=False)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=BATCH_SIZE, shuffle=False, **gpu_args)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=1, shuffle=False, **gpu_args)

## optimizer
optimizer = torch.optim.Adam([SLM_phase], lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

## system
phase_path = 'white.bmp'
im = cv2.imread(phase_path,0)
im = im.T
rate = 213/255
im = 1.0*im*rate
im = np.array(im, dtype = np.uint8)
im_gray = np.reshape(im, -1)
showOn2ndDisplay(monitorNo, windowNo, x, xShift, y, yShift, im_gray.ctypes.data_as(POINTER(c_ubyte)))
time.sleep(0.5)

over_phase2 = torch.from_numpy(0.5*2*np.pi * im / 213)
over_phase2 = over_phase2[440:840, 312:712]

##### ----------initialize---------- #####
cart = RZern(6)
L, K = 400, 400
ddx = np.linspace(-1.0, 1.0, K)
ddy = np.linspace(-1.0, 1.0, L)
xv, yv = np.meshgrid(ddx, ddy)
cart.make_cart_grid(xv, yv)

zern_list = []
clist = np.zeros(cart.nk)
mask_valid = np.ones((400,400))
for i in range(cart.nk):
    clist *= 0.0
    clist[i] = 1.0

    Phi = cart.eval_grid(clist, matrix=True)
    idx = np.where(np.isnan(Phi) == True)
    Phi[idx] = 0
    mask_valid[idx] = 0
    zern_list.append(Phi)

## phase create from vector
def create_phase(para):
    outs = np.zeros((400,400))
    for i in range(len(para)):
        x_temp = para['x'+str(i)]
        outs += x_temp *zern_list[i]
    return torch.from_numpy(outs)

target_scene = torch.from_numpy(np.load('target_real_scene.npy'))

## forward model 
def model(para):
    function_list = ['Scene_change_region_up1()',
                        'Scene_change_region_down1()',
                         'Scene_change_region_down1()',
                         'Scene_change_region_left1()',
                         'Scene_change_region_up1()',
                         'Scene_change_region_up1()',
                         'Scene_change_region_left1()',
                         'Scene_change_region_down1()',
                         'Scene_change_region_down1()']
    
    Scene_change_region_right1()
    time.sleep(0.5)

    loss = 0
    for ii in range(9):
        eval(function_list[ii])
        time.sleep(0.5)

        SLM_phase = create_phase(para)

        SLM_input_p = SLM_create_phase(-over_phase2+SLM_phase, 0)
        cv2.imwrite(temp_stack_path, np.array(SLM_input_p))
        success = eng.set_img(temp_stack_path)
        time.sleep(0.3)

        intensity = cam1.GetNextImage(1000)
        image_captured = intensity.GetNDArray() / 255
        out_inten = get_crop(image_captured, cali_1_M, flip=False)
        outmap = torch.tensor(np.sqrt(out_inten).copy())

        np.save(os.path.join(out_put_dir, 'outmap_%03d.npy'%ii), torch.square(outmap))

        x_abs = torch.square(outmap)
        loss += F.mse_loss(x_abs, target_scene[ii,0])
    
    Scene_change_region_right1()
    time.sleep(0.5)
    Scene_change_region_up1()

    return -loss.item()

##### ---------train--------- #####
search_space = {}
for i in range(cart.nk):
    search_space['x'+str(i)] = np.arange(-2, 2, 0.0001)

opt = SimulatedAnnealingOptimizer(search_space)
opt.search(model, n_iter=500)

best_para = opt.best_para
memory_dict = opt.memory_dict