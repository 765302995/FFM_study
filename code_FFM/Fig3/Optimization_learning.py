import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from zernike import RZern
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import pdb
import random
import numpy as np
import scipy
import cv2
import PySpin
import matlab.engine 
from utils import showOn2ndDisplay
from optics import ASM_propagate
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_wine
from gradient_free_optimizers import HillClimbingOptimizer, ParticleSwarmOptimizer, SimulatedAnnealingOptimizer, SimulatedAnnealingOptimizer
from Datasets import NLOSdatafocus
import matplotlib.pyplot as plt
from PIL import Image
from tensorboardX import SummaryWriter
from ALP4 import *
from ctypes import *
from func import *

###### camera preparation ######
target_fps = 30
max_exposure = 1000000.0 / target_fps
exposure_time = 100
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


###### SLM preparation ######
eng = matlab.engine.start_matlab()  
success = eng.setup_SLM()
temp_stack_path = 'temp.bmp'

eng2 = matlab.engine.start_matlab()
success = eng2.setup_SLM_1024()
temp_stack_obj_path = 'temp0.bmp'


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

## system
phase_path = 'white.bmp' #shiftnew_cal_400px_125_80cm.bmp
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

phase_scatter = torch.zeros(over_phase2.shape)
object_input = phase_scatter
SLM_input_p = np.array(SLM_create_object_phase(object_input))
cv2.imwrite(temp_stack_obj_path, SLM_input_p)
success = eng2.set_img_1024(temp_stack_obj_path)
time.sleep(0.5)

cali_1_M = np.load('./basedata/cali_M_1.npy')

## phase create from vector
def create_phase(para):
    outs = np.zeros((400,400))
    for i in range(len(para)):
        x_temp = para['x'+str(i)]
        outs += x_temp *zern_list[i]
    return torch.from_numpy(outs)

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

use_gpu = False
device = torch.device(f"cuda:{0}" if use_gpu else "cpu")

gpu_args = {'num_workers': 12, 'pin_memory': True} if use_gpu else {}
exp_name = 'debug'
writer = SummaryWriter(os.path.join('./runs', exp_name))
out_put_dir = os.path.join('./output', exp_name)
debugidx = 0
debug_dir = os.path.join('./output', 'debug' + str(debugidx))
if not os.path.exists(debug_dir):
    os.makedirs(debug_dir)
if not os.path.exists(out_put_dir):
    os.makedirs(out_put_dir)

target = torch.zeros((400,400))
target[190:210,190:210] = 3

## forward model 
def model(para):
    SLM_phase = create_phase(para)

    SLM_input_p = SLM_create_phase(-over_phase2+SLM_phase, 0)
    cv2.imwrite(temp_stack_path, np.array(SLM_input_p))
    success = eng.set_img(temp_stack_path)
    time.sleep(0.25)

    intensity = cam1.GetNextImage(1000)
    image_captured = intensity.GetNDArray() / 255
    out_inten = get_crop(image_captured, cali_1_M, flip=False)
    outmap = torch.tensor(np.sqrt(out_inten).copy())

    x_abs = torch.square(outmap)
    loss = F.mse_loss(x_abs, target)

    return -loss.item()

##### ---------optimization--------- #####
search_space = {}
for i in range(cart.nk):
    search_space['x'+str(i)] = np.arange(-2, 2, 0.0001)

opt = SimulatedAnnealingOptimizer(search_space)
opt.search(model, n_iter=2000)

best_para = opt.best_para
memory_dict = opt.memory_dict
train_loss = []
for i,data in enumerate(memory_dict):
    loss_temp = memory_dict[data]
    train_loss.append(loss_temp)

##### ---------test--------- #####
SLM_phase = create_phase(best_para)

SLM_input_p = SLM_create_phase(-over_phase2+SLM_phase, 0)
cv2.imwrite(temp_stack_path, np.array(SLM_input_p))
success = eng.set_img(temp_stack_path)
time.sleep(0.25)

intensity = cam1.GetNextImage(1000)
image_captured = intensity.GetNDArray() / 255
out_inten = get_crop(image_captured, cali_1_M, flip=False)
outmap = torch.tensor(np.sqrt(out_inten).copy())

x_abs = torch.square(outmap)
loss = F.mse_loss(x_abs, target)
print(loss)