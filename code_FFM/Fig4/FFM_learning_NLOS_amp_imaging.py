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
from Datasets import *
from utils import showOn2ndDisplay
from func import *
from tensorboardX import SummaryWriter
from optics import *
import matlab.engine 
from ALP4 import *
import pyautogui
import time
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

##### hardware control #####
x_idx = 1601
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

##### phase calculation #####
def measure_phase_multi(in_pi2, in_0, in_pi):
    theta = torch.arctan(2*(in_pi2 - 0.5*in_0 - 0.5*in_pi) / (in_0 - in_pi))

    sin_theta = in_pi2 - 0.5*in_0 - 0.5*in_pi
    cos_theta = in_0 - in_pi
    idx = torch.where(cos_theta < 0)
    theta[idx] += np.pi

    error_flag = in_0 - in_pi
    idx = torch.where(error_flag == 0)
    theta[idx] = 0.5*np.pi

    return theta

##### measure phase of plane wave #####
def cal_plane_phase(shift_phase_pi2, shift_phase_pi, cali_0_M, cam, gt_plane_phase):
    phase_path = 'white.bmp'
    im = cv2.imread(phase_path,0)
    im = im.T
    rate = 213/255
    im = 1.0*im*rate
    im = np.array(im, dtype = np.uint8)
    im_gray = np.reshape(im, -1)
    showOn2ndDisplay(monitorNo, windowNo, x, xShift, y, yShift, im_gray.ctypes.data_as(POINTER(c_ubyte)))
    time.sleep(0.15)

    over_phase2 = torch.from_numpy(0.5*2*np.pi * im / 213)
    over_phase2 = over_phase2[440:840, 312:712]

    SLM_input_p = SLM_create_phase(-over_phase2, 0)
    cv2.imwrite(temp_stack_path, np.array(SLM_input_p))
    success = eng.set_img(temp_stack_path)
    time.sleep(0.15)

    out_inten_0_all = 0
    for i in range(repeat_num):
        intensity = cam.GetNextImage(1000)
        image_captured = intensity.GetNDArray() / 255
        out_inten2 = get_crop(image_captured, cali_0_M, flip=True)
        out_inten_0_temp = torch.tensor(out_inten2.copy())
        out_inten_0_all += out_inten_0_temp
    out_inten_0 = out_inten_0_all / repeat_num

    SLM_input_p = SLM_create_phase(-over_phase2-shift_phase_pi2, 0)
    cv2.imwrite(temp_stack_path, np.array(SLM_input_p))
    success = eng.set_img(temp_stack_path)
    time.sleep(0.15)

    out_inten_pi2_all = 0
    for i in range(repeat_num):
        intensity = cam.GetNextImage(1000)
        image_captured = intensity.GetNDArray() / 255
        out_inten2 = get_crop(image_captured, cali_0_M, flip=True)
        out_inten_pi2_temp = torch.tensor(out_inten2.copy())
        out_inten_pi2_all += out_inten_pi2_temp
    out_inten_pi2 = out_inten_pi2_all / repeat_num

    SLM_input_p = SLM_create_phase(-over_phase2-shift_phase_pi, 0)
    cv2.imwrite(temp_stack_path, np.array(SLM_input_p))
    success = eng.set_img(temp_stack_path)
    time.sleep(0.15)

    out_inten_pi_all = 0
    for i in range(repeat_num):
        intensity = cam.GetNextImage(1000)
        image_captured = intensity.GetNDArray() / 255
        out_inten2 = get_crop(image_captured, cali_0_M, flip=True)
        out_inten_pi_temp = torch.tensor(out_inten2.copy())
        out_inten_pi_all += out_inten_pi_temp
    out_inten_pi = out_inten_pi_all / repeat_num

    out_phase_plane = measure_phase_multi(out_inten_pi2, out_inten_0, out_inten_pi)
    shift = out_phase_plane - gt_plane_phase

    return shift

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


cam = cam_list[1]
result = True
cam.Init()
nodemap_tldevice = cam.GetTLDeviceNodeMap()
nodemap_tlstream = cam.GetTLStreamNodeMap()
nodemap = cam.GetNodeMap()
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
exposureTimePtr.SetValue(exposure_time*2)
gainAutoPtr = PySpin.CEnumerationPtr(nodemap.GetNode("GainAuto"))
gainAutoPtr.SetIntValue(gainAutoPtr.GetEntryByName("Off").GetValue())
gainPtr = PySpin.CFloatPtr(nodemap.GetNode("Gain"))
gainPtr.SetValue(0)
cam.BeginAcquisition()
_ = cam.GetNextImage(1000)
device_serial_number = ''
node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
if PySpin.IsAvailable(node_device_serial_number) and PySpin.IsReadable(node_device_serial_number):
    device_serial_number = node_device_serial_number.GetValue()
    print('Device serial number retrieved as %s...' % device_serial_number)


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
cali_0_M = np.load('./basedata/cali_M_0.npy')

gt_plane_phase = torch.from_numpy(np.load('gt_plane_phase.npy'))

exp_name = 'debug'
writer = SummaryWriter(os.path.join('./runs', exp_name))
out_put_dir = os.path.join('./output', exp_name)
debugidx = 1
debug_dir = os.path.join('./output', 'debug' + str(debugidx))
if not os.path.exists(debug_dir):
    os.makedirs(debug_dir)
if not os.path.exists(out_put_dir):
    os.makedirs(out_put_dir)

## SLM phase
phase_init = 2 * np.pi * np.random.random(size=(MASK_SIZE, MASK_SIZE)).astype('float32')
grainSize = 4
phase_init_blur = scipy.ndimage.filters.gaussian_filter(phase_init, grainSize/2)
SLM_phase = torch.nn.Parameter(torch.from_numpy(phase_init_blur))

shift_phase_pi2 = 0.5*np.pi*torch.ones(SLM_phase.shape)
shift_phase_pi = np.pi*torch.ones(SLM_phase.shape)
repeat_num = 1

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

##### ---------train--------- #####
for epoch in range(MAX_EPOCH):
    train_loss_all = 0
    train_correct_all = 0

    for b, (data, target) in enumerate(train_loader):
        grad_1_all = 0
        grad_1_amp_all = 0
        train_loss_all_batch = 0 
        data_batch, target_batch = data, target

        ## hradware move
        Scene_change_region_right1()
        time.sleep(0.5)

        function_list = ['Scene_change_region_up1()',
                        'Scene_change_region_down1()',
                         'Scene_change_region_down1()',
                         'Scene_change_region_left1()',
                         'Scene_change_region_up1()',
                         'Scene_change_region_up1()',
                         'Scene_change_region_left1()',
                         'Scene_change_region_down1()',
                         'Scene_change_region_down1()']
        
        save_idx = np.random.randint(0,data_batch.shape[0]-1)

        for ii in range(data_batch.shape[0]):
            shift = torch.from_numpy(np.load('phase_shift_new.npy'))

            eval(function_list[ii])
            time.sleep(1.0)

            target_temp = target_batch[ii,0].to(torch.float64)
            SLM_outinten = data_batch[ii,0]
            SLM_out_phase = SLM_phase

            ##### forward data #####
            phase_path = 'white.bmp'
            im = cv2.imread(phase_path,0)
            im = im.T
            rate = 213/255
            im = 1.0*im*rate
            im = np.array(im, dtype = np.uint8)
            im_gray = np.reshape(im, -1)
            showOn2ndDisplay(monitorNo, windowNo, x, xShift, y, yShift, im_gray.ctypes.data_as(POINTER(c_ubyte)))
            time.sleep(0.15)

            over_phase = torch.from_numpy(0.5*2*np.pi * im / 213)
            over_phase = over_phase[440:840, 312:712]

            SLM_input_p = SLM_create_phase(-over_phase+SLM_out_phase, 0)
            cv2.imwrite(temp_stack_path, np.array(SLM_input_p))
            success = eng.set_img(temp_stack_path)
            time.sleep(0.15)

            ### measure ###
            cam1.EndAcquisition()
            nodemap = cam1.GetNodeMap()
            exposureModePtr = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureMode"))
            exposureModePtr.SetIntValue(exposureModePtr.GetEntryByName("Timed").GetValue())
            exposureTimePtr = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime"))
            exposureTimePtr.SetValue(exposure_time)
            cam1.BeginAcquisition()

            ## amp
            intensity = cam1.GetNextImage(1000)
            image_captured = intensity.GetNDArray() / 255
            out_inten = get_crop(image_captured, cali_1_M, flip=False)
            out_amp = torch.tensor(np.sqrt(out_inten).copy())

            ## phase
            out_inten_0_all = 0
            for i in range(repeat_num):
                intensity = cam.GetNextImage(1000)
                image_captured = intensity.GetNDArray() / 255
                out_inten2 = get_crop(image_captured, cali_0_M, flip=False)
                out_inten_0_temp = torch.tensor(out_inten2.copy())
                out_inten_0_all += out_inten_0_temp
            out_inten_0 = out_inten_0_all / repeat_num

            SLM_input_p = SLM_create_phase(-over_phase+SLM_out_phase - shift_phase_pi2, 0)
            cv2.imwrite(temp_stack_path, np.array(SLM_input_p))
            success = eng.set_img(temp_stack_path)
            time.sleep(0.15)

            out_inten_pi2_all = 0
            for i in range(repeat_num):
                intensity = cam.GetNextImage(1000)
                image_captured = intensity.GetNDArray() / 255
                out_inten2 = get_crop(image_captured, cali_0_M, flip=False)
                out_inten_pi2_temp = torch.tensor(out_inten2.copy())
                out_inten_pi2_all += out_inten_pi2_temp
            out_inten_pi2 = out_inten_pi2_all / repeat_num

            SLM_input_p = SLM_create_phase(-over_phase+SLM_out_phase - shift_phase_pi, 0)
            cv2.imwrite(temp_stack_path, np.array(SLM_input_p))
            success = eng.set_img(temp_stack_path)
            time.sleep(0.15)
            out_inten_pi_all = 0
            for i in range(repeat_num):
                intensity = cam.GetNextImage(1000)
                image_captured = intensity.GetNDArray() / 255
                out_inten2 = get_crop(image_captured, cali_0_M, flip=False)
                out_inten_pi_temp = torch.tensor(out_inten2.copy())
                out_inten_pi_all += out_inten_pi_temp
            out_inten_pi = out_inten_pi_all / repeat_num

            out_phase = measure_phase_multi(out_inten_pi2, out_inten_0, out_inten_pi)
            out_phase = out_phase - shift

            Pkd_1 = torch.ones(SLM_out_phase.shape) * torch.exp(1j * SLM_out_phase)
            
            U_No = out_amp * torch.exp(1j * out_phase)

            ##### Error field generation #####
            grad_record = {}
            def save_grad(name):
                def hook(grad):
                    grad_record[name] = grad
                return hook
            
            x_abs = torch.square(torch.abs(U_No))
            x_abs.requires_grad = True
            U_No_av = x_abs
            U_No_av.register_hook(save_grad('U_No_av'))

            loss = 1-ssim(x_abs.unsqueeze(0).unsqueeze(0), target_temp.unsqueeze(0).unsqueeze(0))
            train_loss_all += loss.item()
            train_loss_all_batch += loss.item()
            loss.backward()

            grad_UNo = grad_record['U_No_av']
            UNo_conj = torch.conj(U_No)
            E_field = torch.mul(grad_UNo, UNo_conj)

            E_field = 2.0 * E_field / torch.abs(E_field).max().detach()
            E_field_angle = E_field.angle()

            ##### forward error #####
            SLM_input, over_phase, input_amp = SLM_create_amp(torch.abs(E_field))
            im_gray = np.reshape(np.array(SLM_input),-1)
            showOn2ndDisplay(monitorNo, windowNo, x, xShift, y, yShift, im_gray.ctypes.data_as(POINTER(c_ubyte)))
            time.sleep(0.15)

            SLM_input_p = SLM_create_phase(-over_phase+E_field_angle, 0)
            cv2.imwrite(temp_stack_path, np.array(SLM_input_p))
            success = eng.set_img(temp_stack_path)
            time.sleep(0.15)

            ### measure ###
            cam1.EndAcquisition()
            nodemap = cam1.GetNodeMap()
            exposureModePtr = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureMode"))
            exposureModePtr.SetIntValue(exposureModePtr.GetEntryByName("Timed").GetValue())
            exposureTimePtr = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime"))
            exposureTimePtr.SetValue(exposure_time*2)
            cam1.BeginAcquisition()

            ## amp
            intensity = cam1.GetNextImage(1000)
            image_captured = intensity.GetNDArray() / 255
            out_e_inten = get_crop(image_captured, cali_1_M, flip=False)
            out_e_amp = torch.tensor(np.sqrt(out_e_inten).copy())

            ## phase 
            out_e_inten_0_all = 0
            for i in range(repeat_num):
                intensity = cam.GetNextImage(1000)
                image_captured = intensity.GetNDArray() / 255

                out_e_inten2 = get_crop(image_captured, cali_0_M, flip=False)
                out_e_inten_0_temp = torch.tensor(out_e_inten2.copy())
                out_e_inten_0_all += out_e_inten_0_temp
            out_e_inten_0 = out_e_inten_0_all / repeat_num

            SLM_input_p = SLM_create_phase(-over_phase+E_field_angle - shift_phase_pi2, 0)
            cv2.imwrite(temp_stack_path, np.array(SLM_input_p))
            success = eng.set_img(temp_stack_path)
            time.sleep(0.15)
            out_e_inten_pi2_all = 0
            for i in range(repeat_num):
                intensity = cam.GetNextImage(1000)
                image_captured = intensity.GetNDArray() / 255
                out_e_inten2 = get_crop(image_captured, cali_0_M, flip=False)
                out_e_inten_pi2_temp = torch.tensor(out_e_inten2.copy())
                out_e_inten_pi2_all += out_e_inten_pi2_temp
            out_e_inten_pi2 = out_e_inten_pi2_all / repeat_num

            SLM_input_p = SLM_create_phase(-over_phase+E_field_angle - shift_phase_pi, 0)
            cv2.imwrite(temp_stack_path, np.array(SLM_input_p))
            success = eng.set_img(temp_stack_path)
            time.sleep(0.15)
            out_e_inten_pi_all = 0
            for i in range(repeat_num):
                intensity = cam.GetNextImage(1000)
                image_captured = intensity.GetNDArray() / 255
                out_e_inten2 = get_crop(image_captured, cali_0_M, flip=False)
                out_e_inten_pi_temp = torch.tensor(out_e_inten2.copy())
                out_e_inten_pi_all += out_e_inten_pi_temp
            out_e_inten_pi = out_e_inten_pi_all / repeat_num

            out_e_phase = measure_phase_multi(out_e_inten_pi2, out_e_inten_0, out_e_inten_pi)
            out_e_phase = out_e_phase - shift

            Pke_1 = out_e_amp * torch.exp(1j * out_e_phase)
            
            grad_1 = 2*torch.mul(1j * Pkd_1.detach(), Pke_1).real
            grad_1 = grad_1.to(torch.float32)

            rate = 0.002/abs(grad_1).max()
            grad_1 = grad_1 * rate
            grad_1_all += grad_1

        grad_1_mean = grad_1_all / target_batch.shape[0]

        ##### update #####
        optimizer.zero_grad()
        SLM_phase.grad = grad_1_mean
        optimizer.step()
        print(f"Train---batch {b}: loss {train_loss_all_batch/target_batch.shape[0]:.6f}.", end='\r')

    loss_epoch = train_loss_all/len(train_loader)/BATCH_SIZE

    writer.add_scalar('train_loss', loss_epoch, global_step=epoch)
    print(f"Train---Epoch {epoch}: loss {loss_epoch:.6f}.")

    ## hardware move back
    Scene_change_region_right1()
    time.sleep(0.5)
    Scene_change_region_up1()