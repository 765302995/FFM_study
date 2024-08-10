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
from utils import prop_simu
from func import *
from tensorboardX import SummaryWriter
from optics import *
import matlab.engine 
from ALP4 import *

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
def cal_plane_phase(shift_phase_pi2, shift_phase_pi, cali_0_M, cam, gt_plane_phase, tcpclient):
    #DMD
    object_input = torch.zeros((400,400))
    data_send = np.array(object_input)
    data_string = pickle.dumps(data_send)
    tcpclient.send(data_string)
    time.sleep(1.0)

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

    kernel_size = 50
    out_inten_0_all = 0
    for i in range(repeat_num):
        intensity = cam.GetNextImage(1000)
        image_captured = intensity.GetNDArray() / 255
        out_inten2 = get_crop(image_captured, cali_0_M, flip=True)
        out_inten2 = cv2.blur(out_inten2, (kernel_size,kernel_size))
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
        out_inten2 = cv2.blur(out_inten2, (kernel_size,kernel_size))
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
        out_inten2 = cv2.blur(out_inten2, (kernel_size,kernel_size))
        out_inten_pi_temp = torch.tensor(out_inten2.copy())
        out_inten_pi_all += out_inten_pi_temp
    out_inten_pi = out_inten_pi_all / repeat_num

    out_phase_plane = measure_phase_multi(out_inten_pi2, out_inten_0, out_inten_pi)
    shift = out_phase_plane - gt_plane_phase

    return shift

##### detection region of output #####
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

cam = cam_list[0]
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
exposureTimePtr.SetValue(5000)
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


cam1 = cam_list[1]
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
exposureTimePtr.SetValue(200)
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

host = "XXX" 
port = 30000
tcpclient = socket.socket()
tcpclient.connect((host, port))

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
LR = 0.03
MAX_EPOCH = 500
DECAY = 0.99
BATCH_SIZE = 2

cali_1_M = np.load('./basedata/cali_M_1.npy')
cali_0_M = np.load('./basedata/cali_M_0.npy')

gt_plane_phase = torch.from_numpy(np.load('gt_plane_phase.npy'))

exp_name = 'debug'
writer = SummaryWriter(os.path.join('./runs', exp_name))
out_put_dir = os.path.join('./output', exp_name)
debugidx = 8
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

phase_scatter_init = np.load('phase_init_blur_s.npy')
phase_scatter_init = torch.from_numpy(phase_scatter_init)
phase_scatter_init = F.interpolate(phase_scatter_init.unsqueeze(0).unsqueeze(0), (100,100), mode='nearest')
phase_scatter = F.interpolate(phase_scatter_init, (400,400), mode='nearest').squeeze()

shift_phase_pi2 = 0.5*np.pi*torch.ones(SLM_phase.shape)
shift_phase_pi = np.pi*torch.ones(SLM_phase.shape)
repeat_num = 1
rotate_angle = 1

## dataset
use_gpu = False
gpu_args = {'num_workers': 12, 'pin_memory': True} if use_gpu else {}

train_dataset = NLOSdata2i2(train=True)
test_dataset = NLOSdata2i2(train=False)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=BATCH_SIZE, shuffle=True, **gpu_args)
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
        train_correct_all_batch = 0
        data_batch, target_batch = data, target

        if b%1 == 0:
            shift = cal_plane_phase(shift_phase_pi2, shift_phase_pi, cali_0_M, cam, gt_plane_phase, tcpclient)

        for i in range(data_batch.shape[0]):
            target_temp = target_batch[i].to(torch.float64)
            SLM_outinten = data_batch[i,0]

            SLM_out_phase = SLM_phase

            SLM_outinten = F.interpolate(SLM_outinten.unsqueeze(0).unsqueeze(0), (100,100), mode='nearest')
            SLM_outinten = F.interpolate(SLM_outinten, (400,400), mode='nearest').squeeze()

            ## roate phase
            phase_scatter = transforms.functional.rotate(phase_scatter, rotate_angle, fill=-1) 
            phase_random = 2*np.pi*np.random.random(phase_scatter.shape)
            phase_random = scipy.ndimage.filters.gaussian_filter(phase_random, 0.5)
            phase_random = torch.from_numpy(phase_random).to(torch.float32)
            idx = torch.where(phase_scatter == -1)
            phase_scatter[idx] = phase_random[idx]

            object_input = SLM_outinten * np.pi
            object_input = object_input + phase_scatter
            data_send = np.array(object_input)
            data_string = pickle.dumps(data_send)
            tcpclient.send(data_string)
            time.sleep(1.0)

            ##### forward data #####
            phase_path = 'white.bmp' #shiftnew_cal_400px_125_80cm.bmp
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
            exposureTimePtr.SetValue(200)
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

            loss = F.mse_loss(x_abs.unsqueeze(0), target_temp.unsqueeze(0))
            target_out = detector_region_10(target_temp.unsqueeze(0))
            target_label = target_out.argmax(dim=1, keepdim=True)
            output = detector_region_10(x_abs.unsqueeze(0))
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target_label.view_as(pred)).sum().item()

            train_loss_all += loss.item()
            train_loss_all_batch += loss.item()
            train_correct_all += correct
            train_correct_all_batch += correct
            loss.backward()

            grad_UNo = grad_record['U_No_av']
            UNo_conj = torch.conj(U_No)
            E_field = torch.mul(grad_UNo, UNo_conj)

            E_field = E_field / torch.abs(E_field).max().detach()
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
            ## amp
            cam1.EndAcquisition()
            nodemap = cam1.GetNodeMap()
            exposureModePtr = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureMode"))
            exposureModePtr.SetIntValue(exposureModePtr.GetEntryByName("Timed").GetValue())
            exposureTimePtr = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime"))
            exposureTimePtr.SetValue(exposure_time*8)
            cam1.BeginAcquisition()

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

            rate = 0.1/abs(grad_1).max()
            grad_1 = grad_1 * rate

            grad_1_all += grad_1

        grad_1_mean = grad_1_all / target_batch.shape[0]

        ##### update #####
        optimizer.zero_grad()
        SLM_phase.grad = grad_1_mean
        optimizer.step()
        print(f"Train---batch {b}: loss {train_loss_all_batch/target_batch.shape[0]:.6f} correct {train_correct_all_batch/target_batch.shape[0]:.2f}. ", end=' ') #\r

    loss_epoch = train_loss_all/len(train_loader)/BATCH_SIZE
    correct_epoch = train_correct_all/len(train_loader)/BATCH_SIZE

    writer.add_scalar('train_loss', loss_epoch, global_step=epoch)
    writer.add_scalar('correct', correct_epoch, global_step=epoch)
    print(f"Train---Epoch {epoch}: loss {loss_epoch:.6f}, accuracy {correct_epoch}.")

##### ---------test--------- #####
    if epoch % 5 == 0:
        test_loss_all = 0
        test_correct_all = 0

        cam1.EndAcquisition()
        nodemap = cam1.GetNodeMap()
        exposureModePtr = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureMode"))
        exposureModePtr.SetIntValue(exposureModePtr.GetEntryByName("Timed").GetValue())
        exposureTimePtr = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime"))
        exposureTimePtr.SetValue(200)
        cam1.BeginAcquisition()

        for b, (data, target) in enumerate(test_loader):
            SLM_outinten = data[0,0]
            SLM_out_phase = SLM_phase

            SLM_outinten = F.interpolate(SLM_outinten.unsqueeze(0).unsqueeze(0), (100,100), mode='nearest')
            SLM_outinten = F.interpolate(SLM_outinten, (400,400), mode='nearest').squeeze()

            object_input = SLM_outinten * np.pi
            object_input = object_input + phase_scatter
            data_send = np.array(object_input)
            data_string = pickle.dumps(data_send)
            tcpclient.send(data_string)
            time.sleep(1.0)

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
            ## amp
            intensity = cam1.GetNextImage(1000)
            image_captured = intensity.GetNDArray() / 255

            out_inten = get_crop(image_captured, cali_1_M, flip=False)
            out_amp = torch.tensor(np.sqrt(out_inten).copy())

            x_abs = torch.square(out_amp)

            loss = F.mse_loss(x_abs.unsqueeze(0), target)
            target_out = detector_region_10(target)
            target_label = target_out.argmax(dim=1, keepdim=True)
            output = detector_region_10(x_abs.unsqueeze(0))
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target_label.view_as(pred)).sum().item()

            test_loss_all += loss.item()
            test_correct_all += correct
        
        writer.add_scalar('test_loss', test_loss_all/len(test_loader), global_step=epoch)
        writer.add_scalar('test_correct', test_correct_all/len(test_loader), global_step=epoch)
        print(f"Test---Epoch {epoch}: loss {test_loss_all/len(test_loader):.6f}, accuracy {test_correct_all/len(test_loader)}.")


