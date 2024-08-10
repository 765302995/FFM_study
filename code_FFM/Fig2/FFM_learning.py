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
import torchvision
import torchvision.transforms as transforms
import scipy
import scipy.ndimage
import PySpin
import matplotlib.pyplot as plt
from Datasets import *
from utils import showOn2ndDisplay
from func import *
from tensorboardX import SummaryWriter
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
def cal_plane_phase(shift_phase_pi2, shift_phase_pi, cali_0_M, cam):
    #plane phase
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
        out_inten2 = get_crop(image_captured, cali_0_M, flip=False)
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
        out_inten2 = get_crop(image_captured, cali_0_M, flip=False)
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
        out_inten2 = get_crop(image_captured, cali_0_M, flip=False)
        out_inten2 = cv2.blur(out_inten2, (kernel_size,kernel_size))
        out_inten_pi_temp = torch.tensor(out_inten2.copy())
        out_inten_pi_all += out_inten_pi_temp
    out_inten_pi = out_inten_pi_all / repeat_num

    out_phase_plane = measure_phase_multi(out_inten_pi2, out_inten_0, out_inten_pi)
    shift = out_phase_plane - gt_plane_phase

    return shift

##### forward propagation for one layer #####
def forward_layer(SLM_outinten, SLM_out_phase, shift_phase_pi2, shift_phase_pi, cali_1_M, cali_0_M, shift, cam1, cam):
    ##### load params #####
    SLM_input, over_phase, input_amp = SLM_create_amp(SLM_outinten)
    im_gray = np.reshape(np.array(SLM_input),-1)
    showOn2ndDisplay(monitorNo, windowNo, x, xShift, y, yShift, im_gray.ctypes.data_as(POINTER(c_ubyte)))
    time.sleep(0.15)

    SLM_input_p = SLM_create_phase(-over_phase+SLM_out_phase, 0)
    cv2.imwrite(temp_stack_path, np.array(SLM_input_p))
    success = eng.set_img(temp_stack_path)
    time.sleep(0.15)

    ##### measure #####
    ## intensity 
    intensity = cam1.GetNextImage(1000)
    image_captured = intensity.GetNDArray() / 255

    out_inten = get_crop(image_captured, cali_1_M, flip=False)
    out_amp = torch.tensor(np.sqrt(out_inten).copy())

    ## calculate the phase 
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
    return out_amp, out_phase

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
LR = 0.05
MAX_EPOCH = 200
DECAY = 0.99
BATCH_SIZE = 10
EPOCH_SVAE = 5

OBJECT_SIZE = 28
UP_SAMPLE = 14
RESIZE = int(OBJECT_SIZE * UP_SAMPLE)
PADDING_SZIE = (MASK_SIZE - RESIZE) // 2
img_transforms = transforms.Compose([
        transforms.Resize((RESIZE, RESIZE), interpolation=Image.BICUBIC),
        transforms.Pad((PADDING_SZIE,PADDING_SZIE)),
        transforms.ToTensor(),
    ])

cali_1_M = np.load('./basedata/cali_M_1.npy')
cali_0_M = np.load('./basedata/cali_M_0.npy')

gt_plane_phase = np.load('gt_plane_phase.npy')

exp_name = 'debug'
writer = SummaryWriter(os.path.join('./runs', exp_name))
out_put_dir = os.path.join('./output', exp_name)
debugidx = 10
debug_dir = os.path.join('./output', 'debug' + str(debugidx))
if not os.path.exists(debug_dir):
    os.makedirs(debug_dir)
if not os.path.exists(out_put_dir):
    os.makedirs(out_put_dir)

device = torch.device("cpu")
## SLM phase
layer_num = 8
SLM_phase_list = []
for i in range(layer_num):
    phase_init = 2 * np.pi * np.random.random(size=(MASK_SIZE, MASK_SIZE)).astype('float32')
    grainSize = 4
    phase_init_blur = scipy.ndimage.filters.gaussian_filter(phase_init, grainSize/2)
    SLM_phase_list.append(torch.nn.Parameter(torch.from_numpy(phase_init_blur)))
SLM_shape = SLM_phase_list[0].shape

shift_phase_pi2 = 0.5*np.pi*torch.ones(SLM_shape)
shift_phase_pi = np.pi*torch.ones(SLM_shape)
repeat_num = 1
rate_shape = 1
save_middle = True

## dataset
use_gpu = False
gpu_args = {'num_workers': 12, 'pin_memory': True} if use_gpu else {}

train_dataset = NLOSdataFM(train=True)
test_dataset = NLOSdataFM(train=False)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=BATCH_SIZE, shuffle=True, **gpu_args)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=1, shuffle=False, **gpu_args)

## optimizer
optimizer = torch.optim.Adam(SLM_phase_list, lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

##### ---------train--------- #####
for epoch in range(MAX_EPOCH):
    train_loss_all = 0
    train_correct_all = 0

    for b, (data, target) in enumerate(train_loader):
        grad_all_list = []
        for ii in range(layer_num):
            grad_all_list.append(0)
        
        train_loss_all_batch = 0 
        train_correct_all_batch = 0 
        data_batch, target_batch = data, target

        if b%1 == 0:
            shift = cal_plane_phase(shift_phase_pi2, shift_phase_pi, cali_0_M, cam)

        for i in range(data_batch.shape[0]):
            optimizer.zero_grad()

            target_temp = target_batch[i]
            SLM_outinten = data_batch[i,0]

            ##### forward data #####
            Pkd_list = []
            Pkd_gt_list = []
            Pke_list = []
            Pke_gt_list = []
            rate_f_list = [1.0]
            rate_b_list = [1.0]

            for ii in range(layer_num):
                if ii == 0:
                    input_amp = SLM_outinten
                    input_phase = SLM_phase_list[0]
                    Pkd_temp = input_amp * torch.exp(1j * input_phase)
                else:
                    input_amp = out_amp
                    input_phase = out_phase + SLM_phase_list[ii]
                    Pkd_temp = input_amp * torch.exp(1j * input_phase)
                
                Pkd_list.append(Pkd_temp)
                out_amp, out_phase = forward_layer(input_amp, input_phase, shift_phase_pi2, shift_phase_pi, cali_1_M, cali_0_M, shift, cam1, cam)

                if ii != layer_num - 1:
                    rate_f_list.append(rate_shape / out_amp.max())
                    out_amp = rate_shape * out_amp / out_amp.max()

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

            rate = torch.sum(torch.mul(x_abs,target_temp)) / torch.sum(torch.mul(x_abs,x_abs))
            x_abs = rate.detach()*x_abs
            
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
            
            E_field = 1.5 * E_field / torch.abs(E_field).max().detach()
            E_field = rate_shape*E_field
            E_field_angle = E_field.angle()

            ##### forward error #####
            for ii in range(layer_num-1,-1,-1):
                if ii == layer_num-1:
                    input_amp = torch.abs(E_field)
                    input_phase = E_field_angle
                else:
                    input_amp = out_e_amp
                    input_phase = out_e_phase + SLM_phase_list[ii+1]
                
                if ii == layer_num-1:
                    cam1.EndAcquisition()
                    nodemap = cam1.GetNodeMap()
                    exposureModePtr = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureMode"))
                    exposureModePtr.SetIntValue(exposureModePtr.GetEntryByName("Timed").GetValue())
                    exposureTimePtr = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime"))
                    exposureTimePtr.SetValue(exposure_time*3)
                    cam1.BeginAcquisition()
                else:
                    cam1.EndAcquisition()
                    nodemap = cam1.GetNodeMap()
                    exposureModePtr = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureMode"))
                    exposureModePtr.SetIntValue(exposureModePtr.GetEntryByName("Timed").GetValue())
                    exposureTimePtr = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime"))
                    exposureTimePtr.SetValue(exposure_time)
                    cam1.BeginAcquisition()

                out_e_amp, out_e_phase = forward_layer(input_amp, input_phase, shift_phase_pi2, shift_phase_pi, cali_1_M, cali_0_M, shift, cam1, cam)

                Pke_temp = out_e_amp * torch.exp(1j * out_e_phase)
                Pke_list.append(Pke_temp)

                if ii != 0:
                    rate_b_list.append(rate_shape / out_e_amp.max())
                    out_e_amp = rate_shape * out_e_amp / out_e_amp.max()
            
            Pke_list.reverse()
            rate_b_list.reverse()

            for ii in range(layer_num):
                grad_temp = 2*torch.mul(1j * Pkd_list[ii].detach(), Pke_list[ii]).real.to(torch.float32)

                rate = 0.02/abs(grad_temp).max()
                grad_temp = grad_temp * rate

                grad_all_list[ii] += grad_temp
        
        grad_1_mean_list = []
        for ii in range(layer_num):
            grad_1_mean_list.append(grad_all_list[ii] / target_batch.shape[0])
        
        ##### update #####
        optimizer.zero_grad()
        for ii in range(layer_num):
            SLM_phase_list[ii].grad = grad_1_mean_list[ii]
        optimizer.step()
        print(f"Train---batch {b}: loss {train_loss_all_batch/target_batch.shape[0]:.6f} correct {train_correct_all_batch/target_batch.shape[0]:.2f}. ", end=' ') #\r

    loss_epoch = train_loss_all/len(train_loader)/BATCH_SIZE
    correct_epoch = train_correct_all/len(train_loader)/BATCH_SIZE

    writer.add_scalar('train_loss', loss_epoch, global_step=epoch)
    writer.add_scalar('correct', correct_epoch, global_step=epoch)
    print(f"Train---Epoch {epoch}: loss {loss_epoch:.6f}, accuracy {correct_epoch}.")

##### ---------test--------- #####
    if epoch % 1 == 0:
        test_loss_all = 0
        test_correct_all = 0

        for b, (data, target) in enumerate(test_loader):
            if b%1 == 0:
                shift = cal_plane_phase(shift_phase_pi2, shift_phase_pi, cali_0_M, cam)
                
            SLM_outinten = data[0,0]

            for ii in range(layer_num):
                if ii == 0:
                    input_amp = SLM_outinten
                    input_phase = torch.zeros(SLM_outinten.shape) + SLM_phase_list[0]
                else:
                    input_amp = out_amp
                    input_phase = out_phase + SLM_phase_list[ii]
                
                out_amp, out_phase = forward_layer(input_amp, input_phase, shift_phase_pi2, shift_phase_pi, cali_1_M, cali_0_M, shift, cam1, cam)
                if ii != layer_num - 1:
                    out_amp = out_amp / out_amp.max()

            U_No = out_amp * torch.exp(1j * out_phase)
            x_abs = torch.square(torch.abs(U_No))
                        
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