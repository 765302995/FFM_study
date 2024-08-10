from PIL import Image
import numpy as np
from ctypes import *
import ctypes
import copy
import os
import cv2
import time
import pdb
import PySpin
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from skimage import measure

def calibrate_M(img_cal, M_size = 200):
    img_cal = cv2.threshold(img_cal, 254, 255, cv2.THRESH_BINARY)[1]
    img_cal = cv2.erode(img_cal, None, iterations=5)
    img_cal = cv2.dilate(img_cal, None, iterations=5)

    labels = measure.label(img_cal, connectivity=2, background=0)

    x = np.linspace(0,labels.shape[1]-1,labels.shape[1])
    y = np.linspace(0,labels.shape[0]-1,labels.shape[0])
    xx,yy = np.meshgrid(x,y)

    loc_4 = np.zeros((4,2))
    for ii in range(4):
        mask = (labels==(ii+1))
        x_i = np.sum(mask*xx) / np.sum(mask)
        y_i = np.sum(mask*yy) / np.sum(mask)
        loc_4[ii,0] = x_i
        loc_4[ii,1] = y_i

    #print(loc_4)
    source_points = loc_4.astype(np.float32)
    destination_points = np.array([[0, 0], [M_size, 0], [0, M_size], [M_size, M_size]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(source_points, destination_points)
    return M

def calibrate_M_all(cam,cam1,exposure_time0, exposure_time1):
    cam1.EndAcquisition()
    nodemap = cam1.GetNodeMap()
    exposureModePtr = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureMode"))
    exposureModePtr.SetIntValue(exposureModePtr.GetEntryByName("Timed").GetValue())
    exposureTimePtr = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime"))
    exposureTimePtr.SetValue(10)
    cam1.BeginAcquisition()

    intensity = cam1.GetNextImage(1000)
    image_captured = intensity.GetNDArray()
    cali_1_M = calibrate_M(image_captured, M_size = 200)

    cam1.EndAcquisition()
    nodemap = cam1.GetNodeMap()
    exposureModePtr = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureMode"))
    exposureModePtr.SetIntValue(exposureModePtr.GetEntryByName("Timed").GetValue())
    exposureTimePtr = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime"))
    exposureTimePtr.SetValue(exposure_time1)
    cam1.BeginAcquisition()
    
    cam.EndAcquisition()
    nodemap = cam.GetNodeMap()
    exposureModePtr = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureMode"))
    exposureModePtr.SetIntValue(exposureModePtr.GetEntryByName("Timed").GetValue())
    exposureTimePtr = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime"))
    exposureTimePtr.SetValue(10)
    cam.BeginAcquisition()
    
    intensity = cam.GetNextImage(1000)
    image_captured = intensity.GetNDArray()
    cali_0_M = calibrate_M(image_captured, M_size = 200)

    cam.EndAcquisition()
    nodemap = cam.GetNodeMap()
    exposureModePtr = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureMode"))
    exposureModePtr.SetIntValue(exposureModePtr.GetEntryByName("Timed").GetValue())
    exposureTimePtr = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime"))
    exposureTimePtr.SetValue(exposure_time0)
    cam.BeginAcquisition()

    return cali_1_M, cali_0_M

def showOn2ndDisplay(monitorNo, windowNo, x, xShift, y, yShift, array):
    Lcoslib = windll.LoadLibrary("Image_Control.dll")
    
    #Select LCOS window
    Window_Settings = Lcoslib.Window_Settings
    Window_Settings.argtypes = [c_int, c_int, c_int, c_int]
    Window_Settings.restype = c_int
    Window_Settings(monitorNo, windowNo, xShift, yShift)
    
    #Show pattern
    Window_Array_to_Display = Lcoslib.Window_Array_to_Display
    Window_Array_to_Display.argtypes = [c_void_p, c_int, c_int, c_int, c_int]
    Window_Array_to_Display.restype = c_int
    Window_Array_to_Display(array, x, y, windowNo, x*y)
    
    return 0

def prop_simu_ODP(ori_in, prop_layer):
    mode_type = 'replicate'  #replicate reflect
    z_in = ori_in

    z_in = F.pad(z_in.unsqueeze(0).unsqueeze(0), ((400-ori_in.shape[1])//2, (400-ori_in.shape[1])//2, (400-ori_in.shape[0])//2, (400-ori_in.shape[0])//2), mode=mode_type).squeeze() #constant/reflect
    z_in = F.pad(z_in.unsqueeze(0).unsqueeze(0), ((800-400)//2, (800-400)//2, (800-400)//2, (800-400)//2), mode=mode_type).squeeze() #constant/reflect

    zout_2 = prop_layer(z_in)

    zout = zout_2[300:500,300:500]
    return zout

def prop_simu(ori_in, prop_layer):
    mode_type = 'constant'
    z_in = ori_in

    z_in = F.pad(z_in.unsqueeze(0).unsqueeze(0), ((800-ori_in.shape[1])//2, (800-ori_in.shape[1])//2, (800-ori_in.shape[0])//2, (800-ori_in.shape[0])//2), mode=mode_type).squeeze() #constant/reflect

    zout_2 = prop_layer(z_in)

    zout = zout_2[200:600,200:600]
    return zout