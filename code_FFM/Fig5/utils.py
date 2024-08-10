import socket
import struct
import numpy as np
import pdb
import time
from tqdm import *
import matplotlib.pyplot as plt
from power_readout import *
import scipy.io
from scipy.optimize import fsolve
from awg import *
from channnel_shift import ChannelShifter
import pyautogui
import torch

def isnan(input):
    return input != input

def dB2inten(dB):
    inten = 1000*(10 ** (dB / 10))
    return inten

def data2inten(data):
    return 1000*np.mean(data[:100000])

def changeshift_1():
    on_off = (1824,747)
    position = [on_off]  
    for i in position:
        pyautogui.moveTo(i)
        pyautogui.click(clicks=1)
        time.sleep(0.1)

def input_value_shift(x1,x2):
    x1_o = 2.5 + x1 * 0.5
    x2_o = 2.5 + x2 * 0.5
    return x1_o, x2_o

def output_value_shift(x):
    x_o = 2*(x - 5)
    return x_o

def find_weight(CS, op, op_reader):
    weight_modu_list = [391, 385, 431, 437]
    voltage_set_list = [0.0,0.0,0.0,0.0]
    op.set_voltage(voltage_set_list, weight_modu_list)
    time.sleep(1.0)
    ## 1 : 2-2
    ## 2 : 2-1
    ## 3 : 1-1
    ## 4 : 1-2
    CS.change(1)
    set_power = dB2inten(get_mean(op_reader))
    print('set_power is ', set_power)
    t_v_1_1 = 0.0 
    threshold = 0.1
    
    v_list_list = np.linspace(0.8,1.0,41)
    last_power = 100
    last_t_v = 0
    for t_v in v_list_list:
        voltage_set_list[3] = t_v
        op.set_voltage(voltage_set_list, weight_modu_list)
        time.sleep(0.1)
        t_power = dB2inten(get_mean_2(op_reader))
        print(t_power)
        if last_power >= set_power and t_power <= set_power:
            if abs(last_power - set_power) < abs(t_power - set_power):
                best_t_v = last_t_v
            else:
                best_t_v = t_v
            break
        last_power = t_power
        last_t_v = t_v
    
    t_v_1_2 = best_t_v
    
    CS.change(2)
    changeshift_1()
    
    v_list_list = np.linspace(0.8,1.0,41)
    last_power = 100
    last_t_v = 0
    for t_v in v_list_list:
        voltage_set_list[0] = t_v
        op.set_voltage(voltage_set_list, weight_modu_list)
        time.sleep(0.1)
        t_power = dB2inten(get_mean_2(op_reader))
        print(t_power)
        if last_power >= set_power and t_power <= set_power:
            if abs(last_power - set_power) < abs(t_power - set_power):
                best_t_v = last_t_v
            else:
                best_t_v = t_v
            break
        last_power = t_power
        last_t_v = t_v
    t_v_2_2 = best_t_v
    
    last_power = 100
    last_t_v = 0
    for t_v in v_list_list:
        voltage_set_list[1] = t_v
        op.set_voltage(voltage_set_list, weight_modu_list)
        time.sleep(0.1)
        t_power = dB2inten(get_mean(op_reader))
        print(t_power)
        if last_power >= set_power and t_power <= set_power:
            if abs(last_power - set_power) < abs(t_power - set_power):
                best_t_v = last_t_v
            else:
                best_t_v = t_v
            break
        last_power = t_power
        last_t_v = t_v
    t_v_2_1 = best_t_v
    
    changeshift_1()
    
    return t_v_1_1, t_v_1_2, t_v_2_1, t_v_2_2

def find_voltage_input(ref_inten, set_inten, cureve_input, v_list):
    cureve_measure = cureve_input * (ref_inten / cureve_input[0])

    for ii in range(cureve_measure.shape[0]-1):
        if cureve_measure[ii] >= set_inten and cureve_measure[ii+1] <= set_inten:
            rate = (set_inten - cureve_measure[ii]) / (cureve_measure[ii+1] - cureve_measure[ii])
            setting_voltage = v_list[ii] + rate * (v_list[ii+1] - v_list[ii])
            break
    
    if cureve_measure[0] < set_inten:
        setting_voltage = v_list[0]
    if cureve_measure[ii+1] > set_inten:
        setting_voltage = v_list[ii+1]
    return setting_voltage

def calculate_optical_2_2(x_0, awg, op_reader, input_curve, ref_inten_list, v_list_input):
    ref_inten_1 = ref_inten_list[0]
    ref_inten_2 = ref_inten_list[1]
    
    x1 = x_0[0]
    x2 = x_0[1]
    
    #w_0[0] * x_0[0] + w_0[1] * x_0[1]
    if x1 == 0 and x2 == 0:
        result = 0
        result2 = 0 
    else:
        if x1 >=0 and x2 >=0:
            rate = min(np.ceil(np.log10(0.1/abs(x1))), np.ceil(np.log10(0.1/abs(x2))))
            rate = 10 ** rate
            rate = rate * 5
            x1 = x1*rate
            x2 = x2*rate
            if isnan(x1) or isnan(x2):
                pdb.set_trace()
            input_set_1 = find_voltage_input(ref_inten_1, x1, input_curve[0], v_list_input)
            input_set_2 = find_voltage_input(ref_inten_2, x2, input_curve[1], v_list_input)

            awg.dc(ch=1, offset=input_set_1)
            awg.dc(ch=2, offset=input_set_2)
            
            time.sleep(0.1)
            temp_op_power = get_mean(op_reader)
            result = dB2inten(temp_op_power)
            result = result / rate
            temp_op_power2 = get_mean_2(op_reader)
            result2 = dB2inten(temp_op_power2)
            result2 = result2 / rate
            
        elif x1 < 0 and x2 < 0:
            x1 = -x1
            x2 = -x2
            rate = min(np.ceil(np.log10(0.1/abs(x1))), np.ceil(np.log10(0.1/abs(x2))))
            rate = 10 ** rate
            rate = rate * 5
            x1 = x1*rate
            x2 = x2*rate
            if isnan(x1) or isnan(x2):
                pdb.set_trace()
            input_set_1 = find_voltage_input(ref_inten_1, x1, input_curve[0], v_list_input)
            input_set_2 = find_voltage_input(ref_inten_2, x2, input_curve[1], v_list_input)

            awg.dc(ch=1, offset=input_set_1)
            awg.dc(ch=2, offset=input_set_2)

            time.sleep(0.1)
            temp_op_power = get_mean(op_reader)
            result = dB2inten(temp_op_power)
            result = -result / rate
            temp_op_power2 = get_mean_2(op_reader)
            result2 = dB2inten(temp_op_power2)
            result2 = -result2 / rate
        else:
            rate = min(np.ceil(np.log10(0.1/abs(x1))), np.ceil(np.log10(0.1/abs(x2))))
            rate = 10 ** rate
            rate = rate * 5
            x1 = x1*rate
            x2 = x2*rate
            if isnan(x1) or isnan(x2):
                pdb.set_trace()
            x1, x2 = input_value_shift(x1, x2)
            
            input_set_1 = find_voltage_input(ref_inten_1, x1, input_curve[0], v_list_input)
            input_set_2 = find_voltage_input(ref_inten_2, x2, input_curve[1], v_list_input)

            awg.dc(ch=1, offset=input_set_1)
            awg.dc(ch=2, offset=input_set_2)

            time.sleep(0.1)
            temp_op_power = get_mean(op_reader)
            result = dB2inten(temp_op_power)
            result = output_value_shift(result)
            result = result / rate
            temp_op_power2 = get_mean_2(op_reader)
            result2 = dB2inten(temp_op_power2)
            result2 = output_value_shift(result2)
            result2 = result2 / rate
    
    result_list = torch.zeros([2,])
    result_list[0] = result
    result_list[1] = result2
    return result_list


def calibrate_input_curve(awg, v_list, CS, op_reader):
    CS.change(1)

    ### 1 - 1 ###
    attenuation_curve = []
    channel = 1
    awg.set_state(state=1, ch=channel)

    for t_v in tqdm(v_list):
        awg.dc(ch=channel, offset=t_v)
        
        time.sleep(0.1)
        temp_op_power = get_mean(op_reader)
        attenuation_curve.append(dB2inten(temp_op_power))

    np.save('input_1.npy', attenuation_curve)
    awg.dc(ch=channel, offset=0.0)
    awg.set_state(state=0, ch=channel)

    CS.change(2)
    changeshift_1()

    time.sleep(60.0)
    ### 2 - 1 ###
    attenuation_curve = []
    channel = 2
    awg.set_state(state=1, ch=channel)

    for t_v in tqdm(v_list):
        awg.dc(ch=channel, offset=t_v)
        
        time.sleep(0.1)
        temp_op_power = get_mean_2(op_reader)
        attenuation_curve.append(dB2inten(temp_op_power))

    np.save('input_2.npy', attenuation_curve)

    awg.dc(ch=channel, offset=0.0)
    awg.set_state(state=0, ch=channel)

    changeshift_1()
    return 0

def get_input_inten(awg, op_reader, channel, v_set):
    awg.dc(ch=channel, offset=v_set)
    awg.set_state(state=1, ch=channel)

    time.sleep(0.1)
    temp_op_power = get_mean(op_reader)
    ref_inten = dB2inten(temp_op_power)
    
    return ref_inten

def get_ref_inten(awg, op_reader, CS):
    awg.dc(ch=1, offset=0.0)
    awg.dc(ch=2, offset=0.0)
    time.sleep(60)
    ## set independent
    CS.change(1)
    ref_inten_1 = get_input_inten(awg, op_reader, 1, 0.0)
    
    ## set independent
    CS.change(2)
    changeshift_1()
    ref_inten_2 = get_input_inten(awg, op_reader, 2, 0.0)
    
    changeshift_1()
    
    return [ref_inten_1, ref_inten_2]