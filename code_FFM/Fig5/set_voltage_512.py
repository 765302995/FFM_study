# -*- coding: utf-8 -*-
import socket
import struct
import time

import numpy as np
import pdb
from power_readout import *

def dB2inten(dB):
    inten = 1000 * (10 ** (dB / 10))
    return inten

class VoltageOperator(object):
    def __init__(self):
        self._addr = ("XXX", 7)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        self._sock.bind(('XXX', 8006))
        self._ch_num = 512
    
    def init(self, ip, port, ch_num):
        self._addr = (ip, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(('XXX', 8006))
        self._ch_num = ch_num
        C_power_begin = 385
        self.weight_modu_list = [C_power_begin+5, C_power_begin+11, C_power_begin+13, C_power_begin+19]
    
    def sendVoltageToDev(self, voltages):
        if self._ch_num != len(voltages):
            return False
        bytes_data = bytes()

        bytes_data += struct.pack('B', 0xAA)
        bytes_data += struct.pack('B', 0x55)
        # bytes_data += struct.pack('B', 0x00)
        bytes_data += struct.pack('B', 0xD2)
        bytes_data += struct.pack('B', 0x04)
        bytes_data += struct.pack('B', 0x04)

        check_sum = int('0xAA', 16) + int('0x55', 16) + int('0xD2', 16) + int('0x04', 16) + int('0x04', 16)
        
        for v in voltages:
            high = v//256
            low = v % 256
            bytes_data += struct.pack('B', high)
            bytes_data += struct.pack('B', low)
            check_sum += int(high)
            check_sum += int(low)
        
        check_sum = hex(check_sum).replace('0x', '').zfill(8)
        
        bytes_data += struct.pack('B', int('0x' + check_sum[-8:-6],16))
        bytes_data += struct.pack('B', int('0x' + check_sum[-6:-4],16))
        bytes_data += struct.pack('B', int('0x' + check_sum[-4:-2],16))
        bytes_data += struct.pack('B', int('0x' + check_sum[-2:],16))  
        self._sock.sendto(bytes_data, self._addr)
        return bytes_data

    def voltage_change(self, input):
        input_c = int(input * 1000)
        return input_c

    def set_voltage(self, voltage_set_list, weight_modu_list):
        if not weight_modu_list:
            weight_modu_list = self.weight_modu_list
        volts = np.zeros([512], dtype=np.int)
        for ii in range(len(weight_modu_list)):
            volts[weight_modu_list[ii]] = self.voltage_change(voltage_set_list[ii])
        volts = volts.tolist()
        v = self.sendVoltageToDev(volts)
        return v
    
    def set_voltage_test(self):
        volts = np.zeros([512], dtype=np.int)
        volts = volts.tolist()
        v = self.sendVoltageToDev(volts)
        return v

if __name__ == '__main__':
    op = VoltageOperator()
    op.init("XXX", 7, 512)
    
    weight_modu_list = [436, 401, 426, 391]
    voltage_set_list = [0.0, 0.0, 0.0, 0.0]
    op.set_voltage(voltage_set_list, weight_modu_list)