from numpy.lib.function_base import re
import pyvisa
from pyvisa.constants import *
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import struct
from time import sleep
import socket
import serial
import os
import scipy.io as scio
import math
import cv2
import pdb
import time

class OSCP:
    def __init__(self, rm, address="XXX"):
        self.oscp = rm.open_resource(address, VI_EXCLUSIVE_LOCK, VI_NULL)
        # self.oscp.timeout = 100000
        # self.mdo.read_termination = '\n'
        # self.mdo.write_termination = '\n'

    def read_waveform(self, ch=2, return_x=False, start_stop=None):
        oscp = self.oscp
        oscp.set_visa_attribute(VI_ATTR_WR_BUF_OPER_MODE, VI_FLUSH_ON_ACCESS)
        oscp.set_visa_attribute(VI_ATTR_RD_BUF_OPER_MODE, VI_FLUSH_ON_ACCESS)
        oscp.write("header off")
        
        oscp.write(f"DATa:SOUrce CH{ch}")
        elements = self.query_reco()
        if start_stop is None:
            oscp.write(f'DATa:STARt 0')
            oscp.write(f'DATa:STOP {elements - 1}')
        else:
            oscp.write(f'DATa:STARt {start_stop[0]}')
            oscp.write(f'DATa:STOP {start_stop[1]}')
        yoffset = oscp.query_ascii_values("WFMOutpre:YOFF?", 'f')[0]
        ymult = oscp.query_ascii_values("WFMOutpre:YMULT?", 'f')[0]
        yzero = oscp.query_ascii_values('WFMOutpre:YZEro?')[0]
        oscp.write("DATA:ENCDG RIBINARY;WIDTH 1")
        oscp.write("CURVE?")
        oscp.flush(VI_WRITE_BUF|VI_READ_BUF_DISCARD)
        oscp.set_visa_attribute(VI_ATTR_RD_BUF_OPER_MODE, VI_FLUSH_DISABLE)
        c = oscp.read_bytes(1)
        assert(c==b'#')
        c = oscp.read_bytes(1)
        assert(b'0' <= c <= b'9')
        count = int(c) - int(b'0')
        c = oscp.read_bytes(count)
        elements = int(c)
        c = oscp.read_bytes(elements)
        oscp.flush(VI_WRITE_BUF | VI_READ_BUF_DISCARD)
        res = np.array(struct.unpack('b' * elements, c))
        res = (res - yoffset) * ymult + yzero
        # time.sleep(1.0)
        if return_x:
            return np.r_[:elements] * self.query_xspec(), res
        else:
            return res
        

    def query_xspec(self):
        return self.oscp.query_ascii_values('WFMOutpre:XINcr?')[0]

    def query_reco(self):
        return self.oscp.query_ascii_values("hor:reco?", 'd')[0]

    def set_xscale(self, scale):
        self.oscp.write(f'HORIZONTAL:SCALE {scale:.2e}')

    def set_yscale(self, ch, scale):
        self.oscp.write(f'CH{ch}:Scale {scale:.2e}')
        
    def set_recordlength(self, length):
        self.oscp.write(f'HORizontal:RECOrdlength {length}')


if __name__ == '__main__':
    rm = pyvisa.ResourceManager()
    oscp = OSCP(rm, address='XXX')
    data = oscp.read_waveform(ch=1)
    data2 = oscp.read_waveform(ch=2)