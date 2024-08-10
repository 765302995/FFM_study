import pyvisa
from pyvisa.constants import *
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import sleep

class AWG:
    wait_time = 0.05
    def __init__(self, rm, address, ch=[1, 2]):
        self.awg = rm.open_resource(address, VI_EXCLUSIVE_LOCK)
        
        self.awg.write_termination = '\n'
        self.awg.read_termination = '\n'
        if isinstance(ch, list):
            for c in ch:
                self.awg.write(f':OUTP{c} ON')
                self.awg.write(f':OUTP{c} OFF')
                self.awg.write(f':SOUR{c}:BURS OFF')
                # self.awg.write(f':OUTP{c}:LOAD 50')
        else:
            self.awg.write(f':OUTP{ch} ON')
            self.awg.write(f':OUTP{ch} OFF')
            self.awg.write(f':SOUR{ch}:BURS OFF')
            # self.awg.write(f':OUTP{ch}:LOAD 50')
    
    
    def write_wait(self, order):
        if isinstance(order, list):
            for o in order:
                self.awg.write(o)
                sleep(self.wait_time)
        else:
            self.awg.write(order)
            sleep(self.wait_time)
        return 
    
    
    def ramp(self, ch=1, freq=1e3, amp=5, offset=0, sym=50):
        self.write_wait([f':SOUR{ch}:APPL:RAMP {freq},{amp},{offset}',
                         f':SOUR{ch}:FUNC:RAMP:SYMM {sym}'])
    
    
    def arbitrary(self, data01, block_len=6000, ch=1, sample_rate=1607600, amp=5, offset=2.5, wait_time=0.0001):
        self.write_wait(f':SOUR{ch}:APPL:ARB {sample_rate},{amp},{offset}')
        len_d = data01.shape[0]
        prefix = f':SOUR{ch}:DATA:DAC16 VOLATILE,'
        d_l = []
        block_num = int(np.ceil(len_d / block_len))
        for i in range(block_num):
            tmp = data01[(i * block_len):(i * block_len + block_len)]
            tmp = np.round(tmp * 0x3fff).astype(np.int_)            
            if i == block_num - 1:
                d_str = prefix + 'END,#'
            else:
                d_str = prefix + 'CON,#'
            bytes_len = f'{2 * tmp.shape[0]}'
            d_str += f'{len(bytes_len)}{bytes_len},'
            d_str = d_str.encode() + bytes.fromhex(''.join([f'{_:04x}' for _ in tmp])) # + '\n'.encode()
            d_l.append(d_str)
        for d_str in d_l:
            self.awg.write_raw(d_str)
            sleep(wait_time)
            
            
    def burst(self, ch=1, nc=2,
              mode='TRIG', # TRIG INF GAT
              slope='POS', # POS NEG
              source='INT', # EXT MAN INT
              period=10e-3,
              trigo='POS', # POS NEG OFF
              idle='FPT', # FPT TOP CENTER BOTTOM
              delay=0
              ):
        self.write_wait([f':SOUR{ch}:BURS ON',                  f':SOUR{ch}:BURS:MODE {mode}',
                        f':SOUR{ch}:BURS:NCYC {nc}',            f':SOUR{ch}:BURS:TRIG:SLOP {slope}',
                        f':SOUR{ch}:BURS:TRIG:SOUR {source}',   f':SOUR{ch}:BURS:INT:PER {period:f}',
                        f':SOUR{ch}:BURS:TRIG:TRIGO {trigo}',   f':SOUR{ch}:BURS:IDLE {idle}'])
        self.write_wait([f':SOUR{ch}:BURS:TDEL {delay}'])
        
    def square(self, ch=1, freq=1000, amp=4, offset=2, duty=50):
        self.write_wait(f':SOUR{ch}:APPL:SQU {freq},{amp},{offset},0')
        self.write_wait(f':SOUR{ch}:FUNC:SQU:DCYC {duty}')
        
    def dc(self, ch=1, offset=2):
        self.write_wait(f':SOUR{ch}:APPL:DC 1,1,{offset}')
    
    def set_state(self, state=[1, 1], ch=[1, 2], burst=False):
        if isinstance(ch, list):
            for c, s in zip(ch, state):
                ss = 'ON' if s == 1 else 'OFF'
                self.write_wait(f':OUTP{c} {ss}')
                if burst:
                    self.write_wait(f':SOUR{c}:BURS {ss}')
        else:
            ss = 'ON' if state == 1 else 'OFF'
            self.write_wait(f':OUTP{ch} {ss}')
            if burst:
                self.write_wait(f':SOUR{ch}:BURS {ss}')
    
            
    def reset(self, ch=[1, 2]):
        if isinstance(ch, list):
            if 1 in ch:
                self.write_wait([':SOUR1:BURS OFF',  ':OUTP1 OFF'])
            if 2 in ch:
                self.write_wait([':SOUR2:BURS OFF',  ':OUTP2 OFF'])
        else:
            if ch == 1:
                self.write_wait([':SOUR1:BURS OFF',  ':OUTP1 OFF'])
            elif ch == 2:
                self.write_wait([':SOUR2:BURS OFF',  ':OUTP2 OFF'])
    
    
    def exp_reshape(self, raw_data):
        def re(r):
            # 2, fm_10, b, od, oh*ow
            r = r.reshape((*r.shape[:3], -1, 5, r.shape[-1]))
            # 2, fm_10, b, od_5, 5, oh*ow
            r = np.transpose(r, [2, 5, 0, 1, 3, 4])
            r_s = r.shape
            # b, oh*ow, 2, fm_10, od_5, 5
            r = np.reshape(r, (-1, 5))
            # b*oh*ow*2*fm_10*od_5, 5
            return r, r_s
            # loop_num, 5
        if isinstance(raw_data, tuple):
            res = []
            for rr in raw_data:
                r, raw_shape = re(rr)
                res.append(r)
            return (*res, raw_shape)
        else:
            res, raw_shape = re(raw_data)
            return res, raw_shape
 

def fmt(s):
    if s == 0 or s == 1 or s == -1:
        return str(int(s))
    else:
        return f'{s:.4f}'
        
class AWG4(AWG):
    def __init__(self, rm, address, ch=[1, 2]):
        super().__init__(rm, address, ch)
    
    def arbitrary(self, data01, block_len=6000, ch=1, sample_rate=1607600, amp=5, offset=2.5, wait_time=0.0001):
        self.write_wait(f':SOUR{ch}:APPL:USER {sample_rate/data01.shape[0]},{amp},{offset}')
        data01 = data01 * 2 - 1
        self.write_wait(f':SOUR{ch}:DATA VOLATILE,' + ','.join([fmt(_) for _ in data01]))
        sleep(wait_time)
    
    def set_amp_off(self, ch=1, amp=1, offset=1./2, sample_rate=1607600):
        self.write_wait([f':SOUR{ch}:APPL:USER {sample_rate/5000},{amp},{offset}'])

if __name__ == '__main__':
    rm = pyvisa.ResourceManager()
    awg2 = AWG4(rm, 'XXX')
    awg = AWG4(rm, 'XXX')
    awg.arbitrary(0*np.ones((220,)), ch=1, sample_rate=3e3)
    awg.arbitrary(0*np.ones((220,)), ch=2, sample_rate=3e3)
    awg.set_state(state=[0,0], ch=[1,2])
    awg2.set_state(1,1)
    