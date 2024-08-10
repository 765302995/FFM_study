import serial
from time import sleep
import numpy as np
import pdb

class ChannelShifter:
    open_flag = False
    def __init__(self, port='COM9', flag=True):
        self.port = port
        if flag:
            self.open()
    
    def query_hex(self, hex_ord):
        self.op.write(bytes.fromhex(hex_ord))
        self.op.flush()
        sleep(.1)
        return self.op.read(2)
    
    def change(self, ch = 1):
        if ch == 1:
            r = self.query_hex('EE AA 01 00 00 00')
            return r
        elif ch == 2:
            r = self.query_hex('EE AA 02 00 00 00')
            return r
    
    def open(self):
        if not self.open_flag:
            self.op = serial.Serial(port=self.port, baudrate=9600, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, 
                    stopbits=serial.STOPBITS_ONE, timeout=5)
            assert(self.op.isOpen())
            self.open_flag = True
    
    def close(self):
        self.op.close()
        self.open_flag = False

if __name__ == '__main__':
    CS = ChannelShifter(port='COM9', flag=True)
    CS.change(2)
    