import serial
from time import sleep
import numpy as np

class OpticalPower:
    open_flag = False
    def __init__(self, port='COM8', flag=True):
        self.port = port
        if flag:
            self.open()
    
    def query_hex(self, hex_ord):
        self.op.write(bytes.fromhex(hex_ord))
        self.op.flush()
        sleep(.1)
        return self.op.read(2)
    
    def get(self, mode='l'):
        if mode == 'l':
            r = self.query_hex('EE AA 01 01 00 00')
            return r[0] * 100 + r[1]
        elif mode == 'p':
            r = self.query_hex('EE AA 03 01 00 00')
            return (int(r.hex(), 16) - 10000) / 100
        # return struct.unpack('<f', r[1:5])
        
    def get_2(self, mode='l'):
        if mode == 'l':
            r = self.query_hex('EE AA 01 02 00 00')
            return r[0] * 100 + r[1]
        elif mode == 'p':
            r = self.query_hex('EE AA 03 02 00 00')
            return (int(r.hex(), 16) - 10000) / 100
        # return struct.unpack('<f', r[1:5])   
    
    def open(self):
        if not self.open_flag:
            self.op = serial.Serial(port=self.port, baudrate=57600, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, 
                    stopbits=serial.STOPBITS_ONE, timeout=5)
            assert(self.op.isOpen())
            self.open_flag = True
    
    def close(self):
        self.op.close()
        self.open_flag = False

def get_mean_2(op, num=5):
    res = []
    for _ in range(num):
        res.append(op.get_2(mode='p'))
        sleep(0.1)
    res = np.array(res)
    # res = res[res>-84]
    res = np.mean(res)
    return res

def get_mean(op, num=5):
    res = []
    for _ in range(num):
        res.append(op.get(mode='p'))
        sleep(0.1)
    res = np.array(res)
    # res = res[res>-84]
    res = np.mean(res)
    return res

if __name__ == '__main__':
    op = OpticalPower(port='COM16', flag=True)
    print(f"OPM lambda: {op.get(mode='l')}")
    print(f"OPM power: {op.get(mode='p')}")
    get_mean(op)