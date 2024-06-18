import time
import serial
import binascii
import multiprocessing as mp
import numpy as np
import enum
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
"""
此代码为夹爪代码
"""

class Command(enum.Enum):
    STOP = 0
    MOVE_TGT = 1
    MOVE_MAX = 2
    MOVE_MIN = 3
    MOVE_MIN_HOLD = 4
    

class inspire_gripper(mp.Process):
    def __init__(self,
                shm_manager: SharedMemoryManager, 
                get_max_k=128,
                frequency=125, 
                launch_timeout=3,
                com_port='com1',
                baudrate=9600,
                act_position_=-1,
                gripper_id=0,
                gripper_state_=0xff):
        super().__init__()
        self.ser = serial.Serial(port=com_port, baudrate=baudrate,bytesize=8, parity='N', stopbits=1,timeout=2)
        self.gripper_id =gripper_id
        self.launch_timeout = launch_timeout

        if self.ser.is_open:
            print("Gripper: Serial port com2 openned")
            id_state = 0
            while (1):
                id_state = self.gripper_start()
                if (id_state == 1):
                    break
                self.gripper_id = self.gripper_id + 1
                if (self.gripper_id > 254):
                    print("Id error!!!")
                    self.gripper_id = 1
            # Get initial state and discard input buffer
            while (gripper_state_ == 0xff):
                gripper_state_ = self.getState(self.gripper_id)
                gripper_state_ = 0
                time.sleep(1)
        else:
            print("Gripper: Serial port com2 not opened")

        
        # build input queue
        example = {
            'cmd': Command.MOVE_TGT.value,
            'target_pose': 100, # '0-1000'
            'speed' : 500, # '1-1000'
            'power' : 100 # '50-1000'
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # build ring buffer
        example = {
            'curopen': 0,
            'power': 500,
            'robot_receive_timestamp': time.time()
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples( # 环形缓冲区
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k, # 最大k
            get_time_budget=0.2, # 获取时间的预算
            put_desired_frequency=frequency
        )

        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer

    # ========== start stop API ===========

    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()
    
    def __enter__(self): # 当使用with语句创建一个对象的上下文时，__enter__方法会被自动调用。
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= command methods ============
    def moveTgt_cmd(self, movetgt):
        message = {
            'cmd': Command.MOVE_TGT.value,
            'target_pose': movetgt
        }
        self.input_queue.put(message)
    
    def moveMax_cmd(self, speed):
        message = {
            'cmd': Command.MOVE_MAX.value,
            'speed': speed
        }
        self.input_queue.put(message)

    def moveMin_cmd(self, speed, power):
        message = {
            'cmd': Command.MOVE_MIN.value,
            'speed': speed,
            'power': power
        }
        self.input_queue.put(message)
    
    def moveMinHold_cmd(self, speed, power):
        message = {
            'cmd': Command.MOVE_MIN_HOLD.value,
            'speed': speed,
            'power': power
        }
        self.input_queue.put(message)

    # ========= receive APIs =============
    def get_gripper_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_gripper_all_state(self):
        return self.ring_buffer.get_all()

    # ========= 夹爪程序 ==========
    def gripper_start(self):
        output = bytearray()
        output.append(0xEB)
        output.append(0x90)
        output.append(self.gripper_id)
        output.append(1)
        output.append(0x41)
        check_num = 0
        leno = output[3] + 5
        for i in range(2,leno - 1):
            check_num = check_num + output[i]

        # Add checksum to the output buffer
        output.append(check_num & 0xff)
        # Send message to the module and wait for response
        self.ser.write(output)
        time.sleep(0.015)
        inputData=self.ser.read(64)
        if len(inputData) ==0:
            result =0
        else:
            result =1
        return result

    def getState(self,gripper_id):
        # print("Reading current module state...")
        output = bytearray()
        # messagefrom master to module
        output.append(0xEB)
        output.append(0x90)

        #module id
        output.append(gripper_id)
        #Data Length
        output.append(1)

        # Command get state
        output.append(0x41)

        #Checksum calculation unsigned int
        check_num = 0
        lenData = output[3] + 5
        for i in range(2,lenData-1):
            check_num = check_num + output[i]
            # Add checksum to the output buffer
        output.append(check_num & 0xff)
        # Send message to the module
        self.ser.write(output)
        time.sleep(0.015)

        inputtmp = self.ser.read(64)
        inputdata =[]
        for i in range(len(inputtmp)):
            # inputdata.append(int(inputtmp[i].encode('hex'),16))
            inputdata.append(inputtmp[i])
        temp = inputdata[7]
        
        curopen = ((inputdata[9] << 8) & 0xff00) + inputdata[8]
        power = ((inputdata[11] << 8) & 0xff00) + inputdata[10]
        error = list(range(5))
        error[0] = inputdata[6] & 0x01
        error[1] = inputdata[6] & 0x02
        error[2] = inputdata[6] & 0x04
        error[3] = inputdata[6] & 0x08
        error[4] = inputdata[6] & 0x10

        if (error[0] == 1):
            print("runing stop fault")
        if (error[1] == 2):
            print("overheat fault")
        if (error[2] == 4):
            print("Over Current Fault")
        if (error[3] == 8):
            print("running fault")
        if (error[4] == 16):
            print("communication fault")

        # if (inputdata[6] == 0):
        #     print("Gripper: Temperature: {} [C] Current open: {} Power: {} [g]".format(temp,curopen,power))
        return curopen, power

    def moveTgt(self, movetgt):
        output = bytearray()
        # message from master to module
        output.append(0xEB)
        output.append(0x90)
        #module id
        output.append(self.gripper_id)
        #Data Length   
        output.append(0x03)
        #Command get state
        output.append(0x54)
        temp_int1 = movetgt

        output.append(temp_int1 & 0xff)
        output.append((temp_int1 >> 8) & 0xff)

        leno = output[3] + 5
        check_num = 0
        for i in range(2,leno-1):
            check_num = check_num + output[i]
        #Add checksum to the output buffer
        output.append(check_num & 0xff)

        #Send message to the module
        self.ser.write(output)
        time.sleep(0.015)

        inputtmp = self.ser.read(64)
        inputdata =[]
        for i in range(len(inputtmp)):
            inputdata.append(int(inputtmp[i]))
        temp = inputdata[5]
        if (temp == 1):
            result = True
        else:
            result = False
        return result
    
    def moveMax(self,speed):
        output = bytearray()
        #message from master to module
        output.append(0xEB)
        output.append(0x90)
        #module id
        output.append(self.gripper_id)
        #Data Length   
        output.append(0x03)
        #Command get state
        output.append(0x11)

        temp_int1 = speed
        output.append(temp_int1 & 0xff)
        output.append((temp_int1 >> 8) & 0xff)

        #Checksum calculation
        check_num = 0
        leno = output[3] + 5

        for i in range(2,leno-1):
            check_num = check_num + output[i]

        #Add checksum to the output buffer
        output.append(check_num & 0xff)

        #Send message to the module
        self.ser.write(output)
        time.sleep(0.015)

        inputtmp =self.ser.read(64)
        inputdata =[]
        for i in range(len(inputtmp)):
            inputdata.append(int(inputtmp[i]))

        temp = inputdata[5]
        if (temp == 1):
            result = True
        else:
            result = False
        return result
    
    # def moveMax(self,speed):
    #     output = bytearray()
    #     #message from master to module
    #     output.append(0xEB)
    #     output.append(0x90)
    #     #module id
    #     output.append(self.gripper_id)
    #     #Data Length   
    #     output.append(0x03)
    #     #Command get state
    #     output.append(0x11)

    #     temp_int1 = speed
    #     output.append(temp_int1 & 0xff)
    #     check_num = 0
    #     leno = output[3] + 5

    #     for i in range(2,leno-1):
    #         check_num = check_num + output[i]

    #     #Add checksum to the output buffer
    #     output.append(check_num & 0xff)

    #     #Send message to the module
    #     self.ser.write(output)
    #     time.sleep(0.015)

    #     inputtmp =self.ser.read(64)
    #     inputdata =[]
    #     for i in range(len(inputtmp)):
    #         inputdata.append(int(inputtmp[i]))

    #     temp = inputdata[5]
    #     if (temp == 1):
    #         result = True
    #     else:
    #         result = False
    #     return result

    def moveMin(self, speed, power):
        output = bytearray()
        # message from master to module
        output.append(0xEB)
        output.append(0x90)
        # module id
        output.append(self.gripper_id)
        # Data Length   
        output.append(0x05)
        # Command get state
        output.append(0x10)

        temp_int1 = speed
        temp_int2 = power

        output.append(temp_int1 & 0xff)
        output.append((temp_int1 >> 8) & 0xff)
        output.append(temp_int2 & 0xff)
        output.append((temp_int2 >> 8) & 0xff)

        # Checksum calculation
        check_num = 0
        leno = output[3] + 5

        for i in range(2,leno - 1):
            check_num = check_num + output[i]

        # Add checksum to the output buffer
        output.append(check_num & 0xff)

        #Send message to the module
        self.ser.write(output)
        time.sleep(0.015)

        inputtmp =self.ser.read(64)
        inputdata =[]
        for i in range(len(inputtmp)):
            inputdata.append(int(inputtmp[i]))

        temp = inputdata[5]
        if (temp == 1):
            result = True
        else:
            result = False
        return result

    def moveMinHold(self, speed, power):
        output = bytearray()
        # message from master to module
        output.append(0xEB)
        output.append(0x90)
        # module id
        output.append(self.gripper_id)
        # Data Length   
        output.append(0x05)
        # Command get state
        output.append(0x18)

        temp_int1 = speed
        temp_int2 = power

        output.append(temp_int1 & 0xff)
        output.append((temp_int1 >> 8) & 0xff)
        output.append(temp_int2 & 0xff)
        output.append((temp_int2 >> 8) & 0xff)

        #Checksum calculation

        check_num = 0
        leno = output[3] + 5

        for i in range(2,leno-1):
            check_num = check_num + output[i]

        # Add checksum to the output buffer
        output.append(check_num & 0xff)

        #Send message to the module
        self.ser.write(output)
        time.sleep(0.015)

        inputtmp =self.ser.read(64)
        inputdata =[]
        for i in range(len(inputtmp)):
            inputdata.append(int(inputtmp[i]))

        temp = inputdata[5]
        if (temp == 1):
            result = True
        else:
            result = False
        return result
    
    def close(self):
        self.ser.close()

    # ========= main loop in process ============
    def run(self):
        # 新增夹爪控制
        self.gripper_start()
        self.moveMax(speed=500)

        try:
            # send one message immediately so client can start reading
            curopen, power = self.getState(self.gripper_id)
            self.ring_buffer.put({
                'curopen': curopen,
                'power': power,
                'robot_receive_timestamp': time.time()
            })
            self.ready_event.set()

            keep_running = True
            while keep_running:

                curopen, power = self.getState(self.gripper_id)
                self.ring_buffer.put({
                'curopen': curopen,
                'power': power,
                'robot_receive_timestamp': time.time()
                })

                # fetch command from queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break

                    elif cmd == Command.MOVE_TGT.value:
                        movetgt = command['target_pose']
                        self.moveTgt(movetgt)
                    
                    elif cmd == Command.MOVE_MAX.value:
                        speed = command['speed']
                        self.moveMax(speed)
                    
                    elif cmd == Command.MOVE_MIN.value:
                        speed = command['speed']
                        power = command['power']
                        self.moveMin(speed, power)
                    
                    elif cmd == Command.MOVE_MIN_HOLD.value:
                        speed = command['speed']
                        power = command['power']
                        self.moveMinHold(speed, power)
                    
                    else:
                        keep_running = False
                        break
        finally:
            self.close()

if __name__ == "__main__":
    gripper = inspire_gripper(com_port='/dev/ttyUSB0',baudrate=115200)
    # gripper.close()
    gripper.gripper_start()
    gripper.moveMax(speed=500)