from threading import Thread
# from tensorforce import TensorforceError
# from tensorforce.environments import Environment
import gym
import re
import socket
from echo_server import EchoServer
import time
import pandas as pd
import numpy as np
import struct
import json

class RemoteEnvironmentClient(gym.Env):
    """Used to communicate with a RemoteEnvironmentServer. The idea is that the pair
    (RemoteEnvironmentClient, RemoteEnvironmentServer) allows to transmit information
    through a socket seamlessly.

    The RemoteEnvironmentClient can be directly given to the Runner.

    The RemoteEnvironmentServer herits from a valid Environment add adds the socketing.
    用于与远程环境服务器通信。这个想法是这对（RemoteEnvironmentClient， RemoteEnvironmentServer） 允许传输信息通过插座无缝衔接。
    远程环境客户端可以直接提供给运行器。
    远程环境服务器从有效的环境添加套接字。
    """

    def __init__(self,
                 # env,
                 port=12230,
                 host=None,
                 crrt_simu=0,
                 number_servers=0,
                 verbose=1,
                 buffer_size=10000,
                 timing_print=False,
                 ):
        """(port, host) is the necessary info for connecting to the Server socket.
        （端口、主机）是连接到服务器套接字的必要信息。
        """
        # templated tensorforce stuff
        self.observation = None
        self.number_servers=number_servers
        self.thread = None

        self.buffer_size = buffer_size

        self.timing_print = timing_print

        # make arguments available to the class
        # socket
        self.port = port + crrt_simu
        self.host = host
        # misc
        self.verbose = verbose
        # states and actions  可以不需要   只是提供state action space
        # self.env=env

        # start the socket
        self.valid_socket = False
        socket.setdefaulttimeout(60)
        self.socket = socket.socket()
        self.socket.setblocking(0)  #设置为非阻塞模式  不然recv返回none会报错
        self.socket.settimeout(60000.0)    #设置大尺度超时区间   防止超时未能接受到数据出错  240

        #if crrt_simu<int(self.number_servers):    #env1~5在飞哥运行
           # self.host = "10.249.159.249"

        # if crrt_simu<int(self.number_servers/7):    #env1~5在飞哥运行
        #     self.host = "10.249.159.181"
        # # elif crrt_simu<int(self.number_servers/7*2):   #env6~10在贾哥运行
        # #     self.host = "10.249.158.229"
        # elif crrt_simu<int(self.number_servers/7*3):   #env11~15在老服务器运行
        #     self.host = "10.249.156.0"
        # # elif crrt_simu<int(self.number_servers/7*4):   #env11~15在老服务器运行
        # #     self.host = "10.249.158.84"
        # elif crrt_simu<int(self.number_servers/7*5):   #env11~15在老服务器运行
        #     self.host = "10.249.152.93"
        # # elif crrt_simu<int(self.number_servers/7*6):   #env11~15在老服务器运行
        # #     self.host = "10.249.153.184"
        # elif crrt_simu<int(self.number_servers/7*7):   #env11~15在老服务器运行
        #     self.host = "10.249.158.75"
        # elif crrt_simu<int(self.number_servers/7*5):   #env11~15在老服务器运行
        #     self.host = "10.249.158.102"

        #re22000
        
        if crrt_simu<int(self.number_servers/5):    #env1~2在飞哥运行 0 1<2
            self.host = "10.249.159.181"
        elif crrt_simu<int(self.number_servers/5*2):   #env3~4在新服务器运行 2 3<4
            self.host = "10.249.158.84"
        elif crrt_simu<int(self.number_servers/5*3):   #env5~6在junle服务器运行 4 5<6
            self.host = "10.249.159.249"
        elif crrt_simu<int(self.number_servers/5*4):   #env7~8在xyy服务器运行
            self.host = "10.249.158.102"
        elif crrt_simu<int(self.number_servers/5*5):   #env9~10在jxh服务器运行
            self.host = "10.249.157.17"
	
        # connect to the socket
        # self.host="10.249.159.181"
        print(self.host, self.port)
        self.socket.connect((self.host, self.port))  #linux默认最高是70s，如果超过70s没有意义

        # for a reason I cannot understand, need next line to be able to use forwarding between dockers
        #由于我无法理解的原因，需要下一行才能在码头工人之间使用转发
        # self.socket.connect(('localhost', self.port))
        # self.host = socket.gethostbyname('localhost')
        # self.socket.connect((self.host, self.port))
        if self.verbose > 0:
            print('Connected to {}:{}'.format(self.host, self.port))
        # now the socket is ok
        self.valid_socket = True

        self.episode = 0
        self.step = 0

        self.time_start = 0
        self.crrt_time = 0
        self.armed_time_measurement = False
        self.start_function = 0
        self.end_function = 0
        self.crrt_time_function = 0
        self.total_function_time = 0
        self.proportion_env_time = 0
        self.timeout=[]

        self.name={'RESET':26,'STATE':3284, 'CONTROL':28,'EVOLVE':27,
                   'REWARD':129,'TERMINAL':28,'INFO':160,}   #注意随时更改
        self.headerdata_len={'RESET':49,'STATE':51, 'CONTROL':49,'EVOLVE':49,
                   'REWARD':50,'TERMINAL':49,'INFO':50,}   #应该不用更改

    def close(self):
        # TODO: think about sending a killing message to the server? Maybe not necessary - can reuse the
        # server maybe - but may be needed if want to clean after us.
        if self.valid_socket:
            self.socket.close()

    def reset(self):
        self.update_time_function_start()

        # perform the reset
        _ = self.communicate_socket("RESET", np.array([0.1]))

        # get the state
        _, init_state = self.communicate_socket("STATE", np.array([0.1]))

        # Updating episode and step numbers
        self.episode += 1
        self.step = 0

        # if self.verbose > 1:
        #     print("reset done; init_state:",init_state)
            # print(init_state)

        self.update_time_function_end()

        self.print_time_information()

        return (init_state)

    def update_time_function_start(self):
        if self.armed_time_measurement:
            self.start_function = time.time()

    def update_time_function_end(self):
        if self.armed_time_measurement:
            self.end_function = time.time()
            self.crrt_time_function = self.end_function - self.start_function
            self.start_function = None
            self.total_function_time += self.crrt_time_function

    def arm_time_measurements(self):
        if not self.armed_time_measurement:
            self.armed_time_measurement = True
            if self.timing_print:
                print("arming time measurements...")
            self.time_start = time.time()

    def print_time_information(self):
        if self.timing_print and self.armed_time_measurement:
            print("summary timing measurements...")
            self.total_time_since_arming = time.time() - self.time_start
            print("total time since arming: {}".format(self.total_time_since_arming))
            print("total time in env functions: {}".format(self.total_function_time))
            print("proportion in env functions: {}".format(float(self.total_function_time) / float(self.total_time_since_arming)))

    def execute(self,actions: np.ndarray):
        # arm the time measurements the first time execute is hit 布防第一次执行时的时间测量被击中
        self.arm_time_measurements()

        self.update_time_function_start()

        # send the control message
        self.communicate_socket("CONTROL", actions)

        # ask to evolve
        self.communicate_socket("EVOLVE", np.array([0.1]))

        # obtain the next state
        _, next_state = self.communicate_socket("STATE",  np.array([0.1]))

        # check if terminal
        _, terminal = self.communicate_socket("TERMINAL",  np.array([0.1]))

        # get the reward
        _, reward = self.communicate_socket("REWARD",  np.array([0.1]))

        # get the info
        _, info = self.communicate_socket("INFO",  np.array([0.1]))

        # now we have done one more step
        self.step += 1

        self.update_time_function_end()

        # return (next_state,  reward, terminal,_)
        return (next_state, reward, terminal, info)

    def communicate_socket(self, request, data):
        """Send a request through the socket, and wait for the answer message.
        通过套接字发送请求，然后等待应答消息。
        """
        # if request=='CONTROL':
        #     print(data,len(data), type(data))
        #控制端传输的为actions或 1   这里发送信息也要传递data_shape
        to_send = EchoServer.encode_message(request, data, verbose=self.verbose)    #打包数据

        cmd_len = len(to_send)
        while cmd_len == 0:
            print("command has nothing return")
            cmd_len = len(to_send)
        # 制作报头 对要发送的内容用字典进行描述
        header_dic = {
            'total_size': cmd_len,  # 总共的大小
            'request': None,
            'md5': None}
        header_json = json.dumps(header_dic)  # 字符串类型
        header_bytes = header_json.encode('utf-8')  # 转成bytes类型(但是长度是可变的)
        while cmd_len == 0:
            print("command has nothing return")  # 发现这里如果cmd_len为0会导致异常，有些是没有返回值的command
            cmd_len = len(header_bytes)
        # 先发报头的长度
        self.socket.send(struct.pack('i', len(header_bytes)))  # 发送固定长度的报头   这种可能丢包
        # 再发报头
        self.socket.send(header_bytes)  # 这种传递肯定有数据

        self.socket.send(to_send)


        # TODO: the recv argument gives the max size of the buffer, can be a source of missouts if
        # a message is larger than this; add some checks to verify that no overflow
        # recv 参数给出缓冲区的最大大小，如果消息大于此大小，则可能是错过的来源;添加一些检查以验证没有溢出
        shape=self.name[f'{request}']
        header_shape=self.headerdata_len[f'{request}']
        starttime = time.time()

        #检测数据shape 先收报头
        # header = self.socket.recv(4)  # 收四个
        # while True:
        #     header = self.socket.recv(4)  # 收四个
        #     if len(header)==4:
        #         unpack_res = struct.unpack('i', header)
        #         break

        # 先收报头的长度    这里可能陷入死循环  删去
        # header_size=0
        header=b""
        # time.sleep(0.1)
        a = self.socket.recv(4)
        # header_size += len(a)
        header += a

        try:
            header_len = struct.unpack('i', header)[0]  # 吧bytes类型的反解  header长度出错  这里会跳过
            # print('header_len',header_len,header_shape)
            recv_size = 0
            header_bytes = b""
            while recv_size < header_len:  # 循环的收
                # header_bytes = self.socket.recv(header_len)  # 收过来的也是bytes类型
                received_msg = self.socket.recv(header_len)  # 接收数据
                recv_size += len(received_msg)
                header_bytes += received_msg
        except:  #一旦出错
            recv_size = 0
            header_bytes = b""
            while recv_size < header_shape:  # 循环的收  此时无header_len
                received_msg = self.socket.recv(header_shape)  # 接收数据
                recv_size += len(received_msg)
                header_bytes += received_msg

            # header_bytes = self.socket.recv(header_shape)  # 收过来的也是bytes类型   继续收集数据

        # 在收报头
        # header_bytes =  self.socket.recv(1024)  # 收过来的也是bytes类型
        header_json = header_bytes.decode('utf-8')  # 拿到json格式的字典
        header_dic = json.loads(header_json)  # 反序列化拿到字典了
        total_size = header_dic['total_size']  # 就拿到数据的总长度了

        # 最后收数据
        recv_size=0
        data_set=b""
        # print(request,total_size)
        while recv_size<total_size: #循环的收
            # time.sleep(0.1)
            print('client',request,recv_size, total_size)
            received_msg = self.socket.recv(self.buffer_size)  # 接收数据
            recv_size += len(received_msg)
            data_set += received_msg

        # while True:
        #     time.sleep(0.2)
        #     received_msg = self.socket.recv(self.buffer_size)  # 接收数据
        #     if received_msg is not None:
        #         data_set += received_msg
        #     if received_msg is None or len(data_set) < shape:  #这个shape必须动态调整  envs数量不同带来的shape大小不同
        #         print(len(data_set), shape)
        #         # break
        #         continue
        #     if len(data_set) == shape:
        #         break

        # received_msg = self.socket.recv(self.buffer_size)  # 接收数据
        # data_set += received_msg
        endtime = time.time()

        timeout=endtime-starttime
        # print('timeout',timeout)
        self.timeout.append(timeout)
        pd.DataFrame(
            self.timeout
        ).to_csv('timeout.csv', index=False, header=False)
        # print(request,len(data_set),)

        request, data = EchoServer.decode_message(data_set, verbose=self.verbose)  # 此处调用echo_server

        return (request, data)


