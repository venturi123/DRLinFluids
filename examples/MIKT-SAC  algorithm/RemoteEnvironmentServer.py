from echo_server import EchoServer
import socket
import pickle
import time
import struct
import json

class RemoteEnvironmentServer(EchoServer):

    def __init__(self,
                 tensorforce_environment,
                 host=None,
                 port=12230,
                 buffer_size=10000,
                 verbose=1):

        # tensorforce_environment should be a ready-to-use environment tensorforce_environment应该是即用型环境
        # host, port is where making available

        self.tensorforce_environment = tensorforce_environment
        self.state = None
        self.terminal = False
        self.reward = None
        self.info={}
        self.nbr_reset = 0

        self.buffer_size = buffer_size

        self.headerdata_len={'RESET':51,'STATE':51, 'CONTROL':54,'EVOLVE':52,
                   'REWARD':52,'TERMINAL':54,'INFO':50,}   #应该不用更改

        EchoServer.__init__(self, verbose)

        # set up the socket
        socket_instance = socket.socket()

        if host is None:
            host = socket.gethostname()

        # host="10.249.159.181"
        socket_instance.bind((host, port))
        # for a reason I cannot understand, need next line to be able to use forwarding between dockers
        # socket_instance.bind(('localhost', port))

        socket_instance.listen(1)  # Buffer only one request

        connection = None

        while True:      #此处一直循环  每个step reset的运行
            if connection is None:
                if verbose > 1:
                    print('[Waiting for connection...]')
                connection, address = socket_instance.accept()    #此处返回self.socket  address为链接IP:端口
                if verbose > 1:
                    print('Got connection from {}'.format(address))
            else:
                # if verbose > 1:
                #     print('[Waiting for request...]')
                # time.sleep(0.1)
                # message = connection.recv(self.buffer_size)      #接受数据

                # header_shape = self.headerdata_len[f'{request}']

                # 先收报头的长度
                # header_size = 0
                header_shape = 49  # control最大49   将所有request大小转为49
                header = b""
                # time.sleep(0.1)
                a = connection.recv(4)
                # header_size += len(a)
                header += a
                # print(header, len(header))

                try:
                    header_len = struct.unpack('i', header)[0]  # bytes类型的反解  header长度出错  这里会跳过
                    recv_size = 0
                    header_bytes = b""
                    while recv_size < header_len:  # 循环的收
                        # header_bytes = self.socket.recv(header_len)  # 收过来的也是bytes类型
                        received_msg = connection.recv(header_len)  # 接收数据
                        recv_size += len(received_msg)
                        header_bytes += received_msg
                except:  # 一旦出错
                    recv_size = 0
                    header_bytes = b""
                    while recv_size < header_shape:  # 循环的收  此时无header_len
                        received_msg = connection.recv(header_shape)  # 接收数据
                        recv_size += len(received_msg)
                        header_bytes += received_msg
                # 在收报头
                header_json = header_bytes.decode('utf-8')  # 拿到json格式的字典
                header_dic = json.loads(header_json)  # 反序列化拿到字典了
                total_size = header_dic['total_size']  # 就拿到数据的总长度了

                # 最后收数据
                recv_size = 0
                message = b""
                while recv_size < total_size:  # 循环的收
                    # message = connection.recv(self.buffer_size)  # 接受数据
                    received_msg = connection.recv(self.buffer_size)  # 接收数据
                    recv_size += len(received_msg)
                    message += received_msg

                request, data = self.decode_message(message, verbose=self.verbose)
                # print('server',request, header_len,header_shape,total_size )

                response = self.handle_message(request, data)
                # response = self.handle_message(message)  # this is given by the EchoServer base class 这是由 EchoServer 基类给出的


                # 发现这里如果cmd_len为0会导致异常，有些是没有返回值的command
                cmd_len = len(response)
                while cmd_len == 0:
                    print("command has nothing return")
                    cmd_len = len(response)

                # 制作报头 对要发送的内容用字典进行描述
                header_dic = {
                    'total_size': cmd_len,  # 总共的大小
                    'filename': None,
                    'md5': None}
                header_json = json.dumps(header_dic)  # 字符串类型
                header_bytes = header_json.encode('utf-8')  # 转成bytes类型(但是长度是可变的)

                while cmd_len == 0:
                    print("command has nothing return")   # 发现这里如果cmd_len为0会导致异常，有些是没有返回值的command
                    cmd_len = len(header_bytes)
                # 先发报头的长度
                connection.send(struct.pack('i', len(header_bytes)))  # 发送固定长度的报头   这种可能丢包
                # 再发报头
                connection.send(header_bytes)      #这种传递肯定有数据

                # time.sleep(0.1)
                connection.send(response)   #发送算的每一步的信息

        # TODO: do we really get here? Should we clean the while True loop somehow to allow to stop completely?
        #我们真的到了这里吗？我们是否应该以某种方式清理 while True 循环以允许完全停止？
        socket_instance.close()

    def RESET(self, data):
        self.nbr_reset += 1
        self.state = self.tensorforce_environment.reset()
        return(1)  # went fine

    def STATE(self, data):
        return(self.state)

    def TERMINAL(self, data):
        return(self.terminal)

    def REWARD(self, data):
        return(self.reward)

    def INFO(self, data):
        return(self.info)

    def CONTROL(self, data):
        self.actions = data
        return(1)  # went fine

    def EVOLVE(self, data):
        # self.state, self.reward,self.terminal, self.info = self.tensorforce_environment.step(self.actions)
        self.state, self.reward, self.terminal, self.info = self.tensorforce_environment.execute(self.actions)  #step
        return(1)  # went fine

    # TODO: add one to close the server when done
