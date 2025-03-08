from __future__ import print_function
import pickle


class EchoServer(object):
    '''Implement a simple echo server for sending data and instruction through a socket.
    实现一个简单的回显服务器，用于通过套接字发送数据和指令。
    '''

    def __init__(self, verbose=0):

        # the commands that are supported; should correspond to methods
        # in the RemoteEnvironmentServer class. Those commands can be
        # used by the RemoteEnvironmentClient.
        #支持的命令； 应对应于 RemoteEnvironmentServer 类中的方法。
        # RemoteEnvironmentClient 可以使用这些命令。

        self.supported_requests = (
            # Put the simulation to the state from which learning begins.将模拟置于学习开始的状态。
            # If successfull RESET: 1RESET, fail empty  如果RESET成功：1 RESET，失败为空
            'RESET',
            # Respond with the state of Simulation (some vector in state space) 响应模拟状态（状态空间中的一些向量）
            'STATE',
            # CONTROL: valuesCONTROL, success CONTROL: 1CONTROL, fail empty  CONTROL: valuesCONTROL, 成功 CONTROL: 1CONTROL, 失败为空
            'CONTROL',
            # Evolve using the set control, success EVOLVE: 1EVOLVE, fail empty  使用set控制进化，成功EVOLVE：1EVOLVE，失败为空
            'EVOLVE',
            # Response to reward, sucess REWARD: valueREWARD, fail empty  响应奖励，成功REWARD：valueREWARD，失败为空
            'REWARD',
            # Is the solver done? value 0 1, empty fail  求解器完成了吗？ 值 0 1，空失败
            'TERMINAL',
            # other information
            'INFO',
            )

        self.verbose = verbose

    @staticmethod
    def decode_message(msg, verbose=1):
        msg = pickle.loads(msg)

        assert(isinstance(msg, (list,)))
        assert(len(msg) == 2)

        request = msg[0]
        data = msg[1]

        # if verbose > 1:
        #     print("decode message --------------")
        #     print(" request:")
        #     print(request)
        #     print(" data:")
        #     print(data)
        #     print("-----------------------------")

        return request, data

    @staticmethod
    def encode_message(request, data, verbose=0):
        '''Encode data (a list) as message'''
        #将数据（列表）编码为消息

        complete_message = [request, data]
        msg = pickle.dumps(complete_message)

        # if verbose > 1:
        #     print("encode message --------------")
        #     print(" request:")
        #     print(request)
        #     print(" data:")
        #     print(data)

        return msg

    # def handle_message(self, msg):
    def handle_message(self, request, data):
        '''Trigger action base on client message.'''
        #基于客户端消息的触发操作

        # request, data = EchoServer.decode_message(msg, verbose=self.verbose)

        if request not in self.supported_requests:
            RuntimeError("unknown request; no support for {}".format(request))
            return EchoServer.encode_message(request, [])

        # Dispatch to action which returns the data. This is what
        # children need to implement
        # so this calls the method of the EchoSolver (or the class derived from it)
        # such as RemoteEnvironmentServer
        #调度返回数据的动作。 这是孩子们需要实现的，所以这调用了 EchoSolver（或从它派生的类）的方法，例如 RemoteEnvironmentServer
        result = getattr(self, request)(data)    #这里调用server中的函数来获取结果result
        msg=EchoServer.encode_message(request, result, verbose=self.verbose)

        # Wrap
        return msg   #压缩信息并发送
