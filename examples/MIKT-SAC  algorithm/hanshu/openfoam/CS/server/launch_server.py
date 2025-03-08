# coding = UTF-8
import argparse
import os
import time
from importlib import import_module
from multiprocessing import Process
from threading import Thread
import re

from tensorforce import Agent, Environment, Runner

envclass = import_module('envclass')
utilities = import_module('utilities')
var = import_module('var')

# parser = argparse.ArgumentParser()
# # parser.add_argument("-n", "--number-servers", required=True, help="number of servers to spawn", type=int)
# parser.add_argument('-t', '--host', default='127.0.0.1', help='the host; default is local host; string either internet domain or IPv4', type=str)
# parser.add_argument('-p', '--ports-start', required=True, help='the start of the range of ports to use', type=int)
# parser.add_argument('-n', '--num-episodes', required=True, help='Number of episodes', type=int)
# shell_args = vars(parser.parse_args())

shell_args = {
    'ports_start': 65432,
    'num_episodes': 100,
    'host': '127.0.0.1'
}

# 获取工作目录路径
root_path = os.getcwd()
# 获取Environment文件夹名称，并按照升序排列，root_path + env_path_list就能获取每一个环境文件夹的绝对路径
env_dir_list = sorted([dir for dir in os.listdir() if re.search('^env\d+', dir)])
# 新建一个environment对象列表
environments = []
# 新建一个server对象列表
processes = []
# 新建一个port对象列表
port_list = [shell_args['ports_start'] + i for i in range(len(env_dir_list))]
# 新建一个index，用于循环输出env_list对象
index = 0

# 检查端口是否可用
assert utilities.check_ports(shell_args['host'], port_list)

# 设置Environments list
for env_dir in env_dir_list:
    jet_info = {
        'name': ['hblclot', 'hbrclot'],
        'min_value': -20,
        'max_value': 20,
    }

    learning_params = {
        'state_type': 'pressure',
        'reward_function_type': 'drag_plain_lift',
        # 选取读取state历史数，一般只取最后一次观察到的数据
        'number_history_value': 1,
        'INTERVENTION_T': var.INTERVENTION_T,
        'DELTA_T': var.DELTA_T,
        'SOLVER': var.SOLVER,
        'N_PROCESSOR': var.N_PROCESSOR,
        'OF_INIT': var.OF_INIT,
        'NICE': var.NICE,
        'INIT_START_TIME': 1
    }

    probe_info = {
        'position': utilities.read_foam_file('/'.join([root_path, env_dir, 'system', 'probes']))
    }

    env = envclass.Env(
        path_root='/'.join([root_path, env_dir]),
        jet_info=jet_info,
        probe_info=probe_info,
        learning_params=learning_params
    )

    environments.append(env)


# 新建一个函数对象，用以建立新进程
def launch_one_server(environment, port):
    Environment.create(environment=environment, remote='socket-server', port=port)

for environment in environments:
    proc = Process(target=launch_one_server, args=(environment, port_list[index]))
    proc.start()
    processes.append(proc)
    print(f'Successful launching process {index+1}')
    time.sleep(2.0)  # just to avoid collisions in the terminal printing
    index += 1

print("all processes started, ready to serve...")