# coding = UTF-8
import argparse
import os
import time
from importlib import import_module
from multiprocessing import Process
from threading import Thread

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

# 设置Environments list
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
    'position': utilities.read_foam_file('/'.join([root_path, 'env01', 'system', 'probes']))
}

env = envclass.Env(
    path_root=None,
    jet_info=jet_info,
    probe_info=probe_info,
    learning_params=learning_params,
    server=False
)

# 定义神经网络形状
network_spec = [
    dict(type='dense', size=512),
    dict(type='dense', size=512)
]

# 初始化Agent
agent = Agent.create(
    agent='ppo', environment=env, batch_size=20, network=network_spec, learning_rate=1e-3, state_preprocessing=None,
    entropy_regularization=0.01, likelihood_ratio_clipping=0.2, subsampling_fraction=0.2, parallel_interactions=2,
    saver=dict(directory='saved_models/checkpoint', frequency=1), summarizer=dict(directory='summary')
)
print('Agent defined DONE!')

# 开始训练
runner = Runner(
    agent=agent, num_parallel=2, remote='socket-client', host='127.0.0.1', port=shell_args['ports_start']
)
print('Runner defined DONE!')
# input('Press any key to continue')
runner.run(num_episodes=shell_args['num_episodes'])
runner.close()
