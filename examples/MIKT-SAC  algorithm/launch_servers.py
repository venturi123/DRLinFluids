from multiprocessing import Process
import time
import argparse
import socket
import os
import re
from utils import check_ports_avail
from RemoteEnvironmentServer import RemoteEnvironmentServer
import envobject
from hanshu.pendulum import PendulumEnv
import numpy as np
import torch
import pprint

import gym
# from hanshu.registration import make
import pandas as pd
import torch
# from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
from hanshu.worker import DummyEnvWorker,RayEnvWorker,SubprocEnvWorker,EnvWorker
from hanshu.collector import Collector, VectorReplayBuffer,AsyncCollector
from hanshu.sac import SACPolicy
from hanshu.venvs import DummyVectorEnv,SubprocVectorEnv,RayVectorEnv,ShmemVectorEnv
from hanshu.offpolicy import offpolicy_trainer
from hanshu.tensorboard import TensorboardLogger
from hanshu.common import Net
from hanshu.continuous import Actor, ActorProb, Critic,RecurrentActorProb,RecurrentCritic

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--number-servers", required=True, help="number of servers to spawn", type=int)
ap.add_argument("-p", "--ports-start", required=True, help="the start of the range of ports to use", type=int)
ap.add_argument("-t", "--host", default="None", help="the host; default is local host; string either internet domain or IPv4", type=str)

args = vars(ap.parse_args())

number_servers = args["number_servers"]
ports_start = args["ports_start"]
host = args["host"]

foam_params = {
    'delta_t': 0.00005,
    'solver': 'pisoFoam',
    'num_processor': 32,
    'of_env_init': 'source ~/OpenFOAM/OpenFOAM-8/etc/bashrc',
    'cfd_init_time': 0.0001,  # 初始化流场，初始化state
    'num_dimension': 2,
    'verbose': False
}

entry_dict_q01 = {
    'U': {
        'JET1': {
            'v0': '({x} 0 0)',
        },
        'JET8': {
            'v0': '(0 {x} 0)',
        },
    }
}
entry_dict_q11 = {
    'U': {
        'JET1': {
            'v1': '({x} 0 0)',
        },
        'JET8': {
            'v1': '(0 {x} 0)',
        },
    }
}

entry_dict_q02 = {
    'U': {
        'JET2': {
            'v0': '({x} 0 0)',
        },
        'JET3': {
            'v0': '(0 {-x} 0)',
        },
    }
}
entry_dict_q12 = {
    'U': {
        'JET2': {
            'v1': '({x} 0 0)',
        },
        'JET3': {
            'v1': '(0 {-x} 0)',
        },
    }
}

entry_dict_q03 = {
    'U': {
        'JET5': {
            'v0': '({-x} 0 0)',
        },
        'JET4': {
            'v0': '(0 {-x} 0)',
        },
    }
}
entry_dict_q13 = {
    'U': {
        'JET5': {
            'v1': '({-x} 0 0)',
        },
        'JET4': {
            'v1': '(0 {-x} 0)',
        },
    }
}

entry_dict_q04 = {
    'U': {
        'JET6': {
            'v0': '({-x} 0 0)',
        },
        'JET7': {
            'v0': '(0 {x} 0)',
        },
    }
}
entry_dict_q14 = {
    'U': {
        'JET6': {
            'v1': '({-x} 0 0)',
        },
        'JET7': {
            'v1': '(0 {x} 0)',
        },
    }
}

entry_dict_t0 = {
    'U': {
        'JET1': {
            't0': '{t}'
        },
        'JET2': {
            't0': '{t}'
        },
        'JET3': {
            't0': '{t}'
        },
        'JET4': {
            't0': '{t}'
        },
        'JET5': {
            't0': '{t}'
        },
        'JET6': {
            't0': '{t}'
        },
        'JET7': {
            't0': '{t}'
        },
        'JET8': {
            't0': '{t}'
        },
    }
}

agent_params = {
    'entry_dict_q01': entry_dict_q01,
    'entry_dict_q11': entry_dict_q11,
    'entry_dict_q02': entry_dict_q02,
    'entry_dict_q12': entry_dict_q12,
    'entry_dict_t0': entry_dict_t0,
    'deltaA': 0.05,
    'minmax_value': (-1, 1),
    'interaction_period': 0.005,
    'vortex_shedding': 0.06,
    'action_discount': 0.1,
    'cd_0': 2.27,
    'purgeWrite_numbers': 0,
    'writeInterval': 0.005,
    'deltaT': 0.00005,
    'variables_q0': ('x',),
    'variables_q1': ('y',),
    'variables_t0': ('t',),
    'verbose': False,
    "zero_net_Qs": True,
}
state_params = {
    'type': 'pressure'
}
root_path = os.getcwd()
env_name_list = sorted([envs for envs in os.listdir(root_path) if re.search(r'^env\d+$', envs)])
env_path_list = ['/'.join([root_path, i]) for i in env_name_list]

if host == 'None':
    host = socket.gethostname()

list_ports = [ind_server + ports_start for ind_server in range(number_servers)]

# check for the availability of the ports 检查端口的可用性
if not check_ports_avail(host, list_ports):
    quit()

def launch_one_server(rank, host, port):
    train_env = gym.make('OpenFoam-v0', foam_root_path=env_path_list[rank],
             foam_params=foam_params,
             agent_params=agent_params,
             state_params=state_params,)
    print(type(train_env))

    #tensorforce_environment为定义好的环境   host=localhost   port为开放的端口
    RemoteEnvironmentServer(train_env, host=host, port=port,verbose=2)

processes = []

# launch all the servers one after the other  一个接一个地启动所有服务器
# 按照环境数定义proc
for rank, port in enumerate(list_ports):
    print("launching process of rank {}".format(rank))
    print(host, port)
    proc = Process(target=launch_one_server, args=(rank, host, port))
    proc.start()
    processes.append(proc)
    time.sleep(2.0)  # just to avoid collisions in the terminal printing  只是为了避免终端打印中的碰撞

print("all processes started, ready to serve...")

for proc in processes:
    proc.join()
