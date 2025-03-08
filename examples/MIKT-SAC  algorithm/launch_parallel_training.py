import argparse
import os
import sys
import csv
import socket
import numpy as np
import re
import envobject

# from simulation_base.env import resume_env, nb_actuations
from RemoteEnvironmentClient import RemoteEnvironmentClient
from hanshu.pendulum import PendulumEnv
import numpy as np
import gym
from gym import spaces, logger
import pandas as pd
import torch
import warnings
warnings.filterwarnings("ignore")
import pickle
import random
from datetime import datetime
# from tianshou.data import VectorReplayBuffer,AsyncCollector,Collector
# from tianshou.policy import ImitationPolicy, SACPolicy
# from tianshou.env import DummyVectorEnv,SubprocVectorEnv,RayVectorEnv,ShmemVectorEnv
# from tianshou.trainer import offpolicy_trainer
# from tianshou.utils import TensorboardLogger
# from tianshou.utils.net.common import Net
# from tianshou.utils.net.continuous import Actor, ActorProb, Critic,Critic,RecurrentActorProb,RecurrentCritic

from hanshu.worker import DummyEnvWorker,RayEnvWorker,SubprocEnvWorker,EnvWorker
from hanshu.collector import Collector, VectorReplayBuffer,AsyncCollector
from hanshu.sac import SACPolicy
from hanshu.venvs import DummyVectorEnv,SubprocVectorEnv,RayVectorEnv,ShmemVectorEnv
from hanshu.offpolicy import offpolicy_trainer
from hanshu.tensorboard import TensorboardLogger
from hanshu.common import Net
from hanshu.continuous import Actor, ActorProb, Critic,RecurrentActorProb,RecurrentCritic

from hanshu.pendulum import PendulumEnv
from hanshu.openfoam.environments import OpenFoam
from hanshu.openfoam import cfd, utils
# from hanshu.core import Env

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--number-servers", required=True, help="number of servers to spawn", type=int)
ap.add_argument("-p", "--ports-start", required=True, help="the start of the range of ports to use", type=int)
ap.add_argument("-t", "--host", default="None", help="the host; default is local host; string either internet domain or IPv4", type=str)

args = vars(ap.parse_args())

number_servers = args["number_servers"]
ports_start = args["ports_start"]
host = args["host"]

if host == 'None':
    host = socket.gethostname()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='OpenFoam-v0')
    parser.add_argument('--reward-threshold', type=float, default=15.8)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--buffer-size', type=int, default=200000)
    parser.add_argument('--actor-lr', type=float, default=2e-4)
    parser.add_argument('--critic-lr', type=float, default=3e-4)
    parser.add_argument('--il-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.97)
    parser.add_argument('--tau', type=float, default=0.005)
    #温度系数
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--auto-alpha', type=int, default=1)
    parser.add_argument('--alpha-lr', type=float, default=3e-4)
    parser.add_argument('--epoch', type=int, default=800)
    #所有环境一起计算epoch
    parser.add_argument('--step-per-epoch', type=int, default=500)
    parser.add_argument('--il-step-per-epoch', type=int, default=1)
    parser.add_argument('--step-per-collect', type=int, default=10)
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[512,256])
    parser.add_argument(
        '--imitation-hidden-sizes', type=int, nargs='*', default=[512, 256]
    )
    parser.add_argument('--training-num', type=int, default=5)
    parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--rew-norm', action="store_true", default=False)
    parser.add_argument('--n-step', type=int, default=5)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--save-buffer-name", type=str, default=None)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="atari.benchmark")
    parser.add_argument(
        "--icm-lr-scale",
        type=float,
        default=0.,
        help="use intrinsic curiosity module with this lr scale"
    )
    parser.add_argument(
        "--resume",
        default=False,
        action="store_true",
        help="restart"
    )
    args = parser.parse_known_args()[0]
    return args

def test_sac_with_il(args=get_args()):
    foam_params = {
        'delta_t': 0.00005,
        'solver': 'pisoFoam',
        'num_processor': 16,
        'of_env_init': 'source ~/OpenFOAM/OpenFOAM-8/etc/bashrc',
        'cfd_init_time': 0.005,  # 初始化流场，初始化state
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
        'vortex_shedding':0.06,
        'action_discount':0.1,
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

    train_envs  = SubprocVectorEnv(
        [lambda x=i: RemoteEnvironmentClient(
            verbose=2, port=ports_start , host=host, crrt_simu=x, number_servers=number_servers,
        timing_print=(x == 0), ) for i in range(number_servers)],
        wait_num=number_servers, timeout=0.2
    )
    # train_envs  = DummyVectorEnv(
    #     [lambda x=i: RemoteEnvironmentClient(
    #          verbose=2, port=ports_start , host=host, crrt_simu=x, number_servers=number_servers,
    #     timing_print=(x == 0),) for i in range(number_servers)],
    #     wait_num=args.training_num, timeout=0.2
    # )

    state_params['probe_info'] = utils.read_foam_file(
        '/'.join([env_path_list[0], 'system', 'probes'])
    )
    state = np.zeros((int(state_params['probe_info'].shape[0] + 4),
                          30), dtype=int)
    observation_space = spaces.Box(low=-np.Inf, high=np.Inf,
                                  shape=(state.shape), dtype=np.float32)
    action_space = spaces.Box(agent_params['minmax_value'][0], agent_params['minmax_value'][1],
                              shape=(len(agent_params['variables_q0'])*4,), dtype=np.float32)

    args.state_shape = observation_space.shape or observation_space.n
    args.action_shape = action_space.shape or action_space.n
    print(args.state_shape,args.action_shape)
    args.max_action = action_space.high[0]
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    net = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = ActorProb(
        net,
        args.action_shape,
        max_action=args.max_action,
        device=args.device,
        unbounded=True
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c1 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device
    )
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    net_c2 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device
    )
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)
    if args.auto_alpha:
        target_entropy = -np.prod(action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)


    # print(actor,type(actor))
    policy = SACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        reward_normalization=args.rew_norm,
        estimation_step=args.n_step,
        action_space=action_space
    )
    # collector
    buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    train_collector = AsyncCollector(policy,  train_envs,   buffer, exploration_noise=True )    #这一步调用reset
    # train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)

    log_path = os.path.join(args.logdir, args.task, 'sac')

    if args.resume:
        # load from existing checkpoint
        print(f"Loading agent under {log_path}")
        ckpt_path = os.path.join(log_path, 'best_model.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=args.device)
            policy.load_state_dict(checkpoint['model'])
            # policy.optim.load_state_dict(checkpoint['optim'])
            print("Successfully restore policy and optim.")
        else:
            print("Fail to restore policy and optim.")
        buffer_path = os.path.join( 'SAC.pkl')
        if os.path.exists(buffer_path):
            train_collector.buffer = pickle.load(open(buffer_path, "rb"))
            print("Successfully restore buffer.")
        else:
            print("Fail to restore buffer.")

    print(policy.training,policy.updating)

    for i in range(int(1e6)):  # total step
        #每个循环每个envs都跑一个action
        train_collector.collect(n_step=number_servers)
        policy.update(512, train_collector.buffer, batch_size=512, repeat=10)
        if i % 100 == 0:
            torch.save({'model': policy.state_dict(),}, os.path.join(log_path, f'best_model_{i}.pth'))
            pickle.dump(buffer, open(os.path.join(os.getcwd(), "SAC.pkl"), "wb"))


test_sac_with_il()