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
from tianshou.policy import ImitationPolicy, SACPolicy,DDPGPolicy
# from tianshou.env import DummyVectorEnv,SubprocVectorEnv,RayVectorEnv,ShmemVectorEnv
# from tianshou.trainer import offpolicy_trainer
# from tianshou.utils import TensorboardLogger
# from tianshou.utils.net.common import Net
# from tianshou.utils.net.continuous import Actor, ActorProb, Critic,Critic,RecurrentActorProb,RecurrentCritic
from hanshu.collector import Collector, VectorReplayBuffer,AsyncCollector
from hanshu.sac import SACPolicy
from hanshu.venvs import DummyVectorEnv,SubprocVectorEnv,RayVectorEnv,ShmemVectorEnv
from hanshu.offpolicy import offpolicy_trainer
from hanshu.tensorboard import TensorboardLogger
from hanshu.common import Net
from hanshu.continuous import Actor, ActorProb, Critic,RecurrentActorProb,RecurrentCritic
from hanshu.student_policy import student_Net,mlp,teacher_mlp,student_mlp

from torch import nn
from hanshu.openfoam.environments import OpenFoam
from test_sac_with_il_old import test_sac_old
import warnings
warnings.filterwarnings("ignore")
import sys
from hanshu.openfoam import cfd, utils

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
    parser.add_argument('--task', type=str, default='OpenFoam-v0')  # Pendulum-v1
    parser.add_argument('--reward-threshold', type=float, default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--buffer-size', type=int, default=200000)
    parser.add_argument('--actor-lr', type=float, default=2e-4)
    parser.add_argument('--critic-lr', type=float, default=3e-4)
    parser.add_argument('--il-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.97)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--alpha', type=float, default=0.04)
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
        '--device', type=str, default='cpu'
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

# @pytest.mark.skipif(envpool is None, reason="EnvPool doesn't support this platform")
def test_sac_with_il(args=get_args()):
    foam_params = {
        'delta_t': 0.00005,
        'solver': 'pisoFoam',
        'num_processor': 32,
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
    # if you want to use python vector env, please refer to other test scripts
    root_path = os.getcwd()
    env_name_list = sorted([envs for envs in os.listdir(root_path) if re.search(r'^env\d+$', envs)])
    env_path_list = ['/'.join([root_path, i]) for i in env_name_list]

    train_envs  = SubprocVectorEnv(
        [lambda x=i: RemoteEnvironmentClient(
            verbose=2, port=ports_start , host=host, crrt_simu=x, number_servers=number_servers,
        timing_print=(x == 0), ) for i in range(number_servers)],
        wait_num=number_servers, timeout=0.2
    )
    state_params['probe_info'] = utils.read_foam_file(
        '/'.join([env_path_list[0], 'system', 'probes'])
    )
    state = np.zeros((int(state_params['probe_info'].shape[0]),30), dtype=int)
    observation_space = spaces.Box(low=-np.Inf, high=np.Inf,
                                  shape=(state.shape), dtype=np.float32)
    action_space = spaces.Box(agent_params['minmax_value'][0], agent_params['minmax_value'][1],
                              shape=(len(agent_params['variables_q0'])*4,), dtype=np.float32)   #8action
    args.state_shape = observation_space.shape or observation_space.n   #（120，30）
    args.action_shape = action_space.shape or action_space.n
    print(args.state_shape,args.action_shape)
    args.max_action = action_space.high[0]
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    #old_policy提前定义
    old_policy=test_sac_old()
    source_env_observation_space_shape=(24,30)
    source_env_action_space_shape=(4,)
    for name, parameters in old_policy.actor.named_parameters():  # 打印出每一层的参数的大小
        print(name, ':', parameters.size())
    for name, parameters in old_policy.critic1.named_parameters():  # 打印出每一层的参数的大小
        print(name, ':', parameters.size())
    print(old_policy.actor.parameters(),old_policy.critic1.parameters())

    # model
    # 在这里提前定义encoder  每个模块可以调用并更新  encoder_actor与variational_actor设置为120 24
    encoder_actor = mlp(int(np.prod(args.state_shape[0])), int(np.prod(source_env_observation_space_shape[0])), device=args.device)   #这里设置120→24
    variational_actor = mlp(int(np.prod(source_env_observation_space_shape[0])), int(np.prod(args.state_shape[0])),device=args.device)  # 这里input_dim=source_env  输出为env  3到15
    variational_logstd = nn.Parameter(
        torch.zeros(1, int(np.prod(args.state_shape[0])), device=args.device))  # env_state_shape=input_dim
    student_net = student_Net(args.state_shape,old_policy=old_policy, hidden_sizes=args.hidden_sizes,
                              device=args.device,encoder=encoder_actor,variational_net=variational_actor,variational_logstd=variational_logstd)
    actor = ActorProb(
        student_net,
        args.action_shape,
        max_action=args.max_action,
        device=args.device,
        unbounded=True
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    encoder_critic1 = mlp(int(np.prod(args.state_shape[0])), int(np.prod(source_env_observation_space_shape[0])), device=args.device)   #这里设置120→24
    variational_critic1 = mlp(int(np.prod(source_env_observation_space_shape[0])), int(np.prod(args.state_shape[0])),device=args.device)  # 这里input_dim=source_env  输出为env  3到15
    variational_logstd_critic1 = nn.Parameter(
        torch.zeros(1, int(np.prod(args.state_shape[0])), device=args.device))  # env_state_shape=input_dim
    net_c1 = student_Net(
        args.state_shape,
        args.action_shape,
        old_policy=old_policy,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
        critic_name=1,encoder=encoder_critic1,variational_net=variational_critic1,variational_logstd=variational_logstd_critic1)
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)

    encoder_critic2 = mlp(int(np.prod(args.state_shape[0])), int(np.prod(source_env_observation_space_shape[0])), device=args.device)   #这里设置120→24
    variational_critic2 = mlp(int(np.prod(source_env_observation_space_shape[0])), int(np.prod(args.state_shape[0])),device=args.device)  # 这里input_dim=source_env  输出为env  3到15
    variational_logstd_critic2 = nn.Parameter(
        torch.zeros(1, int(np.prod(args.state_shape[0])), device=args.device))  # env_state_shape=input_dim
    net_c2 = student_Net(
        args.state_shape,
        args.action_shape,
        old_policy=old_policy,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
        critic_name=2,encoder=encoder_critic2,variational_net=variational_critic2,variational_logstd=variational_logstd_critic2)
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)
    for name, parameters in actor.named_parameters():  # 打印出每一层的参数的大小
        print(name, ':', parameters.size())
    for name, parameters in critic1.named_parameters():  # 打印出每一层的参数的大小
        print(name, ':', parameters.size())
    print(actor.parameters(),critic1.parameters())
    # sys.exit()

    if args.auto_alpha:  #进入这里
        target_entropy = -np.prod(args.action_shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    kwargs= {'action_space':args.action_shape,
        'env_state_shape':args.state_shape,
        'sourceenv_state_shape':source_env_observation_space_shape,
        'old_policy':old_policy,}
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
        **kwargs,
    )

    # collector
    buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    train_collector = AsyncCollector(policy,  train_envs,   buffer, exploration_noise=True )    #这一步调用reset

    # log
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
        buffer_path = os.path.join( 'SAC1.pkl')   #好像是空文件
        print(buffer_path)
        if os.path.exists(buffer_path):
            train_collector.buffer = pickle.load(open(buffer_path, "rb"))
            print("Successfully restore buffer.")
        else:
            print("Fail to restore buffer.")
    # sys.exit()

    print(policy.training,policy.updating)
    info_list = []
    start_student=5*80    #10 epoch*env_num
    Magnification_factor=20   #20
    # for i in range(int(1e6)):  # total step
    #     #每个循环每个envs都跑一个action  # 不包括训练前半段的耦合损耗
    #
    #     student_ps_coef_now = 0.0
    #     vf_student_ps_coef_now = 0.0
    #     mutual_info_coef = 0
    #
    #     kwargs = {'student_ps_coef_now': student_ps_coef_now,
    #               'vf_student_ps_coef_now': vf_student_ps_coef_now,
    #               'mutual_info_coef': mutual_info_coef,}
    #     collect_result=train_collector.collect(n_step=args.training_num)
    #     result=policy.update(512, train_collector.buffer,batch_size=512, repeat=10, **kwargs,)
    #
    #     if i % 4 == 0:
    #         info_list = pd.DataFrame(result)
    #         info_list.to_csv('800_20.csv')
    #     if i % 80 == 0:
    #         torch.save({'model': policy.state_dict(),}, os.path.join(log_path, f'model_{int(i/80+295)}.pth'))
    #         pickle.dump(buffer, open(os.path.join(os.getcwd(), "SAC.pkl"), "wb"))
    policy.eval()
    for i in range(int(1e6)):  # total step
        # collector = Collector(policy, test_envs)
        env_step = 80
        result = train_collector.collect(n_step=env_step,)
        # if i % 4 == 0:
        #     info_list = pd.DataFrame(result)
        #     info_list.to_csv('800_20.csv')
        if i % 1 == 0:
            torch.save({'model': policy.state_dict(),}, os.path.join(log_path, f'model_{int(i/80)}.pth'))
            pickle.dump(buffer, open(os.path.join(os.getcwd(), "SAC.pkl"), "wb"))


if __name__ == '__main__':
    test_sac_with_il()
