import argparse
import os

import numpy as np
import gym
import pandas as pd
import torch
# from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import pickle
from torch import nn
from old_hanshu.collector import Collector, VectorReplayBuffer,AsyncCollector
from old_hanshu.venvs import DummyVectorEnv,SubprocVectorEnv,RayVectorEnv,ShmemVectorEnv
from collections import OrderedDict

from old_hanshu.sac import SACPolicy
from old_hanshu.common import Net
from old_hanshu.continuous import ActorProb, Critic

import sys

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='OpenFoam-v0')  # Pendulum-v1
    parser.add_argument('--reward-threshold', type=float, default=None)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--actor-lr', type=float, default=2e-4)
    parser.add_argument('--critic-lr', type=float, default=3e-4)
    parser.add_argument('--il-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.97)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--auto-alpha', type=int, default=1)
    parser.add_argument('--alpha-lr', type=float, default=3e-4)
    parser.add_argument('--epoch', type=int, default=2000)
    parser.add_argument('--step-per-epoch', type=int, default=500)
    parser.add_argument('--il-step-per-epoch', type=int, default=1)
    parser.add_argument('--step-per-collect', type=int, default=10)
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[512, 256])
    parser.add_argument(
        '--imitation-hidden-sizes', type=int, nargs='*', default=[512, 256]
    )
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=0)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--rew-norm', action="store_true", default=False)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument(
        '--device', type=str, default='cpu'
    )
    parser.add_argument(
        "--resume",
        default=True,
        action="store_true",
        help="restart"
    )
    args = parser.parse_known_args()[0]
    return args

def test_sac_old(args=get_args()):
    # if you want to use python vector env, please refer to other test scripts
    #之前的策略初始化
    state_shape=(24,30)
    # action_shape=(4,)
    action_shape=(4,)   #0609使用8action
    net = Net(state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = ActorProb(
        net,
        action_shape,
        max_action=1,
        device=args.device,
        unbounded=True
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c2 = Net(
        state_shape,
        action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device
    )
    critic1 = Critic(net_c2, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)
    old_policy = SACPolicy(
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
        action_space=action_shape,
        # **kwargs,
    )
    # log
    log_path = os.path.join(args.logdir, args.task, 'sac')
    if args.resume:
        # load from existing checkpoint
        print(f"Loading agent under {log_path}")
        ckpt_path = os.path.join(log_path, 'best_model.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=args.device)
            old_policy.load_state_dict(checkpoint['model'])
            print("Successfully restore policy and optim.")
        else:
            print("Fail to restore policy and optim.")
        buffer_path = os.path.join('expert_SAC_OpenFoam-v0.pkl')
        if os.path.exists(buffer_path):
            train_collector.buffer = pickle.load(open(buffer_path, "rb"))
            print("Successfully restore buffer.")
        else:
            print("Fail to restore buffer.")


    return old_policy


test_sac_old()