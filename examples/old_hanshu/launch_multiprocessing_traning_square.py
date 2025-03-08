# coding = UTF-8
import re
import datetime
import envobject
import argparse
import os
import numpy as np
import torch
import pprint
from torch.utils.tensorboard import SummaryWriter
import gym
import pickle
from tianshou.data import VectorReplayBuffer,Collector,AsyncCollector
from tianshou.policy import SACPolicy,ImitationPolicy
from tianshou.env import DummyVectorEnv,SubprocVectorEnv,RayVectorEnv,ShmemVectorEnv
# from tianshou.trainer import offpolicy_trainer
# from tensorfoam.openfoam import SubprocVectorEnv,offpolicy_trainer,DummyVectorEnv
# from tensorfoam.openfoam import offpolicy_trainer
from tianshou.trainer import offpolicy_trainer
# from tensorfoam.openfoam import AsyncCollector,Collector
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
import warnings
warnings.filterwarnings("ignore")

# define parameters  Define some hyper-parameters:
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='OpenFoam-v0')
    parser.add_argument('--reward-threshold', type=float, default=15.8)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--buffer-size', type=int, default=100000)
    parser.add_argument('--actor-lr', type=float, default=2e-4)
    parser.add_argument('--critic-lr', type=float, default=3e-4)
    parser.add_argument('--il-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.97)
    parser.add_argument('--tau', type=float, default=0.005)
    #温度系数
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--auto-alpha', type=int, default=1)
    parser.add_argument('--alpha-lr', type=float, default=3e-4)
    parser.add_argument('--epoch', type=int, default=2000)
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
    #parser.add_argument('-n', '--num-episodes', required=True, type=int, help='Number of episodes')
    #parser.add_argument('-m', '--max-episode-timesteps', required=True, type=int, help='Maximum number of timesteps per episode')
    shell_args = vars(parser.parse_args())
    args = parser.parse_known_args()[0]

    shell_args['num_episodes']=1000
    shell_args['max_episode_timesteps']=100
    return args


#SAC算法
def test_sac_with_il(args=get_args()):
    # torch.set_num_threads(1)  # we just need only one thread for NN
    # env = gym.make(args.task)
    # define parameters
    # Make environments:
    # you can also try with SubprocVectorEnv
    foam_params = {
        'delta_t': 0.00011,
        'solver': 'pimpleFoam',
        'num_processor': 4,
        'of_env_init': 'source ~/OpenFOAM/OpenFOAM-8/etc/bashrc',
        'cfd_init_time': 0.001,  # 初始化流场，初始化state
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
        'interaction_period': 0.017,
        'vortex_shedding':0.21,
        'action_discount':0.1,
        'cd_0': 2.27,
        'purgeWrite_numbers': 0,
        'writeInterval': 0.017,
        'deltaT': 0.00011,
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

    env = envobject.FlowAroundSquareCylinder2D(
        foam_root_path=env_path_list[0],
        foam_params=foam_params,
        agent_params=agent_params,
        state_params=state_params,
    )
    # # env.spec.reward_threshold = 20
    # # you can also use tianshou.env.SubprocVectorEnv
    train_envs = SubprocVectorEnv(
        [lambda x=i: gym.make(args.task,foam_root_path=x,
                              foam_params=foam_params,
                              agent_params=agent_params,
                              state_params=state_params,
                              size=x, sleep=x) for i in env_path_list[0:args.training_num]],
        wait_num=args.training_num, timeout=0.2
    )
    # # test_envs = gym.make(args.task)
    # test_envs = SubprocVectorEnv(
    #     [lambda x=i: gym.make(args.task,foam_root_path=x,
    #                           foam_params=foam_params,
    #                           agent_params=agent_params,
    #                           state_params=state_params,
    #                           size=x, sleep=x) for i in env_path_list[args.training_num:(args.training_num+args.test_num)]]
    # )
    test_envs=None
    print(env.observation_space.shape,env.action_space.shape)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    if args.reward_threshold is None:
        default_reward_threshold = {"OpenFoam-v0": 30, "Pendulum-v1": -250}
        args.reward_threshold = default_reward_threshold.get(
            args.task, env.spec.reward_threshold
        )
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed([1, 2,3,4,5])
    # test_envs.seed([5])

    # model
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
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

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
        action_space=env.action_space
    )

    # collector
    buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    train_collector = AsyncCollector(
        policy,train_envs,buffer,exploration_noise=True
    )
    test_collector=None

    log_path = os.path.join(args.logdir, args.task, 'sac')
    # logger
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    if args.logger == 'wandb':
        logger = WandbLogger(
            save_interval=1,
            name=log_name.replace(os.path.sep, "__"),
            run_id=args.resume_id,
            update_interval=10,
            config=args,
            project=args.wandb_project,
        )
    if args.logger == "tensorboard":
        logger = TensorboardLogger(writer,update_interval=1,save_interval=args.save_interval,)
    else:  # wandb
        logger.load(writer)

    def save_best_fn(policy):
        torch.save(
                    {
                        'model': policy.state_dict(),
                        # 'optim': optim.state_dict(),
                    }, os.path.join(log_path, 'best_model.pth')
                   )

    def stop_fn(mean_rewards):
        return mean_rewards >= args.reward_threshold

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        torch.save(
            {
                'model': policy.state_dict(),
                # 'optim': optim.state_dict(),
            }, os.path.join(log_path, 'checkpoint.pth')
        )
        # pickle.dump(
        #     train_collector.buffer,
        #     open(os.path.join(log_path, 'train_buffer.pkl'), "wb")
        # )

    def train_fn(epoch, env_step):
        # eps annnealing, just a demo
        if env_step <= 10000:
            policy.set_eps(args.eps_train)
        elif env_step <= 50000:
            eps = args.eps_train - (env_step - 10000) / \
                40000 * (0.9 * args.eps_train)
            policy.set_eps(eps)
        else:
            policy.set_eps(0.1 * args.eps_train)

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    if args.resume:
        # load from existing checkpoint
        print(f"Loading agent under {log_path}")
        ckpt_path = os.path.join(log_path, 'best_model.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=args.device)
            policy.load_state_dict(checkpoint['model'])
            print("Successfully restore policy and optim.")
        else:
            print("Fail to restore policy and optim.")
        buffer_path = os.path.join( 'expert_SAC_OpenFoam-v0.pkl')
        if os.path.exists(buffer_path):
            train_collector.buffer = pickle.load(open(buffer_path, "rb"))
            print("Successfully restore buffer.")
        else:
            print("Fail to restore buffer.")

    print(policy.training,policy.updating)
    # trainer
    # offpolicy_trainer(
    #     policy,
    #     train_collector,
    #     test_collector,
    #     args.epoch,
    #     args.step_per_epoch,
    #     args.step_per_collect,
    #     args.test_num,
    #     args.batch_size,
    #     update_per_step=args.update_per_step,
    #     stop_fn=stop_fn,
    #     save_best_fn=save_best_fn,
    #     save_checkpoint_fn=save_checkpoint_fn,
    #     logger=logger,
    #     resume_from_log=args.resume,
    # )
    # pickle.dump(buffer, open(os.path.join(os.getcwd(), "expert_SAC_OpenFoam-v0.pkl"), "wb"))
    #断言可以在条件不满足程序运行的情况下直接返回错误，而不必等待程序运行后出现崩溃
    #存在test_collector才有best_reward
    # assert stop_fn(result['best_reward'])
    # print("anything done")
    # pre-collect at least 5000 transitions with random action before training
    for i in range(int(1e6)):  # total step
        #每个循环每个envs都跑一个action
        train_collector.collect(n_step=args.training_num)
        # train policy with a sampled batch data from buffer
        # train_collector.buffer.obs=mask_obs(train_collector.buffer.obs)    #需要更改声明 buffer.base.py assert
        policy.update(512, train_collector.buffer,batch_size=512, repeat=10)
        # print(collect_result)
        if i % 80 == 0:
            torch.save(
                {
                    'model': policy.state_dict(),
                }, os.path.join(log_path, 'best_model.pth')
            )
            pickle.dump(buffer, open(os.path.join(os.getcwd(), "expert_SAC_OpenFoam-v0.pkl"), "wb"))

    # if __name__ == '__main__':
    #     pprint.pprint(result)
        # Let's watch its performance!
        # env = train_envs
        # policy.eval()
        # collector = Collector(policy, env)
        # result = collector.collect(n_episode=1, render=args.render)
        # rews, lens = result["rews"], result["lens"]
        # print(f"Final reward: {rews.mean()}, length: {lens.mean()}")



if __name__ == '__main__':

    test_sac_with_il()
