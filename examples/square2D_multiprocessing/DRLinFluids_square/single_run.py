# coding = UTF-8
import re
import argparse
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import gym

from tianshou.data import AsyncCollector
from tianshou.policy import SACPolicy
from tianshou.env import SubprocVectorEnv
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic


# define parameters  Define some hyper-parameters:
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="OpenFoam-v0")
    parser.add_argument("--reward-threshold", type=float, default=15.8)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--il-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", type=int, default=1)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--epoch", type=int, default=3000)
    parser.add_argument("--step-per-epoch", type=int, default=100)
    parser.add_argument("--il-step-per-epoch", type=int, default=1)
    parser.add_argument("--step-per-collect", type=int, default=20)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[512, 512])
    parser.add_argument(
        "--imitation-hidden-sizes", type=int, nargs="*", default=[512, 512]
    )
    parser.add_argument("--training-num", type=int, default=5)
    parser.add_argument("--test-num", type=int, default=1)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument("--rew-norm", action="store_true", default=False)
    parser.add_argument("--n-step", type=int, default=5)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
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
        default=0.0,
        help="use intrinsic curiosity module with this lr scale",
    )
    parser.add_argument("--resume", default=True, action="store_true", help="restart")
    # parser.add_argument('-n', '--num-episodes', required=True, type=int, help='Number of episodes')
    # parser.add_argument('-m', '--max-episode-timesteps', required=True, type=int, help='Maximum number of timesteps per episode')
    shell_args = vars(parser.parse_args())
    args = parser.parse_known_args()[0]

    shell_args["num_episodes"] = 1000
    shell_args["max_episode_timesteps"] = 100
    return args


def test_sac_with_il(args=get_args()):
    # define parameters
    # Make environments:
    # you can also try with SubprocVectorEnv
    foam_params = {
        "delta_t": 0.0005,
        "solver": "pimpleFoam",
        "num_processor": 5,
        "of_env_init": ". /opt/openfoam8/etc/bashrc",
        "cfd_init_time": 0.005,  # 初始化流场，初始化state
        "num_dimension": 2,
        "verbose": False,
    }

    entry_dict_q0 = {
        "U": {
            "JET4": {
                "v0": "(0 {x} 0)",
            },
            "JET7": {
                "v0": "(0 {x} 0)",
            },
        }
    }

    entry_dict_q1 = {
        "U": {
            "JET4": {
                "v1": "(0 {y} 0)",
            },
            "JET7": {
                "v1": "(0 {y} 0)",
            },
        }
    }

    entry_dict_t0 = {"U": {"JET4": {"t0": "{t}"}, "JET7": {"t0": "{t}"}}}

    agent_params = {
        "entry_dict_q0": entry_dict_q0,
        "entry_dict_q1": entry_dict_q1,
        "entry_dict_t0": entry_dict_t0,
        "deltaA": 0.05,
        "minmax_value": (-4, 4),
        "interaction_period": 0.013,
        "purgeWrite_numbers": 0,
        "writeInterval": 0.013,
        "deltaT": 0.0005,
        "variables_q0": ("x",),
        "variables_q1": ("y",),
        "variables_t0": ("t",),
        "verbose": False,
        "zero_net_Qs": True,
    }
    state_params = {"type": "velocity"}
    root_path = os.getcwd()
    env_name_list = sorted(
        [envs for envs in os.listdir(root_path) if re.search(r"^env\d+$", envs)]
    )
    env_path_list = ["/".join([root_path, i]) for i in env_name_list]

    env = envobject.FlowAroundSquareCylinder2D(
        foam_root_path=env_path_list[0],
        foam_params=foam_params,
        agent_params=agent_params,
        state_params=state_params,
    )
    # # env.spec.reward_threshold = 20
    # # you can also use tianshou.env.SubprocVectorEnv
    # train_envs = SubprocVectorEnv(
    #     [lambda x=i: gym.make(args.task,foam_root_path=x,
    #                           foam_params=foam_params,
    #                           agent_params=agent_params,
    #                           state_params=state_params,
    #                           size=x, sleep=x) for i in env_path_list[0:args.training_num]],
    #     wait_num=4, timeout=0.2
    # )
    # # test_envs = gym.make(args.task)
    test_envs = SubprocVectorEnv(
        [
            lambda x=i: gym.make(
                args.task,
                foam_root_path=x,
                foam_params=foam_params,
                agent_params=agent_params,
                state_params=state_params,
                size=x,
                sleep=x,
            )
            for i in env_path_list[
                args.training_num : (args.training_num + args.test_num)
            ]
        ]
    )
    print(env.observation_space.shape, env.action_space.shape)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    if args.reward_threshold is None:
        default_reward_threshold = {"OpenFoam-v0": 15.84, "Pendulum-v1": -250}
        args.reward_threshold = default_reward_threshold.get(
            args.task, env.spec.reward_threshold
        )
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # train_envs.seed([1, 2])
    test_envs.seed([5])

    # model
    net = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = ActorProb(
        net,
        args.action_shape,
        max_action=args.max_action,
        device=args.device,
        unbounded=True,
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c1 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    net_c2 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
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
        action_space=env.action_space,
    )

    # collector
    train_collector = None
    test_collector = AsyncCollector(
        policy,
        test_envs,
    )

    # log
    log_path = os.path.join(args.logdir, args.task, "sac")

    # logger
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    if args.logger == "wandb":
        logger = WandbLogger(
            save_interval=1,
            name=log_name.replace(os.path.sep, "__"),
            run_id=args.resume_id,
            update_interval=10,
            config=args,
            project=args.wandb_project,
        )
    if args.logger == "tensorboard":
        logger = TensorboardLogger(
            writer,
            update_interval=1,
            save_interval=args.save_interval,
        )
    else:  # wandb
        logger.load(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "best_model.pth"))

    def stop_fn(mean_rewards):
        return mean_rewards >= args.reward_threshold

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        torch.save(
            {
                "model": policy.state_dict(),
                # 'optim': optim.state_dict(),
            },
            os.path.join(log_path, "checkpoint.pth"),
        )

    def train_fn(epoch, env_step):
        # eps annnealing, just a demo
        if env_step <= 10000:
            policy.set_eps(args.eps_train)
        elif env_step <= 50000:
            eps = args.eps_train - (env_step - 10000) / 40000 * (0.9 * args.eps_train)
            policy.set_eps(eps)
        else:
            policy.set_eps(0.1 * args.eps_train)

    # def test_fn(epoch, env_step):
    #     policy.set_eps(args.eps_test)

    if args.resume:
        # load from existing checkpoint
        print(f"Loading agent under {log_path}")
        ckpt_path = os.path.join(
            log_path, "best_model.pth"
        )  # choose your best policy to test
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=args.device)
            policy.load_state_dict(checkpoint["model"])
            # optim.load_state_dict(checkpoint['optim'])
            print("Successfully restore policy and optim.")
        else:
            print("Fail to restore policy and optim.")

    print(policy.training, policy.updating)
    # trainer
    start_epoch = 0
    env_step = 0
    gradient_step = 0
    if args.resume:
        start_epoch, env_step, gradient_step = logger.restore_data()
    best_reward = 0
    best_reward_std = 0.0
    episode_per_test = args.test_num
    reward_metric = None
    test_fn = None

    # Let's watch its performance!
    env = test_envs
    # evaluation
    policy.eval()
    collector = Collector(policy, env)
    env_step = (
        400  # NOTE: Pay attention to changing the number of steps per epoch to 400
    )
    result = collector.collect(n_step=env_step, n_episode=None, render=args.render)
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews.mean()}, length: {lens.mean()}")


if __name__ == "__main__":
    test_sac_with_il()
