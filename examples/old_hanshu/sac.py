from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.distributions import Independent, Normal
import pandas as pd
from tianshou.data import Batch, ReplayBuffer
from tianshou.exploration import BaseNoise
from hanshu import DDPGPolicy

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class SACPolicy(DDPGPolicy):
    """Implementation of Soft Actor-Critic. arXiv:1812.05905.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module critic1: the first critic network. (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic1_optim: the optimizer for the first
        critic network.
    :param torch.nn.Module critic2: the second critic network. (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic2_optim: the optimizer for the second
        critic network.
    :param float tau: param for soft update of the target network. Default to 0.005.  #用于目标网络软更新的参数
    :param float gamma: discount factor, in [0, 1]. Default to 0.99.
    :param (float, torch.Tensor, torch.optim.Optimizer) or float alpha: entropy
        regularization coefficient. Default to 0.2.
        If a tuple (target_entropy, log_alpha, alpha_optim) is provided, then
        alpha is automatically tuned.
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.
    :param BaseNoise exploration_noise: add a noise to action for exploration.
        Default to None. This is useful when solving hard-exploration problem.
    :param bool deterministic_eval: whether to use deterministic action (mean
        of Gaussian policy) instead of stochastic action sampled by the policy.
        Default to True.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action) or empty string for no bounding.
        Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1: torch.nn.Module,
        critic1_optim: torch.optim.Optimizer,
        critic2: torch.nn.Module,
        critic2_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        reward_normalization: bool = False,
        estimation_step: int = 1,
        exploration_noise: Optional[BaseNoise] = None,
        deterministic_eval: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            None, None, None, None, tau, gamma, exploration_noise,
            reward_normalization, estimation_step, **kwargs
        )
        self.actor, self.actor_optim = actor, actor_optim
        self.critic1, self.critic1_old = critic1, deepcopy(critic1)
        self.critic1_old.eval()
        self.critic1_optim = critic1_optim
        self.critic2, self.critic2_old = critic2, deepcopy(critic2)   #old为复制网络
        self.critic2_old.eval()
        self.critic2_optim = critic2_optim
        self.gengxincishu=0
        self.max_grad_norm=5

        self._is_auto_alpha = False
        self._alpha: Union[float, torch.Tensor]
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self._alpha_optim = alpha
            assert alpha[1].shape == torch.Size([1]) and alpha[1].requires_grad
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = alpha   #使用

        self._deterministic_eval = deterministic_eval
        self.__eps = np.finfo(np.float32).eps.item()
        self.all_featurenumber = pd.DataFrame()
        self.all_featureindex = pd.DataFrame()
        # self.integrated_gradients = IntegratedGradients(self.actor)

    def train(self, mode: bool = True) -> "SACPolicy":
        self.training = mode
        self.actor.train(mode)    #Sets the module in training mode.
        self.critic1.train(mode)
        self.critic2.train(mode)
        return self

    def sync_weight(self) -> None:
        # tau 用于目标网络软更新的参数
        #将目标模块的参数软更新为源模块的参数
        # tau=0.005
        # def soft_update(self, tgt: nn.Module, src: nn.Module, tau: float) -> None:
        #     """Softly update the parameters of target module towards the parameters \
        #     of source module."""
        #     for tgt_param, src_param in zip(tgt.parameters(), src.parameters()):
        #         tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)
        self.soft_update(self.critic1_old, self.critic1, self.tau)   # critic1_old使用 1-tau 的数据进行更新
        self.soft_update(self.critic2_old, self.critic2, self.tau)

    def forward(  # type: ignore
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        """
在SAC算法中，策略网络的输出是一个高斯分布，由均值和标准差两个参数组成。
在这段代码中，首先通过调用self.actor(obs, state=state, info=batch.info)来获取均值和标准差两个参数，
然后根据这两个参数构建一个高斯分布。如果是在评估模式下，那么直接返回均值作为输出；
否则，从高斯分布中采样一个动作作为输出。同时，这段代码还计算了动作的log_prob和IG值。
其中，log_prob是动作在高斯分布下的概率密度，IG是Integrated Gradients的值，用于解释模型的预测结果。
        """
        obs = batch[input]
        mu,obs_all_actor = self.actor(obs, state=state, info=batch.info)
        print('obs_all_actor',obs_all_actor)
        # shape = [1] * len(mu.shape)
        # shape[1] = - 1
        # sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
        logits = (mu, self.actor.sigma)
        # print(type(self.actor.sigma),type(self.actor.hidden),(obs).shape,state)   #(mu, sigma)   (10, 3)    hidden=None  state=None
        assert isinstance(logits, tuple)
        dist = Independent(Normal(*logits), 1)
        # print(Normal(*logits), type(Normal(*logits)))
        # print(type(dist), dist)
        if self._deterministic_eval and not self.training:
            act = logits[0]     # logits[0]=mu
        else:
            #有超过限值的action
            act = dist.rsample()
            # print(len(act))
        log_prob = dist.log_prob(act).unsqueeze(-1)
        # apply correction for Tanh squashing when computing logprob from Gaussian
        # You can check out the original SAC paper (arXiv 1801.01290): Eq 21.
        # in appendix C to get some understanding of this equation.
        squashed_action = torch.tanh(act)
        log_prob = log_prob - torch.log((1 - squashed_action.pow(2)) +
                                        self.__eps).sum(-1, keepdim=True)

        return Batch(
            logits=logits,
            act=squashed_action,
            state=self.actor.hidden,
            dist=dist,
            log_prob=log_prob,
        )
    """logits是actor网络的输出，act是根据logits采样得到的动作，
    hidden是actor网络的状态，dist是根据logits构建的分布，
    log_prob是动作act在分布dist下的对数概率，IG值是使用IntegratedGradients算法计算得到的输入obs对输出act的梯度 """

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        """
SAC算法的主要思想是在策略优化的同时，学习一个Q函数，用于评估策略的好坏。
在SAC算法中，策略网络是一个高斯分布，它的输出是一个均值和一个标准差，用于生成动作。
在训练过程中，SAC算法使用两个critic网络来评估Q值，使用一个actor网络来生成动作，
使用一个alpha参数来控制策略的熵，从而实现探索和利用的平衡。

在训练过程中，SAC算法使用一个重放缓冲区来存储经验，使用一个目标网络来计算Q值，
使用软更新来更新目标网络的参数。在训练过程中，SAC算法使用一个梯度下降算法来更新actor和critic网络的参数，
使用一个梯度裁剪算法来防止梯度爆炸，使用一个自适应alpha算法来调整策略的熵。
        """
        batch = buffer[indices]  # batch.obs: s_{t+n}
        obs_next_result = self(batch, input="obs_next")
        act_ = obs_next_result.act
        target_q_critic1_old,obs_all_critic1_old=self.critic1_old(batch.obs_next, act_)
        target_q_critic2_old, obs_all_critic2_old = self.critic2_old(batch.obs_next, act_)
        print('obs_all_critic1_old',obs_all_critic1_old,obs_all_critic2_old)
        target_q = torch.min(
            target_q_critic1_old,
            target_q_critic2_old,
        ) - self._alpha * obs_next_result.log_prob
        return target_q

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        """
这段代码是SACPolicy类中的learn方法，用于训练SAC算法中的actor、critic1和critic2网络。
其中，actor网络的训练需要用到critic1和critic2网络的Q值，因此需要先训练critic1和critic2网络。
在训练critic1和critic2网络时，需要计算TD误差和critic_loss，并更新网络参数。
在训练actor网络时，需要计算actor_loss，并更新网络参数。此外，还需要进行目标网络的软更新。

在训练critic1和critic2网络时，使用了_mse_optimizer函数进行训练，该函数用于更新评论家网络的简单包装脚本。
在训练actor网络时，需要计算actor_loss，并通过反向传播计算梯度并更新策略网络的参数。
如果alpha是一个元组，则需要自动调整alpha的值。在训练过程中，还需要进行目标网络的软更新。
        """
        self.gengxincishu+=1
        # critic 1&2
        td1, critic1_loss = self._mse_optimizer(
            batch, self.critic1, self.critic1_optim
        )   #size=256,  float数  #用于更新评论家网络的简单包装脚本。
        td2, critic2_loss = self._mse_optimizer(
            batch, self.critic2, self.critic2_optim
        )      #size=256,  float数, critic_optim与actor_optim相同的方式， zero_grad  backward  step
        batch.weight = (td1 + td2) / 2.0  # prio-buffer

        # actor
        # 首先通过调用self(batch)来获取当前状态下的动作act，然后计算当前状态下的Q值current_q1a和current_q2a。
        # 接着，根据SAC算法的公式计算actor_loss，即策略网络的损失。最后，通过反向传播计算梯度并更新策略网络的参数。

        # 在SAC算法中，actor的梯度需要通过critic的Q值来计算，具体来说，需要计算log_prob和Q值的加权和，
        # 然后对actor进行反向传播，得到actor的梯度。这个函数的输入是一个batch，输出是一个字典，其中包含了actor的梯度。
        obs_result = self(batch)   #batch的key值都有，
        act = obs_result.act       #256
        current_q1a,obs_all_critic1 = self.critic1(batch.obs, act).flatten()  #critic1(batch.obs, act)用了两次
        current_q2a,obs_all_critic2 = self.critic2(batch.obs, act).flatten()    #[256]
        print('obs_all_critic1',obs_all_critic1,obs_all_critic1)
        # print(torch.min(current_q1a, current_q2a))
        actor_loss = (
            self._alpha * obs_result.log_prob.flatten() -
            torch.min(current_q1a, current_q2a)
        ).mean()       #  actor_loss=tensor(0.1171, grad_fn=<MeanBackward0>)    self._alpha=tensor([0.9943])
        # print(actor_loss,actor_loss.shape)

        self.actor_optim.zero_grad()   #首先进行梯度清零   None  zero_grad()方法将所有模型参数的梯度重置为零
        actor_loss.backward()     #计算损失值    None     backward()方法计算相对于模型参数的actor损失的梯度。
        self.actor_optim.step()  # 进行梯度下降   None

        # print(f"Model structure: {self.actor}\n\n")
        # for name, param in self.actor.named_parameters():
        #     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
        #
        # print(f"Model structure: {self.critic1}\n\n")
        # for name, param in self.critic1.named_parameters():
        #     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")


        # convert numpy array to PyTorch tensor
        # a1_index, a2_index, a3_index, a4_index = [], [], [], [],
        # if int(self.gengxincishu%200)==0 :
        #     print(self.gengxincishu)
        #     self.actor.eval()  # Sets the module in evaluation mode.
        #     a = torch.tensor(batch.obs, requires_grad=True, dtype=torch.float32)  # 随机的会更好
        #     baselines = torch.zeros(a.shape)
        #     ig = IntegratedGradients(self.actor, )
        #     ig_attr_test, delta_IG = ig.attribute(a, baselines=baselines, n_steps=100, target=0,
        #                                           return_convergence_delta=True)
        #     ig_attr_test_sum = ig_attr_test.detach().numpy().sum(0)
        #     ig_attr_test_norm_sum = ig_attr_test_sum / np.linalg.norm(ig_attr_test_sum, ord=1)
        #     # print(a.shape,ig_attr_test_norm_sum.shape)     #torch.Size([256, 3]) (3,)
        #     ig_attr_test_norm_sum_lor = np.abs(ig_attr_test_norm_sum)  # sum = mean
        #     a1_all = list(np.abs(ig_attr_test_norm_sum_lor))
        #     for i in range(len(a1_all)):
        #         a1 = np.max(a1_all)
        #         a1_index.append(int(list(np.abs(ig_attr_test_norm_sum_lor)).index(a1)) + 1)
        #         a1_all.remove(a1)
        #
        #     ig_nt = NoiseTunnel(ig)
        #     ig_nt_attr_test = ig_nt.attribute(a, baselines=baselines, n_steps=100, target=0, )
        #     ig_nt_attr_test_sum = ig_nt_attr_test.detach().numpy().sum(0)
        #     ig_nt_attr_test_norm_sum = ig_nt_attr_test_sum / np.linalg.norm(ig_nt_attr_test_sum, ord=1)
        #     ig_nt_attr_test_norm_sum_lor =np.abs(ig_nt_attr_test_norm_sum)
        #     a2_all = list(np.abs(ig_nt_attr_test_norm_sum_lor))
        #     for i in range(len(a2_all)):
        #         a2 = np.max(a2_all)
        #         a2_index.append(int(list(np.abs(ig_nt_attr_test_norm_sum_lor)).index(a2)) + 1)
        #         a2_all.remove(a2)
        #
        #     dl = DeepLift(self.actor)
        #     dl_attr_test = dl.attribute(a, baselines=baselines, target=0, )
        #     dl_attr_test_sum = dl_attr_test.detach().numpy().sum(0)
        #     dl_attr_test_norm_sum = dl_attr_test_sum / np.linalg.norm(dl_attr_test_sum, ord=1)
        #     dl_attr_test_norm_sum_lor =np.abs(dl_attr_test_norm_sum)
        #     a3_all = list(np.abs(dl_attr_test_norm_sum_lor))
        #     for i in range(len(a3_all)):
        #         a3 = np.max(a3_all)
        #         a3_index.append(int(list(np.abs(dl_attr_test_norm_sum_lor)).index(a3)) + 1)
        #         a3_all.remove(a3)
        #
        #     fa = FeatureAblation(self.actor)
        #     fa_attr_test = fa.attribute(a, baselines=baselines, n_steps=100, target=0, )
        #     fa_attr_test_sum = fa_attr_test.detach().numpy().sum(0)
        #     fa_attr_test_norm_sum = fa_attr_test_sum / np.linalg.norm(fa_attr_test_sum, ord=1)
        #     fa_attr_test_norm_sum_lor =np.abs(fa_attr_test_norm_sum)
        #     a4_all = list(np.abs(fa_attr_test_norm_sum_lor))
        #     for i in range(len(a4_all)):
        #         a4 = np.max(a4_all)
        #         a4_index.append(int(list(np.abs(fa_attr_test_norm_sum_lor)).index(a4)) + 1)
        #         a4_all.remove(a4)
        #
        #     number=int(self.gengxincishu/500)
        #     self.all_featurenumber[number * 4-3] = pd.DataFrame(ig_attr_test_norm_sum_lor)
        #     self.all_featurenumber[number * 4-2] = pd.DataFrame(ig_nt_attr_test_norm_sum_lor)
        #     self.all_featurenumber[number * 4 - 1] = pd.DataFrame(dl_attr_test_norm_sum)
        #     self.all_featurenumber[number * 4 ] = pd.DataFrame(fa_attr_test_norm_sum)
        #     self.all_featurenumber.to_csv('all_featurenumber.csv', index=False, header=False)
        #     self.all_featureindex[number * 4-3] = pd.DataFrame(a1_index)
        #     self.all_featureindex[number * 4-2] = pd.DataFrame(a2_index)
        #     self.all_featureindex[number * 4 - 1] = pd.DataFrame(a3_index)
        #     self.all_featureindex[number * 4 ] = pd.DataFrame(a4_index)
        #     self.all_featureindex.to_csv('all_featureindex.csv', index=False, header=False)
        #
        #     a.requires_grad_(False)

        #通过将p.grad设置为每个参数p的计算梯度，我们将局部梯度应用于模型参数。我们使用torch.nn.utils.clip_grad_norm_()剪辑梯度norm以避免梯度爆炸，
        # 然后使用优化器的step()方法更新模型参数。
        # if self.gengxincishu%2==0:
        #     print(len(td1), len(batch),td2, critic2_loss)   256
        #     print(len(act),self._alpha,actor_loss,actor_loss.item(),a,b,c)
            # print(obs_result,batch.shape)
            # print(current_q2a,current_q2a.shape)      #256


        if self._is_auto_alpha:   #alpha自动优化   未使用
            log_prob = obs_result.log_prob.detach() + self._target_entropy
            # please take a look at issue #258 if you'd like to change this line
            alpha_loss = -(self._log_alpha * log_prob).mean()
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.sync_weight()   #用于目标网络软更新的参数

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
        }
        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()  # type: ignore

        return result

