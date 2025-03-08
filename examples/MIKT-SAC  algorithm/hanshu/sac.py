from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.distributions import Independent, Normal
import pandas as pd
from tianshou.data import Batch, ReplayBuffer
from tianshou.exploration import BaseNoise
from hanshu.ddpg import DDPGPolicy
# from captum.attr import LayerConductance, LayerActivation, LayerIntegratedGradients
# from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from hanshu.student_policy import student_Net,mlp,teacher_mlp,student_mlp
import sys

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
        # env_state_shape=None,
        # sourceenv_state_shape=None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            None, None, None, None, tau, gamma, exploration_noise,
            reward_normalization, estimation_step,**kwargs
        )
        self.actor, self.actor_optim = actor, actor_optim
        self.critic1, self.critic1_old = critic1, deepcopy(critic1)
        self.critic1_old.eval()   #Sets the module in evaluation mode
        self.critic1_optim = critic1_optim
        self.critic2, self.critic2_old = critic2, deepcopy(critic2)   #old为复制网络
        self.critic2_old.eval()
        self.critic2_optim = critic2_optim
        self.max_grad_norm=5
        # print(kwargs)
        self.env_state_shape=kwargs['env_state_shape']
        self.sourceenv_state_shape=kwargs['sourceenv_state_shape']
        self.old_policy = kwargs['old_policy']
        self.a_l,self.a_b,self.a_a,self.c1_l,self.c2_l,\
        self.c1_b,self.c2_b,self.c1_a,self.c2_a,self.s_l,self.m_l,self.alpha_loss,self.alpha\
         ,self.s_ps,self.vfs1_ps, self.vfs2_ps ,self.c1_ml,self.c2_ml,self.c1_sl,self.c2_sl =[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
        # self.variational_net = mlp(self.sourceenv_state_shape, int(np.prod(self.env_state_shape)),)   #这里input_dim=source_env  输出为env  3到15

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

    def train(self, mode: bool = True) -> "SACPolicy":
        self.training = mode
        self.actor.train(mode)    #Sets the module in training mode.
        self.critic1.train(mode)
        self.critic2.train(mode)
        return self

    def sync_weight(self) -> None:
        # tau 用于目标网络软更新的参数
        #将目标模块的参数软更新为源模块的参数
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
        mu, _, _,_,_= self.actor(obs, state=state, info=batch.info)
        # self.student_ps=student_ps
        # self.mapped_state_train=embedding
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
    log_prob是动作act在分布dist下的对数概率"""

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
        critic1_old_loss,_ ,_ ,_ ,_=self.critic1_old(batch.obs_next, act_)
        critic2_old_loss,_ ,_ ,_ ,_= self.critic2_old(batch.obs_next, act_)
        target_q = torch.min(critic1_old_loss, critic2_old_loss,) - self._alpha * obs_next_result.log_prob
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
        # critic 1&2 在这里更新
        student_ps_coef_now = kwargs['student_ps_coef_now']
        vf_student_ps_coef_now =  kwargs['vf_student_ps_coef_now']
        mutual_info_coef = kwargs['mutual_info_coef']
        self.VF_STUDENT_PS_COEF = torch.tensor([vf_student_ps_coef_now], dtype=torch.float32)
        self.STUDENT_PS_COEF = torch.tensor([student_ps_coef_now], dtype=torch.float32)  #设置为1
        self.MUTUAL_INFO_COEF = torch.tensor([mutual_info_coef], dtype=torch.float32)   #设置为1
        td1, critic1_loss,critic1_loss_before,critic1_loss_after ,vf_student_ps_mean1,vf_student_ps_loss1,mutual_info_critic1_loss= self._mse_optimizer(
            batch, self.critic1, self.critic1_optim,self.VF_STUDENT_PS_COEF,self.MUTUAL_INFO_COEF )   #size=256,  float数  #用于更新评论家网络的简单包装脚本。
        td2, critic2_loss,critic2_loss_before,critic2_loss_after ,vf_student_ps_mean2,vf_student_ps_loss2,mutual_info_critic2_loss = self._mse_optimizer(
            batch, self.critic2, self.critic2_optim,self.VF_STUDENT_PS_COEF,self.MUTUAL_INFO_COEF )    #size=256,  float数, critic_optim与actor_optim相同的方式， zero_grad  backward  step
        batch.weight = (td1 + td2) / 2.0  # pr-buffer  #用于训练智能体在给定状态下采取行动的价值估计

        # actor
        # 首先通过调用self(batch)来获取当前状态下的动作act，然后计算当前状态下的Q值current_q1a和current_q2a。
        # 接着，根据SAC算法的公式计算actor_loss，即策略网络的损失。最后，通过反向传播计算梯度并更新策略网络的参数。
        # 在SAC算法中，actor的梯度需要通过critic的Q值来计算，具体来说，需要计算log_prob和Q值的加权和，
        # 然后对actor进行反向传播，得到actor的梯度。这个函数的输入是一个batch，输出是一个字典，其中包含了actor的梯度。
        obs_result = self(batch)   #batch的key值都有，
        act = obs_result.act       #256
        # current_q1a ,_ = self.critic1(batch.obs, act).flatten()  #critic1(batch.obs, act)用了两次
        # current_q2a ,_= self.critic2(batch.obs, act).flatten()    #[256]
        current_q1a ,_ ,_ ,_ ,_ = self.critic1(batch.obs, act)  #critic1(batch.obs, act)用了两次
        current_q2a ,_ ,_ ,_ ,_= self.critic2(batch.obs, act)    #[256]

        # 此处更改   不能在梯度计算的Tensor上直接调用numpy()方法
        _, student_ps, mapped_state_train,variational_mu,variational_logstd= self.actor(batch.obs,)    #获得student_ps

        # variational_mu = self.variational_net(mapped_state_train)
        # variational_logstd = nn.Parameter(torch.zeros(1,int(np.prod(self.env_state_shape))))
        # print(variational_mu.detach().numpy().shape,variational_logstd.detach().numpy().shape,batch.obs.shape)   #(512, 15) (1, 15) (512, 15)
        #MIloss 归一化处理
        # h = torch.from_numpy(batch.obs)[:, :, -1]    # 之前h=16 (1, 120，30)或者(512, 120，30)   这里obs不展平
        # max = torch.max(torch.abs(h))
        # h = h / max   #这里的值与embedding一致  传输最新状态(当前状态)给variation
        mutual_info_loss = torch.mean(variational_logstd + ((torch.from_numpy(batch.obs)[:, :, -1] - variational_mu) ** 2) /
            (2 * (torch.exp(variational_logstd)) ** 2))   #tensor(0.1742, dtype=torch.float64, grad_fn=<MeanBackward0>)

        # for i in range(len(self.student_ps)):  #这样操作无法求导
        #     self.student_ps[i]=self.student_ps[i].detach().numpy()   #Can't call numpy() on Tensor that requires grad
        # self.student_ps = torch.tensor( np.array(self.student_ps))   #[array(0.47252765, dtype=float32), array(0.47252765, dtype=float32)]
        student_ps_loss = -torch.log(student_ps).mean()  #自然对数
        student_ps_mean= (student_ps).mean()             #需要考虑VF_STUDENT_PS_COEF的取值

        # a= (self._alpha * obs_result.log_prob.flatten()-torch.min(current_q1a.flatten() , current_q2a.flatten() )  ).mean()
        # print(a ,student_ps_mean,student_ps_loss,)  #tensor(-1.9867, grad_fn=<MeanBackward0>) tensor(0.4725) tensor(0.7497)

        # loss=(self._alpha * obs_result.log_prob.flatten() -torch.min(current_q1a.flatten(), current_q2a.flatten())).mean()
        if student_ps_loss.item()==0:
            self.MUTUAL_INFO_COEF=0
        #这里自动求导与系数无关
        actor_loss = (self._alpha * obs_result.log_prob.flatten() -torch.min(current_q1a.flatten(), current_q2a.flatten())).mean()\
                     +self.MUTUAL_INFO_COEF*mutual_info_loss +self.STUDENT_PS_COEF*student_ps_loss
        # print('actor_loss',actor_loss,)      #tensor([-1.8125], grad_fn=<AddBackward0>)
        # print((self._alpha * obs_result.log_prob.flatten()).requires_grad,  mutual_info_loss.requires_grad,student_ps_loss.requires_grad ) #True True True
        actor_before=(self._alpha * obs_result.log_prob.flatten() -torch.min(current_q1a.flatten(), current_q2a.flatten())).mean()
        actor_after = self.MUTUAL_INFO_COEF*mutual_info_loss +self.STUDENT_PS_COEF*student_ps_loss

        # actor_loss = (
        #     self._alpha * obs_result.log_prob.flatten() -
        #     torch.min(current_q1a.flatten() , current_q2a.flatten() )
        # ).mean()       #  actor_loss=tensor(0.1171, grad_fn=<MeanBackward0>)
        self.actor_optim.zero_grad()   #首先进行梯度清零   None  zero_grad()方法将所有模型参数的梯度重置为零
        actor_loss.backward()     #计算损失值    None     backward()方法计算相对于模型参数的actor损失的梯度。
        self.actor_optim.step()   #进行梯度下降   None

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

        self.a_l.append(actor_loss.item())
        self.a_b.append(actor_before.item())
        self.a_a.append(actor_after.item())
        self.c1_l.append(critic1_loss.item())
        self.c2_l.append(critic2_loss.item())
        self.c1_b.append(critic1_loss_before.item())
        self.c2_b.append(critic2_loss_before.item())
        self.c1_a.append(critic1_loss_after.item())
        self.c2_a.append(critic2_loss_after.item())
        self.s_l.append(student_ps_loss.item())
        self.c1_sl.append(vf_student_ps_loss1.item())
        self.c2_sl.append(vf_student_ps_loss2.item())
        self.m_l.append(mutual_info_loss.item())
        self.c1_ml.append(mutual_info_critic1_loss.item())
        self.c2_ml.append(mutual_info_critic2_loss.item())

        self.alpha_loss.append(alpha_loss.item())
        self.alpha.append(self._alpha.item())
        self.s_ps.append(student_ps_mean.item())
        self.vfs1_ps.append(vf_student_ps_mean1.item())
        self.vfs2_ps.append(vf_student_ps_mean2.item())
        result = {
            "actor": self.a_l,
            "actor_before": self.a_b,
            "actor_after": self.a_a,
            "critic1": self.c1_l,
            "critic2": self.c2_l,
            "critic1_before": self.c1_b,
            "critic2_before": self.c2_b,
            "critic1_after": self.c1_a,
            "critic2_after": self.c2_a,
            "student_ps_loss": self.s_l,
            "critic1vf_student_ps_loss": self.c1_sl,
            "critic2vf_student_ps_loss": self.c2_sl,
            "actor_mutual_info_loss": self.m_l,
            "critic1_mutual_info_loss": self.c1_ml,
            "critic2_mutual_info_loss": self.c2_ml,
            'actor_student_ps':self.s_ps,
            'critic1_vf_student_ps':self.vfs1_ps,
            'critic2_vf_student_ps':self.vfs2_ps,
        }
        if self._is_auto_alpha:
            result["loss/alpha"] = self.alpha_loss
            result["alpha"] = self.alpha  # type: ignore

        return result

