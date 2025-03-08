import warnings
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from tianshou.data import Batch, ReplayBuffer
from tianshou.exploration import BaseNoise, GaussianNoise
# from tianshou.policy import BasePolicy
from hanshu.base import BasePolicy
import sys

class DDPGPolicy(BasePolicy):
    """Implementation of Deep Deterministic Policy Gradient. arXiv:1509.02971.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module critic: the critic network. (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic_optim: the optimizer for critic network.
    :param float tau: param for soft update of the target network. Default to 0.005.
    :param float gamma: discount factor, in [0, 1]. Default to 0.99.
    :param BaseNoise exploration_noise: the exploration noise,
        add to the action. Default to ``GaussianNoise(sigma=0.1)``.
    :param bool reward_normalization: normalize the reward to Normal(0, 1),
        Default to False.
    :param int estimation_step: the number of steps to look ahead. Default to 1.
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
        actor: Optional[torch.nn.Module],
        actor_optim: Optional[torch.optim.Optimizer],
        critic: Optional[torch.nn.Module],
        critic_optim: Optional[torch.optim.Optimizer],
        tau: float = 0.005,
        gamma: float = 0.99,
        exploration_noise: Optional[BaseNoise] = GaussianNoise(sigma=0.1),
        reward_normalization: bool = False,
        estimation_step: int = 1,
        action_scaling: bool = True,
        action_bound_method: str = "clip",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            **kwargs
        )
        assert action_bound_method != "tanh", "tanh mapping is not supported" \
            "in policies where action is used as input of critic , because" \
            "raw action in range (-inf, inf) will cause instability in training"
        if actor is not None and actor_optim is not None:
            self.actor: torch.nn.Module = actor
            self.actor_old = deepcopy(actor)
            self.actor_old.eval()
            self.actor_optim: torch.optim.Optimizer = actor_optim
        if critic is not None and critic_optim is not None:
            self.critic: torch.nn.Module = critic
            self.critic_old = deepcopy(critic)
            self.critic_old.eval()
            self.critic_optim: torch.optim.Optimizer = critic_optim
        assert 0.0 <= tau <= 1.0, "tau should be in [0, 1]"
        self.tau = tau
        assert 0.0 <= gamma <= 1.0, "gamma should be in [0, 1]"
        self._gamma = gamma
        self._noise = exploration_noise
        # it is only a little difference to use GaussianNoise
        # self.noise = OUNoise()
        self._rew_norm = reward_normalization
        self._n_step = estimation_step

    def set_exp_noise(self, noise: Optional[BaseNoise]) -> None:
        """Set the exploration noise."""
        self._noise = noise

    def train(self, mode: bool = True) -> "DDPGPolicy":
        """Set the module in training mode, except for the target network."""
        self.training = mode
        self.actor.train(mode)
        self.critic.train(mode)
        return self

    def sync_weight(self) -> None:
        """Soft-update the weight for the target network."""
        self.soft_update(self.actor_old, self.actor, self.tau)
        self.soft_update(self.critic_old, self.critic, self.tau)

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]  # batch.obs_next: s_{t+n}
        target_q = self.critic_old(
            batch.obs_next,
            self(batch, model='actor_old', input='obs_next').act
        )
        return target_q

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        batch = self.compute_nstep_return(
            batch, buffer, indices, self._target_q, self._gamma, self._n_step,
            self._rew_norm
        )
        return batch

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "actor",
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which has 2 keys:

            * ``act`` the action.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        model = getattr(self, model)
        obs = batch[input]
        actions, hidden = model(obs, state=state, info=batch.info)
        return Batch(act=actions, state=hidden)

    @staticmethod
    def _mse_optimizer(
        batch: Batch, critic: torch.nn.Module, optimizer: torch.optim.Optimizer,
            VF_STUDENT_PS_COEF=None,MUTUAL_INFO_COEF=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        """A simple wrapper script for updating critic network."""
        #用于更新评论家网络的简单包装脚本。
        weight = getattr(batch, "weight", 1.0)
        # current_q = critic(batch.obs, batch.act).flatten()  #current_q需要被flatten()
        #这里mapped_state_train=embedding不包括包括actor
        current_q, vf_student_ps_q1a,mapped_state_train,variational_mu,variational_logstd = critic(batch.obs, batch.act)    #使用与174   175相同的处理方式
        target_q = batch.returns.flatten()
        td = current_q.flatten() - target_q
        # critic_loss = F.mse_loss(current_q1, target_q)
        #MI loss
        # print(torch.as_tensor(batch.obs, ).flatten(1).detach().numpy().shape,variational_mu.detach().numpy().shape,variational_logstd.detach().numpy().shape)   #(512, 15) (512, 15)  (1, 16)
        #这里计算MI loss时进行归一化  归一化似乎没用
        # h = torch.from_numpy(batch.obs)[:, :, -1]    # 之前h=16 (1, 120，30)或者(512, 120，30)   这里obs不展平
        # max = torch.max(torch.abs(h))
        # h = h / max   #这里的值与embedding一致  传输最新状态(当前状态)给variation
        mutual_info_loss = torch.mean(variational_logstd + ((torch.from_numpy(batch.obs)[:, :, -1] - variational_mu) ** 2) /
            (2 * (torch.exp(variational_logstd)) ** 2))   #tensor(0.1742, dtype=torch.float64, grad_fn=<MeanBackward0>)  #(512, 15) (512, 15)  (1, 15)

        vf_student_ps_loss = -torch.log(vf_student_ps_q1a).mean()
        vf_student_ps_mean= (vf_student_ps_q1a).mean()             #需要考虑VF_STUDENT_PS_COEF的取值
        if vf_student_ps_loss.item()==0:
            MUTUAL_INFO_COEF=0
        # print((a ).mean(),vf_student_ps_mean,vf_student_ps_loss,0.05*vf_student_ps_loss)  #tensor(0.8085, grad_fn=<MeanBackward0>) tensor(0.4725) tensor(0.7497) tensor(0.0375)
        # loss=(td.pow(2) * weight ).mean()
        critic_loss = (td.pow(2) * weight ).mean() + VF_STUDENT_PS_COEF*vf_student_ps_loss +MUTUAL_INFO_COEF*mutual_info_loss
        critic_loss_before=(td.pow(2) * weight ).mean()
        critic_loss_after= VF_STUDENT_PS_COEF*vf_student_ps_loss \
                           +MUTUAL_INFO_COEF*mutual_info_loss
        # print('critic_loss',critic_loss)

        optimizer.zero_grad()
        critic_loss.backward()
        optimizer.step()
        return td, critic_loss,critic_loss_before,critic_loss_after,vf_student_ps_mean,vf_student_ps_loss,mutual_info_loss

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        # critic
        td, critic_loss = self._mse_optimizer(batch, self.critic, self.critic_optim)
        batch.weight = td  # prio-buffer
        # actor
        actor_loss = -self.critic(batch.obs, self(batch).act).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.sync_weight()
        return {
            "loss/actor": actor_loss.item(),
            "loss/critic": critic_loss.item(),
        }

    def exploration_noise(self, act: Union[np.ndarray, Batch],
                          batch: Batch) -> Union[np.ndarray, Batch]:
        if self._noise is None:
            return act
        if isinstance(act, np.ndarray):
            return act + self._noise(act.shape)
        warnings.warn("Cannot add exploration noise to non-numpy_array action.")
        return act
