from typing import Optional, Union, Dict, Type, Any
import torch
from torch import nn
from stable_baselines3.common.torch_layers import create_mlp


class SelfImitationLearning(nn.Module):
    """
    Self-Imitation Learning (SIL) for encouraging agents to learn from good trajectories in PPO.
    """
    def __init__(self, input_dim, action_dim):
        super(SelfImitationLearning, self).__init__()
        self.value_net = create_mlp(input_dim, 1, [128, 128], nn.ReLU)
        self.policy_net = create_mlp(input_dim, action_dim, [128, 128], nn.ReLU)

    def forward(self, state):
        value = self.value_net(state)
        action_logits = self.policy_net(state)
        return value, action_logits


class PPOWithSIL(BasePPO):
    """
    PPO implementation with Self-Imitation Learning (SIL).
    """
    def __init__(
            self,
            policy: Union[str, Type[BasePolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 3e-4,
            n_steps: int = 2048,
            batch_size: int = 64,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: Union[float, Schedule] = 0.2,
            normalize_advantage: bool = True,
            vf_coef: float = 0.5,
            max_grad_norm: Optional[float] = None,
            rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
            rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
            tensorboard_log: Optional[str] = None,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[torch.device, str] = "auto",
            _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            n_steps,
            batch_size,
            n_epochs,
            gamma,
            gae_lambda,
            clip_range,
            normalize_advantage,
            vf_coef,
            max_grad_norm,
            rollout_buffer_class,
            rollout_buffer_kwargs,
            tensorboard_log,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model=False,
        )
        if _init_setup_model:
            self._setup_model()
        self.sil = SelfImitationLearning(input_dim=self.observation_space.shape[0], action_dim=self.action_space.n)

    def train(self):
        """
        Update policy using the current rollout buffer.
        Incorporate SIL for self-imitation learning.
        """
        super().train()  # Call the base class train method
        # Integrate SIL-based loss into training here
        for rollout_data in self.rollout_buffer.get(self.batch_size):
            value, action_logits = self.sil(rollout_data.observations)
            # Compute SIL loss based on positive advantage trajectories
            advantage = rollout_data.returns - value.detach()
            mask = advantage > 0
            sil_policy_loss = -(torch.log_softmax(action_logits, dim=-1) * rollout_data.actions).sum(dim=-1)
            sil_policy_loss = (sil_policy_loss * mask).mean()
            sil_value_loss = F.mse_loss(value[mask], rollout_data.returns[mask])
            sil_loss = sil_policy_loss + sil_value_loss
            self.policy.optimizer.zero_grad()
            sil_loss.backward()
            self.policy.optimizer.step()
