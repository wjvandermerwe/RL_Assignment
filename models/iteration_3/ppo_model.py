import torch
from torch import nn
from stable_baselines3.common.torch_layers import create_mlp
import torch.nn.functional as F
from models.iteration_2.ppo_model import PPOWithICM


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


class PPOWithSIL(PPOWithICM):
    """
    PPO implementation with Self-Imitation Learning (SIL).
    """
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(
            **kwargs
        )
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
