import torch
from torch import nn
from stable_baselines3.common.torch_layers import create_mlp
import torch.nn.functional as F
from models.iteration_1.ppo_model import TrulyProximalPPO

class ICMModule(nn.Module):
    """
    Intrinsic Curiosity Module (ICM) for encouraging exploration in PPO.
    """
    def __init__(self, input_dim, action_dim):
        super(ICMModule, self).__init__()
        self.feature_net = create_mlp(input_dim, 256, [128, 128], nn.ReLU)
        self.inverse_net = create_mlp(256 * 2, action_dim, [128], nn.ReLU)
        self.forward_net = create_mlp(256 + action_dim, 256, [128], nn.ReLU)

    def forward(self, state, next_state, action):
        # Extract features
        phi_state = self.feature_net(state)
        phi_next_state = self.feature_net(next_state)
        # Predict action based on features (inverse model)
        pred_action = self.inverse_net(torch.cat([phi_state, phi_next_state], dim=-1))
        # Predict next features based on action and current features (forward model)
        pred_phi_next_state = self.forward_net(torch.cat([phi_state, action], dim=-1))
        return pred_action, pred_phi_next_state, phi_next_state


class PPOWithICM(TrulyProximalPPO):
    """
    PPO implementation with Intrinsic Curiosity Module (ICM).
    """
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(
            **kwargs
        )

        self.icm = ICMModule(input_dim=self.observation_space.shape[0], action_dim=self.action_space.n)

    def train(self):
        """
        Update policy using the current rollout buffer.
        Incorporate ICM for intrinsic motivation.
        """
        super().train()  # Call the base class train method
        # Integrate ICM-based intrinsic rewards into training here
        for rollout_data in self.rollout_buffer.get(self.batch_size):
            pred_action, pred_phi_next_state, phi_next_state = self.icm(
                rollout_data.observations, rollout_data.next_observations, rollout_data.actions
            )
            # Compute ICM loss and backpropagate
            inverse_loss = F.mse_loss(pred_action, rollout_data.actions)
            forward_loss = F.mse_loss(pred_phi_next_state, phi_next_state)
            icm_loss = inverse_loss + forward_loss
            self.policy.optimizer.zero_grad()
            icm_loss.backward()
            self.policy.optimizer.step()
