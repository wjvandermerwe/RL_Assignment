from typing import  Optional, Type,  List

import torch
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.policies import BasePolicy
from torch import nn
from models.dqn_model import BaseDQN, DQNPolicy
import torch.nn.functional as F

class DuelingQNetwork(BasePolicy):
    def __init__(self, obs_shape, observation_space, action_space, net_arch=None, activation_fn=nn.ReLU):
        """
        Q-Value Network for Dueling DQN.

        :param obs_shape: Shape of the observation space.
        :param action_space: Action space.
        :param net_arch: Network architecture as a list of layer sizes.
        :param activation_fn: Activation function to use between layers.
        """
        super().__init__(observation_space=observation_space, action_space=action_space)

        # If no custom architecture is provided, use default [64, 64]
        if net_arch is None:
            net_arch = [64, 64]

        self.obs_shape = obs_shape
        self.action_dim = action_space.n
        self.net_arch = net_arch
        self.activation_fn = activation_fn

        # Shared layers for both value and advantage streams
        layers = []
        input_dim = obs_shape[0]
        for layer_size in net_arch:
            layers.append(nn.Linear(input_dim, layer_size))
            layers.append(activation_fn())
            input_dim = layer_size
        self.shared_layers = nn.Sequential(*layers)

        # Value stream
        self.value_layer = nn.Linear(input_dim, 1)

        # Advantage stream
        self.advantage_layer = nn.Linear(input_dim, self.action_dim)

    def forward(self, obs):
        x = self.shared_layers(obs)
        value = self.value_layer(x)
        advantage = self.advantage_layer(x)
        # Combine value and advantage to get Q-values
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

class DuelingDQN(BaseDQN):
    """
    A custom Dueling DQN agent that extends the CustomDQN.
    Uses DuelingDQNPolicy to implement the Dueling architecture.
    """
    def __init__(self, **kwargs):
        super().__init__(
            policy=DuelingQNetwork,
            replay_buffer_class=ReplayBuffer,
            **kwargs
        )