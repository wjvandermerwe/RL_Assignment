
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.torch_layers import create_mlp

from torch import nn
from models.dqn_model import BaseDQN
import torch.nn.functional as F

from models.iteration_1.dqn_model import DoubleDQNPolicy, DoubleDQN


class DuelingQNetwork(DoubleDQNPolicy):
    def __init__(self, observation_space, action_space,
                 lr_schedule,
                 net_arch=None, activation_fn=nn.ReLU):
        """
        Q-Value Network for Dueling DQN.

        :param obs_shape: Shape of the observation space.
        :param action_space: Action space.
        :param net_arch: Network architecture as a list of layer sizes.
        :param activation_fn: Activation function to use between layers.
        """
        super().__init__(observation_space=observation_space, action_space=action_space, lr_schedule=lr_schedule)

        # self.obs_shape = obs_shape
        self.action_dim = action_space.n
        self.activation_fn = activation_fn
        input_dim = observation_space.shape[0]
        net_arch=[64,64]
        # Shared layers for both value and advantage streams
        # input_dim = obs_shape[0]
        shared_layers = create_mlp(input_dim, net_arch[-1], net_arch, activation_fn=activation_fn)
        self.shared_layers = nn.Sequential(*shared_layers)

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

class DuelingDQN(DoubleDQN):
    """
    A custom Dueling DQN agent that extends the CustomDQN.
    Uses DuelingDQNPolicy to implement the Dueling architecture.
    """
    def __init__(self, **kwargs):
        super().__init__(
            **kwargs
        )