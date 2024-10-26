from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.torch_layers import create_mlp
from torch import nn
from models.dqn_model import BaseDQN
import torch.nn.functional as F
from models.iteration_1.dqn_model import DoubleDQNPolicy, DoubleDQN

class DuelingQNetwork(DoubleDQNPolicy):
    def __init__(self, observation_space, action_space,
                 lr_schedule,
                 activation_fn=nn.ReLU):
        super().__init__(observation_space=observation_space, action_space=action_space, lr_schedule=lr_schedule)

        self.action_dim = action_space.n
        self.activation_fn = activation_fn
        input_dim = observation_space.shape[0]
        net_arch=[64,64]
        shared_layers = create_mlp(input_dim, net_arch[-1], net_arch, activation_fn=activation_fn)
        self.shared_layers = nn.Sequential(*shared_layers)

        self.value_layer = nn.Linear(input_dim, 1)
        self.advantage_layer = nn.Linear(input_dim, self.action_dim)

    def forward(self, obs):
        x = self.shared_layers(obs)
        value = self.value_layer(x)
        advantage = self.advantage_layer(x)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

class DuelingDQN(DoubleDQN):
    def __init__(self, **kwargs):
        super().__init__(
            **kwargs
        )