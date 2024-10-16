import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import BasePolicy
from typing import Optional, List, Dict, Any, Type
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.buffers import ReplayBuffer
import numpy as np


class QNetwork(BasePolicy):
    def __init__(self, obs_shape, observation_space, action_space, net_arch=None, activation_fn=nn.ReLU):
        """
        Q-Value Network for DQN.

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
        self.activation_fn = activation_fn
        self.net_arch = net_arch

        layers = []
        input_dim = obs_shape[0]

        for layer_size in net_arch:
            layers.append(nn.Linear(input_dim, layer_size))
            layers.append(activation_fn())
            input_dim = layer_size

        layers.append(nn.Linear(input_dim, self.action_dim))

        #unpack
        self.q_net = nn.Sequential(*layers)


    def forward(self, obs):
        return self.q_net(obs)





class DQNPolicy(BasePolicy):
    """
    DQN Policy with Q-Network and Target Network.

    :param observation_space: Observation space.
    :param action_space: Action space.
    :param net_arch: Architecture of the Q-Network.
    :param activation_fn: Activation function to use between layers.
    :param normalize_images: Whether to normalize images or not.
    """
    q_net: QNetwork
    q_net_target: QNetwork

    def __init__(
            self,
            observation_space,
            action_space,
            lr_schedule,
            net_arch: Optional[List[int]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            # normalize_images: bool = True,
            # optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            # optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(observation_space=observation_space, action_space=action_space)

        # Network arguments
        self.net_args = {
            "obs_shape": observation_space.shape,
            "observation_space": observation_space,
            "action_space": action_space,
            "net_arch": net_arch,
            "activation_fn": activation_fn,
        }

        # Build the Q-Networks
        self.q_net = self.make_q_net()
        self.q_net_target = self.make_q_net()
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.q_net_target.eval()

        # Optimizer setup
        self.optimizer = self.optimizer_class(self.q_net.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def make_q_net(self) -> QNetwork:
        """
        Creates the Q-Network using the specified architecture.
        """
        return QNetwork(**self.net_args).to(self.device)

    def forward(self, obs, deterministic: bool = True) -> torch.Tensor:
        return self._predict(obs, deterministic)

    def _predict(self, obs, deterministic: bool = True) -> torch.Tensor:
        q_values = self.q_net(obs)
        action = q_values.argmax(dim=1) if deterministic else torch.multinomial(F.softmax(q_values, dim=1), num_samples=1)
        return action

    def update_target_network(self):
        """
        Update the target network weights with the Q-network weights.
        """
        self.q_net_target.load_state_dict(self.q_net.state_dict())

class BaseDQN(OffPolicyAlgorithm):
    """
    A custom DQN agent built on top of OffPolicyAlgorithm.
    """

    def __init__(self, policy, env, learning_rate=1e-3, buffer_size=10000, tau=1.0, gamma=0.99,
                 train_freq=4, gradient_steps=1, target_update_interval=1000, replay_buffer_class=None,
                 exploration_fraction=0.1, exploration_initial_eps=1.0,
                 exploration_final_eps=0.05, max_grad_norm=10, policy_kwargs=None,
                 verbose=0, device="cuda"):

        super(BaseDQN, self).__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=None,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=None,
            sde_support=False,
            optimize_memory_usage=False
        )

        self.device = torch.device(device)
        self.epsilon = exploration_initial_eps
        self.epsilon_decay = (exploration_initial_eps - exploration_final_eps) / exploration_fraction
        self.min_epsilon = exploration_final_eps
        self.max_grad_norm = max_grad_norm
        self.target_update_interval = target_update_interval

    def act(self, state: np.ndarray, deterministic: bool = False):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        if not deterministic and np.random.rand() < self.epsilon:
            action = np.random.randint(self.policy.q_net.q_net[-1].out_features)
        else:
            with torch.no_grad():
                action = self.policy.forward(state_tensor).item()

        return action

    def train_step(self, batch):
        states, actions, rewards, next_states, dones = self._process_batch(batch)

        # Q-values for actions taken
        q_values = self.policy.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values using target network
        with torch.no_grad():
            next_q_values = self.policy.q_net_target(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Loss calculation
        loss = F.mse_loss(q_values, target_q_values)

        # Backpropagation step
        self._optimize_model(loss)

        return loss.item()

    def _process_batch(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        return states, actions, rewards, next_states, dones

    def _optimize_model(self, loss):
        self.policy.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.q_net.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

    def _adjust_exploration_rate(self):
        self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay)

    def _on_step(self):
        self._n_calls += 1
        if self._n_calls % self.target_update_interval == 0:
            self.policy.update_target_network()
        self._adjust_exploration_rate()

