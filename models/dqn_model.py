import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.spaces import flatten_space
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.torch_layers import create_mlp


class DQNNetwork(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_shape[0], 64)
        self.fc2 = nn.Linear(64, 64)
        self.q_values = nn.Linear(64, n_actions)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q_values(x)


class CustomDQN(OffPolicyAlgorithm):
    """
    A custom DQN algorithm built on top of the OffPolicyAlgorithm from Stable Baselines3.
    This class defines the DQN model, the optimizer, and the training process.
    """

    def __init__(self, policy, env, learning_rate=1e-3, buffer_size=10000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        max_grad_norm=10,
        policy_kwargs=None,
        verbose=0,
        device="cuda"):

        super(CustomDQN, self).__init__(policy, env, learning_rate, buffer_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            replay_buffer_class=ReplayBuffer,
            replay_buffer_kwargs=None,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=None,
            sde_support=False,
            optimize_memory_usage=False)

        self.device = torch.device(device)
        self.q_net = DQNNetwork(env.observation_space.shape, env.action_space.n).to(self.device)
        self.q_net_target = DQNNetwork(env.observation_space.shape, env.action_space.n).to(self.device)
        self.q_net_target.load_state_dict(self.q_net.state_dict())  # Sync target network with main network
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.target_update_interval = target_update_interval
        self.epsilon = exploration_initial_eps
        self.epsilon_decay = (exploration_initial_eps - exploration_final_eps) / exploration_fraction
        self.min_epsilon = exploration_final_eps
        self.max_grad_norm = max_grad_norm

        self.update_target_network()
        self._n_calls = 0
        
        
    def update_target_network(self):
        polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)

    def act(self, state: np.ndarray, deterministic: bool = False):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        if not deterministic and np.random.rand() < self.epsilon:
            action = np.random.randint(self.q_net.q_values.out_features)
        else:
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
                action = torch.argmax(q_values, dim=1).item()
        if not deterministic:
            self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay)
        return action

    def train_step(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.q_net_target(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return loss.item()

    def train(self, gradient_steps: int, batch_size: int = 32):
        for _ in range(gradient_steps):
            batch = self.replay_buffer.sample(batch_size)
            loss = self.train_step(batch)
        return loss

    def _on_step(self):
        """
        Update the target network and adjust the exploration rate after every environment step.
        """
        self._n_calls += 1

        # Update target network
        if self._n_calls % self.target_update_interval == 0:
            self.update_target_network()

        # Adjust exploration rate
        self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay)

