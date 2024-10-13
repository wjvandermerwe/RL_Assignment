import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from torch.distributions import Categorical


# Custom PPO Policy Network
class PPOPolicyNetwork(BasePolicy):
    """
    Feed-forward network with separate heads for policy and value function.
    """

    def __init__(self, observation_space, action_space, lr_schedule):
        super(PPOPolicyNetwork, self).__init__(observation_space, action_space)

        obs_shape = observation_space.shape[0]
        n_actions = action_space.n

        # Network layers
        self.fc1 = nn.Linear(obs_shape, 64)
        self.fc2 = nn.Linear(64, 64)

        # Policy and value heads
        self.policy_head = nn.Linear(64, n_actions)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, deterministic=False):
        """
        Forward pass through the network.
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))

        # Policy head (action probabilities)
        action_logits = self.policy_head(x)
        dist = Categorical(logits=action_logits)

        # Sample action based on policy distribution
        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
        else:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), self.value_head(x)

    def evaluate_actions(self, obs, actions):
        """
        Evaluate given actions for PPO loss calculation.
        """
        _, log_probs, entropy, values = self.forward(obs)
        dist = Categorical(logits=log_probs)
        return dist.log_prob(actions), entropy, values


# Basic PPO Algorithm
class BasicPPO(OnPolicyAlgorithm):
    """
    A simple PPO implementation extending from OnPolicyAlgorithm (based on Stable Baselines3).
    """

    def __init__(self, policy, env, learning_rate=3e-4, gamma=0.99, clip_ratio=0.2, **kwargs):
        # Super constructor calls the parent class's initialization
        super(BasicPPO, self).__init__(policy=policy, env=env, learning_rate=learning_rate, gamma=gamma, **kwargs)

        # PPO-specific attributes
        self.clip_ratio = clip_ratio
        self.ent_coef = 0.01  # Entropy coefficient
        self.vf_coef = 0.5  # Value function coefficient

        # Optimizer for the policy network
        self.policy = policy
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

    def ppo_loss(self, data):
        """
        Compute PPO loss based on the collected rollouts.
        """
        obs, actions, old_log_probs, returns, advantages = data

        # Get new log probs, entropy, and values from the current policy
        log_probs, entropy, values = self.policy.evaluate_actions(obs, actions)

        # Calculate PPO loss with clipping
        ratio = torch.exp(log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        policy_loss = torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # Value function loss
        value_loss = F.mse_loss(returns, values)

        # Combined loss (including entropy to encourage exploration)
        loss = -policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy.mean()
        return loss

    def train(self, rollout_buffer):
        """
        Execute one training step of PPO.
        """
        data = rollout_buffer.get()
        loss = self.ppo_loss(data)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def collect_rollouts(self, env, n_steps):
        """
        Collect rollouts from the environment and store them for training.
        """
        rollout_buffer = self.rollout_buffer
        rollout_buffer.reset()

        obs = env.reset()

        for step in range(n_steps):
            # Get action and value from policy
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                action, log_prob, entropy, value = self.policy(obs_tensor)

            # Execute action in the environment
            next_obs, reward, done, _ = env.step(action.item())

            # Store transition in rollout buffer
            rollout_buffer.add(obs, action, reward, done, value, log_prob)

            obs = next_obs
            if done:
                obs = env.reset()

        return rollout_buffer