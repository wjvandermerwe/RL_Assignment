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

    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
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

    def _predict(self, obs, deterministic=False):
        """
        Predict action given an observation.
        """
        action, _, _, _ = self.forward(obs, deterministic=deterministic)
        return action


# Basic PPO Algorithm
class CustomPPO(OnPolicyAlgorithm):
    """
    PPO implementation extending from OnPolicyAlgorithm.
    """

    def __init__(self,
                 policy,
                 env,
                 learning_rate=3e-4,
                 n_steps=2048,
                 batch_size=64,
                 n_epochs=10,
                 gamma=0.99,
                 clip_range=0.2,
                 **kwargs):
        super().__init__(policy=policy,
                                        env=env,
                                        learning_rate=learning_rate,
                                        n_steps=n_steps,
                                        gamma=gamma,
                                        # gae_lambda=0.95,
                                        # ent_coef=0.0,
                                        # vf_coef=0.5,
                                        # max_grad_norm=0.5,
                                        # use_sde=False,
                                        # sde_sample_freq=-1,
                                        **kwargs)

        self.clip_range = clip_range
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # Initialize the policy
        # self.policy = policy(observation_space=env.observation_space, action_space=env.action_space,
        #                      lr_schedule=lambda _: learning_rate)

        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

    def ppo_loss(self, data):
        obs, actions, old_log_probs, returns, advantages = data

        # Get new log probs, entropy, and values from the current policy
        log_probs, entropy, values = self.policy.evaluate_actions(obs, actions)

        # Calculate PPO loss with clipping
        ratio = torch.exp(log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
        policy_loss = torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # Value function loss
        value_loss = F.mse_loss(returns, values)

        # Combined loss
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

    # def act(self, obs, deterministic=False):
    #     obs_tensor = torch.tensor(obs, dtype=torch.float32)
    #     with torch.no_grad():
    #         action, _, _, _ = self.policy(obs_tensor)
    #     return action.item()

    # def collect_rollouts(self, env, n_steps):
    #     """
    #     Collect rollouts from the environment.
    #     """
    #     rollout_buffer = RolloutBuffer(n_steps, env.observation_space, env.action_space, device=self.device, gamma=self.gamma, gae_lambda=self.gae_lambda)
    #
    #     obs = env.reset()
    #     for step in range(n_steps):
    #         action = self.act(obs)
    #
    #         # Execute action in the environment
    #         next_obs, reward, done, _ = env.step(action)
    #
    #         # Store transition
    #         rollout_buffer.add(obs, action, reward, done)
    #
    #         obs = next_obs
    #         if done:
    #             obs = env.reset()
    #
    #     rollout_buffer.compute_returns_and_advantage(last_values=None, dones=done)
    #     return rollout_buffer