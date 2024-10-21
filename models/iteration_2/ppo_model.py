from typing import NamedTuple, Optional

import numpy as np
import torch
from gymnasium.vector.utils import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from torch import nn
from stable_baselines3.common.torch_layers import create_mlp
import torch.nn.functional as F
from models.iteration_1.ppo_model import TrulyProximalPPO
from models.ppo_model import RolloutBuffer


class RolloutBufferSamplesWithNextObs(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    next_observations: torch.Tensor
    indices: torch.Tensor

class RolloutBufferWithNextObs(RolloutBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.next_observations = None

    def reset(self):
        super().reset()
        self.next_observations = torch.zeros_like(torch.tensor(self.observations))

    def add(self, obs, action, reward, episode_start, value, log_prob, next_obs):
        # Add the next observation to the buffer
        super().add(obs, action, reward, episode_start, value, log_prob)
        self.next_observations[self.pos - 1] = torch.tensor(next_obs)

    def get(self, batch_size: Optional[int] = None):
        # Extend the get method to include next_observations
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.next_observations = self.swap_and_flatten(self.next_observations)
            self.generator_ready = True
        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            batch_indices = indices[start_idx: start_idx + batch_size]
            yield RolloutBufferSamplesWithNextObs(
                observations=self.to_torch(self.observations[batch_indices]).clone().detach(),
                actions=self.to_torch(self.actions[batch_indices]).clone().detach(),
                old_values=self.to_torch(self.values[batch_indices]).clone().detach(),
                old_log_prob=self.to_torch(self.log_probs[batch_indices]).clone().detach(),
                advantages=self.to_torch(self.advantages[batch_indices]).clone().detach(),
                returns=self.to_torch(self.returns[batch_indices]).clone().detach(),
                next_observations=self.to_torch(self.next_observations[batch_indices]).clone().detach(),
                indices=batch_indices
            )
            start_idx += batch_size


class ICMModule(nn.Module):
    """
    Intrinsic Curiosity Module (ICM) for encouraging exploration in PPO.
    """
    def __init__(self, input_dim, action_dim, device):
        super().__init__()
        self.device = device
        self.feature_net = nn.Sequential(*create_mlp(input_dim, 256, [128, 128], nn.ReLU)).to(device)
        self.inverse_net = nn.Sequential(*create_mlp(256 * 2, action_dim, [128], nn.ReLU)).to(device)
        self.forward_net = nn.Sequential(*create_mlp(256 + action_dim, 256, [128], nn.ReLU)).to(device)

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

        self.icm = ICMModule(input_dim=self.observation_space.shape[0], action_dim=self.action_space.n, device=self.device)

    def train(self) -> None:
        """
        Update policy using the current rollout buffer.
        Incorporate ICM for intrinsic motivation.
        """
        # Set policy to training mode
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        entropy_losses, pg_losses, value_losses, clip_fractions, icm_losses = [], [], [], [], []
        continue_training = True

        # Train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            # Iterate over the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                # Policy training part
                policy_loss, value_loss, entropy_loss, approx_kl_div, clip_fraction = self._train_policy(rollout_data)
                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                clip_fractions.append(clip_fraction)
                approx_kl_divs.append(approx_kl_div)

                # Stop training early if KL divergence is too large
                if not self._check_continue_training(approx_kl_div, epoch):
                    continue_training = False
                    break

                # Intrinsic Curiosity Module (ICM) training part
                icm_loss = self._train_icm(rollout_data)
                icm_losses.append(icm_loss.item())

                # Total loss
                total_loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss + icm_loss

                # Optimize the policy and ICM networks
                self._optimize_policy(total_loss)

            self._n_updates += 1
            if not continue_training:
                break

        self._record_training_metrics(entropy_losses, pg_losses, value_losses, approx_kl_divs, clip_fractions)

    def _train_policy(self, rollout_data):
        actions = self._get_actions(rollout_data)
        values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)

        advantages = self._normalize_advantage(rollout_data.advantages)
        ratio = torch.exp(log_prob - rollout_data.old_log_prob)

        # Policy loss
        policy_loss, clip_fraction = self._compute_policy_loss(advantages, ratio, log_prob, rollout_data.old_log_prob)

        # Value loss
        value_loss = self._compute_value_loss(values, rollout_data)

        # Entropy loss
        entropy_loss = self._compute_entropy_loss(log_prob, entropy)

        # Compute approximate KL divergence for early stopping
        approx_kl_div = self._compute_approx_kl(log_prob, rollout_data.old_log_prob)

        return policy_loss, value_loss, entropy_loss, approx_kl_div, clip_fraction

    def _train_icm(self, rollout_data):
        """
        Train the Intrinsic Curiosity Module (ICM) using the current rollout buffer data.
        """
        # Predict action and next state features using ICM
        pred_action, pred_phi_next_state, phi_next_state = self.icm(
            rollout_data.observations, rollout_data.next_observations, rollout_data.actions
        )

        # Compute the inverse and forward loss
        inverse_loss = F.mse_loss(pred_action, rollout_data.actions)
        forward_loss = F.mse_loss(pred_phi_next_state, phi_next_state)
        icm_loss = inverse_loss + forward_loss

        return icm_loss

    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            rollout_buffer: RolloutBuffer,
            n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBufferWithNextObs``.
        Modified to include next observations in the rollout buffer.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with torch.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with torch.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            # Add the current observation, action, reward, episode start, value, log probability, and next observation to the buffer
            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
                next_obs=new_obs
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with torch.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True
