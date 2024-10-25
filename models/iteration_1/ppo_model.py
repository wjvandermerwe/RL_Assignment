import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, Dict, Type, Any
from gymnasium import spaces
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import get_schedule_fn, explained_variance

from models.ppo_model import BasePPO


class TrulyProximalPPO(BasePPO):
    """
    Truly Proximal Policy Optimization (TPPO) implementation.
    This class extends PPO with adaptive clipping and KL divergence monitoring to ensure truly proximal updates.
    """

    def __init__(
            self,
            target_kl: Optional[float] = 0.01,
            _init_setup_model: bool = True,
            **kwargs,
    ):

        self.target_kl = target_kl
        super(TrulyProximalPPO, self).__init__(
            **kwargs,
            supported_action_spaces=(spaces.Box, spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary),
        )

    def _setup_model(self) -> None:
        super()._setup_model()
        # Initialize schedules for policy and value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def train(self) -> None:
        """
        Update policy using the current rollout buffer with adaptive clipping and KL divergence monitoring.
        """
        # Set policy to training mode
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        continue_training = True

        # Train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            entropy_losses, pg_losses, value_losses, clip_fractions = [], [], [], []
            # Iterate over the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = self._get_actions(rollout_data)
                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                advantages = self._normalize_advantage(rollout_data.advantages)
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss, clip_fraction = self._compute_policy_loss(advantages, ratio, log_prob,
                                                                       rollout_data.old_log_prob)

                clip_fractions.append(clip_fraction)
                pg_losses.append(policy_loss.item())

                value_loss = self._compute_value_loss(values, rollout_data)
                value_losses.append(value_loss.item())

                entropy_loss = self._compute_entropy_loss(log_prob, entropy)
                entropy_losses.append(entropy_loss.item())

                # Total loss
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                # Compute approximate KL divergence for early stopping
                approx_kl_div = self._compute_approx_kl(log_prob, rollout_data.old_log_prob)
                approx_kl_divs.append(approx_kl_div)

                # Stop training early if KL divergence is too large
                if not self._check_continue_training(approx_kl_div, epoch):
                    continue_training = False
                    break

                # Optimize the policy
                self._optimize_policy(loss)

            self._record_training_metrics(entropy_losses, pg_losses, value_losses, approx_kl_divs, clip_fractions)
            self._n_updates += 1
            if not continue_training:
                break

    def _compute_value_loss(self, values, rollout_data):
        clip_range_vf = self.clip_range_vf(self._current_progress_remaining) if self.clip_range_vf is not None else None
        if clip_range_vf is None:
            values_pred = values
        else:
            values_pred = rollout_data.old_values + torch.clamp(values - rollout_data.old_values, -clip_range_vf,
                                                                clip_range_vf)
        values_pred = values_pred.view(-1)
        returns = rollout_data.returns.view(-1)
        value_loss = F.mse_loss(returns, values_pred)
        return value_loss

    def _compute_entropy_loss(self, log_prob, entropy):
        return -log_prob.mean() if entropy is None else -entropy.mean()

    def _compute_approx_kl(self, log_prob, old_log_prob):
        with torch.no_grad():
            log_ratio = log_prob - old_log_prob
            approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
        return approx_kl_div

    def _check_continue_training(self, approx_kl_div, epoch):
        if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
            if self.verbose >= 1:
                print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
            return False
        return True


