from typing import Optional, Union, Dict, Type, Any, ClassVar

import numpy as np
import torch
from gymnasium.vector.utils import spaces
from stable_baselines3.common.buffers import BaseBuffer, RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ActorCriticPolicy
from stable_baselines3.common.torch_layers import create_mlp, FlattenExtractor
from stable_baselines3.common.type_aliases import GymEnv, Schedule, MaybeCallback
from stable_baselines3.common.utils import get_schedule_fn, explained_variance
from torch import nn
import torch.nn.functional as F

class RolloutBuffer(RolloutBuffer):
    def __init__(
            self,
            n_steps,
            observation_space,
            action_space,
            **kwargs
    ):
        super().__init__(
            n_steps,
            observation_space,  # type: ignore[arg-type]
            action_space,
            # device=self.device,
            # gamma=self.gamma,
            # gae_lambda=self.gae_lambda,
            # n_envs=self.n_envs,
            **kwargs)

    def compute_returns_and_advantage(self, last_values: torch.Tensor, dones: np.ndarray) -> None:
        """
        Compute the lambda-return (TD(lambda) estimate) and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate, set gae_lambda=1.0.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        last_values = last_values.clone().cpu().numpy().flatten()
        last_gae_lam = 0
        T = self.buffer_size  # Truncation length, equivalent to trajectory segment length
        for step in reversed(range(T)):
            if step == T - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

class ActorCriticPolicy(ActorCriticPolicy):
    """
    Custom Actor-Critic Policy for PPO with simplified structure.
    """
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        use_sde,
        net_arch: Optional[Union[Dict[str, list], list]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule=lr_schedule,
            use_sde=use_sde,
            features_extractor_class=FlattenExtractor,
            features_extractor_kwargs=None,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=False,
            normalize_images=normalize_images,
        )

        # Default network architecture for both actor and critic
        if net_arch is None:
            net_arch = dict(pi=[64, 64], vf=[64, 64])

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_dim = observation_space.shape[0]

        # Create actor and critic networks
        self._build_networks()
        self.optimizer = optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _build_networks(self) -> None:
        """
        Create the actor (policy) and critic (value) networks using given architecture.
        """
        pi_layers = create_mlp(self.features_dim, self.action_space.n, self.net_arch['pi'], self.activation_fn)
        vf_layers = create_mlp(self.features_dim, 1, self.net_arch['vf'], self.activation_fn)

        self.actor_net = nn.Sequential(*pi_layers)
        self.critic_net = nn.Sequential(*vf_layers)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Forward pass for policy network.

        :param obs: Observation tensor
        :param deterministic: Whether to use deterministic actions
        :return: Tuple containing action, value, and log probability of the action
        """
        features = self.extract_features(obs)
        action_logits = self.actor_net(features)
        values = self.critic_net(features)

        if deterministic:
            actions = torch.argmax(action_logits, dim=-1)
        else:
            actions = torch.multinomial(torch.softmax(action_logits, dim=-1), 1).squeeze(-1)

        dist = torch.distributions.Categorical(logits=action_logits)
        log_prob = dist.log_prob(actions)

        return actions, values, log_prob

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Predict value estimates from critic network.

        :param obs: Observation tensor
        :return: Value tensor
        """
        features = self.extract_features(obs)
        return self.critic_net(features)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> (torch.Tensor, torch.Tensor, Optional[torch.Tensor]):
        """
        Evaluate actions using the current policy.

        :param obs: Observation tensor
        :param actions: Actions tensor
        :return: Estimated values, log probability of actions, entropy of action distribution
        """
        features = self.extract_features(obs)
        action_logits = self.actor_net(features)
        values = self.critic_net(features)

        dist = torch.distributions.Categorical(logits=action_logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return values, log_prob, entropy

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        features = self.extract_features(observation)
        action_logits = self.actor_net(features)
        if deterministic:
            return torch.argmax(action_logits, dim=-1)
        else:
            return torch.multinomial(torch.softmax(action_logits, dim=-1), 1).squeeze(-1)

class BasePPO(OnPolicyAlgorithm):
    """
    Simplified PPO implementation that removes additional optimizations.
    Retains the core PPO features: clipped surrogate objective and GAE.
    """

    def __init__(
            self,
            policy: Union[str, Type[ActorCriticPolicy]],
            env: Union[GymEnv, str],
            batch_size: int = 64,
            n_epochs: int = 10,
            clip_range: Union[float, Schedule] = 0.2,
            clip_range_vf: Union[None, float, Schedule] = None,
            normalize_advantage: bool = True,
            rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
            _init_setup_model: bool = True,
            **kwargs
    ):
        # Sanity check: `batch_size` must be greater than 1 to avoid NaN during normalization
        if normalize_advantage:
            assert batch_size > 1, "`batch_size` must be greater than 1."
        self.clip_range_vf = clip_range_vf
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.normalize_advantage = normalize_advantage
        super().__init__(
            policy,
            env,
            rollout_buffer_class=rollout_buffer_class,
            # tb_log_name="PPO",
            **kwargs,
        )

    def _setup_model(self) -> None:
        super()._setup_model()
        # Initialize schedules for policy clipping
        self.clip_range = get_schedule_fn(self.clip_range)

    def train(self) -> None:
        """
        Update policy using the current rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]

        pg_losses, value_losses, clip_fractions = [], [], []
        losses = []
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = self._get_actions(rollout_data)

                # Evaluate the policy on the current batch of observations and actions
                values, log_prob, _ = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = self._normalize_advantage(rollout_data.advantages)

                # Ratio between the new policy and the old policy
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # cliped surrogate
                policy_loss, clip_fraction = self._compute_policy_loss(advantages, ratio, log_prob,
                                                                       rollout_data.old_log_prob)

                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                # Value loss using TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values)
                value_losses.append(value_loss.item())

                # Total loss
                loss = policy_loss + self.vf_coef * value_loss

                self._optimize_policy(loss)

            self._n_updates += 1
            self._record_training_metrics(None, pg_losses, value_losses,None, clip_fractions)

    def _normalize_advantage(self, advantages):
        if self.normalize_advantage and len(advantages) > 1:
            return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def _compute_policy_loss(self, advantages, ratio, log_prob, old_log_prob):
        clip_range = self.clip_range(self._current_progress_remaining)
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
        clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
        return policy_loss, clip_fraction

    def _record_training_metrics(self, entropy_losses, pg_losses, value_losses, approx_kl_divs, clip_fractions,
                                 icm_losses=None, sil_losses=None):
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Policy gradient loss
        if pg_losses is not None:
            pg_loss_mean = np.mean(pg_losses)
            self.logger.record("train/policy_gradient_loss", pg_loss_mean)
        else:
            pg_loss_mean = 0

        # Value loss
        if value_losses is not None:
            value_loss_mean = np.mean(value_losses)
            self.logger.record("train/value_loss", value_loss_mean)
        else:
            value_loss_mean = 0

        # Entropy loss (for PPO)
        if entropy_losses is not None:
            entropy_loss_mean = np.mean(entropy_losses)
            self.logger.record("train/entropy_loss", entropy_loss_mean)
        else:
            entropy_loss_mean = 0  # Assume 0 if entropy losses are not used

        # ICM loss (if ICM is used)
        if icm_losses is not None:
            icm_loss_mean = np.mean(icm_losses)
            self.logger.record("train/icm_loss", icm_loss_mean)
        else:
            icm_loss_mean = 0  # If ICM is not used, set to 0

        # SIL loss (if SIL is used)
        if sil_losses is not None:
            sil_loss_mean = np.mean(sil_losses)
            self.logger.record("train/sil_loss", sil_loss_mean)
        else:
            sil_loss_mean = 0  # If SIL is not used, set to 0

        # KL divergence for early stopping (if used)
        if approx_kl_divs is not None:
            self.logger.record("train/approx_kl", np.mean(approx_kl_divs))

        # Clip fraction
        if clip_fractions is not None:
            self.logger.record("train/clip_fraction", np.mean(clip_fractions))

        # Calculate and log the total loss (including optional components like entropy, ICM, and SIL)
        total_loss = pg_loss_mean + self.vf_coef * value_loss_mean + self.ent_coef * entropy_loss_mean + icm_loss_mean + sil_loss_mean
        self.logger.record("train/loss", total_loss)

        # Record explained variance
        self.logger.record("train/explained_variance", explained_var)

        # Record standard deviation if the policy has log_std (for continuous action spaces)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        # Number of updates
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

        # Record the clipping ranges
        self.logger.record("train/clip_range", self.clip_range(self._current_progress_remaining))
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", self.clip_range_vf(self._current_progress_remaining))

    def _optimize_policy(self, loss):
        self.policy.optimizer.zero_grad()
        loss.backward()
        # Clip gradient norms
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

    def _get_actions(self, rollout_data):
        actions = rollout_data.actions
        if isinstance(self.action_space, spaces.Discrete):
            actions = rollout_data.actions.long().flatten()
        return actions

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 4,
            tb_log_name: str = "PPO",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ):
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )