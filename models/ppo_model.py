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
            observation_space,
            action_space,
            **kwargs)

    # extend to show calculation
    def compute_returns_and_advantage(self, last_values: torch.Tensor, dones: np.ndarray) -> None:
        last_values = last_values.clone().cpu().numpy().flatten()
        last_gae_lam = 0
        T = self.buffer_size
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
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        use_sde,
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
        net_arch = dict(pi=[64, 64], vf=[64, 64])

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_dim = observation_space.shape[0]

        self._build_networks()
        self.optimizer = optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _build_networks(self) -> None:
        pi_layers = create_mlp(self.features_dim, self.action_space.n, self.net_arch['pi'], self.activation_fn)
        vf_layers = create_mlp(self.features_dim, 1, self.net_arch['vf'], self.activation_fn)

        self.actor_net = nn.Sequential(*pi_layers)
        self.critic_net = nn.Sequential(*vf_layers)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> (torch.Tensor, torch.Tensor, torch.Tensor):
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
        features = self.extract_features(obs)
        return self.critic_net(features)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> (torch.Tensor, torch.Tensor, Optional[torch.Tensor]):
        features = self.extract_features(obs)
        action_logits = self.actor_net(features)
        values = self.critic_net(features)

        dist = torch.distributions.Categorical(logits=action_logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return values, log_prob, entropy

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        features = self.extract_features(observation)
        action_logits = self.actor_net(features)
        if deterministic:
            return torch.argmax(action_logits, dim=-1)
        else:
            return torch.multinomial(torch.softmax(action_logits, dim=-1), 1).squeeze(-1)

class BasePPO(OnPolicyAlgorithm):
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
            **kwargs,
        )

    def _setup_model(self) -> None:
        super()._setup_model()
        self.clip_range = get_schedule_fn(self.clip_range)

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        pg_losses, value_losses, clip_fractions = [], [], []

        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = self._get_actions(rollout_data)
                values, log_prob, _ = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()

                advantages = self._normalize_advantage(rollout_data.advantages)

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # cliped surrogate
                policy_loss, clip_fraction = self._compute_policy_loss(advantages, ratio, log_prob,
                                                                       rollout_data.old_log_prob)
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)
                value_loss = F.mse_loss(rollout_data.returns, values)
                value_losses.append(value_loss.item())

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
        if pg_losses is not None:
            pg_loss_mean = np.mean(pg_losses)
            self.logger.record("train/policy_gradient_loss", pg_loss_mean)
        else:
            pg_loss_mean = 0
        if value_losses is not None:
            value_loss_mean = np.mean(value_losses)
            self.logger.record("train/value_loss", value_loss_mean)
        else:
            value_loss_mean = 0
        if entropy_losses is not None:
            entropy_loss_mean = np.mean(entropy_losses)
            self.logger.record("train/entropy_loss", entropy_loss_mean)
        else:
            entropy_loss_mean = 0
        if icm_losses is not None:
            icm_loss_mean = np.mean(icm_losses)
            self.logger.record("train/icm_loss", icm_loss_mean)
        else:
            icm_loss_mean = 0
        if sil_losses is not None:
            sil_loss_mean = np.mean(sil_losses)
            self.logger.record("train/sil_loss", sil_loss_mean)
        else:
            sil_loss_mean = 0
        if approx_kl_divs is not None:
            self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        if clip_fractions is not None:
            self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        total_loss = pg_loss_mean + self.vf_coef * value_loss_mean + self.ent_coef * entropy_loss_mean + icm_loss_mean + sil_loss_mean
        self.logger.record("train/loss", total_loss)
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", self.clip_range(self._current_progress_remaining))
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", self.clip_range_vf(self._current_progress_remaining))

    def _optimize_policy(self, loss):
        self.policy.optimizer.zero_grad()
        loss.backward()
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