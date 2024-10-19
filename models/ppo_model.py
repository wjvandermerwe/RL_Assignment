from typing import Optional, Union, Dict, Type, Any, ClassVar

import numpy as np
import torch
from gymnasium.vector.utils import spaces
from stable_baselines3.common.buffers import BaseBuffer, RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ActorCriticPolicy
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import get_schedule_fn, explained_variance
from torch import nn
import torch.nn.functional as F

class ComputeAdvantageRolloutBuffer(RolloutBuffer):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

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




class ActorCriticPolicy(BasePolicy):
    """
    Custom Actor-Critic Policy for PPO with simplified structure.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not
    :param optimizer_class: The optimizer to use, ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments, excluding the learning rate, to pass to the optimizer
    """
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        net_arch: Optional[Union[Dict[str, list], list]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class=None,
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

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Forward pass for policy network.

        :param obs: Observation tensor
        :param deterministic: Whether to use deterministic actions
        :return: Action tensor
        """
        action_logits = self.actor_net(obs)
        if deterministic:
            return torch.argmax(action_logits, dim=-1)
        else:
            return torch.multinomial(torch.softmax(action_logits, dim=-1), 1)

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Predict value estimates from critic network.

        :param obs: Observation tensor
        :return: Value tensor
        """
        return self.critic_net(obs)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Evaluate actions using the current policy.

        :param obs: Observation tensor
        :param actions: Actions tensor
        :return: Estimated values, log probability of actions, entropy of action distribution
        """
        action_logits = self.actor_net(obs)
        values = self.critic_net(obs)
        dist = torch.distributions.Categorical(logits=action_logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return values, log_prob, entropy


class BasePPO(OnPolicyAlgorithm):
    """
    Simplified PPO implementation that removes additional optimizations.
    Retains the core PPO features: clipped surrogate objective and GAE.
    """

    def __init__(
            self,
            policy: Union[str, Type[ActorCriticPolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 3e-4,
            n_steps: int = 2048,
            batch_size: int = 64,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: Union[float, Schedule] = 0.2,
            normalize_advantage: bool = True,
            vf_coef: float = 0.5,
            max_grad_norm: Optional[float] = None,
            rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
            rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
            tensorboard_log: Optional[str] = None,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[torch.device, str] = "auto",
            _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(spaces.Box, spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary),
        )

        # Sanity check: `batch_size` must be greater than 1 to avoid NaN during normalization
        if normalize_advantage:
            assert batch_size > 1, "`batch_size` must be greater than 1."

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.normalize_advantage = normalize_advantage

        if _init_setup_model:
            self._setup_model()

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

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Evaluate the policy on the current batch of observations and actions
                values, log_prob, _ = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()

                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Ratio between the new policy and the old policy
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # Clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())

                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                # Value loss using TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values)
                value_losses.append(value_loss.item())

                # Total loss
                loss = policy_loss + self.vf_coef * value_loss

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()

                # Optional: Clip gradient norms to stabilize training
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

                self.policy.optimizer.step()

            self._n_updates += 1

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logging
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)