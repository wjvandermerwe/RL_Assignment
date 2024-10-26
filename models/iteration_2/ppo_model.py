from typing import NamedTuple, Optional
import numpy as np
import torch
from gymnasium.vector.utils import spaces
from numpy.core.numeric import indices
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from torch import nn
from stable_baselines3.common.torch_layers import create_mlp
import torch.nn.functional as F
from models.iteration_1.ppo_model import TrulyProximalPPO
from models.ppo_model import RolloutBuffer, BasePPO


class RolloutBufferSamplesWithNextObs(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    next_observations: torch.Tensor
    indices: torch.Tensor

# extend to include next state and indices
class RolloutBufferWithNextObs(RolloutBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.next_observations = None

    def reset(self):
        super().reset()
        self.next_observations = torch.zeros_like(torch.tensor(self.observations))

    def add(self, obs, action, reward, episode_start, value, log_prob, next_obs):
        super().add(obs, action, reward, episode_start, value, log_prob)
        self.next_observations[self.pos - 1] = torch.tensor(next_obs)

    def get(self, batch_size: Optional[int] = None):
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
    def __init__(self, input_dim, action_dim, device):
        super().__init__()
        self.device = device
        self.feature_net = nn.Sequential(*create_mlp(input_dim, 256, [128, 128], nn.ReLU)).to(device)
        self.inverse_net = nn.Sequential(*create_mlp(256 * 2, action_dim, [128], nn.ReLU)).to(device)
        self.forward_net = nn.Sequential(*create_mlp(256 + action_dim, 256, [128], nn.ReLU)).to(device)

    def forward(self, state, next_state, action):
        phi_state = self.feature_net(state)
        phi_next_state = self.feature_net(next_state)
        if len(action.shape) == 1:
            action = F.one_hot(action.long(), num_classes=self.inverse_net[-1].out_features).float()
        pred_action = self.inverse_net(torch.cat([phi_state, phi_next_state], dim=-1))
        pred_phi_next_state = self.forward_net(torch.cat([phi_state, action], dim=-1))
        return pred_action, pred_phi_next_state, phi_next_state


class PPOWithICM(TrulyProximalPPO):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(
            **kwargs
        )

        self.icm = ICMModule(input_dim=self.observation_space.shape[0], action_dim=self.action_space.n, device=self.device)

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            entropy_losses, pg_losses, value_losses, clip_fractions, icm_losses = [], [], [], [], []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                policy_loss, value_loss, entropy_loss, approx_kl_div, clip_fraction = self._train_policy(rollout_data)
                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                clip_fractions.append(clip_fraction)
                approx_kl_divs.append(approx_kl_div)
                if not self._check_continue_training(approx_kl_div, epoch):
                    continue_training = False
                    break
                icm_loss = self._train_icm(rollout_data)
                icm_losses.append(icm_loss.item())

                total_loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss + icm_loss

                self._optimize_policy(total_loss)
            self._record_training_metrics(entropy_losses, pg_losses, value_losses, approx_kl_divs, clip_fractions, icm_losses)
            self._n_updates += 1
            if not continue_training:
                break


    def _train_policy(self, rollout_data):
        actions = self._get_actions(rollout_data)
        values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)

        advantages = self._normalize_advantage(rollout_data.advantages)
        ratio = torch.exp(log_prob - rollout_data.old_log_prob)
        policy_loss, clip_fraction = self._compute_policy_loss(advantages, ratio, log_prob, rollout_data.old_log_prob)
        value_loss = self._compute_value_loss(values, rollout_data)
        entropy_loss = self._compute_entropy_loss(log_prob, entropy)

        approx_kl_div = self._compute_approx_kl(log_prob, rollout_data.old_log_prob)

        return policy_loss, value_loss, entropy_loss, approx_kl_div, clip_fraction

    def _train_icm(self, rollout_data):
        pred_action, pred_phi_next_state, phi_next_state = self.icm(
            rollout_data.observations, rollout_data.next_observations, rollout_data.actions
        )
        inverse_loss = F.mse_loss(pred_action, rollout_data.actions)
        forward_loss = F.mse_loss(pred_phi_next_state, phi_next_state)
        icm_loss = inverse_loss + forward_loss

        return icm_loss


    # copied form baselines to simply include the next observation into the new buffer
    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            rollout_buffer: RolloutBuffer,
            n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"

        self.policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with torch.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:

                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)
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
            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                next_obs=new_obs
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with torch.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def load(self, **kwargs):
        self.use_sde=False
        super().load(**kwargs)

