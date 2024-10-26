import torch
from torch import nn
from stable_baselines3.common.torch_layers import create_mlp
import torch.nn.functional as F
from models.iteration_1.ppo_model import TrulyProximalPPO


class SelfImitationLearning(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(SelfImitationLearning, self).__init__()
        self.value_net = nn.Sequential(*create_mlp(input_dim, 1, [128, 128], nn.ReLU))
        self.policy_net = nn.Sequential(*create_mlp(input_dim, action_dim, [128, 128], nn.ReLU))

    def forward(self, state):
        value = self.value_net(state)
        action_logits = self.policy_net(state)
        return value, action_logits


class PPOWithSIL(TrulyProximalPPO):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sil = SelfImitationLearning(input_dim=self.observation_space.shape[0], action_dim=self.action_space.n).to(self.device)

    def train(self):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            entropy_losses, pg_losses, value_losses, clip_fractions, sil_losses = [], [], [], [], []

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
                sil_loss = self._train_sil(rollout_data)
                sil_losses.append(sil_loss.item())
                total_loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss + sil_loss
                self.policy.optimizer.zero_grad()
                total_loss.backward()
                self.policy.optimizer.step()

            self._record_training_metrics(entropy_losses, pg_losses, value_losses, approx_kl_divs, clip_fractions,sil_losses= sil_losses)
            self._n_updates += 1

            if not continue_training:
                break

    def _train_sil(self, rollout_data):
        value, action_logits = self.sil(rollout_data.observations)
        advantage = rollout_data.returns - value.detach()
        mask = (advantage > 0).float()
        action_probabilities = F.log_softmax(action_logits, dim=-1)
        selected_action_probabilities = (action_probabilities * rollout_data.actions).sum(dim=-1)
        sil_policy_loss = -(selected_action_probabilities * mask).mean()
        sil_value_loss = F.mse_loss(value.squeeze(), rollout_data.returns, reduction='none')
        sil_value_loss = (sil_value_loss * mask).mean()
        sil_loss = sil_policy_loss + sil_value_loss
        return sil_loss

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

