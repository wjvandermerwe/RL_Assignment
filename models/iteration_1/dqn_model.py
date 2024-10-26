from models.dqn_model import BaseDQN, DQNPolicy
import torch
import torch.nn.functional as F

class DoubleDQNPolicy(DQNPolicy):
    def __init__(
            self,
            observation_space,
            action_space,
            lr_schedule,
            **kwargs
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            **kwargs
        )

    def compute_double_dqn_target(self, rewards, next_states, dones, gamma):
        # extend dqn policy with double dqn pass
        with torch.no_grad():
            next_q_values = self.q_net(next_states)
            best_actions = next_q_values.argmax(dim=1, keepdim=True)
            target_next_q_values = self.q_net_target(next_states).gather(1, best_actions).squeeze(1)
            target_next_q_values = target_next_q_values.reshape(-1, 1)
            target_q_values = rewards + gamma * target_next_q_values * (1 - dones)
            target_q_values = target_q_values.squeeze(1)
        return target_q_values


class DoubleDQN(BaseDQN):
    def __init__(self, **kwargs):
        super().__init__(
            **kwargs
        )

    def train_step(self, batch):

        states, actions, next_states, dones, rewards = batch
        q_values = self.policy.q_net(states).gather(1, actions.view(-1, 1)).squeeze(1)
        target_q_values = self.policy.compute_double_dqn_target(rewards, next_states, dones, self.gamma)
        loss = F.mse_loss(q_values, target_q_values)
        self._optimize_model(loss)

        return loss.item()
