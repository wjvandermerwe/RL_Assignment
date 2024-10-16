from typing import Optional, Type, List, Dict, Any

from stable_baselines3.common.buffers import ReplayBuffer

from models.dqn_model import BaseDQN, DQNPolicy
import torch
from torch import nn
import torch.nn.functional as F

class DoubleDQNPolicy(DQNPolicy):
    """
    Double DQN Policy that modifies the target value calculation for Double DQN.
    Inherits from the base DQNPolicy.
    """

    def __init__(
            self,
            observation_space,
            action_space,
            lr_schedule,
            net_arch: Optional[List[int]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
    ) -> None:
        # Initialize base DQNPolicy
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
        )

    def compute_double_dqn_target(self, rewards, next_states, dones):
        """
        Compute the Double DQN target values.

        :param rewards: Tensor of rewards from the batch.
        :param next_states: Tensor of next states from the batch.
        :param dones: Tensor indicating the end of an episode.
        :return: Computed target Q-values for Double DQN.
        """
        with torch.no_grad():
            # Get the best action from the current Q-network (q_net)
            next_q_values = self.q_net(next_states)
            best_actions = next_q_values.argmax(dim=1, keepdim=True)

            # Evaluate the value of those actions using the target network
            target_next_q_values = self.q_net_target(next_states).gather(1, best_actions).squeeze(1)

            # Compute target Q-value
            target_q_values = rewards + self.gamma * target_next_q_values * (1 - dones)
        return target_q_values


class DoubleDQN(BaseDQN):
    """
    A custom Double DQN agent that extends the CustomDQN.
    Uses DoubleDQNPolicy to implement the Double DQN target value calculation.
    """
    def __init__(self, **kwargs):
        super().__init__(
            policy=DoubleDQNPolicy,
            replay_buffer_class=ReplayBuffer,
            **kwargs
        )

    def train_step(self, batch):
        states, actions, rewards, next_states, dones = self._process_batch(batch)
        q_values = self.policy.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # double DQN Target Q-values using policy's Double DQN logic
        target_q_values = self.policy.compute_double_dqn_target(rewards, next_states, dones)
        loss = F.mse_loss(q_values, target_q_values)
        self._optimize_model(loss)
        return loss.item()
