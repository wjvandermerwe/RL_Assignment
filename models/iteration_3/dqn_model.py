import torch
from stable_baselines3.common.buffers import ReplayBuffer
import numpy as np
from models.iteration_2.dqn_model import DuelingDQN

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, observation_space, action_space, alpha=0.6, **kwargs):
        super().__init__(buffer_size, observation_space, action_space, **kwargs)
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.alpha = alpha
        self.max_priority = 1.0

    def add(self, *args, **kwargs):
        idx = self.pos
        super().add(*args, **kwargs)
        self.priorities[idx] = self.max_priority

    def sample(self, batch_size, beta=0.4):
        current_size = self.size()
        if current_size == 0:
            raise ValueError("Cannot sample from an empty buffer!")
        if batch_size > current_size:
            batch_size = current_size

        scaled_priorities = self.priorities[:current_size] ** self.alpha
        sampling_probabilities = scaled_priorities / scaled_priorities.sum()
        indices = np.random.choice(current_size, batch_size, p=sampling_probabilities, replace=False)
        batch = super()._get_samples(indices)
        total = current_size
        weights = (total * sampling_probabilities[indices]) ** (-beta)
        weights /= weights.max()
        batch = batch + (indices, weights)
        return batch

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = max(priority, 1e-5)  # threshold
        self.max_priority = max(self.max_priority, priorities.max())

class PER_DQN(DuelingDQN):
    def __init__(self, **kwargs):
        super().__init__(
            **kwargs
        )

    def train_step(self, batch):
        states, actions, next_states, dones, rewards, indices, weights = batch
        q_values = self.policy.q_net(states).gather(1, actions.view(-1, 1)).squeeze(1)
        target_q_values = self.policy.compute_double_dqn_target(rewards, next_states, dones, self.gamma)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        td_errors = q_values - target_q_values
        loss = (weights * td_errors.pow(2)).mean()

        self._optimize_model(loss)
        new_priorities = td_errors.abs().detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, new_priorities)

        return loss.item()