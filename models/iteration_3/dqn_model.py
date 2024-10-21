import torch
from stable_baselines3.common.buffers import ReplayBuffer
import numpy as np

from models.dqn_model import BaseDQN, DQNPolicy
from models.iteration_2.dqn_model import DuelingDQN
import torch.nn.functional as F


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, observation_space, action_space, alpha=0.6, **kwargs):
        """
        Extend the ReplayBuffer from Stable Baselines3 to include prioritization.

        :param buffer_size: Maximum number of transitions to store.
        :param observation_space: Observation space of the environment.
        :param action_space: Action space of the environment.
        :param alpha: The exponent determining how much prioritization is used (0 - no prioritization, 1 - full prioritization).
        :param kwargs: Other arguments passed to the base ReplayBuffer.
        """
        super().__init__(buffer_size, observation_space, action_space, **kwargs)
        # Initialize an array to store priorities for each experience
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.alpha = alpha  # How much prioritization is used
        self.max_priority = 1.0  # Initial priority

    def add(self, *args, **kwargs):
        """
        Add a new transition to the buffer with the default maximum priority.

        :param args: Arguments for adding an experience.
        :param kwargs: Keyword arguments for adding an experience.
        """
        # Get the current index in the buffer where the new experience will be added
        idx = self.pos
        # Use the original add method to add the experience
        super().add(*args, **kwargs)
        # Set the priority of the added experience to the maximum priority value
        self.priorities[idx] = self.max_priority

    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch of experiences based on priority.

        :param batch_size: Number of samples to draw from the buffer.
        :param beta: Importance-sampling weight exponent, used to compensate for non-uniform sampling.
        :return: Sampled batch along with importance sampling weights and indices.
        """
        current_size = self.size()
        if current_size == 0:
            raise ValueError("Cannot sample from an empty buffer!")
        if batch_size > current_size:
            batch_size = current_size
        # Calculate probabilities for each experience in the buffer
        scaled_priorities = self.priorities[:current_size] ** self.alpha
        sampling_probabilities = scaled_priorities / scaled_priorities.sum()

        # Sample indices based on these probabilities
        indices = np.random.choice(current_size, batch_size, p=sampling_probabilities, replace=False)

        # Sample the transitions using the base buffer's internal sampling method
        batch = super()._get_samples(indices)

        # Calculate importance sampling weights
        total = current_size
        weights = (total * sampling_probabilities[indices]) ** (-beta)
        weights /= weights.max()  # Normalize weights to avoid instability

        # Append the weights and indices to the batch for updating priorities later
        batch = batch + (indices, weights)

        return batch

    def update_priorities(self, indices, priorities):
        """
        Update the priorities of sampled transitions.

        :param indices: The indices of the sampled transitions.
        :param priorities: The new priorities based on TD-errors.
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = max(priority, 1e-5)  # To avoid zero priority issues
        self.max_priority = max(self.max_priority, priorities.max())

class PER_DQN(DuelingDQN):
    """
    A custom Dueling DQN agent that extends the CustomDQN.
    Uses DuelingDQNPolicy to implement the Dueling architecture.
    """
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

        # loss = F.mse_loss(q_values, target_q_values)
        self._optimize_model(loss)
        new_priorities = td_errors.abs().detach().cpu().numpy()  # Priorities are the absolute TD-errors
        self.replay_buffer.update_priorities(indices, new_priorities)

        return loss.item()