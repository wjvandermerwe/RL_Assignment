import torch
import torch.nn.functional as F

from models.dqn_model import CustomDQN


class Imp1_DQN(CustomDQN):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train_step(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Get Q-values for the actions taken
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Use target network to compute the next Q-values
        with torch.no_grad():
            next_q_values = self.q_net_target(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss
        loss = F.mse_loss(q_values, target_q_values)

        # Backpropagation step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return loss.item()

    def _on_step(self):
        """
        Update the target network and adjust the exploration rate after every environment step.
        """
        self._n_calls += 1

        # Update target network after fixed number of steps (Fixed Q-Targets)
        if self._n_calls % self.target_update_interval == 0:
            self.update_target_network()

        # Adjust exploration rate
        self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay)
