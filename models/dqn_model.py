import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import BasePolicy
from typing import Optional, List, Dict, Any, Type
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.buffers import ReplayBuffer
import numpy as np
from stable_baselines3.common.torch_layers import FlattenExtractor, BaseFeaturesExtractor, create_mlp
from stable_baselines3.common.type_aliases import TrainFreq, PyTorchObs, TrainFrequencyUnit, ReplayBufferSamples, \
    MaybeCallback


class QNetwork(BasePolicy):
    def __init__(self, obs_shape, observation_space, action_space, features_extractor: BaseFeaturesExtractor,features_dim=None, net_arch=None, activation_fn=nn.ReLU):
        """
        Q-Value Network for DQN.

        :param obs_shape: Shape of the observation space.
        :param action_space: Action space.
        :param net_arch: Network architecture as a list of layer sizes.
        :param activation_fn: Activation function to use between layers.
        """
        super().__init__(observation_space=observation_space, action_space=action_space,features_extractor=features_extractor)

        # If no custom architecture is provided, use default [64, 64]
        if net_arch is None:
            net_arch = [64, 64]

        self.obs_shape = obs_shape
        self.action_dim = action_space.n
        self.activation_fn = activation_fn
        self.net_arch = net_arch
        self.features_dim = features_dim
        q_net = create_mlp(self.features_dim, self.action_dim, self.net_arch, self.activation_fn)

        #unpack
        self.q_net = nn.Sequential(*q_net)

    # def forward(self, obs):
    #     return self.q_net(obs)

    def forward(self, obs: PyTorchObs) -> torch.Tensor:
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        return self.q_net(self.extract_features(obs, self.features_extractor))


    def _predict(self, observation: PyTorchObs, deterministic: bool = True) -> torch.Tensor:
        q_values = self(observation)
        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action




class DQNPolicy(BasePolicy):
    """
    DQN Policy with Q-Network and Target Network.

    :param observation_space: Observation space.
    :param action_space: Action space.
    :param net_arch: Architecture of the Q-Network.
    :param activation_fn: Activation function to use between layers.
    :param normalize_images: Whether to normalize images or not.
    """
    q_net: QNetwork
    q_net_target: QNetwork

    def __init__(
            self,
            observation_space,
            action_space,
            lr_schedule,
            net_arch: Optional[List[int]] = None,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            activation_fn: Type[nn.Module] = nn.ReLU,
            # normalize_images: bool = True,
            # optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            # optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(observation_space=observation_space, action_space=action_space, features_extractor_class=features_extractor_class)

        # Network arguments
        self.net_args = {
            "obs_shape": observation_space.shape,
            "observation_space": observation_space,
            "action_space": action_space,
            "net_arch": net_arch,
            "activation_fn": activation_fn,
        }

        # Build the Q-Networks
        self.q_net = self.make_q_net()
        self.q_net_target = self.make_q_net()
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.q_net_target.eval()

        # Optimizer setup
        self.optimizer = self.optimizer_class(self.q_net.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)



    def make_q_net(self) -> QNetwork:
        """
        Creates the Q-Network using the specified architecture.
        """
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return QNetwork(**net_args).to(self.device)

    def forward(self, obs, deterministic: bool = True) -> torch.Tensor:
        return self._predict(obs, deterministic)

    def _predict(self, obs, deterministic: bool = True) -> torch.Tensor:
        q_values = self.q_net(obs)
        action = q_values.argmax(dim=1) if deterministic else torch.multinomial(F.softmax(q_values, dim=1), num_samples=1)
        return action

    def update_target_network(self):
        """
        Update the target network weights with the Q-network weights.
        """
        self.q_net_target.load_state_dict(self.q_net.state_dict())

class BaseDQN(OffPolicyAlgorithm):
    """
    A custom DQN agent built on top of OffPolicyAlgorithm.
    """

    def __init__(self, policy, env, learning_rate=1e-3,
                 buffer_size=10000, tau=1.0,
                 gamma=0.99, gradient_steps=1, target_update_interval=1000,
                 replay_buffer_class=None,
                 exploration_fraction=0.1, exploration_initial_eps=1.0,
                 exploration_final_eps=0.05, max_grad_norm=10, policy_kwargs=None,
                 verbose=0, device="cuda", _init_setup_model=None, **kwargs):

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            tau=tau,
            gamma=gamma,
            train_freq=TrainFreq(frequency=1, unit=TrainFrequencyUnit.STEP),
            gradient_steps=gradient_steps,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=None,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=None,
            sde_support=False,
            optimize_memory_usage=False,
            **kwargs
        )
        self._n_calls = 0
        self.device = torch.device(device)
        self.epsilon = exploration_initial_eps
        self.epsilon_decay = (exploration_initial_eps - exploration_final_eps) / exploration_fraction
        self.min_epsilon = exploration_final_eps
        self.max_grad_norm = max_grad_norm
        self.target_update_interval = target_update_interval
        self._setup_model()

    def train_step(self, batch):

        states, actions, next_states, dones, rewards = batch

        # Q-values for actions taken
        q_values = self.policy.q_net(states).gather(1, actions.view(-1, 1)).squeeze(1)

        # q_values = self.policy.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values using target network
        with torch.no_grad():
            # next_q_values = self.policy.q_net_target(next_states).max(dim=1)[0]

            next_q_values = self.policy.q_net_target(next_states)
            # Follow greedy policy: use the one with the highest value
            next_q_values, _ = next_q_values.max(dim=1)
            # Avoid potential broadcast issue
            next_q_values = next_q_values.reshape(-1, 1)
            # next_q_values = next_q_values.reshape(-1,1).squeeze(0)
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
            target_q_values = target_q_values.squeeze(1)

        # Loss calculation
        loss = F.mse_loss(q_values, target_q_values)

        # Backpropagation step
        self._optimize_model(loss)

        return loss.item()

    def _optimize_model(self, loss):
        self.policy.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.q_net.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

    def _adjust_exploration_rate(self):
        self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay)

    def _on_step(self):
        self._n_calls += 1
        if self._n_calls % self.target_update_interval == 0:
            self.policy.update_target_network()


    def train(self, gradient_steps: int, batch_size: int = 32):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses = []
        for _ in range(gradient_steps):
            self._adjust_exploration_rate()
            batch = self.replay_buffer.sample(batch_size)
            loss = self.train_step(batch)
            losses.append(loss)
        self.logger.record("train/loss", np.mean(losses))
        return loss

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 4,
            tb_log_name: str = "DQN",
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