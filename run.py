import argparse
import torch
from grid2op.Episode import EpisodeReplay
from grid2op.utils import EpisodeStatistics
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import device

from env.env import Gym2OpEnv  # Assuming the environment setup code is in env_setup.py
from grid2op.Runner import Runner
from models.agent import RLAgent
from models.dqn_model import BaseDQN, DQNPolicy

import warnings

from models.iteration_1.dqn_model import DoubleDQN, DoubleDQNPolicy
from models.iteration_1.ppo_model import TrulyProximalPPO
from models.iteration_2.dqn_model import DuelingQNetwork, DuelingDQN
from models.iteration_2.ppo_model import PPOWithICM
from models.iteration_3.dqn_model import PrioritizedReplayBuffer, PER_DQN
from models.ppo_model import BasePPO, RolloutBuffer, ActorCriticPolicy


def main(args):
    # Step 1: Initialize the environment
    env = Gym2OpEnv()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    models_to_train = args.type.split(",")
    if args.mode == 'train':
        for model_type in models_to_train:
            model=None
            dqn_net_args = {
                "env" : env,
                "learning_rate" : 1e-3,
                "buffer_size" : 10000,
                "tau" : 0.005,
                "gamma" : 0.99
            }
            ppo_net_args = {
                "env": env,
                "learning_rate": 1e-3,
                "gamma": 0.99,
                "n_steps": 2048,
                "gae_lambda": 0.95,
                "ent_coef": 0.01,
                "vf_coef":0.5,
                "max_grad_norm": 0.5,
                "use_sde":False,
                "sde_sample_freq": -1,
            }
            if model_type == 'dqn':
                model = BaseDQN(
                    policy=DQNPolicy,
                    replay_buffer_class=ReplayBuffer,
                    **dqn_net_args
                )
            elif model_type == '2dqn':
                model = DoubleDQN(
                    policy=DoubleDQNPolicy,
                    replay_buffer_class=ReplayBuffer,
                    **dqn_net_args)
            elif model_type == 'ddqn':
                model = DuelingDQN(
                    policy=DuelingQNetwork,
                    replay_buffer_class=ReplayBuffer,
                    **dqn_net_args)
            elif model_type == 'per-dqn':
                model = PER_DQN(
                    policy=DuelingQNetwork,
                    replay_buffer_class=PrioritizedReplayBuffer,
                    **dqn_net_args)
            if model_type == 'ppo':
                model = BasePPO(
                    policy=ActorCriticPolicy,
                    rollout_buffer_class=RolloutBuffer,
                    **ppo_net_args
                )
            elif model_type == 'tppo':
                model = TrulyProximalPPO(
                    policy=ActorCriticPolicy,
                    rollout_buffer_class=RolloutBuffer,
                    **ppo_net_args)
            elif model_type == 'icm-ppo':
                model = PPOWithICM(
                    policy=ActorCriticPolicy,
                    rollout_buffer_class=RolloutBuffer,
                    **ppo_net_args)
            elif model_type == 'sil-ppo':
                model = PER_DQN(
                    policy=DuelingQNetwork,
                    replay_buffer_class=PrioritizedReplayBuffer,
                    **ppo_net_args)
            model.learn(total_timesteps=20000, progress_bar=True)
            model.save(model_type)
            print("Training completed and model saved.")

    elif args.mode == 'inference':
        # Inference mode
        # Load the saved model
        dqn_model = BaseDQN.load("dqn_base", device="cuda", env=env)

        # Step 4: Create the Grid2Op-compatible DQN agent
        agent = RLAgent(model=dqn_model, gym_env=env)
        params = env._g2op_env.get_params_for_runner()
        del params["verbose"]

        # Step 5: Set up the Runner
        runner = Runner(**params, agentInstance=agent, agentClass=None, verbose=False)

        # Step 6: Run the agent in the environment for a set number of episodes
        results = runner.run(nb_episode=100, path_save="runs")

        # Replay an episode
        plot_epi = EpisodeReplay("runs")
        plot_epi.replay_episode(results[15][1], gif_name="test")

        # Step 7: Compute statistics and print results
        stats = EpisodeStatistics(env)
        stats.compute(nb_scenario=100)
        print("Runner results:")
        for ep_idx, episode in enumerate(results):
            env.render()
            # print(f"Episode {ep_idx + 1}:")
            # print(f"  Total Reward: {episode.reward}")
            # print(f"  Total Steps: {episode.nb_step}")
            # print(f"  Done: {episode.done}")

    else:
        print("Invalid mode selected. Please use 'train' or 'inference'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN Training and Inference")
    # parser.add_argument('--mode', type=str, required=True, choices=['train', 'inference'],
    #                     help="Options are 'dqn', 'ddqn', '2dqn','per-dqn'.")
    # parser.add_argument('--type', type=str, required=True, choices=['dqn', 'ddqn', '2dqn','per-dqn', 'ppo', 'tppo','icm-ppo','sil-ppo'],)
    args = parser.parse_args()
    args.type="tppo"
    args.mode="train"
    main(args)
