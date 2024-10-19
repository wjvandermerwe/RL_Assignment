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

def main(args):
    # Step 1: Initialize the environment
    env = Gym2OpEnv()
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if args.mode == 'train':
        model=None
        if args.type == 'dqn':
            model = BaseDQN(
                policy=DQNPolicy,
                env=env,
                replay_buffer_class=ReplayBuffer,
                learning_rate=1e-3,
                buffer_size=10000,
                tau=0.005,
                gamma=0.99
            )
        elif args.type == 'ddqn':
            model = BaseDQN()
        elif args.type == '2dqn':
            model = BaseDQN()
        elif args.type == 'per-dqn':
            model = BaseDQN()
        model.learn(total_timesteps=20000, progress_bar=True)
        model.save("dqn_base")
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
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'inference'],
                        help="Options are 'dqn', 'ddqn', '2dqn','per-dqn'")
    parser.add_argument('--type', type=str, required=True, choices=['dqn', 'ddqn', '2dqn','per-dqn'])
    args = parser.parse_args()
    main(args)
