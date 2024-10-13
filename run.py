import torch
from grid2op.Episode import EpisodeReplay

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.dqn import MultiInputPolicy

from env.env import Gym2OpEnv  # Assuming the environment setup code is in env_setup.py
from grid2op.Runner import Runner

from models.agent import RLAgent
from models.dqn_model import CustomDQN

import logging


def main():
    # Step 1: Initialize the environment wrapper
    env = Gym2OpEnv()

    logging.basicConfig(level=logging.DEBUG)
    # Step 2: Set up the observation and action spaces
    # observation_space = env_wrapper.observation_space
    # action_space_size = env_wrapper.action_space
    # flattened_env = FlattenObservation(env)
    # vec_env = DummyVecEnv([lambda: flattened_env])
    # Step 3: Create the DQN model
    dqn_model = CustomDQN(
        policy=MultiInputPolicy,
        env=env,
        learning_rate=1e-3,
        buffer_size=10000,
        tau=0.005,
        gamma=0.99,
    )

    # Step 4: Create the Grid2Op-compatible DQN agent
    agent = RLAgent(model=dqn_model, gym_env=env)

    # Step 5: Set up the Runner
    runner = Runner(**env._g2op_env.get_params_for_runner(), agentInstance = agent, agentClass=None)

    # Step 6: Run the agent in the environment for a set number of episodes
    results = runner.run(nb_episode=5, path_save="runs", add_detailed_output=True)
    plot_epi = EpisodeReplay("runs")
    plot_epi.replay_episode(results[0][1], gif_name="test")
    # Step 7: Print out the performance results for each episode
    print("Runner results:")
    for ep_idx, episode in enumerate(results):
        env.render()
        # print(f"Episode {ep_idx + 1}:")
        # print(f"  Total Reward: {episode.reward}")
        # print(f"  Total Steps: {episode.nb_step}")
        # print(f"  Done: {episode.done}")

if __name__ == "__main__":
    main()
