import torch
from grid2op.Episode import EpisodeReplay
from stable_baselines3.common.buffers import ReplayBuffer

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MultiInputPolicy, MlpPolicy

# from stable_baselines3.dqn import MultiInputPolicy

from env.env import Gym2OpEnv  # Assuming the environment setup code is in env_setup.py
from grid2op.Runner import Runner

from models.agent import RLAgent
from models.dqn_model import BaseDQN, DQNPolicy
from models.iteration_1.dqn_model import DoubleDQN
from models.iteration_2.dqn_model import DuelingDQN
from models.iteration_3.dqn_model import PER_DQN

import logging

from models.ppo_model import CustomPPO, PPOPolicyNetwork

# logging.getLogger('pandapower').setLevel(logging.WARNING)
#
# # Suppress grid2op Runner logs
# logging.getLogger('grid2op.Environment').setLevel(logging.WARNING)
#
# # Optionally, if you want to suppress all logs (not recommended in most cases):
# logging.getLogger().setLevel(logging.CRITICAL)


def main():
    # Step 1: Initialize the environment wrapper
    env = Gym2OpEnv()

    logging.basicConfig(level=logging.DEBUG)


    # observation_space = env_wrapper.observation_space
    # action_space_size = env_wrapper.action_space
    # flattened_env = FlattenObservation(env)
    # vec_env = DummyVecEnv([lambda: flattened_env])

    # Step 3: Create the DQN model
    args = {
        "env" : env,
        "learning_rate" : 1e-3,
        "buffer_size" : 10000,
        "tau" : 0.005,
        "gamma" : 0.99
    }

    dqn_model = BaseDQN(policy=DQNPolicy,replay_buffer_class=ReplayBuffer,**args)
    dqn_model = DoubleDQN(**args)
    dqn_model = DuelingDQN(**args)
    dqn_model = PER_DQN(**args)

    # ppo = CustomPPO(policy=MlpPolicy,
    #                env=env,
    #                ent_coef = 0.0,
    #                vf_coef= 0.5,
    #                gae_lambda = 0.95,
    #                n_steps = 2048,
    #                max_grad_norm=0.5,
    #                use_sde = False,
    #                sde_sample_freq=-1)

    # Step 4: Create the Grid2Op-compatible DQN agent
    agent = RLAgent(model=dqn_model, gym_env=env)
    params =  env._g2op_env.get_params_for_runner()
    del params["verbose"]
    # Step 5: Set up the Runner
    runner = Runner(**params, agentInstance= agent, agentClass=None, verbose=False)

    # Step 6: Run the agent in the environment for a set number of episodes
    results = runner.run(nb_episode=100, path_save="runs")
    plot_epi = EpisodeReplay("runs")
    plot_epi.replay_episode(results[15][1], gif_name="test")
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
