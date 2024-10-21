import argparse


from grid2op.Episode import EpisodeData
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import ReplayBuffer

from env.env import Gym2OpEnv  # Assuming the environment setup code is in env_setup.py
from grid2op.Runner import Runner
from env.agent import RLAgent
from models.dqn_model import BaseDQN, DQNPolicy

import warnings
from stable_baselines3.common.utils import get_schedule_fn

from models.iteration_1.dqn_model import DoubleDQN, DoubleDQNPolicy
from models.iteration_1.ppo_model import TrulyProximalPPO
from models.iteration_2.dqn_model import DuelingQNetwork, DuelingDQN
from models.iteration_2.ppo_model import PPOWithICM, RolloutBufferWithNextObs
from models.iteration_3.dqn_model import PrioritizedReplayBuffer, PER_DQN
from models.iteration_3.ppo_model import PPOWithSIL
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
                "learning_rate" : get_schedule_fn(1e-3),
                "buffer_size" : 1000,
                "tau" : 0.005,
                "gamma" : 0.90,
                "tensorboard_log": "train_output/"+model_type,
            }
            ppo_net_args = {
                "env": env,
                "device":"cuda",
                "learning_rate": 1e-5,
                "gamma": 0.80,
                "n_steps": 2048,
                "gae_lambda": 0.95,
                "ent_coef": 0.01,
                "vf_coef":0.5,
                "max_grad_norm": 0.5,
                "use_sde":False,
                "sde_sample_freq": 4,
                "tensorboard_log": "train_output/"+model_type,
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
            elif model_type == 'baseppo':
                model = PPO(
                    # policy=ActorCriticPolicy,
                    # rollout_buffer_class=RolloutBuffer,
                    # **ppo_net_args
                    env=env,
                    tensorboard_log="train_output/baseppo",
                )
            elif model_type == 'ppo':
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
                    rollout_buffer_class=RolloutBufferWithNextObs,
                    **ppo_net_args)
            elif model_type == 'sil-ppo':
                model = PPOWithSIL(
                    policy=ActorCriticPolicy,
                    replay_buffer_class=RolloutBufferWithNextObs,
                    **ppo_net_args)


            model.learn(total_timesteps=100000, progress_bar=True)
            model.save("outputs/small/"+model_type)
            print("Training completed and model saved.")

    elif args.mode == 'inference':
        # Inference mode
        # Load the saved model
        if args.inference_run == True:
            dqn_model = BaseDQN.load("outputs/small/2dqn", device="cuda", env=env)

            agent = RLAgent(model=dqn_model, gym_env=env)
            params = env._g2op_env.get_params_for_runner()
            del params["verbose"]

            runner = Runner(**params, agentInstance=agent, agentClass=None, verbose=False)

            runner.run(nb_episode=100, path_save="runs")

        # Replay an episode
        # plot_epi = EpisodeReplay("runs")
        # plot_epi.replay_episode(results[99][1], gif_name="test")

        # Step 7: Compute statistics and print results
        all = EpisodeData.list_episode("runs")
        loaded = [EpisodeData.from_disk(*epi) for epi in all]

        for i, epi in enumerate(loaded):
            line_disc = 0
            line_reco = 0
            line_changed = 0
            for act in epi.actions:
                dict_ = act.as_dict()
                if "set_line_status" in dict_:
                    line_reco += dict_["set_line_status"]["nb_connected"]
                    line_disc += dict_["set_line_status"]["nb_disconnected"]
                if "change_line_status" in dict_:
                    line_changed += dict_["change_line_status"]["nb_changed"]
            print(f'Episode {i}')
            print(f'Total lines set to connected : {line_reco}')
            print(f'Total lines set to disconnected : {line_disc}')
            print(f'Total lines changed: {line_changed}')
            print(f'\n\n')
        # stats.compute(nb_scenario=100)
        print("Runner results:")
        # visualize_rewards(loaded)
        visualize_peak_load_heatmap()
        # visualize_line_disconnections(loaded)
        # visualize_observations(loaded)

    else:
        print("Invalid mode selected. Please use 'train' or 'inference'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN Training and Inference")
    # parser.add_argument('--mode', type=str, required=True, choices=['train', 'inference'],
    #                     help="Options are 'dqn', 'ddqn', '2dqn','per-dqn'.")
    # parser.add_argument('--type', type=str, required=True)
    args = parser.parse_args()
    args.type="dqn"
    args.mode="train"
    args.inference_run = False
    main(args)
