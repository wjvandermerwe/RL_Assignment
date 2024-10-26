import argparse
from grid2op.Episode import EpisodeData
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.save_util import load_from_zip_file

from env.env import Gym2OpEnv
from grid2op.Runner import Runner
from env.agent import RLAgent
from env.utils import  save_log_gif
from models.dqn_model import BaseDQN, DQNPolicy
import warnings
from stable_baselines3.common.utils import get_schedule_fn
from models.iteration_1.dqn_model import DoubleDQN, DoubleDQNPolicy
from models.iteration_1.ppo_model import TrulyProximalPPO
from models.iteration_2.dqn_model import DuelingQNetwork, DuelingDQN
from models.iteration_2.ppo_model import PPOWithICM, RolloutBufferWithNextObs
from models.iteration_3.dqn_model import PrioritizedReplayBuffer, PER_DQN
from models.iteration_3.ppo_model import PPOWithSIL
# from models.iteration_3.ppo_model import PPOWithSIL
from models.ppo_model import BasePPO, RolloutBuffer, ActorCriticPolicy


def init_model(model_type, env):
    model = None
    dqn_net_args = {
        "env": env,
        "learning_rate": get_schedule_fn(1e-5),
        "buffer_size": 2500,
        "tau": 0.005,
        "gamma": 0.95,
        # "tensorboard_log": "tensorb_run2/" + model_type,
    }
    ppo_net_args = {
        "env": env,
        "device": "cuda",
        "learning_rate": get_schedule_fn(1e-5),
        "gamma": 0.90,
        "n_steps": 2048,
        "batch_size": 1000,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "use_sde": False,
        "sde_sample_freq": -1,
        # "tensorboard_log": "tensorb_run2/" + model_type,
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
            **dqn_net_args
        )
    elif model_type == 'ddqn':
        model = DuelingDQN(
            policy=DuelingQNetwork,
            replay_buffer_class=ReplayBuffer,
            **dqn_net_args
        )
    elif model_type == 'per-dqn':
        model = PER_DQN(
            policy=DuelingQNetwork,
            replay_buffer_class=PrioritizedReplayBuffer,
            **dqn_net_args
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
            **ppo_net_args
        )
    elif model_type == 'icm-ppo':
        model = PPOWithICM(
            policy=ActorCriticPolicy,
            rollout_buffer_class=RolloutBufferWithNextObs,
            **ppo_net_args
        )
    elif model_type == 'sil-ppo':
        model = PPOWithSIL(
            policy=ActorCriticPolicy,
            rollout_buffer_class=RolloutBuffer,
            **ppo_net_args
        )

    return model, dqn_net_args, ppo_net_args

def main(args):
    env = Gym2OpEnv()
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if args.mode == 'train':
        for model_type, steps in args.model_steps:

            model,_,_ = init_model(model_type, env)
            model.learn(total_timesteps=steps, progress_bar=True)
            model.save(f"outputs/{model_type}_{steps}")
            print(f"Training for {model_type} completed and model saved.")

    elif args.mode == 'inference':
        for model_type, steps in args.model_steps:
            model,ppo_net_args, ppo_net_args = init_model(model_type, env)
            _, params, _ = load_from_zip_file(f'outputs/{model_type}_{steps}', device="cuda")

            # Set parameters to the existing model instance
            model.set_parameters(params, exact_match=True, device="cuda")

            agent = RLAgent(model=model, gym_env=env)
            params = env._g2op_env.get_params_for_runner()
            del params["verbose"]

            runner = Runner(**params, agentInstance=agent, agentClass=None, verbose=False)
            if args.gif == True:
                res = runner.run(nb_episode=10, path_save=f'runs/{model_type}_{steps}')

                # Track the best result out of the 10 episodes
                best_result = None
                best_reward = -float('inf')

                for result in res:
                    _, chron_name, cum_reward, nb_time_step, max_ts = result

                    if cum_reward > best_reward:
                        best_reward = cum_reward
                        best_result = result

                    msg_tmp = "\tFor chronics located at {}\n".format(chron_name)
                    msg_tmp += "\t\t - cumulative reward: {:.6f}\n".format(cum_reward)
                    msg_tmp += "\t\t - number of time steps completed: {:.0f} / {:.0f}".format(nb_time_step, max_ts)
                    print(msg_tmp)

                # Save only the best result from the 10 runs
                if best_result:
                    save_log_gif(f'runs/{model_type}_{steps}', [best_result])
            else:
                import matplotlib.pyplot as plt
                import numpy as np

                # Initialize lists to store results
                cumulative_rewards = []
                episode_lengths = []

                # Run the episodes and collect data
                res = runner.run(nb_episode=100)
                for _, chron_name, cum_reward, nb_time_step, max_ts in res:
                    cumulative_rewards.append(cum_reward)
                    episode_lengths.append(nb_time_step)

                # Calculate cumulative sum for rewards
                cumulative_rewards = np.cumsum(cumulative_rewards)

                # Overlay the charts
                fig, ax1 = plt.subplots()

                # Plot cumulative rewards on primary y-axis
                ax1.plot(cumulative_rewards, color="b", label="Cumulative Reward")
                ax1.set_xlabel("Episode")
                ax1.set_ylabel("Cumulative Reward", color="b")
                ax1.tick_params(axis="y", labelcolor="b")

                # Create secondary y-axis for episode length
                ax2 = ax1.twinx()
                ax2.plot(episode_lengths, color="g", label="Episode Length")
                ax2.set_ylabel("Episode Length", color="g")
                ax2.tick_params(axis="y", labelcolor="g")

                # Title and legend
                plt.title("Cumulative Reward and Episode Length over Episodes")
                fig.tight_layout()
                plt.savefig(f'overlayed_rewards_lengths_{model_type}_{steps}.png')

                print(f'Completed {model_type}_{steps}')


    else:
        print("Invalid mode selected. Please use 'train' or 'inference'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN Training and Inference")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'inference'],
                        help="Options are 'dqn', 'ddqn', '2dqn','per-dqn'.")
    parser.add_argument('--gif', type=str, required=True, choices=['True', 'False'],
                        help="Options are 'dqn', 'ddqn', '2dqn','per-dqn'.")
    args = parser.parse_args()
    # args.mode = "inference"
    args.model_steps = [
        ("dqn", 100000),
        # ("dqn", 50000),
        # ("dqn", 20000),
        ("2dqn", 100000),
        # ("2dqn", 50000),
        # ("2dqn", 20000),
        ("ddqn", 100000),
        # ("ddqn", 50000),
        # ("ddqn", 20000),
        ("per-dqn", 100000),
        # ("per-dqn", 50000),
        # ("per-dqn", 20000),
        # ("ppo", 200000),
        ("ppo", 100000),
        # ("ppo", 50000),
        # ("tppo", 200000),
        ("tppo", 100000),
        # ("tppo", 50000),
        # ("icm-ppo", 200000),
        ("icm-ppo", 100000),
        # ("icm-ppo", 50000),
        # ("sil-ppo", 200000),
        ("sil-ppo", 100000),
        # ("sil-ppo", 50000)
    ]


    main(args)

