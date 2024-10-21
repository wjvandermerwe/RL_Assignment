import argparse
from grid2op.Episode import EpisodeData
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import ReplayBuffer
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
# from models.iteration_3.ppo_model import PPOWithSIL
from models.ppo_model import BasePPO, RolloutBuffer, ActorCriticPolicy


def main(args):
    env = Gym2OpEnv()
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if args.mode == 'train':
        for model_type, steps in args.model_steps:
            model = None
            dqn_net_args = {
                "env": env,
                "learning_rate": get_schedule_fn(1e-3),
                "buffer_size": 1000,
                "tau": 0.005,
                "gamma": 0.90,
                "tensorboard_log": "tensor_board/" + model_type,
            }
            ppo_net_args = {
                "env": env,
                "device": "cuda",
                "learning_rate": get_schedule_fn(1e-3),
                "gamma": 0.80,
                "n_steps": 2048,
                "gae_lambda": 0.95,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "use_sde": False,
                "sde_sample_freq": 4,
                "tensorboard_log": "tensor_board/" + model_type,
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
            # ran output of time
            # elif model_type == 'sil-ppo':
            #     model = PPOWithSIL(
            #         policy=ActorCriticPolicy,
            #         replay_buffer_class=RolloutBufferWithNextObs,
            #         **ppo_net_args
            #     )

            model.learn(total_timesteps=steps, progress_bar=True)
            model.save(f"outputs/{model_type}_{steps}")
            print(f"Training for {model_type} completed and model saved.")

    elif args.mode == 'inference':
        for model_type, steps in args.model_steps:
            dqn_model = BaseDQN.load(f'outputs/{args.model}_{steps}', device="cuda", env=env)

            agent = RLAgent(model=dqn_model, gym_env=env)
            params = env._g2op_env.get_params_for_runner()
            del params["verbose"]

            runner = Runner(**params, agentInstance=agent, agentClass=None, verbose=False)

            res = runner.run(nb_episode=10, path_save=f'runs/{args.model}')

            for _, chron_name, cum_reward, nb_time_step, max_ts in res:
                msg_tmp = "\tFor chronics located at {}\n".format(chron_name)
                msg_tmp += "\t\t - cumulative reward: {:.6f}\n".format(cum_reward)
                msg_tmp += "\t\t - number of time steps completed: {:.0f} / {:.0f}".format(nb_time_step, max_ts)
                print(msg_tmp)

            save_log_gif("runs", res)

    else:
        print("Invalid mode selected. Please use 'train' or 'inference'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN Training and Inference")
    args = parser.parse_args()

    args.model_steps = [
        ("dqn", 50000),
        ("dqn", 20000),
        ("2dqn", 50000),
        ("2dqn", 20000),
        ("ddqn", 50000),
        ("ddqn", 20000),
        ("per-dqn", 50000),
        ("per-dqn", 20000),
        ("ppo", 50000),
        ("ppo", 20000),
        ("tppo", 50000),
        ("tppo", 20000),
        ("icm-ppo", 50000),
        ("icm-ppo", 20000),
    ]

    args.mode = "train"
    main(args)

