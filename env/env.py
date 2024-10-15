import gymnasium as gym
import grid2op
import numpy as np
import torch
import torch.nn.functional as F
from grid2op import gym_compat
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward
from grid2op.gym_compat import ScalerAttrConverter, ContinuousToDiscreteConverter, DiscreteActSpace, BoxGymObsSpace
from gymnasium.spaces import Box
from gymnasium.wrappers import HumanRendering
from lightsim2grid import LightSimBackend

class Gym2OpEnv(gym.Env):
    def __init__(
            self
    ):
        super().__init__()

        self._backend = LightSimBackend()
        self._env_name = "l2rpn_case14_sandbox"  # DO NOT CHANGE

        action_class = PlayableAction
        observation_class = CompleteObservation
        reward_class = CombinedScaledReward  # Setup further below

        # DO NOT CHANGE Parameters
        # See https://grid2op.readthedocs.io/en/latest/parameters.html
        p = Parameters()
        p.MAX_SUB_CHANGED = 4  # Up to 4 substations can be reconfigured each timestep
        p.MAX_LINE_STATUS_CHANGED = 4  # Up to 4 powerline statuses can be changed each timestep

        # Make grid2op env
        self._g2op_env = grid2op.make(
            self._env_name, backend=self._backend, test=False,
            action_class=action_class, observation_class=observation_class,
            reward_class=reward_class, param=p
        )

        ##########
        # REWARD #
        ##########
        # NOTE: This reward should not be modified when evaluating RL agent
        # See https://grid2op.readthedocs.io/en/latest/reward.html
        cr = self._g2op_env.get_reward_instance()
        cr.addReward("N1", N1Reward(), 1.0)
        cr.addReward("L2RPN", L2RPNReward(), 1.0)
        # reward = N1 + L2RPN
        cr.initialize(self._g2op_env)
        ##########

        self._gym_env = gym_compat.GymEnv(self._g2op_env)

        self.setup_observations()
        self.setup_actions()

        self.observation_space = self._gym_env.observation_space
        self.action_space = self._gym_env.action_space

    def setup_observations(self):
        obs, _ = self._gym_env.reset()
        self._gym_env.observation_space = BoxGymObsSpace(self._g2op_env.observation_space,
                                                         attr_to_keep=[
                                                             "gen_p", "load_p", "topo_vect",
                                                             "rho", "actual_dispatch", "connectivity_matrix",
                                                             "line_status",  # New addition
                                                             "v_or",  # Voltage observation
                                                         ],
                                                         divide={"gen_p": self._g2op_env.gen_pmax,
                                                                 "load_p": obs['load_p'],
                                                                 "actual_dispatch": self._g2op_env.gen_pmax},
                                                         functs={"connectivity_matrix": (
                                                             lambda grid2obs: grid2obs.connectivity_matrix().flatten(),
                                                             0., 1., None, None,
                                                         )}
                                                         )

    def setup_actions(self):
        reencoded_act_space = DiscreteActSpace(self._g2op_env.action_space,
                                               attr_to_keep=[
                                                   "set_line_status",
                                                   "set_bus",
                                                   "redispatch",
                                                   "curtail",  # New addition
                                                   "set_storage"  # New addition
                                               ])
        self._gym_env.action_space = reencoded_act_space


    def reset(self, seed=None):
        return self._gym_env.reset(seed=seed, options=None)

    def step(self, action):
        return self._gym_env.step(action)

    def render(self):
        # TODO: Modify for your own required usage
        return self._g2op_env.render()




