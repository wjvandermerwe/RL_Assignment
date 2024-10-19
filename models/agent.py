from grid2op.Agent import BaseAgent

class RLAgent(BaseAgent):
    def __init__(self, model, gym_env):
        super().__init__(action_space=gym_env.action_space)
        self.gym_env = gym_env
        self.model = model

    def act(self, obs, reward, done):
        gym_obs = self.gym_env.observation_space.to_gym(obs)
        gym_act = self.model._sample_action(gym_obs)
        grid2op_act = self.gym_env.action_space.from_gym(gym_act[0])
        return grid2op_act