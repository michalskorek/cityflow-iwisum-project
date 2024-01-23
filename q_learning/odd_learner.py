import numpy as np
from tqdm import tqdm

from collections import defaultdict

from environment.cityflow_env import CityFlowEnv

"""
Ony chooses odd-numbered actions - skips all red light phases
"""


class OddLearner:
    def __init__(self, roadnet, config, random_steps_number):
        self.roadnet = roadnet
        self.config = config

        self.max_steps = 500
        self.attempt_no = 1
        self.avg_rewards = list()
        self.environment = None
        self._reset_env()
        self.random_steps_number = random_steps_number
        self.action_space = self.environment.get_possible_actions(0)
        self.q = defaultdict(lambda: np.array(self.action_space))
        self.steps = []

    def __str__(self):
        return f"OddLearner_random_steps_number_{self.random_steps_number}"

    def learn(self, steps, progress=False):
        for i in tqdm(range(steps), disable=not progress):
            avg_reward = self.attempt(i)
            self.avg_rewards.append(avg_reward)

    def attempt(self, episode):
        self._reset_env()
        terminated = False
        reward_sum = 0

        steps = 0
        for _ in range(self.random_steps_number):
            self.environment.random_step()

        while steps < self.max_steps and not terminated:
            action = np.random.choice(self.action_space[0::2])
            reward = self.environment.step([action])
            reward = np.sum(reward)
            steps += 1
            terminated = self._terminate()

            reward_sum += reward

        avg_reward = reward_sum / steps

        self.attempt_no += 1
        self.steps.append(steps)
        return avg_reward

    def _get_state_vector(self):
        return self.environment.get_states()[0]

    def _terminate(self):
        full_lane_length = 36
        return np.any(obs == full_lane_length for obs in self._get_state_vector())

    def _reset_env(self):
        self.environment = CityFlowEnv(self.roadnet, self.config)
