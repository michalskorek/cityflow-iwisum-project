import math
import numpy as np
import pickle
from tqdm import tqdm

from collections import defaultdict

from environment.cityflow_env import CityFlowEnv


class QLearner:
    def __init__(self, roadnet, config, alpha, gamma, epsilon, bin_count):
        self.roadnet = roadnet
        self.config = config

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.max_steps = 500

        self.q = defaultdict(lambda: np.random.uniform(0.2, 0.3))
        self.avg_rewards = list()

        self.environment = None
        self._reset_env()

        self.attempt_no = 1
        self.lower_bound = [0] * 20
        self.upper_bounds = [36] * 20
        self.bins = list(np.linspace(0, 36, bin_count + 2)[1:-1])

        self.important_lanes = [
            0,
            1,
            2,
            3,
            4,
            7,
            8,
            9,
            10,
            11,
            42,
            43,
            44,
            45,
            46,
            49,
            50,
            51,
            52,
            53,
        ]

        self.steps = []

    def learn(self, steps, progress=False):
        for i in tqdm(range(steps), disable=not progress):
            avg_reward = self.attempt()
            self.avg_rewards.append(avg_reward)

    def attempt(self):
        self._reset_env()
        observation = self._discretise(self._get_state_vector())
        terminated = False
        reward_sum = 0

        steps = 0

        while steps < self.max_steps and not terminated:
            action = self._pick_action(steps, observation)
            reward = self.environment.step([action])
            reward = sum(reward)
            steps += 1
            terminated = self._terminate()

            new_observation = self._get_state_vector()
            new_observation = self._discretise(new_observation)

            self._update_knowledge(action, observation, new_observation, reward)

            observation = new_observation
            reward_sum += reward

        avg_reward = reward_sum / steps

        self.attempt_no += 1
        self.steps.append(steps)
        return avg_reward

    def _get_state_vector(self):
        return [
            lane
            for i, lane in enumerate(self.environment.get_states()[0])
            if i in self.important_lanes
        ]

    def _terminate(self):
        full_lane_length = 36
        return any(obs == full_lane_length for obs in self._get_state_vector())

    def _discretise(self, observation):
        return tuple(np.digitize(obs, self.bins) for obs in observation)

    def _pick_action(self, step, observation):
        if np.random.random() > self.epsilon:
            best_reward = -1000

            for potential_action in range(8):
                if self.q[(observation, potential_action)] > best_reward:
                    action = potential_action
                    best_reward = self.q[(observation, action)]
        else:
            action = np.random.choice(range(8))

        return action

    def _update_knowledge(self, action, observation, new_observation, reward):
        best_potential_reward = -1000

        for a in range(8):
            best_potential_reward = max(
                best_potential_reward, self.q[(new_observation, a)]
            )

        self.q[(observation, action)] *= 1 - self.alpha
        self.q[(observation, action)] += self.alpha * (
            reward + self.gamma * best_potential_reward
        )

    def _reset_env(self):
        self.environment = CityFlowEnv(self.roadnet, self.config)
