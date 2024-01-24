import dataclasses
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

from collections import defaultdict

from environment.cityflow_env import CityFlowEnv


class SARSALearner:
    def __init__(self, roadnet, config, params, random_steps_number):
        self.roadnet = roadnet
        self.config = config

        self.alpha = params.alpha
        self.gamma = params.gamma
        self.start_epsilon = params.start_epsilon
        self.epsilon_min = params.epsilon_min
        self.epsilon_decay_rate = params.epsilon_decay_rate
        self.bin_count = params.bin_count
        self.bins = np.logspace(0, np.log2(36), params.bin_count, base=2.0)[:-1]

        self.max_steps = 500
        self.attempt_no = 1
        self.avg_rewards = list()
        self.environment = None
        self._reset_env()
        self.random_steps_number = random_steps_number
        self.action_space = self.environment.get_possible_actions(0)
        self.q = defaultdict(lambda: np.zeros(len(self.action_space)))
        self.steps = []

    def __str__(self):
        return f"SARSA_alpha_{self.alpha}_gamma_{self.gamma}_bin_count_{self.bin_count}_random_steps_number_{self.random_steps_number}_epsilon_min_{self.epsilon_min}_epsilon_decay_rate_{self.epsilon_decay_rate}"

    def learn(self, steps, progress=False):
        for i in tqdm(range(steps), disable=not progress):
            avg_reward = self.attempt(i)
            self.avg_rewards.append(avg_reward)

    def attempt(self, episode):
        self._reset_env()
        observation = self._discretise(self._get_state_vector())
        terminated = False
        reward_sum = 0
        epsilon = self._exponential_decay_epsilon(episode)
        steps = 0
        for _ in range(self.random_steps_number):
            self.environment.random_step()

        while steps < self.max_steps and not terminated:
            action = self._pick_action(steps, observation, epsilon)
            reward = self.environment.step([action])
            reward = np.sum(reward)
            next_observation = self._get_state_vector()
            next_observation = self._discretise(next_observation)
            next_action = self._pick_action(steps + 1, next_observation, epsilon)
            self._update_knowledge(action, observation, next_observation, reward, next_action)

            observation = next_observation
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
        state_vector = self._get_state_vector()
        return np.any(state_vector == full_lane_length)

    def _discretise(self, observation):
        return tuple(np.digitize(obs, self.bins) for obs in observation)

    def _pick_action(self, step, observation, epsilon):
        if np.random.random() > epsilon:
            action = np.argmax(self.q[observation])
        else:
            action = np.random.choice(self.action_space)

        return action

    def _update_knowledge(self, action, observation, new_observation, reward, next_action):
        next_potential_reward = self.q[new_observation][next_action]

        self.q[observation][action] *= 1 - self.alpha
        self.q[observation][action] += self.alpha * (
            reward + self.gamma * next_potential_reward
        )

    def _reset_env(self):
        self.environment = CityFlowEnv(self.roadnet, self.config)

    def _exponential_decay_epsilon(self, current_step):
        epsilon = self.start_epsilon * (self.epsilon_decay_rate**current_step)
        return max(epsilon, self.epsilon_min)
