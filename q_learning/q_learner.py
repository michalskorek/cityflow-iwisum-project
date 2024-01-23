import dataclasses
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

from collections import defaultdict

from environment.cityflow_env import CityFlowEnv


@dataclass
class LearningParams:
    alpha: float
    gamma: float
    start_epsilon: float
    epsilon_min: float
    epsilon_decay_rate: float
    bin_count: int

    def to_list(self):
        return [
            self.alpha,
            self.gamma,
            self.start_epsilon,
            self.epsilon_min,
            self.epsilon_decay_rate,
            self.bin_count,
        ]

    @staticmethod
    def from_list(param_list):
        (
            alpha,
            gamma,
            start_epsilon,
            epsilon_min,
            epsilon_decay_rate,
            bin_count,
        ) = param_list
        return LearningParams(
            alpha=alpha,
            gamma=gamma,
            start_epsilon=start_epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay_rate=epsilon_decay_rate,
            bin_count=bin_count,
        )

    @staticmethod
    def get_types():
        return [f.type for f in dataclasses.fields(LearningParams)]

    @staticmethod
    def get_value_space() -> dict:
        lohi = [
            ("alpha", (0, 0.5)),
            ("gamma", (0, 1)),
            ("start_epsilon", (0, 1)),
            ("epsilon_min", (0, 0.15)),
            ("epsilon_decay_rate", (0.9, 1)),
        ]

        value_space = {key: {"low": lo, "high": hi} for (key, (lo, hi)) in lohi}
        value_space["bin_count"] = list(range(1, 19))

        return value_space

    @staticmethod
    def random():
        params = dict()

        for key, value in LearningParams.get_value_space().items():
            if isinstance(value, dict):
                params[key] = np.random.uniform(low=value["low"], high=value["high"])
            elif isinstance(value, list):
                params[key] = np.random.choice(value)

        return LearningParams(**params)


class QLearner:
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
        self.q = defaultdict(lambda: np.array([float("-Inf") for _ in self.action_space]))
        self.steps = []

    def __str__(self):
        return f"QLearner_alpha_{self.alpha}_gamma_{self.gamma}_bin_count_{self.bin_count}_random_steps_number_{self.random_steps_number}_epsilon_min_{self.epsilon_min}_epsilon_decay_rate_{self.epsilon_decay_rate}"

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
        return self.environment.get_states()[0]

    def _terminate(self):
        full_lane_length = 36
        return np.any(obs == full_lane_length for obs in self._get_state_vector())

    def _discretise(self, observation):
        return tuple(np.digitize(obs, self.bins) for obs in observation)

    def _pick_action(self, step, observation, epsilon):
        if np.random.random() > epsilon:
            action = np.argmax(self.q[observation])
        else:
            action = np.random.choice(self.action_space)

        return action

    def _update_knowledge(self, action, observation, new_observation, reward):
        best_potential_reward = np.max(self.q[new_observation])

        self.q[observation][action] *= 1 - self.alpha
        self.q[observation][action] += self.alpha * (
            reward + self.gamma * best_potential_reward
        )

    def _reset_env(self):
        self.environment = CityFlowEnv(self.roadnet, self.config)

    def _exponential_decay_epsilon(self, current_step):
        epsilon = self.start_epsilon * (self.epsilon_decay_rate**current_step)
        return max(epsilon, self.epsilon_min)
