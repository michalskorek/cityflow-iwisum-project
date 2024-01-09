import math
import numpy as np
import pickle
from tqdm import tqdm

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Params:
    steps: int
    alpha: tuple
    gamma: int
    epsilon: tuple
    bins: list[int]

    @staticmethod
    def from_genes(genes):
        return Params(
            steps=genes[0],
            alpha=genes[1],
            gamma=genes[2],
            epsilon=genes[3],
            bins=[int(b) for b in genes[4:]],
        )

    def to_list(self):
        bins = list(self.bins)
        return [self.steps, self.alpha, self.gamma, self.epsilon] + bins

    # @staticmethod
    # def random(alpha, gamma, epsilon, bin_sizes):
    #     return Params(
    #         alpha=np.random.uniform(alpha[0], alpha[1]),
    #         gamma=np.random.uniform(gamma[0], gamma[1]),
    #         epsilon=np.random.uniform(epsilon[0], epsilon[1]),
    #         bins=[np.random.choice(bin_sizes) for _ in range(4)],
    #     )


class QLearner:
    def __init__(self, params: Params, sarsa=False):
        self.params = params
        self.sarsa = sarsa

        self.q = defaultdict(lambda: np.random.uniform(0.2, 0.3))
        self.rewards = list()

        self.environment = None

        self.attempt_no = 1
        self.upper_bounds = [
            2.4,
            0.5,
            0.2095,
            math.radians(50),
        ]
        self.lower_bounds = [
            -2.4,
            -0.5,
            -0.2095,
            -math.radians(50),
        ]

        self.bins = [
            np.linspace(
                self.lower_bounds[i], self.upper_bounds[i], self.params.bins[i] + 1
            )[1:-1]
            for i in range(4)
        ]

        self.steps = []

    def learn(self, progress=False):
        for i in tqdm(range(self.params.steps), disable=not progress):
            reward_sum = self.attempt()
            self.rewards.append(reward_sum)

    def attempt(self):
        observation = self.discretise(self.environment.reset()[0])
        terminated, truncated = False, False
        reward_sum = 0.0

        steps = 0

        while not truncated and not terminated:
            self.environment.render()
            action = self.pick_action(steps, observation)
            (
                new_observation,
                reward,
                terminated,
                truncated,
                info,
            ) = self.environment.step(action)
            steps += 1

            new_observation = self.discretise(new_observation)

            alpha = self.get_alpha(steps)

            if not self.sarsa:
                self.update_knowledge_q_learn(
                    alpha, action, observation, new_observation, reward
                )
            else:
                next_action = self.pick_action(steps, new_observation)
                self.update_knowledge_sarsa(
                    alpha, action, next_action, observation, new_observation, reward
                )

            observation = new_observation
            reward_sum += reward
        self.attempt_no += 1
        self.steps.append(steps)
        return reward_sum

    def discretise(self, observation):
        # Potrzebne bo q learning nie działa na zbiorach ciągłych
        # żeby się nauczyć musimy trafiać na te same wartości
        # przechodzimy z liczb rzeczywistych do całkiwitych
        # np.linspace, np.digitize

        discretised = [
            np.digitize(elem, self.bins[i]) for i, elem in enumerate(observation)
        ]
        return tuple(discretised)

    def pick_action(self, step, observation):
        # Pierwszy krok algorytmu

        # return self.environment.action_space.sample()
        if np.random.random() > self.get_epsilon(step):
            action = 0 if self.q[(observation, 0)] > self.q[(observation, 1)] else 1
        else:
            action = np.random.choice([0, 1])

        return action

    def update_knowledge_q_learn(
        self, alpha, action, observation, new_observation, reward
    ):
        self.q[(observation, action)] *= 1 - alpha
        self.q[(observation, action)] += alpha * (
            reward
            + self.params.gamma
            * max(self.q[(new_observation, 0)], self.q[(new_observation, 1)])
        )

    def update_knowledge_sarsa(
        self, alpha, action, next_action, observation, new_observation, reward
    ):
        self.q[(observation, action)] = self.q[(observation, action)] + alpha * (
            reward + self.params.gamma * self.q[(new_observation, next_action)]
        )

    def save_q(self, filename):
        with open(f"{filename}.pkl", "wb") as outp:
            dict_items = list(self.q.items())

            pickle.dump(dict_items, outp, pickle.HIGHEST_PROTOCOL)

    def load_q(self, filename):
        with open(f"{filename}.pkl", "rb") as inp:
            dict_items = pickle.load(inp)

            for k, v in dict_items:
                self.q[k] = v

    def get_alpha(self, step):
        initial, end, half_steps = self.params.alpha
        return end + (initial - end) * 2 ** (-step / half_steps)

    def get_epsilon(self, step):
        initial, end, half_steps = self.params.alpha
        return end + (initial - end) * 2 ** (-step / half_steps)
