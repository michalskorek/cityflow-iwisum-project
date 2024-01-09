from collections import defaultdict
import cityflow
from config import Config
from models.roadnet import Roadnet
from environment.cityflow_env import CityFlowEnv
import numpy as np
import random
import matplotlib.pyplot as plt


def discretise(state, bins):
    return tuple(np.digitize(s, bins) for s in state)


def pick_action(observation, epsilon, q):
    # Pierwszy krok algorytmu

    # return self.environment.action_space.sample()
    if np.random.random() > epsilon:
        best_reward = -1000

        for potential_action in range(8):
            if q[(observation, potential_action)] > best_reward:
                action = potential_action
                best_reward = q[(observation, action)]
    else:
        action = np.random.choice(range(8))

    return action


def update_knowledge_q_learn(
    alpha, gamma, action, observation, new_observation, reward
):
    best_potential_reward = -1000

    for a in range(8):
        best_potential_reward = max(best_potential_reward, q[(new_observation, a)])

    q[(observation, action)] *= 1 - alpha
    q[(observation, action)] += alpha * (reward + gamma * best_potential_reward)


if __name__ == "__main__":
    config = Config()
    with open(config.roadnet_path) as roadnetFile:
        roadnetJson = roadnetFile.read()
        roadnet = Roadnet.from_json(roadnetJson)

    env = CityFlowEnv(roadnet, config)

    is_ever_non_zero = [False] * len(env.get_states()[0])

    important_lanes = [
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

    rewards = []
    next_action = 0
    idle_action_steps = 3
    active_action_steps = 40
    current_action_steps = 0

    idle_actions = [1, 3, 5, 7]
    active_actions = [0, 2, 4, 6]

    lower_bound = [0] * 20
    upper_bounds = [36] * 20

    q = defaultdict(lambda: np.random.uniform(0.2, 0.3))

    alpha = 0.3
    gamma = 0.9
    epsilon = 0.1
    bins = [2, 10]

    observation = discretise([0] * 20, bins)

    for step in range(10000):
        # states_vect = env.get_states()

        # if (next_action in idle_actions and current_action_steps == idle_action_steps) or (next_action in active_actions and current_action_steps == active_action_steps):
        #     next_action += 1
        #     next_action %= 8
        #     current_action_steps = 0

        # next_action = step // 60 % 8
        # next_action = random.randint(0, 7)

        action = pick_action(observation, epsilon, q)

        reward = sum(env.step([next_action]))
        # current_action_steps += 1

        state_vector = [
            lane for i, lane in enumerate(env.get_states()[0]) if i in important_lanes
        ]
        new_observation = discretise(state_vector, bins)

        update_knowledge_q_learn(
            alpha, gamma, action, observation, new_observation, reward
        )

        observation = new_observation

        if step % 100 == 0:
            print(state_vector)

        rewards.append(reward)

    avg_reward = sum(rewards) / len(rewards)

    plt.plot(rewards)
    plt.title(f"avg_reward={avg_reward}")

    name = "q-learning"
    plt.savefig(f"plots/{name}.png")

    plt.show()
