from collections import defaultdict
import cityflow
from config import Config
from models.roadnet import Roadnet
from environment.cityflow_env import CityFlowEnv
import numpy as np
import matplotlib.pyplot as plt

from q_learning.q_learner import QLearner

if __name__ == "__main__":
    config = Config()
    with open(config.roadnet_path) as roadnetFile:
        roadnetJson = roadnetFile.read()
        roadnet = Roadnet.from_json(roadnetJson)

    learner = QLearner(
        config=config, roadnet=roadnet, alpha=0.15, gamma=0.95, epsilon=0.1, bin_count=5
    )
    learner.learn(steps=500, progress=True)

    avg_rewards = learner.avg_rewards

    plt.plot(avg_rewards)
    plt.title(f"best_reward={max(avg_rewards)}")

    name = "q-learning2"
    plt.savefig(f"plots/{name}.png")

    plt.show()
