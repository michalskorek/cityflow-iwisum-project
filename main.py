from config import Config
from models.roadnet import Roadnet
import matplotlib.pyplot as plt

from q_learning.q_learner import QLearner

def step_average(data, window=3):
    averages = []
    for i in range(len(data) - window + 1):
        averages.append(sum(data[i:i+window]) / window)
    return averages


if __name__ == "__main__":
    config = Config()
    with open(config.roadnet_path) as roadnetFile:
        roadnetJson = roadnetFile.read()
        roadnet = Roadnet.from_json(roadnetJson)

    learner = QLearner(
        config=config, roadnet=roadnet, alpha=0.15, gamma=0.95, start_epsilon=1, bin_count=5, random_steps_number = 100, epsilon_min=0.01, epsilon_decay_rate=0.99
    )
    steps = 500
    learner.learn(steps=steps, progress=True)

    avg_rewards = learner.avg_rewards
    step_avg = step_average(avg_rewards, steps//50)
    plt.plot(step_avg)
    plt.title(f"best_reward={max(avg_rewards)}")

    plt.savefig(f"plots/{str(learner)}.png")

    plt.show()
