import dataclasses
from config import Config
from models.roadnet import Roadnet
import matplotlib.pyplot as plt
import uuid
from q_learning.genetic_optimiser import GeneticOptimizer, OptimizerParams
from q_learning.odd_learner import OddLearner

from q_learning.q_learner import LearningParams, QLearner
from q_learning.random_learner import RandomLearner


def step_average(data, window=3):
    averages = []
    for i in range(len(data) - window + 1):
        averages.append(sum(data[i : i + window]) / window)
    return averages


if __name__ == "__main__":
    # import os
    # from os.path import isfile, join

    # import pandas as pd

    # output_dir = "output"
    # output_path = os.path.join(output_dir, "genetic_optimizer_results_9c80a.hdf5")

    # partial_files = [
    #     f
    #     for f in os.listdir(output_dir)
    #     if isfile(join(output_dir, f))
    #     if f[:3] == "tmp"
    # ]

    # with pd.HDFStore(output_path) as store:
    #     dataframe = store["dataframe"]

    # for filename in partial_files:
    #     path = os.path.join(output_dir, filename)
    #     partial_df = pd.read_csv(path, index_col=0)

    #     dataframe.loc[len(dataframe.index)] = partial_df.loc[0]

    #     os.remove(path)

    # with pd.HDFStore(output_path) as storedata:
    #     storedata.put("dataframe", dataframe)

    config = Config()
    with open(config.roadnet_path) as roadnetFile:
        roadnetJson = roadnetFile.read()
        roadnet = Roadnet.from_json(roadnetJson)

    optimizer_params = OptimizerParams(
        simulation_steps=10,
        fitness_calculation_last_n_steps=50,
        simulation_repetitions=4,
        number_generations=1,
    )

    optimiser = GeneticOptimizer(
        params=optimizer_params,
        qlearner_params={
            "config": config,
            "roadnet": roadnet,
            "random_steps_number": 1,
        },
    )

    optimiser.run()

    # # learner = QLearner(
    # #     config=config, roadnet=roadnet, alpha=0.15, gamma=0.95, start_epsilon=1, bin_count=5, random_steps_number = 100, epsilon_min=0.01, epsilon_decay_rate=0.99
    # # )
    # learner = RandomLearner(config=config, roadnet=roadnet, random_steps_number=100)
    # learner.learn(steps=steps, progress=True)

    # avg_rewards = learner.avg_rewards
    # step_avg = step_average(avg_rewards, steps // 50)
    # plt.plot(step_avg)
    # plt.title(f"best_avg_reward={max(step_avg)}")

    # uuid = str(uuid.uuid4())[:3]
    # plt.savefig(f"plots/{str(learner)}_{uuid}.png")

    # # plt.show()

    # print(list(LearningParams.get_value_space().values()))
