import pandas as pd

from config import Config
from models.roadnet import Roadnet
from q_learning.genetic_optimiser import GeneticOptimizer, OptimizerParams
from q_learning.q_learner import LearningParams


def load_n_best_solutions(filename, n):
    with pd.HDFStore(filename) as store:
        dataframe = store["dataframe"]

    params = []
    n_best = dataframe.sort_values(by="fitness", ascending=False).head(n)
    for solution in n_best.iterrows():
        params.append(LearningParams.from_list(list(solution[1][1:-1])))

    return params


if __name__ == "__main__":
    config = Config()
    with open(config.roadnet_path) as roadnetFile:
        roadnetJson = roadnetFile.read()
        roadnet = Roadnet.from_json(roadnetJson)

    filename = "output/ga_results_01_24_07_08_47.hdf5"
    take_n_best_from_file = 5
    initial_population = load_n_best_solutions(filename, take_n_best_from_file)

    solutions_per_population = 12

    optimizer_params = OptimizerParams(
        simulation_steps=1000,
        fitness_calculation_last_n_steps=200,
        simulation_repetitions=2,
        number_generations=2,
        solutions_per_population=solutions_per_population,
        random_mutation_val=0.05,
        keep_parents=2,
        num_parents_mating=solutions_per_population // 2.25,
        initial_population=initial_population,
    )

    optimiser = GeneticOptimizer(
        params=optimizer_params,
        qlearner_params={
            "config": config,
            "roadnet": roadnet,
            "random_steps_number": 0,
        },
    )

    optimiser.run()
