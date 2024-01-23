from config import Config
from models.roadnet import Roadnet
from q_learning.genetic_optimiser import GeneticOptimizer, OptimizerParams


if __name__ == "__main__":
    config = Config()
    with open(config.roadnet_path) as roadnetFile:
        roadnetJson = roadnetFile.read()
        roadnet = Roadnet.from_json(roadnetJson)

    optimizer_params = OptimizerParams(
        simulation_steps=1000,
        fitness_calculation_last_n_steps=150,
        simulation_repetitions=4,
        number_generations=20,
        solutions_per_population=32,
        random_mutation_val=0.05,
        keep_parents=4,
    )

    optimiser = GeneticOptimizer(
        params=optimizer_params,
        qlearner_params={
            "config": config,
            "roadnet": roadnet,
            "random_steps_number": 100,
        },
    )

    optimiser.run()
