from dataclasses import dataclass
import dataclasses
import multiprocessing
import os
import time
import uuid
import numpy as np
import pandas as pd
import pygad

from os.path import isfile, join
from q_learning.q_learner import LearningParams, QLearner


@dataclass
class OptimizerParams:
    simulation_steps: int
    fitness_calculation_last_n_steps: int

    simulation_repetitions: int

    output_dir = "output"

    # PyGAD GA params

    keep_parents = 2
    solutions_per_population = 32
    number_generations: int = 3
    num_parents_mating = 10

    random_mutation_min_val = 0.0
    random_mutation_max_val = 0.05
    mutation_percent_genes = 30

    parallel_processes = 16


class GeneticOptimizer(pygad.GA):
    def __init__(self, params: OptimizerParams, qlearner_params):
        self.optimizer_params = params
        self.qlearner_params = qlearner_params

        self.uuid = str(uuid.uuid4())[:5]

        self.output_dir = params.output_dir
        self.output_path = (
            f"{params.output_dir}/genetic_optimizer_results_{self.uuid}.hdf5"
        )

        self._init_output_file()

        initial_population = [
            LearningParams.random().to_list()
            for _ in range(params.solutions_per_population)
        ]

        super().__init__(
            num_generations=params.number_generations,
            num_parents_mating=params.num_parents_mating,
            fitness_func=self._fitness_function,
            initial_population=initial_population,
            num_genes=len(LearningParams.get_types()),
            gene_type=LearningParams.get_types(),
            gene_space=list(LearningParams.get_value_space().values()),
            parent_selection_type="rws",
            crossover_type="uniform",
            keep_parents=params.keep_parents,
            mutation_type="random",
            random_mutation_min_val=params.random_mutation_min_val,
            random_mutation_max_val=params.random_mutation_max_val,
            mutation_percent_genes=params.mutation_percent_genes,
            parallel_processing=["process", params.parallel_processes],
            on_fitness=self.on_generation
        )

    @staticmethod
    def _fitness_function(self, solution, solution_idx):
        params = LearningParams.from_list(solution)

        fitnesses = []

        for _ in range(self.optimizer_params.simulation_repetitions):
            learner = QLearner(**{**{"params": params}, **self.qlearner_params})
            learner.learn(self.optimizer_params.simulation_steps)
            rewards = learner.avg_rewards

            fitnesses.append(self._calculate_fitness(rewards))

        fitness = sum(fitnesses) / len(fitnesses)

        df = pd.DataFrame(
            columns=["Fitness"] + [f.name for f in dataclasses.fields(LearningParams)]
        )
        df.loc[0] = [fitness] + params.to_list()
        df.to_csv(f"output/tmp_{str(uuid.uuid4())[:12]}.csv")

        return fitness

    def _calculate_fitness(self, rewards):
        n_steps = self.optimizer_params.fitness_calculation_last_n_steps
        return sum(rewards[-n_steps:]) / n_steps

    def _init_output_file(self):
        columns = ["Fitness"] + [f.name for f in dataclasses.fields(LearningParams)]
        dataframe = pd.DataFrame(columns=columns)

        metadata = {
            "optimizer_params": self.optimizer_params,
            "random_steps_number": self.qlearner_params["random_steps_number"],
        }

        with pd.HDFStore(self.output_path) as store:
            store.put("dataframe", dataframe)
            store.get_storer("dataframe").attrs.metadata = metadata

    def _update_results(self):
        partial_files = [
            f
            for f in os.listdir(self.output_dir)
            if isfile(join(self.output_dir, f))
            if f[:3] == "tmp"
        ]

        with pd.HDFStore(self.output_path) as store:
            dataframe = store["dataframe"]

        for filename in partial_files:
            path = os.path.join(self.output_dir, filename)
            partial_df = pd.read_csv(path, index_col=0)

            dataframe.loc[len(dataframe.index)] = partial_df.loc[0]

            os.remove(path)

        with pd.HDFStore(self.output_path) as storedata:
            storedata.put("dataframe", dataframe)

    def on_generation(self):
        print("on generation")

        try:
            self._update_results()
        except:
            RuntimeError("asdf")
