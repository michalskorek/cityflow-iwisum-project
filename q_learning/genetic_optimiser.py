from dataclasses import dataclass
import dataclasses
import datetime
import os
import time
import uuid
import pandas as pd
import pygad
import warnings

from os.path import isfile, join
from q_learning.q_learner import LearningParams, QLearner


warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


@dataclass
class OptimizerParams:
    simulation_steps: int
    fitness_calculation_last_n_steps: int

    simulation_repetitions: int

    output_dir: str = "output"

    initial_population: list = None

    # PyGAD GA params

    keep_parents: int = 1
    solutions_per_population: int = 32
    number_generations: int = 3
    num_parents_mating: int = 10

    random_mutation_val: int = 0.05
    mutation_num_genes: int = 1

    parallel_processes: int = 16


class GeneticOptimizer(pygad.GA):
    def __init__(self, params: OptimizerParams, qlearner_params):
        self.optimizer_params = params
        self.qlearner_params = qlearner_params

        start_date = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")

        self.output_dir = params.output_dir
        self.output_path = f"{params.output_dir}/ga_results_{start_date}.hdf5"

        self._init_output_file()
        self.metadata = {
            "optimizer_params": self.optimizer_params,
            "random_steps_number": self.qlearner_params["random_steps_number"],
        }
        self.start_time = None

        initial_population = params.initial_population if params.initial_population else []
        initial_population += [
            LearningParams.random()
            for _ in range(params.solutions_per_population)
        ]
        initial_population = initial_population[:params.solutions_per_population]
        initial_population = [p.to_list() for p in initial_population]

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
            mutation_num_genes=params.mutation_num_genes,
            random_mutation_min_val=params.random_mutation_val * -1,
            random_mutation_max_val=params.random_mutation_val,
            parallel_processing=["process", params.parallel_processes],
            on_generation=GeneticOptimizer.on_generation,
        )

    def run(self):
        self.start_time = time.time()
        super().run()

    @staticmethod
    def _fitness_function(self, solution, solution_idx):
        params = LearningParams.from_list(solution)

        all_rewards = []
        fitnesses = []

        for _ in range(self.optimizer_params.simulation_repetitions):
            learner = QLearner(**{**{"params": params}, **self.qlearner_params})
            learner.learn(self.optimizer_params.simulation_steps)
            rewards = learner.avg_rewards

            all_rewards.append(rewards)
            fitnesses.append(self._calculate_fitness(rewards))

        fitness = sum(fitnesses) / len(fitnesses)
        all_rewards_avg = [sum(r) / len(r) for r in zip(*all_rewards)]

        df = pd.DataFrame(columns=GeneticOptimizer._get_columns())
        df.loc[0] = [fitness] + params.to_list() + [all_rewards_avg]
        df.to_csv(f"output/tmp_{str(uuid.uuid4())[:12]}.csv")

        return fitness

    def _calculate_fitness(self, rewards):
        n_steps = self.optimizer_params.fitness_calculation_last_n_steps
        return sum(rewards[-n_steps:]) / n_steps

    def _init_output_file(self):
        if os.path.exists(self.output_path):
            return

        dataframe = pd.DataFrame(columns=GeneticOptimizer._get_columns())

        with pd.HDFStore(self.output_path) as store:
            store.put("dataframe", dataframe)

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

        with pd.HDFStore(self.output_path) as store:
            store.put("dataframe", dataframe)
            store.get_storer("dataframe").attrs.metadata = self.metadata

    @staticmethod
    def _get_columns():
        return (
            ["fitness"]
            + [f.name for f in dataclasses.fields(LearningParams)]
            + ["all_rewards"]
        )

    @staticmethod
    def on_generation(self):
        generation_duration = time.time() - self.start_time
        self.start_time = time.time()
        print(
            f"Generation {self.generations_completed} done in {generation_duration:.2f}s"
        )
        self._update_results()
