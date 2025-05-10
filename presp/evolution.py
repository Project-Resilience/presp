"""
Class responsible for running the overall evolutionary process.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from presp.evaluator import Evaluator
from presp import nsga2_utils
from presp.prescriptor import Prescriptor, PrescriptorFactory


class Evolution:
    """
    Handles the ESP evolutionary process in run_evolution.
    Creates an initial population, then for each time .step() is called, runs a single generation of evolution.
    """
    def __init__(self,
                 n_generations: int,
                 population_size: int,
                 remove_population_pct: float,
                 n_elites: int,
                 mutation_rate: float,
                 mutation_factor: float,
                 save_path: str,
                 prescriptor_factory: PrescriptorFactory,
                 evaluator: Evaluator,
                 seed_path: str = None,
                 validator: Evaluator = None,
                 val_interval: int = 0,
                 save_all: bool = False):
        """
        :param n_generations: The number of generations to run evolution for. Generations start counting at 1.
        :param population_size: The size of the total population including elites.
        :param remove_population_pct: The bottom percentage of the population to not use as parents.
        :param n_elites: The number of elites to keep from the previous generation. When this is set to 0, all
        candidates are considered elite.
        :param mutation_rate: The rate at which to mutate each parameter.
        :param mutation_factor: The factor by which to mutate each parameter.
        :param save_path: The path to save the results to.
        :param prescriptor_factory: The factory to create prescriptors with.
        :param evaluator: The evaluator used to evaluate the prescriptors.
        :param seed_path: Optional path to a file holding seed candidates.
        :param validator: Optional evaluator class used to validate the population every val_interval steps.
        :param val_interval: The interval at which to validate the population.
        :param save_all: Whether to save all candidates or just rank 1 candidates.
        """
        # Evolution parameters
        self.n_generations = n_generations
        self.population_size = population_size
        self.remove_population_pct = remove_population_pct
        self.n_elites = n_elites
        self.mutation_rate = mutation_rate
        self.mutation_factor = mutation_factor
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.save_all = save_all
        self.seed_path = Path(seed_path) if seed_path is not None else None
        self.val_interval = val_interval

        # Implemented classes
        self.prescriptor_factory = prescriptor_factory
        self.evaluator = evaluator
        self.validator = validator

        # Population tracking
        self.population = []
        self.generation = 1

        # Set up results dataframe
        results_df = pd.DataFrame(columns=["gen",
                                           "cand_id",
                                           "parents",
                                           "cv",
                                           "rank",
                                           "distance",
                                           *self.evaluator.outcomes])
        results_df.to_csv(self.save_path / "results.csv", index=False)
        if self.validator:
            val_df = pd.DataFrame(columns=["gen", "cand_id", *self.validator.outcomes])
            val_df.to_csv(self.save_path / "validation.csv", index=False)
        self.save_cache = []

    def create_initial_population(self):
        """
        Creates initial population from seed dir, randomly initializes the rest.
        """
        self.population = []
        if self.seed_path is not None:
            print(f"Loading seeds from {self.seed_path}")
            seed_dict = self.prescriptor_factory.load_population(self.seed_path)

            def sort_cand_id(cand_id):
                return int(cand_id.split("_")[0]), int(cand_id.split("_")[1])
            sorted_ids = sorted(seed_dict.keys(), key=sort_cand_id)

            # We don't want duplicates cand_ids messing things up in our population
            # We make the seeds generation 0 to show they are seeds
            for i, cand_id in enumerate(sorted_ids):
                candidate = seed_dict[cand_id]
                candidate.cand_id = "0_" + str(i)
                self.population.append(candidate)

        print(f"Loaded {len(self.population)} candidates from seed dir.")

        for i in range(len(self.population), self.population_size):
            candidate = self.prescriptor_factory.random_init()
            candidate.cand_id = f"{self.generation}_{i}"
            self.population.append(candidate)

        # TODO: This is a really hacky way to do this.
        self.evaluator.update_predictor([None for _ in range(self.n_elites)])

        self.evaluator.evaluate_population(self.population)
        self.population = self.sort_pop(self.population)
        self.record_results()
        if self.validator:
            self.validate()
        self.generation += 1

    def selection(self, sorted_population: list[Prescriptor]) -> list[Prescriptor]:
        """
        Takes two random parents and compares their indices since this is a measure of their performance.
        NOTE: It is possible for this function to select the same parent twice.
        :param sorted_population: The population of prescriptors sorted by rank and distance.
        :return: A tuple of two parents to be used for crossover.
        """
        idx1 = np.min(np.random.randint(0, len(sorted_population), size=2))
        idx2 = np.min(np.random.randint(0, len(sorted_population), size=2))
        return [sorted_population[idx1], sorted_population[idx2]]

    def create_pop(self, population: list[Prescriptor], remove_population_pct: float = 0) -> list[Prescriptor]:
        """
        Creates a population by selecting parents from the top candidates, crossing them over,
        then mutating the children.
        :param population: The filtered population to select parents from.
        NOTE: The population must be sorted.
        :param remove_population_pct: The percentage of the population to remove from the bottom.
        :return: The new population
        """
        top_pop = population[:int(len(population) * (1 - remove_population_pct))]
        next_pop = []
        n_children = self.population_size - self.n_elites
        while len(next_pop) < n_children:
            # Select, crossover, mutate
            parents = self.selection(top_pop)
            children = self.prescriptor_factory.crossover(parents)
            for child in children:
                self.prescriptor_factory.mutation(child, self.mutation_rate, self.mutation_factor)

            # Tag children with parent ids
            parent_ids = [parent.cand_id for parent in parents]
            for child in children:
                child.parents = parent_ids

            next_pop.extend(children)
        return next_pop[:n_children]

    def sort_pop(self, population: list[Prescriptor]) -> list[Prescriptor]:
        """
        Sort by rank and distance according to NSGA-II.
        :param population: The population to sort.
        :return: The sorted population.
        """
        fronts = nsga2_utils.fast_non_dominated_sort(population)
        for front in fronts:
            nsga2_utils.calculate_crowding_distance(front)

        population.sort(key=lambda c: (c.rank, -c.distance))
        return population

    def record_results(self):
        """
        Record results from current population into a CSV file.
        Also keeps track of the best candidates in self.save_cache so we can serialize them later.
        """
        rows = []
        for candidate in self.population:
            # Save to file if it's new. Only save rank 1 candidates if save_all is False.
            cand_gen = int(candidate.cand_id.split("_")[0])
            if (candidate.rank == 1 or self.save_all) and cand_gen == self.generation:
                self.save_cache.append(candidate)
            # Record candidates' results
            row = {
                "gen": self.generation,
                "cand_id": candidate.cand_id,
                "parents": candidate.parents,
                "cv": candidate.cv,
                "rank": candidate.rank,
                "distance": candidate.distance
            }
            for outcome, metric in zip(candidate.outcomes, candidate.metrics):
                row[outcome] = metric
            rows.append(row)

        results_df = pd.DataFrame(rows)
        results_df.to_csv(self.save_path / "results.csv", mode="a", index=False, header=False)

    def validate(self):
        """
        Validates the current population with a validator evaluator.
        """
        rows = []
        # Get validation results manually without setting cand's metrics or outcomes
        valid_results = self.validator.evaluate_subset(self.population, verbose=1)

        for candidate, val_metrics in zip(self.population, valid_results):
            row = {"gen": self.generation, "cand_id": candidate.cand_id}
            for outcome, metric in zip(self.validator.outcomes, val_metrics):
                row[outcome] = metric
            rows.append(row)
        results_df = pd.DataFrame(rows)
        results_df.to_csv(self.save_path / "validation.csv", mode="a", index=False, header=False)

    def step(self):
        """
        Assumes we have a sorted, evaluated population of prescriptors.
        1. Creates the next population from the kept candidates from the last generation.
        2. Evaluates the new population.
        3. Combines the new population with the elites.
        4. Sorts the population.
        """
        elites = self.population[:self.n_elites] if self.n_elites > 0 else self.population
        # Collect data and train predictor for evaluation
        self.evaluator.update_predictor(elites)

        next_pop = self.create_pop(self.population, self.remove_population_pct)
        # Tag new population with candidate IDs
        for i, candidate in enumerate(next_pop):
            candidate.cand_id = f"{self.generation}_{i}"

        next_pop = elites + next_pop
        # Force evaluation if we are updating predictor
        self.evaluator.evaluate_population(next_pop, force=self.n_elites > 0)
        self.population = self.sort_pop(next_pop)
        self.population = self.population[:self.population_size]

        self.record_results()
        if self.validator and (self.generation % self.val_interval == 0 or self.generation == self.n_generations):
            self.validate()
        self.generation += 1

    def run_evolution(self):
        """
        Runs the evolutionary process for n_generations.
        Then saves the Pareto front from each generation to file.
        """
        self.create_initial_population()
        for _ in tqdm(range(self.generation, self.n_generations+1), desc="Running Evolution"):
            self.step()
        # Save all candidates at the end of evolution
        self.prescriptor_factory.save_population(self.save_cache, self.save_path / "population")
