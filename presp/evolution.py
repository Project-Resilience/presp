"""
Class responsible for running the overall evolutionary process.
"""
from pathlib import Path
import random

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
                 seed_dir: str,
                 prescriptor_factory: PrescriptorFactory,
                 evaluator: Evaluator,
                 validator: Evaluator = None,
                 val_interval: int = 0,
                 save_all: bool = False):

        self.n_generations = n_generations
        self.population_size = population_size
        self.remove_population_pct = remove_population_pct
        self.n_elites = n_elites
        self.mutation_rate = mutation_rate
        self.mutation_factor = mutation_factor
        self.save_path = Path(save_path)
        self.save_all = save_all
        self.seed_dir = Path(seed_dir) if seed_dir is not None else None
        self.val_interval = val_interval

        self.prescriptor_factory = prescriptor_factory
        self.evaluator = evaluator
        self.validator = validator

        self.population = []
        self.generation = 1

    def create_initial_population(self):
        """
        Creates initial population from seed dir, randomly initializes the rest.
        """
        self.population = []
        if self.seed_dir is not None:
            for seed_file in self.seed_dir.glob("*"):
                candidate = self.prescriptor_factory.load(seed_file)
                candidate.cand_id = seed_file.stem
                self.population.append(candidate)

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
        idx1 = min(random.choices(range(len(sorted_population)), k=2))
        idx2 = min(random.choices(range(len(sorted_population)), k=2))
        return sorted_population[idx1], sorted_population[idx2]

    def create_pop(self, population: list[Prescriptor]) -> list[Prescriptor]:
        """
        Creates a population by selecting parents, crossing them over, then mutating the children.
        :param population: The filtered population to select parents from.
        :return: The new population
        """
        next_pop = []
        n_children = self.population_size - self.n_elites
        while len(next_pop) < n_children:
            parents = self.selection(population)
            children = self.prescriptor_factory.crossover(parents, self.mutation_rate, self.mutation_factor)
            next_pop.extend(children)
        return next_pop[:n_children]

    def sort_pop(self, population: list[Prescriptor]) -> list[Prescriptor]:
        """
        Sorts the population by rank and distance according to NSGA-II.
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
        Record results from population.
        """
        rows = []
        (self.save_path / str(self.generation)).mkdir(parents=True, exist_ok=True)
        for candidate in self.population:
            # Save to file if it's new. Only save rank 1 candidates if save_all is False.
            cand_gen = int(candidate.cand_id.split("_")[0])
            if (candidate.rank == 1 or self.save_all) and cand_gen == self.generation:
                candidate.save(self.save_path / str(cand_gen) / f"{candidate.cand_id}")
            # Record candidates' results
            row = {"cand_id": candidate.cand_id, "rank": candidate.rank, "distance": candidate.distance}
            for outcome, metric in zip(candidate.outcomes, candidate.metrics):
                row[outcome] = metric
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(self.save_path / f"{self.generation}.csv", index=False)

    def validate(self):
        """
        Validates the current population with a validator evaluator.
        """
        rows = []
        (self.save_path / "validation").mkdir(parents=True, exist_ok=True)

        # Get validation results manually without setting cand's metrics or outcomes
        valid_results = self.validator.evaluate_subset(self.population, verbose=1)

        for candidate, val_metrics in zip(self.population, valid_results):
            row = {"cand_id": candidate.cand_id}
            for outcome, metric in zip(self.validator.outcomes, val_metrics):
                row[outcome] = metric
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(self.save_path / "validation" / f"{self.generation}.csv", index=False)

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

        n_keep = int(self.population_size * (1 - self.remove_population_pct))
        next_pop = self.create_pop(self.population[:n_keep])
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
        """
        self.create_initial_population()
        for _ in tqdm(range(self.generation, self.n_generations+1), desc="Running Evolution"):
            self.step()
