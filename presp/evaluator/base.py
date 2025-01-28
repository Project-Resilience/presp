"""
Abstract evaluator class that handles evaluating candidates in evolution.
"""
from abc import ABC, abstractmethod

from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm

from presp.prescriptor import Prescriptor


class Evaluator(ABC):
    """
    Abstract class responsible for evaluating a population of prescriptors as well as updating itself with new data in
    the outer ESP loop.
    :param outcomes: What to label the outcomes the prescriptors are evaluated on.
    :param n_jobs: The number of jobs to use in parallel evaluation. Defaults to 1 for sequential evaluation.
    NOTE: The outer ESP loop can be bypassed by not implementing the update_predictor method.
    """
    def __init__(self, outcomes: list[str], n_jobs: int = 1):
        self.outcomes = outcomes
        assert (n_jobs >= -1) and n_jobs != 0
        self.n_jobs = n_jobs

    @abstractmethod
    def update_predictor(self, elites: list[Prescriptor]):
        """
        Trains a predictor by collecting training data from the elites.
        :param elites: The elite prescriptors to collect data from by using them as policies in RL.
        """

    def evaluate_population(self, population: list[Prescriptor], force=False, verbose=1) -> list[np.ndarray]:
        """
        Evaluates an entire population of prescriptors.
        Doesn't evaluate prescriptors that already have metrics unless force is True.
        Sets candidates' metrics and outcomes for evolution.
        :param population: The population of prescriptors to evaluate.
        :param force: Whether to force evaluation of all prescriptors.
        :param verbose: Whether to show a progress bar for sequential evaluation.
        """
        pop_subset = [cand for cand in population if (cand.metrics is None) or force]
        pop_results = self.evaluate_subset(pop_subset, verbose=verbose)
        for candidate, metrics in zip(pop_subset, pop_results):
            candidate.metrics = metrics
            candidate.outcomes = self.outcomes

    def evaluate_subset(self, population: list[Prescriptor], verbose=1) -> list[np.ndarray]:
        """
        Evaluates a subset of the population, returning the corresponding metrics in order.
        :param population: The subset of the population to evaluate.
        :param verbose: Whether to show a progress bar.
        :return: The metrics of the population.
        """
        if self.n_jobs != 1:
            num_jobs = self.n_jobs if self.n_jobs != -1 else len(population)
            pop_results = self.parallel_evaluation(population, n_jobs=num_jobs)
        else:
            pop_results = self.sequential_evaluation(population, verbose=verbose)
        return pop_results

    def sequential_evaluation(self, population: list[Prescriptor], verbose=1) -> list[np.ndarray]:
        """
        Performs sequential evaluation of a given population.
        :param population: The population of prescriptors to evaluate.
        :param verbose: Whether to show a progress bar.
        """
        results = []
        iterator = population if verbose < 1 else tqdm(population, leave=False, desc="Evaluating Population")
        for candidate in iterator:
            results.append(self.evaluate_candidate(candidate))
        return results

    def parallel_evaluation(self, population: list[Prescriptor], n_jobs: int) -> list[np.ndarray]:
        """
        Performs parallel evaluation using joblib's Parallel function.
        :param population: The population of prescriptors to evaluate.
        :param n_jobs: The number of jobs to use in parallel evaluation. NOTE: n_jobs is hard-coded right now but can
        be modified later.
        NOTE: This is still experimental and may not work with all evaluators.
        TODO: Make parallel_evaluation take in force argument and handle evaluating a subset of the population in
        parallel.
        """
        parallel_results = Parallel(n_jobs=n_jobs)(delayed(self.evaluate_candidate)(cand) for cand in population)
        return parallel_results

    @abstractmethod
    def evaluate_candidate(self, candidate: Prescriptor) -> np.ndarray:
        """
        Evaluates a single candidate prescriptor.
        :param candidate: The candidate prescriptor to evaluate.
        :return: The metrics of the candidate.
        """
