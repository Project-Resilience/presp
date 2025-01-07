"""
Abstract evaluator class that handles evaluating candidates in evolution.
"""
from abc import ABC, abstractmethod

import numpy as np
from tqdm import tqdm

from presp.prescriptor import Prescriptor


class Evaluator(ABC):
    """
    Abstract class responsible for evaluating a population of prescriptors as well as updating itself with new data in
    the outer ESP loop.
    NOTE: The outer ESP loop can be bypassed by not implementing the update_predictor method.
    """
    def __init__(self, outcomes: list[str]):
        self.outcomes = outcomes

    @abstractmethod
    def update_predictor(self, elites: list[Prescriptor]):
        """
        Trains a predictor by collecting training data from the elites.
        :param elites: The elite prescriptors to collect data from by using them as policies in RL.
        """

    def evaluate_population(self, population: list[Prescriptor], force=False, verbose=1):
        """
        Evaluates an entire population of prescriptors.
        Doesn't evaluate prescriptors that already have metrics unless force is True.
        :param population: The population of prescriptors to evaluate.
        :param force: Whether to force evaluation of all prescriptors.
        :param verbose: Whether to show a progress bar.
        """
        iterator = population if verbose < 1 else tqdm(population, leave=False, desc="Evaluating Population")
        for candidate in iterator:
            if candidate.metrics is None or force:
                candidate.metrics = self.evaluate_candidate(candidate)
                candidate.outcomes = self.outcomes

    @abstractmethod
    def evaluate_candidate(self, candidate: Prescriptor) -> np.ndarray:
        """
        Evaluates a single candidate prescriptor.
        :param candidate: The candidate prescriptor to evaluate.
        :return: The metrics of the candidate.
        """
