from abc import ABC, abstractmethod

import numpy as np
from tqdm import tqdm

from presp.prescriptor import Prescriptor


class Evaluator(ABC):
    """
    Abstract class responsible for evaluating a population of prescriptors as well as updating itself with new data.
    """
    def __init__(self, outcomes: list[str]):
        self.outcomes = outcomes

    @abstractmethod
    def update_predictor(self, elites: list[Prescriptor]):
        """
        Trains a predictor by collecting training data from the elites.
        """

    def evaluate_population(self, population: list[Prescriptor], force=False, verbose=1) -> np.ndarray:
        """
        Evaluates an entire population of prescriptors.
        Doesn't evaluate prescriptors that already have metrics unless force is True.
        """
        iterator = population if verbose < 1 else tqdm(population, leave=False)
        for candidate in iterator:
            if candidate.metrics is None or force:
                candidate.metrics = self.evaluate_candidate(candidate)
                candidate.outcomes = self.outcomes

    @abstractmethod
    def evaluate_candidate(self, candidate: Prescriptor) -> np.ndarray:
        """
        Evaluates a single candidate prescriptor.
        """
