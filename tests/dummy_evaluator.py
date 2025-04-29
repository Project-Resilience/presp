"""
Dummy evaluator class for testing purposes.
"""
import numpy as np

from presp.evaluator import Evaluator
from tests.dummy_prescriptor import DummyPrescriptor


class DummyEvaluator(Evaluator):
    """
    A dummy evaluator. The score is just the action the prescriptor prescribed times 10.
    """
    def __init__(self):
        super().__init__(["score"])

    def update_predictor(self, elites: list[DummyPrescriptor]):
        pass

    def evaluate_candidate(self, candidate: DummyPrescriptor) -> np.ndarray:
        return np.array([-1 * candidate.forward(None) * 10]), 0
