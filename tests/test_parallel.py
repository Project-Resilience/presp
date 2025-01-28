"""
Unit tests to see if parallel evaluation is working correctly.
"""
import unittest

import numpy as np

from presp.evaluator import Evaluator
from presp.prescriptor import Prescriptor


class IdPrescriptor(Prescriptor):
    """
    Dummy prescriptor that returns the id of the candidate.
    """
    def __init__(self, cand_id: str):
        super().__init__()
        self.cand_id = cand_id

    def forward(self, _):
        return ""

    def save(self, path):
        pass


class IdEvaluator(Evaluator):
    """
    Dummy evaluator that returns the id of the candidate as its metric.
    """
    def __init__(self):
        super().__init__(outcomes=["id"], n_jobs=2)

    def update_predictor(self, elites):
        pass

    def evaluate_candidate(self, candidate: Prescriptor):
        return np.array([int(candidate.cand_id.split("_")[1])])


class TestParallelization(unittest.TestCase):
    """
    Tests parallelization of evaluation.
    """
    def test_results_correct_order(self):
        """
        Tests to make sure results are in the correct order after parallel evaluation.
        """
        n_candidates = 100
        population = [IdPrescriptor(f"0_{i}") for i in range(n_candidates)]
        evaluator = IdEvaluator()
        parallel_results = evaluator.evaluate_subset(population)

        for candidate, result in zip(population, parallel_results):
            cand_id = candidate.cand_id.split("_")[1]
            self.assertEqual(int(cand_id), result[0])
