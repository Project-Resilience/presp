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
        super().__init__(outcomes=["id"], n_jobs=-1)

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

    def test_force_no_overwrite(self):
        """
        Makes sure we don't overwrite metrics if we don't force evaluation.
        """
        n_candidates = 100
        population = [IdPrescriptor(f"0_{i}") for i in range(n_candidates)]
        for i, candidate in enumerate(population):
            if i % 2 == 0:
                candidate.metrics = np.array([-999])

        evaluator = IdEvaluator()
        evaluator.evaluate_population(population, force=False)

        for i, candidate in enumerate(population):
            if i % 2 == 0:
                self.assertEqual(-999, candidate.metrics[0])
            else:
                self.assertEqual(i, candidate.metrics[0])

    def test_force_overwrite(self):
        """
        Makes sure we overwrite metrics when we want to force evaluation.
        """
        n_candidates = 100
        population = [IdPrescriptor(f"0_{i}") for i in range(n_candidates)]
        for i, candidate in enumerate(population):
            if i % 2 == 0:
                candidate.metrics = np.array([-999])

        evaluator = IdEvaluator()
        evaluator.evaluate_population(population, force=True)

        for i, candidate in enumerate(population):
            self.assertEqual(i, candidate.metrics[0])
