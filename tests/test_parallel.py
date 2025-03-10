"""
Unit tests to see if parallel evaluation is working correctly.
"""
import unittest

import numpy as np
import yaml

from presp.evaluator import Evaluator
from presp.prescriptor import Prescriptor
from presp.prescriptor.nn import NNPrescriptorFactory

from examples.cartpole.direct_evaluator import DirectEvaluator


class IdPrescriptor(Prescriptor):
    """
    Dummy prescriptor that returns the id of the candidate.
    """
    def __init__(self, cand_id: str):
        super().__init__()
        self.cand_id = cand_id

    def forward(self, _):
        return ""


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

    def test_same_as_sequential(self):
        """
        We run the same population through the evaluator twice, once in parallel and once sequentially to see
        if they produce the same results.
        We use the cartpole prescriptor and evaluator for this test.
        """
        with open("examples/cartpole/config.yml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        factory = NNPrescriptorFactory(**config["prescriptor_params"])
        population = [factory.random_init() for _ in range(100)]

        sequential_evaluator = DirectEvaluator(n_jobs=1, n_envs=10)
        sequential_results = sequential_evaluator.evaluate_subset(population)

        parallel_evaluator = DirectEvaluator(n_jobs=4, n_envs=10)
        parallel_results = parallel_evaluator.evaluate_subset(population)

        for sequential, parallel in zip(sequential_results, parallel_results):
            self.assertTrue(np.equal(sequential, parallel))
