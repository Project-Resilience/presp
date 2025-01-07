from unittest import TestCase

from presp.evolution import Evolution
from tests.dummy_evaluator import DummyEvaluator
from tests.dummy_prescriptor import DummyPrescriptor, DummyFactory


class TestEvolution(TestCase):

    def test_initial_pop(self):
        factory = DummyFactory(DummyPrescriptor)
        evaluator = DummyEvaluator()
        config = {
            "n_generations": 2,
            "population_size": 10,
            "remove_population_pct": 0.5,
            "n_elites": 2,
            "mutation_rate": 0.1,
            "mutation_factor": 0.1,
            "save_path": "tests/temp",
            "seed_dir": None,
            "prescriptor_factory": factory,
            "evaluator": evaluator
        }
        evolution = Evolution(**config)
