"""
Tests the Evolution class.
"""
from pathlib import Path
import random
import shutil
import unittest

import numpy as np
import pandas as pd

from presp.evolution import Evolution
from tests.dummy_evaluator import DummyEvaluator
from tests.dummy_prescriptor import DummyPrescriptor, DummyFactory


class TestInitialPopulation(unittest.TestCase):
    """
    Tests the initial population creation of the Evolution class.
    """

    def setUp(self):
        if Path("tests/temp").exists():
            shutil.rmtree(Path("tests/temp"))
        Path("tests/seeds").unlink(missing_ok=True)

    def tearDown(self):
        if Path("tests/temp").exists():
            shutil.rmtree(Path("tests/temp"))
        Path("tests/seeds").unlink(missing_ok=True)

    def test_create_initial_population_random(self):
        """
        Tests the creation of a new initial population randomly. Makes sure all of the candidates are initialized.
        Also makes sure the generation is set properly.
        """
        evaluator = DummyEvaluator()
        factory = DummyFactory()
        evolution = Evolution(10, 10, 0.1, 2, 0.1, 0.1, "tests/temp", factory, evaluator)
        evolution.create_initial_population()

        self.assertEqual(len(evolution.population), 10)
        for candidate in evolution.population:
            self.assertTrue(candidate.metrics is not None)
            self.assertTrue(candidate.cand_id is not None)
            self.assertTrue(candidate.outcomes is not None)
            self.assertTrue(candidate.distance is not None)
            self.assertTrue(candidate.rank is not None)

        self.assertEqual(evolution.generation, 2)

    def test_create_initial_population_seeded(self):
        """
        Adds some seeds to the initial population and checks they are loaded correctly.
        Also checks that the rest of the population is initialized randomly.
        """
        factory = DummyFactory()
        seed_path = Path("tests/seeds")
        seeds = []
        for i in range(8):
            candidate = DummyPrescriptor()
            candidate.cand_id = f"0_{i}"
            candidate.number = i
            seeds.append(candidate)
        factory.save_population(seeds, seed_path)

        evaluator = DummyEvaluator()
        evolution = Evolution(10, 10, 0.1, 2, 0.1, 0.1, "tests/temp", factory, evaluator, seed_path="tests/seeds")
        evolution.create_initial_population()

        # Make sure population is the right size
        self.assertEqual(len(evolution.population), 10)
        cand_names = {f"0_{i}" for i in range(8)}
        cand_names.update({f"1_{i}" for i in range(8, 10)})

        # Check that every name is accounted for and that the number is set correctly
        for candidate in evolution.population:
            self.assertTrue(candidate.cand_id in cand_names)
            cand_names.remove(candidate.cand_id)
            if candidate.cand_id.startswith("0"):
                self.assertEqual(candidate.number, int(candidate.cand_id.split("_")[1]))
            else:
                self.assertGreaterEqual(candidate.number, 0)
                self.assertLessEqual(candidate.number, 1)

        self.assertEqual(len(cand_names), 0)


class TestSelection(unittest.TestCase):
    """
    Tests the default parent selection (tournament selection)
    """
    def test_selection(self):
        """
        Tests tournament selection. We choose the min of 2 candidates, therefore with 2 candidates the smaller one
        should get picked 3/4 of the time: (0, 0), (0, 1), (1, 0) and the larger one 1/4 of the time: (1, 1).
        We run this many times and see if we're within 1%.
        """
        random.seed(42)
        np.random.seed(42)

        factory = DummyFactory()
        evaluator = DummyEvaluator()
        evolution = Evolution(10, 10, 0.1, 2, 0.1, 0.1, "tests/temp", factory, evaluator)

        cand_0 = DummyPrescriptor()
        cand_0.number = 0
        cand_1 = DummyPrescriptor()
        cand_1.number = 1

        population = [cand_0, cand_1]
        counts = {0: 0, 1: 0}
        n = 100000
        for _ in range(n):
            parents = evolution.selection(population)
            counts[parents[0].number] += 1
            counts[parents[1].number] += 1

        self.assertTrue(np.isclose(0.75, counts[0] / (n * 2), atol=0.01))
        self.assertTrue(np.isclose(0.25, counts[1] / (n * 2), atol=0.01))


class TestCreatePop(unittest.TestCase):
    """
    Tests the population creation function in evolution. Makes sure remove_population_pct works.
    """
    def test_remove_pop_pct(self):
        """
        Ensures that any children created have parents from the top half of the population.
        """
        n_cands = 10000

        factory = DummyFactory()
        evaluator = DummyEvaluator()
        evolution = Evolution(10, n_cands, 0.5, 2, 0.1, 0.1, "tests/temp", factory, evaluator)

        population = [DummyPrescriptor() for _ in range(n_cands)]
        for i, candidate in enumerate(population):
            candidate.cand_id = f"0_{i}"

        next_pop = evolution.create_pop(population, 0.5)

        for candidate in next_pop:
            parents = candidate.parents
            parent_idxs = [int(parent.split("_")[1]) for parent in parents]
            for parent_idx in parent_idxs:
                self.assertIn(parent_idx, range(int(0.5 * n_cands)))
                self.assertNotIn(parent_idx, range(int(0.5 * n_cands), n_cands))


class TestElites(unittest.TestCase):
    """
    Tests that elites stay elite.
    """
    def setUp(self):
        if Path("tests/temp").exists():
            shutil.rmtree(Path("tests/temp"))

    def tearDown(self):
        if Path("tests/temp").exists():
            shutil.rmtree(Path("tests/temp"))

    def test_elites_consistent(self):
        """
        Checks that when we force elites to always have the best metrics, they stay elite and everything else is created
        fresh every generation.
        """
        factory = DummyFactory()
        evaluator = DummyEvaluator()
        evolution = Evolution(10, 100, 0.5, 10, 0.1, 0.1, "tests/temp", factory, evaluator)

        # Create initial pop, replace some metrics with -999 to make sure they're always elite. Then re-sort.
        evolution.create_initial_population()
        elite_ids = [f"1_{i}" for i in range(10)]
        for candidate in evolution.population:
            if candidate.cand_id in elite_ids:
                candidate.metrics = np.array([-999])
        evolution.population = evolution.sort_pop(evolution.population)

        for i in range(2, 11):
            # Step, then set elite metrics to -999 again. Resort afterwards.
            evolution.step()
            for candidate in evolution.population:
                if candidate.cand_id in elite_ids:
                    candidate.metrics = np.array([-999])
            evolution.population = evolution.sort_pop(evolution.population)

            # Check that elites are elite and non-elites are non-elite.
            for j, candidate in enumerate(evolution.population):
                if j < evolution.n_elites:
                    self.assertIn(candidate.cand_id, elite_ids)
                else:
                    self.assertNotIn(candidate.cand_id, elite_ids)
                    self.assertEqual(candidate.cand_id.split("_")[0], str(i))


class TestSaving(unittest.TestCase):
    """
    Tests the saving of the evolution class at each generation.
    """
    def setUp(self):
        if Path("tests/temp").exists():
            shutil.rmtree(Path("tests/temp"))
        self.factory = DummyFactory()
        evaluator = DummyEvaluator()
        self.pop_size = 10
        self.n_gens = 10

        # Create some seeds too
        seed_path = Path("tests/seeds")
        if seed_path.exists():
            seed_path.unlink()
        seeds = [self.factory.random_init() for _ in range(3)]
        for i, candidate in enumerate(seeds):
            candidate.cand_id = f"0_{i}"
        seeds[0].number = 99999  # Inject a great seed to make sure it gets saved
        self.factory.save_population(seeds, seed_path)

        self.evolution = Evolution(self.n_gens,
                                   self.pop_size,
                                   0.0,
                                   2,
                                   0.1,
                                   0.1,
                                   "tests/temp",
                                   seed_path="tests/seeds",
                                   prescriptor_factory=self.factory,
                                   evaluator=evaluator)

    def tearDown(self):
        if Path("tests/temp").exists():
            shutil.rmtree(Path("tests/temp"))
        Path("tests/seeds").unlink(missing_ok=True)

    def test_results_csv(self):
        """
        Makes sure we're saving the correct number of rows in the results csv.
        """
        for i in range(self.n_gens):
            if i == 0:
                self.evolution.create_initial_population()
            else:
                self.evolution.step()

            results_df = pd.read_csv("tests/temp/results.csv")
            self.assertEqual(len(results_df), self.pop_size * (i + 1))

    def test_population_saving(self):
        """
        Makes sure we're saving the population correctly. We save only the pareto front and check that we aren't
        saving duplicates (eg when 1_0 does well in gen 1 and gen 2, we only save it once).
        Also checks to make sure the "perfect" seed got saved.
        """
        for i in range(self.n_gens):
            if i == 0:
                self.evolution.create_initial_population()
            else:
                self.evolution.step()

            results_df = pd.read_csv("tests/temp/results.csv")
            pareto = results_df[results_df["rank"] == 1]["cand_id"].to_list()
            pop_dict = self.factory.load_population(Path("tests/temp/population"))
            self.assertEqual(set(pop_dict.keys()), set(pareto))
            self.assertTrue("0_0" in pareto)
