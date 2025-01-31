"""
Very simple unit test for the cartpole example to ensure it fully runs evolution as expected.
This also serves as an end-to-end test to see that evolution can be run without error.
"""
import csv
from pathlib import Path
import shutil
import unittest

import yaml

from examples.cartpole.evaluator import CartPoleEvaluator
from examples.cartpole.prescriptor import CartPolePrescriptorFactory
from presp.evolution import Evolution


class TestCartPole(unittest.TestCase):
    """
    Tests the cartpole example by running it all the way through and checking the results.
    """
    def setUp(self):
        if Path("temp").exists():
            shutil.rmtree(Path("temp"))

    def test_cartpole_etoe(self):
        """
        Tests end to end evolution using the cartpole example.
        """
        with open("examples/cartpole/config.yml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        factory = CartPolePrescriptorFactory(**config["prescriptor_params"])
        evaluator = CartPoleEvaluator(**config["eval_params"])

        evolution = Evolution(prescriptor_factory=factory, evaluator=evaluator, **config["evolution_params"])
        evolution.run_evolution()

        with open("tests/temp/10.csv", "r", encoding="utf-8") as csvfile:
            rows = list(csv.reader(csvfile, delimiter=','))

            # Checks the results file is 101x4
            self.assertEqual(len(rows), 101)
            for row in rows:
                self.assertEqual(len(row), 4)

            # Checks that the first candidate in the file has rank 1, inf distance, and 0 score
            self.assertEqual(rows[1][1], "1")
            self.assertEqual(rows[1][2], "inf")
            self.assertEqual(rows[1][3], "-500.0")

    def tearDown(self):
        if Path("temp").exists():
            shutil.rmtree(Path("temp"))
