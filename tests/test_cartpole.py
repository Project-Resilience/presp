"""
Very simple unit test for the cartpole example to ensure it fully runs evolution as expected.
This also serves as an end-to-end test to see that evolution can be run without error.
"""
import csv
from pathlib import Path
import shutil
import unittest

import numpy as np
import torch
import yaml

from examples.cartpole.cartpole_prescriptor import CartPolePrescriptor
from examples.cartpole.direct_evaluator import CartPoleEvaluator
from presp.prescriptor.nn import NNPrescriptorFactory
from presp.evolution import Evolution


class TestCartPole(unittest.TestCase):
    """
    Tests the cartpole example by running it all the way through and checking the results.
    """
    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)
        if Path("temp").exists():
            shutil.rmtree(Path("temp"))

    def test_cartpole_etoe(self):
        """
        Tests end to end evolution using the cartpole example.
        """
        with open("examples/cartpole/config.yml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        factory = NNPrescriptorFactory(CartPolePrescriptor, **config["prescriptor_params"])
        evaluator = CartPoleEvaluator(**config["eval_params"])

        evolution = Evolution(prescriptor_factory=factory, evaluator=evaluator, **config["evolution_params"])
        evolution.run_evolution()

        with open("tests/temp/10.csv", "r", encoding="utf-8") as csvfile:
            rows = list(csv.reader(csvfile, delimiter=','))

            # Checks the results file is 101x5
            self.assertEqual(len(rows), 101)
            for row in rows:
                self.assertEqual(len(row), 6)

            # Checks that the first candidate in the file has rank 1, inf distance, and the highest score possible
            self.assertEqual(rows[1][2], "1")
            self.assertEqual(rows[1][3], "inf")
            self.assertEqual(rows[1][4], "-500.0")

    def tearDown(self):
        if Path("temp").exists():
            shutil.rmtree(Path("temp"))
