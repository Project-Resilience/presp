"""
Very simple unit test for the cartpole example to ensure it fully runs evolution as expected.
This also serves as an end-to-end test to see that evolution can be run without error.
"""
import csv
from pathlib import Path
import shutil
import unittest

import numpy as np
import pandas as pd
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

        results_df = pd.read_csv("tests/temp/10.csv")

        # Makes sure we have the correct number of rows and columns
        self.assertEqual(results_df.shape, (100, 6))

        # There are no constraints so the cv column should be all 0s
        self.assertFalse(results_df["cv"].any())

        # Our top candidate should have a rank of 1
        self.assertEqual(results_df["rank"].iloc[0], 1)

        # The distance of the top candidate should be infinity
        self.assertEqual(results_df["distance"].iloc[0], float("inf"))

        # Check that we have the best possible score after evolution
        self.assertEqual(results_df["score"].iloc[0], -500)

    def tearDown(self):
        if Path("temp").exists():
            shutil.rmtree(Path("temp"))
