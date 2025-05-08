"""
Very simple unit test for the cartpole example to ensure it fully runs evolution as expected.
This also serves as an end-to-end test to see that evolution can be run without error.
"""
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
            config["evolution_params"]["n_generations"] = 7  # We only need 7 or so generations to get a good solution

        factory = NNPrescriptorFactory(CartPolePrescriptor, **config["prescriptor_params"])
        evaluator = CartPoleEvaluator(**config["eval_params"])

        evolution = Evolution(prescriptor_factory=factory, evaluator=evaluator, **config["evolution_params"])
        evolution.run_evolution()

        results_df = pd.read_csv("tests/temp/results.csv")

        # Makes sure we have the correct number of rows and columns
        n_rows = config["evolution_params"]["n_generations"] * config["evolution_params"]["population_size"]
        self.assertEqual(results_df.shape, (n_rows, 7))

        # Make sure the column names are right
        self.assertEqual(
            results_df.columns.tolist(),
            ["gen", "cand_id", "parents", "cv", "rank", "distance", "score"]
        )

        # There are no constraints so the cv column should be all 0s
        self.assertFalse(results_df["cv"].any())

        # Our top candidate should have a rank of 1
        best_row = results_df[results_df["gen"] == config["evolution_params"]["n_generations"]].iloc[0]
        self.assertEqual(best_row["rank"], 1)

        # The distance of the top candidate should be infinity
        self.assertEqual(best_row["distance"], float("inf"))

        # Check that we have the best possible score after evolution
        self.assertEqual(best_row["score"], -500)

    def tearDown(self):
        if Path("temp").exists():
            shutil.rmtree(Path("temp"))
