"""
Very simple unit test for the cartpole example to ensure it fully runs evolution as expected.
This also serves as an end-to-end test to see that evolution can be run without error.
"""
import csv
from pathlib import Path
import shutil
import unittest

import yaml

from examples.cartpole.direct_evaluator import DirectEvaluator
from examples.cartpole.esp_evaluator import ESPEvaluator
from presp.prescriptor.nn import NNPrescriptorFactory
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

        factory = NNPrescriptorFactory(**config["prescriptor_params"])
        evaluator = DirectEvaluator(**config["eval_params"])

        evolution = Evolution(prescriptor_factory=factory, evaluator=evaluator, **config["evolution_params"])
        evolution.run_evolution()

        with open("tests/temp/10.csv", "r", encoding="utf-8") as csvfile:
            rows = list(csv.reader(csvfile, delimiter=','))

            # Checks the results file is 101x5
            self.assertEqual(len(rows), 101)
            for row in rows:
                self.assertEqual(len(row), 5)

            # Checks that the first candidate in the file has rank 1, inf distance, and the highest score possible
            self.assertEqual(rows[1][2], "1")
            self.assertEqual(rows[1][3], "inf")
            self.assertEqual(rows[1][4], "-500.0")

    def tearDown(self):
        if Path("temp").exists():
            shutil.rmtree(Path("temp"))


class TestESP(unittest.TestCase):
    """
    Tests CartPole using an ESP evaluator.
    """
    def setUp(self):
        if Path("temp").exists():
            shutil.rmtree(Path("temp"))

    def test_esp_etoe(self):
        """
        Tests end to end evolution using the cartpole example.
        """
        with open("examples/cartpole/config.yml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        factory = NNPrescriptorFactory(**config["prescriptor_params"])
        evaluator = ESPEvaluator(**config["eval_params"])
        validator = DirectEvaluator(**config["eval_params"])

        config["evolution_params"]["val_interval"] = 1

        evolution = Evolution(prescriptor_factory=factory,
                              evaluator=evaluator,
                              validator=validator,
                              **config["evolution_params"])
        evolution.run_evolution()

        with open("tests/temp/10.csv", "r", encoding="utf-8") as csvfile:
            rows = list(csv.reader(csvfile, delimiter=','))

            # Checks the results file is 101x5
            self.assertEqual(len(rows), 101)
            for row in rows:
                self.assertEqual(len(row), 5)

            # Checks that the first candidate in the file has rank 1, inf distance, and the highest score possible
            self.assertEqual(rows[1][2], "1")
            self.assertEqual(rows[1][3], "inf")
            self.assertEqual(rows[1][4], "-500.0")

    # def tearDown(self):
    #     if Path("temp").exists():
    #         shutil.rmtree(Path("temp"))
