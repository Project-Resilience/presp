"""
Tests the implementation of the neural network prescriptor.
"""
import random
import unittest

import numpy as np
import torch

from presp.prescriptor import NNPrescriptor, NNPrescriptorFactory


def init_uniform(model, c):
    """
    Initializes a model uniformly by filling it with constant c.
    """
    for parameter in model.parameters():
        with torch.no_grad():
            parameter.fill_(c)


class TestNN(unittest.TestCase):
    """
    Tests the neural net prescriptor
    """

    def setUp(self):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

    def test_crossover(self):
        """
        Tests that we get 50/50 parameters from model 0 and model 1 after crossover.
        """
        # Fill a candidate with all 0 and all 1
        factory = NNPrescriptorFactory(NNPrescriptor, {"in_size": 12, "hidden_size": 64, "out_size": 1})
        parent0 = factory.random_init()
        parent1 = factory.random_init()
        init_uniform(parent0.model, 0)
        init_uniform(parent1.model, 1)

        # After crossover we should have 50% 1's
        count = 0
        total_num = 0
        n = 100
        for _ in range(n):
            child = factory.crossover([parent0, parent1], 0, 0)[0]
            for parameter in child.model.parameters():
                count += torch.sum(parameter.data).item()
                total_num += parameter.numel()
                # Also double check that each parameter is either a 0 or 1
                self.assertTrue(torch.all(torch.logical_or(parameter.data == 0, parameter.data == 1)))

        self.assertAlmostEqual(count / total_num, 0.5, places=2)

    def test_mutate(self):
        """
        Checks if the mean and standard deviation of the mutated parameters are correct.
        We are using gaussian percent noise with std = f. If we initialize models to all be c then:
        params ~ c * (1 + N(0, f)) = c * N(1, f) = N(c, f*c)
        We try this with 3 sets of c and f, and test over 100,000 models to get an accurate distribution.
        """
        factory = NNPrescriptorFactory(NNPrescriptor, {"in_size": 12, "hidden_size": 64, "out_size": 1})

        for _ in range(3):
            c = np.random.randint(0, 5)
            f = np.random.randint(0, 5)

            parent0 = factory.random_init()
            parent1 = factory.random_init()
            init_uniform(parent0.model, c)
            init_uniform(parent1.model, c)

            params = []
            n = 100000
            for _ in range(n):
                child = factory.crossover([parent0, parent1], 1, f)[0]
                for parameter in child.model.parameters():
                    params.append(parameter)

            flattened = torch.cat([p.flatten() for p in params])
            mean = torch.mean(flattened).item()
            std = torch.std(flattened).item()

            self.assertAlmostEqual(mean, c, places=2)
            self.assertAlmostEqual(std, f * c, places=2)
