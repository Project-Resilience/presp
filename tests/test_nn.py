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
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(c)


def check_orthogonal(weight: torch.Tensor, atol: float) -> bool:
    """
    See if WTW = I to a certain degree of tolerance.
    """
    identity = torch.eye(weight.shape[1])
    wtw = torch.matmul(weight.T, weight)
    return torch.allclose(wtw, identity, atol=atol)


class TestNNInit(unittest.TestCase):
    """
    Tests the initialization of the neural net prescriptor
    """
    def setUp(self):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

    def test_init(self):
        """
        Checks that the first layer is initialized orthogonally and the last layer is initialized normally. Also checks
        that the biases are initialized normally.
        """
        factory = NNPrescriptorFactory(NNPrescriptor, {"in_size": 12, "hidden_size": 16, "out_size": 1})
        candidates = [factory.random_init() for _ in range(100000)]

        last_layer_params = []
        biases = []

        for candidate in candidates:
            for i, layer in enumerate(candidate.model):
                if isinstance(layer, torch.nn.Linear):
                    if i != len(candidate.model) - 1:
                        self.assertTrue(check_orthogonal(layer.weight.data, 1e-4))
                    else:
                        self.assertFalse(check_orthogonal(layer.weight.data, 1e-4))
                        last_layer_params.append(layer.weight.data)
                    biases.append(layer.bias.data)

        # Check that the last layer is normally distributed
        last_layer_params = torch.cat([p.flatten() for p in last_layer_params])
        last_layer_mean = torch.mean(last_layer_params).item()
        last_layer_std = torch.std(last_layer_params).item()
        self.assertAlmostEqual(last_layer_mean, 0, places=2)
        self.assertAlmostEqual(last_layer_std, 1, places=2)

        # Check that the biases are normally distributed
        biases = torch.cat([b.flatten() for b in biases])
        bias_mean = torch.mean(biases).item()
        bias_std = torch.std(biases).item()
        self.assertAlmostEqual(bias_mean, 0, places=2)
        self.assertAlmostEqual(bias_std, 1, places=2)


class TestNNCrossover(unittest.TestCase):
    """
    Tests the crossover method of the neural net prescriptor factory.
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


class TestNNMutation(unittest.TestCase):
    """
    Tests mutation for the neural net prescriptor.
    """
    def setUp(self):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

    def test_mutate(self):
        """
        Checks if the mean and standard deviation of the mutated parameters are correct.
        We are using gaussian percent noise with std = f. If we initialize models to all be c then:
        params ~ c * (1 + N(0, f)) = c * N(1, f) = N(c, f*c)
        We try this with 3 sets of c and f, and test over 10000 models to get an accurate distribution.
        """
        factory = NNPrescriptorFactory(NNPrescriptor, {"in_size": 12, "hidden_size": 64, "out_size": 1})

        for _ in range(3):
            c = np.random.rand()
            f = np.random.rand()

            parent0 = factory.random_init()
            parent1 = factory.random_init()
            init_uniform(parent0.model, c)
            init_uniform(parent1.model, c)

            # Make sure parents are initialized correctly to all c
            for parameter in parent0.model.parameters():
                self.assertTrue(torch.all(parameter.data == c))
            for parameter in parent1.model.parameters():
                self.assertTrue(torch.all(parameter.data == c))

            params = []
            n = 10000
            for _ in range(n):
                child = factory.crossover([parent0, parent1], 1, f)[0]
                for parameter in child.model.parameters():
                    params.append(parameter)

            flattened = torch.cat([p.flatten() for p in params])
            mean = torch.mean(flattened).item()
            std = torch.std(flattened).item()

            self.assertAlmostEqual(mean, c, places=3)
            self.assertAlmostEqual(std, f * c, places=3)

    def test_mutate_count(self):
        """
        Checks the count of mutated parameters is correct.
        We should have mutation_factor mutated parameters.
        """
        factory = NNPrescriptorFactory(NNPrescriptor, {"in_size": 12, "hidden_size": 100000, "out_size": 1})
        parent0 = factory.random_init()
        parent1 = factory.random_init()
        init_uniform(parent0.model, 1)
        init_uniform(parent1.model, 1)

        child = factory.crossover([parent0, parent1], 0.1, 0.1)[0]
        diff_weight_count = 0
        param_count = 0
        for parameter in child.model.parameters():
            diff_weight_count += torch.sum(parameter.data != 1).item()
            param_count += parameter.numel()
        self.assertAlmostEqual(diff_weight_count / param_count, 0.1, places=3)
