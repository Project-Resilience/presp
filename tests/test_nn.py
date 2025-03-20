"""
Tests the implementation of the neural network prescriptor.
TODO: Make the tolerances based on confidence interval.
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


def check_dist(weight: torch.Tensor, mean: float, std: float, atol: float) -> bool:
    """
    Check if the weight has correct mean and standard deviation.
    """
    weight_mean = torch.mean(weight).item()
    weight_std = torch.std(weight).item()
    return torch.isclose(weight_mean, mean, atol=atol) and torch.isclose(weight_std, std, atol=atol)


def create_factory() -> NNPrescriptorFactory:
    """
    Creates a factory with a simple model.
    """
    factory_params = {
        "model_params": [
            {"type": "linear", "in_features": 12, "out_features": 16},
            {"type": "tanh"},
            {"type": "linear", "in_features": 16, "out_features": 1},
            {"type": "sigmoid"}
        ]
    }
    return NNPrescriptorFactory(NNPrescriptor, **factory_params)


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
        factory = create_factory()
        candidates = [factory.random_init() for _ in range(100000)]

        last_layer_params = []
        biases = []

        for candidate in candidates:
            linear_layers = [layer for layer in candidate.model if isinstance(layer, torch.nn.Linear)]
            for i, layer in enumerate(linear_layers):
                if isinstance(layer, torch.nn.Linear):
                    if i < len(linear_layers) - 1:
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
        factory = create_factory()
        parent0 = factory.random_init()
        parent1 = factory.random_init()
        init_uniform(parent0.model, 0)
        init_uniform(parent1.model, 1)

        # After crossover we should have 50% 1's
        count = 0
        total_num = 0
        n = 100
        for _ in range(n):
            child = factory.crossover([parent0, parent1])[0]
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
        We try this with 3 sets of c and f, and test over 1000 models to get an accurate distribution.
        """
        factory = create_factory()

        for _ in range(3):
            c = np.random.rand()
            f = np.random.rand()

            params = []
            n = 10000
            for _ in range(n):
                candidate = factory.random_init()
                init_uniform(candidate.model, c)

                # Make sure candidate is initialized correctly to all c
                for parameter in candidate.model.parameters():
                    self.assertTrue(torch.all(parameter.data == c))

                factory.mutation(candidate, 1, f)
                for parameter in candidate.model.parameters():
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
        factory = create_factory()

        diff_weight_count = 0
        param_count = 0

        n = 10
        for _ in range(n):
            # Uniformly initialize candidate
            candidate = factory.random_init()
            init_uniform(candidate.model, 1)

            # Mutate candidate
            factory.mutation(candidate, 0.1, 0.1)

            # Count mutated parameters
            for parameter in candidate.model.parameters():
                diff_weight_count += torch.sum(parameter.data != 1).item()
                param_count += parameter.numel()

        self.assertAlmostEqual(diff_weight_count / param_count, 0.1, places=3)
