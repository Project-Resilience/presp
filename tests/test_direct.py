"""
Unit tests for the prescriptor that directly evolves a vector of parameters.
"""
import unittest

import numpy as np

from presp.prescriptor import DirectPrescriptor, DirectFactory


class TestDirect(unittest.TestCase):
    """
    Tests the direct prescriptor and factory.
    """
    def test_random_init_uniform(self):
        """
        Tests the random initialization is uniform between xl and xu.
        TODO: We can do significance testing here to make sure the distribution is uniform to a certain p value.
        """
        xl = np.array([0, 1, 2, 3])
        xu = np.array([4, 4, 4, 4])

        factory = DirectFactory(xl, xu)

        genomes = []
        for _ in range(100_000):
            prescriptor = factory.random_init()
            genomes.append(prescriptor.genome)

        genomes = np.stack(genomes)

        mu = genomes.mean(axis=0)
        sigma = genomes.std(axis=0)

        mu_expected = (xl + xu) / 2
        sigma_expected = np.sqrt(((xu - xl) ** 2) / 12)

        self.assertTrue(np.allclose(mu, mu_expected, atol=0.01))
        self.assertTrue(np.allclose(sigma, sigma_expected, atol=0.01))

    def test_crossover(self):
        """
        Tests the crossover function uniformly selects between the two parents.
        """
        n_param = 1_000_000
        parent1 = DirectPrescriptor(np.zeros(n_param), np.zeros(n_param))

        parent2 = DirectPrescriptor(np.ones(n_param), np.ones(n_param))

        factory = DirectFactory(np.zeros(n_param), np.ones(n_param))
        [child] = factory.crossover([parent1, parent2])

        self.assertEqual(child.genome.min(), 0)
        self.assertEqual(child.genome.max(), 1)
        self.assertAlmostEqual(child.genome.mean(), 0.5, delta=0.001)
        self.assertAlmostEqual(child.genome.std(), 0.5, delta=0.001)

    def test_mutation_bounds(self):
        """
        Tests the mutation function respects the bounds.
        """
        n_param = 1_000_000
        xl = np.zeros(n_param)
        xu = np.ones(n_param)

        factory = DirectFactory(xl, xu)
        prescriptor = factory.random_init()

        # Mutate with a high mutation rate and factor to ensure we go out of bounds
        factory.mutation(prescriptor, mutation_rate=1.0, mutation_factor=1_000_000)

        self.assertTrue(np.all(prescriptor.genome >= xl))
        self.assertTrue(np.all(prescriptor.genome <= xu))
