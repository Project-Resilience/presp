"""
Unit tests for the NSGA-II utils implementation.
"""
import itertools
import unittest

import numpy as np

from tests.dummy_prescriptor import DummyPrescriptor
from presp.nsga2_utils import dominates, calculate_crowding_distance, fast_non_dominated_sort


class TestDominates(unittest.TestCase):
    """
    Tests the domination function.
    """
    def manual_two_obj_dominate(self, presc1: DummyPrescriptor, presc2: DummyPrescriptor) -> bool:
        """
        Manually test all cases of domination for 2 objectives.
        For candidate 1 to dominate candidate 2 it must:
            - Have a lower value in at least one objective
            - Have lower or equal values in all the rest
        """
        if (presc1.metrics[0] < presc2.metrics[0]) and (presc1.metrics[1] <= presc2.metrics[1]):
            return True
        if (presc1.metrics[0] <= presc2.metrics[0]) and (presc1.metrics[1] < presc2.metrics[1]):
            return True
        return False

    def test_domination_two_obj(self):
        """
        Tests domination works in all possible cases.
        Get all combinations of pairs of values [0, 1, 2] for each objective and tests against the manual checker.
        """
        for comb in itertools.combinations([0, 1, 2], 4):
            presc1 = DummyPrescriptor()
            presc1.metrics = [comb[0], comb[1]]
            presc2 = DummyPrescriptor()
            presc2.metrics = [comb[2], comb[3]]
            self.assertEqual(dominates(presc1, presc2), self.manual_two_obj_dominate(presc1, presc2))

    def test_domination_many_obj(self):
        """
        Tests domination works with many dimensions. Sets dimension 0 metric to presc idx and all other dimensions to 0.
        Then checks each candidate dominates the ones before it.
        """
        front = [DummyPrescriptor() for _ in range(10)]
        for i, presc in enumerate(front):
            presc.metrics = np.array([i] + [0] * 9)

        for i, presc_i in enumerate(front):
            for j, presc_j in enumerate(front):
                if i > j:
                    self.assertTrue(dominates(presc_j, presc_i))
                    self.assertFalse(dominates(presc_i, presc_j))
                elif i == j:
                    self.assertFalse(dominates(presc_i, presc_j))
                    self.assertFalse(dominates(presc_j, presc_i))
                else:
                    self.assertFalse(dominates(presc_j, presc_i))
                    self.assertTrue(dominates(presc_i, presc_j))


class TestDistanceCalculation(unittest.TestCase):
    """
    Tests the distance calculation function.
    """
    def test_distance_calculation(self):
        """
        Tests the calculation of crowding distance.
        Objective 1 is the prescriptor index times 2.
        Objective 2 is the prescriptor index squared.
        We manually compute what each distance should be then compare against our implementation.
        """
        # Create a dummy front
        front = []
        tgt_distances = []
        for i in range(4):
            presc = DummyPrescriptor()
            presc.metrics = np.array([i*2, i**2])
            front.append(presc)
            if i in {0, 3}:
                tgt_distances.append(np.inf)
            else:
                dist0 = ((i + 1) * 2 - (i - 1) * 2) / 6
                dist1 = ((i + 1) ** 2 - (i - 1) ** 2) / 9
                tgt_distances.append(dist0 + dist1)

        # Manually shuffle the front
        shuffled_indices = [1, 3, 0, 2]
        shuffled_front = [front[i] for i in shuffled_indices]
        shuffled_tgts = [tgt_distances[i] for i in shuffled_indices]

        # Assign crowding distances
        calculate_crowding_distance(shuffled_front)
        for presc, tgt in zip(shuffled_front, shuffled_tgts):
            self.assertAlmostEqual(presc.distance, tgt)


class TestNonDominatedSort(unittest.TestCase):
    """
    Tests the fast non-dominated sort function.
    """
    def test_fast_nondominated_sort(self):
        """
        Tests a general case of fast non-dominated sort.
        """
        front1 = [DummyPrescriptor() for _ in range(3)]
        for i, presc in enumerate(front1):
            presc.cand_id = f"1_{i}"
            presc.metrics = np.array([i, -i])

        front2 = [DummyPrescriptor() for _ in range(3)]
        for i, presc in enumerate(front2):
            presc.cand_id = f"2_{i}"
            presc.metrics = np.array([10+i, 10-i])

        # "Randomly" shuffle population
        population = [front1[0], front2[0], front1[1], front1[2], front2[2], front2[1]]
        fronts = fast_non_dominated_sort(population)

        # Check we have 2 fronts of size 3
        self.assertEqual(len(fronts), 2)
        self.assertEqual(len(fronts[0]), 3)
        self.assertEqual(len(fronts[1]), 3)

        # Check that the correct candidates are in the correct fronts and their ranks have been set correctly
        for i, front in enumerate(fronts):
            for presc in front:
                self.assertEqual(presc.rank, i+1)
                self.assertTrue(presc.cand_id.startswith(str(i+1)))
