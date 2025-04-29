"""
Tests constraint violation handling.
We added constraint violation handling to domination: cv's take priority before metrics
"""
import unittest

import numpy as np

from presp.nsga2_utils import dominates
from tests.dummy_prescriptor import DummyPrescriptor


class TestConstraintViolation(unittest.TestCase):
    """
    Checks various cases in the new domination function to make sure CV is prioritized over metrics.
    """
    def test_cv_dominates(self):
        """
        Tests that a candidate that would normally get dominated by metrics is dominated by cv.
        """
        cand1 = DummyPrescriptor()
        cand1.metrics = np.array([1, 1])

        cand2 = DummyPrescriptor()
        cand2.cv = 1
        cand2.metrics = np.array([0, 0])

        self.assertTrue(dominates(cand1, cand2))
        self.assertFalse(dominates(cand2, cand1))

    def test_equal_cv_dominates(self):
        """
        Tests that when CV is equal, the regular domination holds.
        """
        cand1 = DummyPrescriptor()
        cand1.cv = 1
        cand1.metrics = np.array([1, 1])

        cand2 = DummyPrescriptor()
        cand2.cv = 1
        cand2.metrics = np.array([0, 0])

        self.assertTrue(dominates(cand2, cand1))
        self.assertFalse(dominates(cand1, cand2))
