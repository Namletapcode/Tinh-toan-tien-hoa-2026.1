"""Solution representation and copying checked on explicit small vectors."""

from copy import copy
import unittest

import numpy as np

from moead_framework.solution import OneDimensionSolution


class SolutionTest(unittest.TestCase):
    def setUp(self):
        self.solution = OneDimensionSolution(np.array([0.0, 1.0, 0.0]), f=[3.0, 9.0])

    def test_set_item(self):
        self.assertEqual(self.solution[0], 3.0)
        self.solution[0] = -42
        self.assertEqual(self.solution[0], -42)
        self.assertEqual(self.solution.F, [-42, 9.0])

    def test_repr(self):
        self.assertEqual(
            repr(self.solution),
            "OneDimensionSolution(decision_vector=[0.0, 1.0, 0.0], f=[3.0, 9.0])",
        )

    def test_copy(self):
        copied = copy(self.solution)
        self.assertIsNot(copied, self.solution)
        self.assertEqual(copied.F, self.solution.F)
        np.testing.assert_array_equal(copied.decision_vector, self.solution.decision_vector)
        copied.F[0] = -1
        self.assertEqual(self.solution.F, [3.0, 9.0])
