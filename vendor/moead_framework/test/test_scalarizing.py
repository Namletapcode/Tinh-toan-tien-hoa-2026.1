"""Scalarization values from a two-objective example calculated by hand."""

import unittest

from moead_framework.aggregation import Tchebycheff, WeightedSum
from moead_framework.test.fixtures import make_solutions


class ScalarizingTest(unittest.TestCase):
    def setUp(self):
        self.solution = make_solutions([[3, 9]])[0]
        self.weights = [[0.25, 0.75], [0.75, 0.25]]
        self.z = [1, 1]

    def test_tchebycheff_params(self):
        aggregation = Tchebycheff()
        with self.assertRaises(TypeError):
            aggregation.run(2, 2, self.weights, 0, self.z)
        with self.assertRaises(TypeError):
            aggregation.run(self.solution, self.weights, self.weights, 0, self.z)
        with self.assertRaises(TypeError):
            aggregation.run(self.solution, 2, 2, 0, self.z)
        with self.assertRaises(TypeError):
            aggregation.run(self.solution, 2, self.weights, self.weights, self.z)
        with self.assertRaises(TypeError):
            aggregation.run(self.solution, 2, self.weights, 0, 3)
        with self.assertRaises(TypeError):
            aggregation.run(self.solution, 2, self.weights, 0, [1, 1, 1, 1])

    def test_tchebycheff(self):
        aggregation = Tchebycheff()
        first = aggregation.run(self.solution, 2, self.weights, 0, self.z)
        second = aggregation.run(self.solution, 2, self.weights, 1, self.z)
        self.assertEqual(first, 6)   # max(0.25 * 2, 0.75 * 8)
        self.assertEqual(second, 2)  # max(0.75 * 2, 0.25 * 8)
        self.assertTrue(aggregation.is_better(first, second))
        self.assertFalse(aggregation.is_better(second, first))

    def test_weighted_sum_params(self):
        aggregation = WeightedSum()
        with self.assertRaises(TypeError):
            aggregation.run(1, 2, self.weights, 0, self.z)
        with self.assertRaises(TypeError):
            aggregation.run(self.solution, [2, 2], self.weights, 0, self.z)
        with self.assertRaises(TypeError):
            aggregation.run(self.solution, 2, 3, 0, self.z)
        with self.assertRaises(TypeError):
            aggregation.run(self.solution, 2, self.weights, [0, 1], self.z)
        with self.assertRaises(TypeError):
            aggregation.run(self.solution, 2, self.weights, 0, 1)
        with self.assertRaises(TypeError):
            aggregation.run(self.solution, 2, self.weights, 0, [1, 9, 0])

    def test_weighted_sum(self):
        aggregation = WeightedSum()
        first = aggregation.run(self.solution, 2, self.weights, 0, self.z)
        second = aggregation.run(self.solution, 2, self.weights, 1, self.z)
        self.assertEqual(first, 7.5)   # 0.25 * 3 + 0.75 * 9
        self.assertEqual(second, 4.5)  # 0.75 * 3 + 0.25 * 9
        self.assertTrue(aggregation.is_better(first, second))
        self.assertFalse(aggregation.is_better(second, first))

    def test_is_better(self):
        for aggregation in (WeightedSum(), Tchebycheff()):
            with self.assertRaises(TypeError):
                aggregation.is_better([99, 1], 10)
            with self.assertRaises(TypeError):
                aggregation.is_better(1.0, [9, 8])
