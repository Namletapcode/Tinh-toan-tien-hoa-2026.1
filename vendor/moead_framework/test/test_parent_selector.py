"""Parent selectors must return the exact members selected from a supplied pool."""

from pathlib import Path
import unittest
from unittest.mock import patch

from moead_framework.aggregation import Tchebycheff
from moead_framework.algorithm import Moead
from moead_framework.core.parent_selector import (
    OneRandomAndCurrentParentSelector, TwoRandomAndCurrentParentSelector,
    TwoRandomParentSelector,
)
from moead_framework.test.fixtures import BinaryTestProblem
from moead_framework.tool.result import set_seed


class ParentSelectorTest(unittest.TestCase):
    def setUp(self):
        set_seed(1)
        self.problem = BinaryTestProblem()
        self.moead = Moead(
            problem=self.problem, max_evaluation=100, number_of_weight_neighborhood=3,
            aggregation_function=Tchebycheff,
            weight_file=str(Path(__file__).parent / "data/weights/SOBOL-2objs-10wei.ws"),
        )
        self.moead.update_current_sub_problem(3)
        self.pool = [7, 2, 5]

    def test_two_random_param(self):
        with self.assertRaises(TypeError):
            TwoRandomParentSelector(algorithm=self.problem)

    def test_two_random_param2(self):
        with self.assertRaises(TypeError):
            TwoRandomParentSelector(algorithm=self.moead).select(self.moead.b)

    def test_two_random(self):
        selector = TwoRandomParentSelector(algorithm=self.moead)
        with patch("moead_framework.core.parent_selector.two_random_parent_selector.random.randint",
                   side_effect=[0, 2]):
            parents = selector.select(self.pool)
        self.assertEqual(len(parents), 2)
        self.assertIs(parents[0], self.moead.population[7])
        self.assertIs(parents[1], self.moead.population[5])

    def test_one_random_and_current_parent_instance(self):
        with self.assertRaises(TypeError):
            OneRandomAndCurrentParentSelector(algorithm=self.problem)

    def test_one_random_and_current_parent_instance2(self):
        with self.assertRaises(TypeError):
            OneRandomAndCurrentParentSelector(algorithm=self.moead).select(self.moead)

    def test_one_random_and_current_parent(self):
        selector = OneRandomAndCurrentParentSelector(algorithm=self.moead)
        with patch("moead_framework.core.parent_selector.one_random_and_current_parent_selector.random.randint",
                   return_value=1):
            parents = selector.select(self.pool)
        self.assertEqual(len(parents), 2)
        self.assertIs(parents[0], self.moead.population[3])
        self.assertIs(parents[1], self.moead.population[2])

    def test_two_random_and_current_parent_instance(self):
        with self.assertRaises(TypeError):
            TwoRandomAndCurrentParentSelector(algorithm=self.problem)

    def test_two_random_and_current_parent_instance2(self):
        with self.assertRaises(TypeError):
            TwoRandomAndCurrentParentSelector(algorithm=self.moead).select(3)

    def test_two_random_and_current_parent(self):
        selector = TwoRandomAndCurrentParentSelector(algorithm=self.moead)
        with patch("moead_framework.core.parent_selector.two_random_and_current_parent_selector.random.randint",
                   side_effect=[2, 0]):
            parents = selector.select(self.pool)
        self.assertEqual(len(parents), 3)
        self.assertIs(parents[0], self.moead.population[3])
        self.assertIs(parents[1], self.moead.population[5])
        self.assertIs(parents[2], self.moead.population[7])
