"""Mating-pool selection checked independently of a benchmark problem."""

from pathlib import Path
import unittest
from unittest.mock import patch

from moead_framework.aggregation import Tchebycheff
from moead_framework.algorithm import Moead, MoeadDeltaNr
from moead_framework.core.selector import DeltaSelector
from moead_framework.test.fixtures import BinaryTestProblem
from moead_framework.tool.result import set_seed


class ParentSelectorTest(unittest.TestCase):
    def setUp(self):
        set_seed(1)
        self.problem = BinaryTestProblem()
        weights = str(Path(__file__).parent / "data/weights/SOBOL-2objs-10wei.ws")
        options = dict(problem=self.problem, max_evaluation=100,
                       number_of_weight_neighborhood=3, aggregation_function=Tchebycheff,
                       weight_file=weights)
        self.moead = Moead(**options)
        self.moead_delta_nr = MoeadDeltaNr(
            **options, number_of_objective=2, delta=0.9, number_of_replacement=2,
        )

    def test_delta_selector1(self):
        selector = DeltaSelector(algorithm_instance=self.moead_delta_nr)
        with patch("moead_framework.core.selector.delta_selector.random.random", return_value=0.1):
            local = selector.select(sub_problem=1)
        self.assertEqual(local, self.moead_delta_nr.b[1])
        self.assertEqual(len(local), 3)
        with patch("moead_framework.core.selector.delta_selector.random.random", return_value=0.95):
            global_pool = selector.select(sub_problem=1)
        self.assertEqual(global_pool, list(range(10)))

    def test_delta_selector2(self):
        selector = DeltaSelector(algorithm_instance=self.moead_delta_nr)
        with self.assertRaises(TypeError):
            selector.select(sub_problem="1")

    def test_delta_selector3(self):
        with self.assertRaises(AttributeError):
            DeltaSelector(algorithm_instance=self.moead)
