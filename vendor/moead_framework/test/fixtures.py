"""Small, explicit inputs for component tests; never used by the FJSP runner."""

import random

import numpy as np

from moead_framework.problem import Problem
from moead_framework.solution import OneDimensionSolution


class BinaryTestProblem(Problem):
    """Six bits, two hand-computable costs, with no external instance files."""

    def __init__(self):
        super().__init__(objective_number=2)

    def f(self, function_id, decision_vector):
        if function_id == 0:
            return float(1 + sum((i + 1) * bit for i, bit in enumerate(decision_vector)))
        return float(1 + sum((6 - i) * (1 - bit) for i, bit in enumerate(decision_vector)))

    def generate_random_solution(self):
        return self.evaluate([random.randrange(2) for _ in range(6)])


def make_solutions(points):
    """Assign distinct decision vectors to supplied objective vectors."""
    return [
        OneDimensionSolution(np.array([index, index + 1], dtype=float), f=list(point))
        for index, point in enumerate(points)
    ]
