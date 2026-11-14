"""Adapter between standalone FJSP decoding and the framework's Problem API."""

from numbers import Integral

import numpy as np

from moead_framework.problem.problem import Problem
from moead_framework.solution import OneDimensionSolution
from moead_framework.solution.base import Solution

from .decoder import Schedule, decode, validate_schedule
from .encoding import Chromosome, random_chromosome
from .instance import FJSPInstance, load_instance


class FJSPProblem(Problem):
    dtype = int

    def __init__(self, instance):
        super().__init__(objective_number=3)
        self.instance = instance if isinstance(instance, FJSPInstance) else load_instance(instance)
        self.evaluation_count = 0

    def evaluate(self, x):
        """Decode once, obtain all three objectives and retain the immutable schedule.

        Invalid genes are rejected rather than rounded or silently repaired.
        Re-evaluation always decodes again; it never trusts an old schedule cache.
        """
        vector = x.decision_vector if isinstance(x, Solution) else x
        chromosome = Chromosome.from_vector(self.instance, vector)
        schedule = decode(self.instance, chromosome)
        validate_schedule(self.instance, schedule)
        solution = x if isinstance(x, OneDimensionSolution) else OneDimensionSolution([])
        solution.decision_vector = np.array(chromosome.to_vector(), dtype=self.dtype)
        solution.F = list(schedule.objectives)
        solution.schedule = schedule
        self.evaluation_count += 1
        return solution

    def f(self, function_id, decision_vector):
        """Compatibility accessor; normal MOEA/D calls evaluate(), not f() three times."""
        if isinstance(function_id, bool) or not isinstance(function_id, Integral) or not 0 <= function_id < 3:
            raise ValueError("FJSP objective index must be 0, 1 or 2")
        return self.evaluate(decision_vector).F[function_id]

    def generate_random_solution(self):
        return self.evaluate(random_chromosome(self.instance).to_vector())

    def validate_solution(self, solution):
        """Check cached results against both genes and feasibility, without another decode/NFE."""
        chromosome = Chromosome.from_vector(self.instance, solution.decision_vector)
        schedule = getattr(solution, "schedule", None)
        if not isinstance(schedule, Schedule):
            raise ValueError("FJSP solution has no evaluated schedule")
        validate_schedule(self.instance, schedule)
        if tuple(op.job for op in schedule.operations) != chromosome.os:
            raise ValueError("Cached schedule does not match OS")
        offsets = self.instance.operation_offsets
        for op in schedule.operations:
            choice = chromosome.ms[offsets[op.job] + op.operation]
            if self.instance.jobs[op.job][op.operation][choice].machine != op.machine:
                raise ValueError("Cached schedule does not match MS")
        if tuple(solution.F) != schedule.objectives:
            raise ValueError("Cached objective vector does not match the schedule")
