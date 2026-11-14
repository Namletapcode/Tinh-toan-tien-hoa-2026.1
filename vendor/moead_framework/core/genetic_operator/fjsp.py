"""Fixed-rate, feasibility-preserving OS/MS variation; no objective evaluations."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real
import random

import numpy as np

from moead_framework.core.genetic_operator import GeneticOperator
from moead_framework.problem.fjsp import Chromosome


@dataclass(frozen=True)
class FJSPOperatorConfig:
    crossover_probability: float = 0.8
    mutation_probability: float = 0.05

    def __post_init__(self):
        for name in ("crossover_probability", "mutation_probability"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{name} must be a finite probability in [0, 1]")
            object.__setattr__(self, name, float(value))


class FJSPCrossoverAndMutation(GeneticOperator):
    def __init__(self, solutions, instance, config=None):
        if not isinstance(solutions, list) or len(solutions) != 2:
            raise ValueError("FJSP variation requires exactly two parent vectors")
        self.instance = instance
        self.config = FJSPOperatorConfig() if config is None else config
        if not isinstance(self.config, FJSPOperatorConfig):
            raise TypeError("config must be FJSPOperatorConfig")
        self.parents = tuple(Chromosome.from_vector(instance, vector) for vector in solutions)
        super().__init__([np.array(parent.to_vector(), dtype=int) for parent in self.parents])
        self.option_counts = tuple(len(options) for job in instance.jobs for options in job)

    def _pox(self, first, second):
        if self.instance.job_count < 2:
            return list(first)
        size = random.randint(1, self.instance.job_count - 1)
        kept_jobs = set(random.sample(range(self.instance.job_count), size))
        fill = iter(job for job in second if job not in kept_jobs)
        return [job if job in kept_jobs else next(fill) for job in first]

    def run(self):
        first, second = self.parents
        sequence, choices = list(first.os), list(first.ms)
        if random.random() < self.config.crossover_probability:
            sequence = self._pox(first.os, second.os)
            choices = [a if random.random() < 0.5 else b for a, b in zip(first.ms, second.ms)]

        # Each segment gets an independent, fixed-probability mutation attempt.
        if len(sequence) >= 2 and random.random() < self.config.mutation_probability:
            a, b = random.sample(range(len(sequence)), 2)
            sequence[a], sequence[b] = sequence[b], sequence[a]
        mutable = [i for i, count in enumerate(self.option_counts) if count > 1]
        if mutable and random.random() < self.config.mutation_probability:
            index = random.choice(mutable)
            choice = random.randrange(self.option_counts[index] - 1)
            choices[index] = choice + (choice >= choices[index])

        child = Chromosome(tuple(sequence), tuple(choices))
        child.validate(self.instance)
        return np.array(child.to_vector(), dtype=int)
