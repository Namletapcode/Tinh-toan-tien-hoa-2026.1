"""Operation-sequence / machine-selection encoding, without framework coupling."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from .instance import FJSPInstance, _integer


@dataclass(frozen=True)
class Chromosome:
    os: tuple[int, ...]  # The kth occurrence of job j dispatches operation (j, k).
    ms: tuple[int, ...]  # Local alternative indices in fixed job/operation order.

    def __post_init__(self):
        object.__setattr__(self, "os", tuple(_integer(v, "OS gene") for v in self.os))
        object.__setattr__(self, "ms", tuple(_integer(v, "MS gene") for v in self.ms))

    def validate(self, instance: FJSPInstance) -> None:
        if len(self.os) != instance.operation_count or len(self.ms) != instance.operation_count:
            raise ValueError("OS and MS must each have one gene per operation")
        if any(not 0 <= job < instance.job_count for job in self.os):
            raise ValueError("OS contains an out-of-range job ID")
        counts = Counter(self.os)
        if any(counts[j] != size for j, size in enumerate(instance.operations_per_job)):
            raise ValueError("OS must repeat each job exactly as often as its operation count")
        index = 0
        for job in instance.jobs:
            for options in job:
                if not 0 <= self.ms[index] < len(options):
                    raise ValueError(f"MS gene {index} is not a valid local machine-choice index")
                index += 1

    def to_vector(self) -> tuple[int, ...]:
        return self.os + self.ms

    @classmethod
    def from_vector(cls, instance: FJSPInstance, vector: Sequence[int]) -> Chromosome:
        size = instance.operation_count
        if len(vector) != 2 * size:
            raise ValueError(f"Expected a one-dimensional OS+MS vector of length {2 * size}")
        chromosome = cls(tuple(vector[:size]), tuple(vector[size:]))
        chromosome.validate(instance)
        return chromosome


def random_chromosome(instance: FJSPInstance, rng: random.Random | None = None) -> Chromosome:
    """Uniformly shuffle the job multiset; independently choose each local MS gene."""
    rng = rng if rng is not None else random
    sequence = [j for j, count in enumerate(instance.operations_per_job) for _ in range(count)]
    rng.shuffle(sequence)
    choices = [rng.randrange(len(options)) for job in instance.jobs for options in job]
    return Chromosome(tuple(sequence), tuple(choices))
