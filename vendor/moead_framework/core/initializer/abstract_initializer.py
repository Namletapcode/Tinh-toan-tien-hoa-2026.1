"""Abstract interface and shared validation for population initialization."""

from abc import ABC, abstractmethod
from numbers import Integral


def validate_population_size(population_size):
    """Ensure population_size is a positive integer (excluding bool)."""
    if isinstance(population_size, bool) or not isinstance(population_size, Integral) or population_size < 1:
        raise ValueError("population_size must be a positive integer")
    return int(population_size)


class AbstractInitializer(ABC):
    """Base interface for MOEA/D population initializers (`Abstract initialize`)."""

    @abstractmethod
    def initialize(self, problem, population_size, weights, on_evaluated):
        """Generate and evaluate `population_size` initial solutions.

        :param problem: Problem instance implementing `evaluate()` and `generate_random_solution()`.
        :param population_size: Exact number of initial solutions (one per subproblem, one NFE each).
        :param weights: Sequence of decomposition weight vectors.
        :param on_evaluated: Callback invoked once per evaluated candidate to update NFE and archive.
        :return: list of evaluated solutions of length `population_size`.
        """
