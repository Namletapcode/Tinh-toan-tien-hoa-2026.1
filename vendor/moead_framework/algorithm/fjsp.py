"""Assemble the existing basic Moead class with FJSP-specific components."""

from numbers import Integral
from pathlib import Path
from tempfile import TemporaryDirectory

from moead_framework.aggregation import Tchebycheff
from moead_framework.algorithm.moead import Moead
from moead_framework.core.genetic_operator.fjsp import (
    FJSPCrossoverAndMutation, FJSPOperatorConfig,
)
from moead_framework.core.offspring_generator.fjsp import FJSPOffspringGenerator
from moead_framework.problem.fjsp import FJSPProblem


def uniform_weights_3d(partitions):
    """Integer simplex lattice: N=(H+1)(H+2)/2, including the three axes."""
    if isinstance(partitions, bool) or not isinstance(partitions, Integral) or partitions < 1:
        raise ValueError("partitions must be a positive integer")
    h = int(partitions)
    return tuple((i / h, j / h, (h - i - j) / h)
                 for i in range(h + 1) for j in range(h - i + 1))


def create_fjsp_moead(problem, max_evaluation=1000, number_of_weight_neighborhood=10,
                      partitions=12, operator_config=None):
    """Return a plain Moead, not a new algorithm variant.

    The upstream API loads weights from a file once during construction. A
    temporary file adapts the generated lattice to that API; no generated
    files or paths need to be kept in the project after construction.
    """
    if not isinstance(problem, FJSPProblem):
        raise TypeError("problem must be FJSPProblem")
    if problem.evaluation_count:
        raise ValueError("Use a fresh FJSPProblem for each MOEA/D run")
    config = FJSPOperatorConfig() if operator_config is None else operator_config
    if not isinstance(config, FJSPOperatorConfig):
        raise TypeError("operator_config must be FJSPOperatorConfig")
    weights = uniform_weights_3d(partitions)
    with TemporaryDirectory(prefix="moead-fjsp-") as directory:
        path = Path(directory) / "weights.ws"
        path.write_text("\n".join(" ".join(map(str, row)) for row in weights) + "\n", encoding="utf-8")
        algorithm = Moead(
            problem=problem, max_evaluation=max_evaluation,
            number_of_weight_neighborhood=number_of_weight_neighborhood,
            weight_file=str(path), aggregation_function=Tchebycheff,
            genetic_operator=FJSPCrossoverAndMutation,
            offspring_generator=FJSPOffspringGenerator,
        )
    algorithm.offspring_generator.config = config
    return algorithm
