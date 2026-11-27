"""Factory and initializer builder for MOEA/D-FJSP with Q-learning initialization."""

from dataclasses import fields
from pathlib import Path
from tempfile import TemporaryDirectory

from moead_framework.aggregation import Tchebycheff
from moead_framework.algorithm.fjsp import uniform_weights_2d
from moead_framework.algorithm.moead_qinit import MoeadQInit
from moead_framework.core.genetic_operator.fjsp import (
    FJSPCrossoverAndMutation,
    FJSPOperatorConfig,
)
from moead_framework.core.initializer import (
    AbstractInitializer,
    HybridQRandomInitializer,
    QInitConfig,
    QLearningInitializer,
)
from moead_framework.core.offspring_generator.fjsp import FJSPOffspringGenerator
from moead_framework.problem.fjsp import FJSPProblem


# "baseline" is the unchanged Moead (`create_fjsp_moead`); the others use MoeadQInit.
INITIALIZER_MODES = ("baseline", "qlearning", "hybrid_q_random")
_Q_INIT_FIELDS = {field.name for field in fields(QInitConfig)}


def parse_q_init_config(raw=None):
    """Parse an optional dict (or QInitConfig) into a validated QInitConfig."""
    if raw is None:
        return QInitConfig()
    if isinstance(raw, QInitConfig):
        return raw
    if not isinstance(raw, dict):
        raise TypeError("q_init settings must be a dictionary or QInitConfig")
    unknown = sorted(set(raw) - _Q_INIT_FIELDS)
    if unknown:
        raise ValueError(f"Unknown q_init parameter(s): {', '.join(unknown)}")
    return QInitConfig(**raw)


def build_initializer(mode="hybrid_q_random", q_config=None):
    """Instantiate a Q-learning initializer by mode name ("qlearning" or "hybrid_q_random")."""
    if mode not in INITIALIZER_MODES or mode == "baseline":
        raise ValueError(f"Unsupported initializer mode: {mode!r}")
    config = parse_q_init_config(q_config)
    if mode == "qlearning":
        return QLearningInitializer(config=config)
    return HybridQRandomInitializer(config=config)


def create_fjsp_qinit_moead(
    problem,
    max_evaluation=1000,
    number_of_weight_neighborhood=10,
    partitions=12,
    operator_config=None,
    initializer=None,
    q_config=None,
):
    """Assemble `MoeadQInit` with FJSP components and a pluggable initializer."""
    if not isinstance(problem, FJSPProblem):
        raise TypeError("problem must be FJSPProblem")
    if problem.evaluation_count:
        raise ValueError("Use a fresh FJSPProblem for each MOEA/D run")
    config = FJSPOperatorConfig() if operator_config is None else operator_config
    if not isinstance(config, FJSPOperatorConfig):
        raise TypeError("operator_config must be FJSPOperatorConfig")

    if initializer is None:
        resolved_initializer = HybridQRandomInitializer(config=parse_q_init_config(q_config))
    elif isinstance(initializer, str):
        resolved_initializer = build_initializer(initializer, q_config=q_config)
    elif isinstance(initializer, AbstractInitializer):
        resolved_initializer = initializer
    else:
        raise TypeError("initializer must be an AbstractInitializer, mode string, or None")

    weights = uniform_weights_2d(partitions)
    with TemporaryDirectory(prefix="moead-fjsp-qinit-") as directory:
        path = Path(directory) / "weights.ws"
        path.write_text("\n".join(" ".join(map(str, row)) for row in weights) + "\n", encoding="utf-8")
        algorithm = MoeadQInit(
            problem=problem,
            max_evaluation=max_evaluation,
            number_of_weight_neighborhood=number_of_weight_neighborhood,
            weight_file=str(path),
            aggregation_function=Tchebycheff,
            genetic_operator=FJSPCrossoverAndMutation,
            offspring_generator=FJSPOffspringGenerator,
            initializer=resolved_initializer,
        )
    algorithm.offspring_generator.config = config
    return algorithm
