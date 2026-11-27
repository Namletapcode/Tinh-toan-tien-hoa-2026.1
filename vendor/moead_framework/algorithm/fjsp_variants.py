"""All runners use thismodule to agree on variant names, group count and actual initialization cost.
"""

from numbers import Integral
from pathlib import Path
from tempfile import TemporaryDirectory

from moead_framework.aggregation import Tchebycheff
from moead_framework.algorithm.fjsp import create_fjsp_moead, uniform_weights_2d
from moead_framework.algorithm.moead_micro_gwo import MoeadMicroGWO
from moead_framework.core.genetic_operator.fjsp import FJSPCrossoverAndMutation, FJSPOperatorConfig
from moead_framework.core.offspring_generator.fjsp import FJSPOffspringGenerator
from moead_framework.problem.fjsp import FJSPProblem


VARIANTS = ("baseline", "micro_random", "micro_gwo")


def population_layout(settings):
    #Kiểm tra tên chế độ, tính số nhóm, tổng cá thể và NFE khởi tạo
    variant = settings.get("variant", "baseline")
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    group_count = len(uniform_weights_2d(settings["partitions"]))
    size = 1 if variant == "baseline" else settings.get("micro_population_size", 4)
    if (isinstance(size, bool) or not isinstance(size, Integral)
            or (variant != "baseline" and size < 4)):
        raise ValueError("micro_population_size must be an integer >= 4")
    return {"variant": variant, "number_of_subproblems": group_count,
            "micro_population_size": int(size), "population_size": group_count * int(size),
            "initial_evaluations": group_count * int(size)}


def create_fjsp_variant(problem, settings):
    #Tạo thuật toán tương ứng cấu hình
    layout = population_layout(settings)
    config = FJSPOperatorConfig(settings["crossover"], settings["mutation"])
    if not isinstance(problem, FJSPProblem):
        raise TypeError("problem must be FJSPProblem")
    if problem.evaluation_count:
        raise ValueError("Use a fresh FJSPProblem for each run")
    kwargs = dict(problem=problem, max_evaluation=settings["evaluations"],
                  number_of_weight_neighborhood=settings["neighbors"])
    if layout["variant"] == "baseline":
        return create_fjsp_moead(
            **kwargs, partitions=settings["partitions"], operator_config=config,
        )
    weights = uniform_weights_2d(settings["partitions"])
    with TemporaryDirectory(prefix="moead-micro-") as directory:
        path = Path(directory) / "weights.ws"
        path.write_text("\n".join(" ".join(map(str, row)) for row in weights), encoding="utf-8")
        algorithm = MoeadMicroGWO(
            **kwargs, micro_population_size=layout["micro_population_size"],
            parent_selection="grey_wolf" if layout["variant"] == "micro_gwo" else "random",
            weight_file=str(path), aggregation_function=Tchebycheff,
            genetic_operator=FJSPCrossoverAndMutation, offspring_generator=FJSPOffspringGenerator,
        )
    algorithm.offspring_generator.config = config
    return algorithm


def population_solutions(algorithm):
    #Lấy toàn bộ cá thể để kiểm tra tính hợp lệ
    if isinstance(algorithm, MoeadMicroGWO):
        return algorithm.all_population_solutions()
    return algorithm.population
