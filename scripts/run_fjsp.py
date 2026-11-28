"""Run MOEA/D on a bi-objective FJSP instance (Makespan, TWT).

The default `--initializer baseline` runs the unchanged basic Moead; the other modes
only replace population initialization (see docs/QINIT.md).
"""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from moead_framework.algorithm.fjsp import create_fjsp_moead
from moead_framework.algorithm.fjsp_qinit import (
    INITIALIZER_MODES,
    build_initializer,
    create_fjsp_qinit_moead,
    parse_q_init_config,
)
from moead_framework.algorithm.fjsp_variants import (
    VARIANTS, create_fjsp_variant, population_layout, population_solutions,
)
from moead_framework.core.genetic_operator.fjsp import FJSPOperatorConfig
from moead_framework.problem.fjsp import FJSPProblem
from moead_framework.tool.result import set_seed


Q_FLAGS = (
    ("--q-alpha", "alpha", float),
    ("--q-gamma", "gamma", float),
    ("--q-epsilon-start", "epsilon_start", float),
    ("--q-epsilon-end", "epsilon_end", float),
    ("--q-random-ratio", "random_ratio", float),
    ("--q-progress-bins", "progress_bins", int),
    ("--q-weight-bins", "weight_bins", int),
    ("--q-max-resample", "max_resample", int),
)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path,
                        default=root / "instance/main/Brandimate1993/Loose/Mk01/Mk01.fjs")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--evaluations", type=int, default=1000, help="Total NFE, including initialization")
    parser.add_argument("--variant", choices=VARIANTS, default="baseline")
    parser.add_argument("--micro-population-size", type=int, default=4,
                        help="Residents per group for micro variants (>=4); ignored by baseline")
    parser.add_argument("--partitions", type=int, default=12, help="H; number of groups is H+1")
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--crossover", type=float, default=0.8)
    parser.add_argument("--mutation", type=float, default=0.05)
    parser.add_argument("--initializer", choices=INITIALIZER_MODES, default="baseline",
                        help="Population initialization strategy (baseline keeps original Moead)")
    # Q-learning hyperparameters; omitted values keep the QInitConfig defaults.
    for flag, field, kind in Q_FLAGS:
        parser.add_argument(flag, dest=field, type=kind, default=None)
    parser.add_argument("--output", type=Path, help="Save full archive, genes and schedules as a new JSON file")
    args = parser.parse_args()
    given_q_flags = [flag for flag, field, _ in Q_FLAGS if getattr(args, field) is not None]
    if args.initializer == "baseline" and given_q_flags:
        parser.error(f"{', '.join(given_q_flags)} require --initializer qlearning or hybrid_q_random")
    if args.output is not None and args.output.exists():
        parser.error("Output already exists; choose a new filename to preserve the previous run")
    try:
        config = FJSPOperatorConfig(args.crossover, args.mutation)
        q_raw = {field: getattr(args, field) for _, field, _ in Q_FLAGS if getattr(args, field) is not None}
        settings = dict(variant=args.variant, micro_population_size=args.micro_population_size,
                        partitions=args.partitions, neighbors=args.neighbors, evaluations=args.evaluations,
                        crossover=args.crossover, mutation=args.mutation, initializer=args.initializer)
        q_config = None
        if args.initializer != "baseline":
            settings["q_init"] = q_raw
            q_config = parse_q_init_config(q_raw)
        layout = population_layout(settings)
        problem = FJSPProblem(args.instance)
        instance_hash = sha256(args.instance.read_bytes()).hexdigest()
        set_seed(args.seed)
        start = perf_counter()
        if args.variant != "baseline":
            algorithm = create_fjsp_variant(problem, settings)
        elif args.initializer == "baseline":
            algorithm = create_fjsp_moead(
                problem, max_evaluation=args.evaluations,
                number_of_weight_neighborhood=args.neighbors,
                partitions=args.partitions, operator_config=config,
            )
        else:
            algorithm = create_fjsp_qinit_moead(
                problem, max_evaluation=args.evaluations,
                number_of_weight_neighborhood=args.neighbors,
                partitions=args.partitions, operator_config=config,
                initializer=build_initializer(args.initializer, q_config=q_config),
            )
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))

    archive = sorted(algorithm.run(), key=lambda solution: tuple(solution.F))
    elapsed = perf_counter() - start
    if algorithm.current_eval != problem.evaluation_count or problem.evaluation_count != args.evaluations:
        raise RuntimeError("Algorithm NFE, actual evaluations and requested budget disagree")
    for solution in archive + population_solutions(algorithm) + algorithm.population:
        problem.validate_solution(solution)
    summary = {
        "instance": problem.instance.name,
        "instance_sha256": instance_hash,
        "algorithm": "Moead" if args.variant == "baseline" and args.initializer == "baseline" else type(algorithm).__name__,
        **layout,
        "aggregation": "Tchebycheff",
        "seed": args.seed,
        "objective_names": ["makespan", "twt"],
        "directions": ["minimize"] * 2,
        "partitions": args.partitions,
        "neighbors": args.neighbors,
        **asdict(config),
        "requested_evaluations": args.evaluations,
        "current_eval": algorithm.current_eval,
        "actual_evaluations": problem.evaluation_count,
        "elapsed_seconds": round(elapsed, 6),
        "archive_size": len(archive),
        "objective_vectors": [solution.F for solution in archive],
        "all_schedules_valid": True,
    }
    if args.initializer != "baseline":
        summary["initializer"] = args.initializer
        summary["initializer_stats"] = algorithm.initializer.last_stats
        summary["q_init"] = asdict(q_config)
    if args.output is not None:
        export = {
            "schema_version": 1,
            "index_base": 0,
            "run": summary,
            "weights": [weight.tolist() for weight in algorithm.weights],
            "solutions": [{
                "objectives": solution.F,
                "decision_vector": solution.decision_vector.tolist(),
                "machine_loads": list(solution.schedule.machine_loads),
                "schedule": [asdict(operation) for operation in solution.schedule.operations],
            } for solution in archive],
        }
        if args.variant != "baseline":
            export["micro_populations"] = [
                [{"objectives": resident.F,
                  "decision_vector": resident.decision_vector.tolist()} for resident in group]
                for group in algorithm.micro_populations
            ]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as file:
            json.dump(export, file, ensure_ascii=False, indent=2)
            file.write("\n")
        summary["output"] = str(args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
