"""Run the basic MOEA/D baseline on a three-objective FJSP instance."""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from moead_framework.algorithm.fjsp import create_fjsp_moead
from moead_framework.core.genetic_operator.fjsp import FJSPOperatorConfig
from moead_framework.problem.fjsp import FJSPProblem
from moead_framework.tool.result import set_seed


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path,
                        default=root / "instance/FattahiMehrabadJolai2007/fattahi1.txt")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--evaluations", type=int, default=1000, help="Total NFE, including initialization")
    parser.add_argument("--partitions", type=int, default=12, help="H; population size is (H+1)(H+2)/2")
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--crossover", type=float, default=0.8)
    parser.add_argument("--mutation", type=float, default=0.05)
    parser.add_argument("--output", type=Path, help="Save full archive, genes and schedules as a new JSON file")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("Output already exists; choose a new filename to preserve the previous run")
    try:
        config = FJSPOperatorConfig(args.crossover, args.mutation)
        problem = FJSPProblem(args.instance)
        instance_hash = sha256(args.instance.read_bytes()).hexdigest()
        set_seed(args.seed)
        start = perf_counter()
        algorithm = create_fjsp_moead(
            problem, max_evaluation=args.evaluations,
            number_of_weight_neighborhood=args.neighbors,
            partitions=args.partitions, operator_config=config,
        )
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))

    archive = sorted(algorithm.run(), key=lambda solution: tuple(solution.F))
    elapsed = perf_counter() - start
    if algorithm.current_eval != problem.evaluation_count or problem.evaluation_count != args.evaluations:
        raise RuntimeError("Algorithm NFE, actual evaluations and requested budget disagree")
    for solution in archive + algorithm.population:
        problem.validate_solution(solution)
    summary = {
        "instance": problem.instance.name,
        "instance_sha256": instance_hash,
        "algorithm": "Moead",
        "aggregation": "Tchebycheff",
        "seed": args.seed,
        "objective_names": ["makespan", "total_workload", "max_machine_workload"],
        "directions": ["minimize"] * 3,
        "partitions": args.partitions,
        "population_size": algorithm.number_of_weight,
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
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as file:
            json.dump(export, file, ensure_ascii=False, indent=2)
            file.write("\n")
        summary["output"] = str(args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
