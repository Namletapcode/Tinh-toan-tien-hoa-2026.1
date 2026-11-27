import argparse
import json
from dataclasses import asdict
from hashlib import sha256
from time import perf_counter
from pathlib import Path
from moead_framework.algorithm.fjsp_dqce import *
from moead_framework.core.genetic_operator.fjsp import *
from moead_framework.core.reinforcement_learning.dqn_agent import *
from moead_framework.problem.fjsp import *
from moead_framework.tool.result import *


def main():
    root = Path(__file__).parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path, default=root / "instance/main/Brandimate1993/Loose/Mk01/Mk01.fjs")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--evaluations", type=int, default=1000)
    parser.add_argument("--partitions", type=int, default=12)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--crossover", type=float, default=0.8)
    parser.add_argument("--mutation", type=float, default=0.05)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--buffer-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--target-update", type=int, default=10)
    parser.add_argument("--no-compaction", action="store_true")
    parser.add_argument("--output", type=Path)

    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("Output already exists, choose a new filename to preserve the previous run")

    try:
        op_config = FJSPOperatorConfig(args.crossover, args.mutation)
        dqn_config = DQNConfig(
            learning_rate=args.lr,
            gamma=args.gamma,
            epsilon=args.epsilon,
            batch_size=args.batch_size,
            experience_pool_size=args.buffer_size,
            epochs=args.epochs,
            target_update_freq=args.target_update
        )
        problem = FJSPProblem(args.instance)
        instance_hash = sha256(args.instance.read_bytes()).hexdigest()
        set_seed(args.seed)
        start = perf_counter()
        algorithm = create_fjsp_dqce(
            problem,
            max_evaluation=args.evaluations,
            number_of_weight_neighborhood=args.neighbors,
            partitions=args.partitions,
            operator_config=op_config,
            dqn_config=dqn_config,
            use_compaction=not args.no_compaction
        )

    except (OSError, ValueError, TypeError, ImportError) as error:
        parser.error(str(error))

    archive = sorted(algorithm.run(), key=lambda solution: tuple(solution.F))
    elapsed = perf_counter() - start

    if algorithm.current_eval != problem.evaluation_count or problem.evaluation_count != args.evaluations:
        raise RuntimeError(
            f"Budget mismatch: current_eval={algorithm.current_eval}, "
            f"problem.evaluation_count={problem.evaluation_count}, requested={args.evaluations}"
        )

    for solution in archive + algorithm.population + algorithm.parasite_population:
        problem.validate_solution(solution)

    summary = {
        "instance": problem.instance.name,
        "instance_sha256": instance_hash,
        "algorithm": "FjspDqceMoead",
        "aggregation": "Tchebycheff",
        "seed": args.seed,
        "objective_names": ["makespan", "twt"],
        "directions": ["minimize"] * 2,
        "partitions": args.partitions,
        "population_size": algorithm.number_of_weight,
        "neighbors": args.neighbors,
        **asdict(op_config),
        "dqn": asdict(dqn_config),
        "use_compaction": not args.no_compaction,
        "requested_evaluations": args.evaluations,
        "current_eval": algorithm.current_eval,
        "actual_evaluations": problem.evaluation_count,
        "elapsed_seconds": round(elapsed, 6),
        "archive_size": len(archive),
        "parasite_size": len(algorithm.parasite_population),
        "objective_vectors": [solution.F for solution in archive],
        "operator_selection_history": algorithm.operator_history,
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
