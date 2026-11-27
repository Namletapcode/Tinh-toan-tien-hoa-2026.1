"""Evaluate FjspDqceMoead across seeds using normalized HV with greywolf's baseline config."""

import argparse
import csv
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, stdev
import sys
from time import perf_counter

import torch

from moead_framework.algorithm.fjsp_dqce import create_fjsp_dqce
from moead_framework.core.genetic_operator.fjsp import FJSPOperatorConfig
from moead_framework.core.reinforcement_learning.dqn_agent import DQNConfig
from moead_framework.problem.fjsp import FJSPProblem
from moead_framework.tool.fjsp_experiments import normalized_hypervolume, objective_bounds
from moead_framework.tool.result import set_seed


ROOT = Path(__file__).resolve().parents[1]
MODERATE_SIX_INSTANCES = [
    ROOT / "instance/main/Brandimate1993/Moderate/Mk01/Mk01.fjs",
    ROOT / "instance/main/Brandimate1993/Moderate/Mk02/Mk02.fjs",
    ROOT / "instance/main/Brandimate1993/Moderate/Mk03/Mk03.fjs",
    ROOT / "instance/main/DauzerePeresPaulli1994/Moderate/01a/01a.fjs",
    ROOT / "instance/main/DauzerePeresPaulli1994/Moderate/02a/02a.fjs",
    ROOT / "instance/main/DauzerePeresPaulli1994/Moderate/03a/03a.fjs",
]
DEFAULT_SEEDS = list(range(10))


def _instance_id(path):
    path = Path(path).resolve()
    due_levels = {"loose", "moderate", "tight"}
    due_candidate = path.parent.parent.name.lower()
    if due_candidate in due_levels:
        return f"{path.stem.lower()}_{due_candidate}"
    return path.stem.lower()


def _mean_std(values):
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _run_dqce_single(instance_path, instance_id, instance_sha256, seed, evaluations, partitions,
                     neighbors, op_config, dqn_config, use_compaction, bounds, hv_reference):
    problem = FJSPProblem(instance_path)
    set_seed(seed)
    torch.manual_seed(seed)

    start = perf_counter()
    algorithm = create_fjsp_dqce(
        problem,
        max_evaluation=evaluations,
        number_of_weight_neighborhood=neighbors,
        partitions=partitions,
        operator_config=op_config,
        dqn_config=dqn_config,
        use_compaction=use_compaction,
    )
    archive = sorted(algorithm.run(), key=lambda sol: tuple(sol.F))
    elapsed = perf_counter() - start

    if algorithm.current_eval != problem.evaluation_count or problem.evaluation_count != evaluations:
        raise RuntimeError(
            f"DQCE {instance_id} seed={seed} budget mismatch: "
            f"current_eval={algorithm.current_eval}, "
            f"actual={problem.evaluation_count}, requested={evaluations}"
        )

    for sol in archive + algorithm.population + algorithm.parasite_population:
        problem.validate_solution(sol)

    front = [list(sol.F) for sol in archive]
    hv = normalized_hypervolume(front, bounds, reference=hv_reference)

    run_meta = {
        "instance_id": instance_id,
        "instance": problem.instance.name,
        "instance_sha256": instance_sha256,
        "algorithm": "FjspDqceMoead",
        "seed": seed,
        "partitions": partitions,
        "population_size": algorithm.number_of_weight,
        "neighbors": neighbors,
        "evaluations": evaluations,
        "actual_evaluations": problem.evaluation_count,
        "crossover_probability": op_config.crossover_probability,
        "mutation_probability": op_config.mutation_probability,
        "dqn": asdict(dqn_config),
        "use_compaction": use_compaction,
        "elapsed_seconds": round(elapsed, 6),
        "archive_size": len(archive),
        "parasite_size": len(algorithm.parasite_population),
        "hv": hv,
        "hv_reference": list(hv_reference),
        "objective_vectors": front,
        "min_objectives": [min(pt[0] for pt in front), min(pt[1] for pt in front)],
        "operator_selection_history": algorithm.operator_history,
        "all_schedules_valid": True,
    }

    return {
        "schema_version": 1,
        "index_base": 0,
        "run": run_meta,
        "solutions": [{
            "objectives": sol.F,
            "decision_vector": sol.decision_vector.tolist(),
            "machine_loads": list(sol.schedule.machine_loads),
            "schedule": [asdict(op) for op in sol.schedule.operations],
        } for sol in archive],
    }


def _write_json(path, value):
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_or_run(run_path, expected_config, reuse, runner):
    if run_path.exists():
        if not reuse:
            raise ValueError(f"Output already exists (use --reuse or a new --output-dir): {run_path}")
        payload = json.loads(run_path.read_text(encoding="utf-8"))
        run = payload["run"]
        for key, value in expected_config.items():
            if run.get(key) != value:
                raise ValueError(
                    f"Existing run {run_path} mismatches {key}: {run.get(key)!r} != {value!r}"
                )
        return payload, True

    payload = runner()
    _write_json(run_path, payload)
    return payload, False


def evaluate_dqce(instance_paths, output_dir, seeds=None, evaluations=1000, partitions=12,
                  neighbors=10, crossover=0.8, mutation=0.05, dqn_config=None,
                  use_compaction=True, hv_reference=(1.1, 1.1), reuse=False):
    resolved_paths = [Path(p).resolve() for p in instance_paths]
    if not resolved_paths:
        raise ValueError("At least one instance path is required")
    output_dir = Path(output_dir).resolve()
    seeds = list(DEFAULT_SEEDS if seeds is None else seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and distinct")

    instance_ids = [_instance_id(p) for p in resolved_paths]
    if len(set(instance_ids)) != len(instance_ids):
        raise ValueError("Instance IDs must be distinct")

    op_config = FJSPOperatorConfig(crossover, mutation)
    dqn_cfg = DQNConfig() if dqn_config is None else dqn_config

    if output_dir.exists() and not reuse:
        raise ValueError(f"Output directory already exists: {output_dir} (pass --reuse to reuse runs)")

    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows = []
    summaries = []
    bounds_by_id = {}
    inputs_meta = {}
    total_runs = len(resolved_paths) * len(seeds)
    done = 0

    for inst_path, inst_id in zip(resolved_paths, instance_ids):
        inst_bytes = inst_path.read_bytes()
        inst_sha256 = sha256(inst_bytes).hexdigest()
        ref_problem = FJSPProblem(inst_path)
        bounds = objective_bounds(ref_problem.instance)
        bounds_by_id[inst_id] = bounds
        inputs_meta[inst_id] = {"path": str(inst_path), "sha256": inst_sha256}

        inst_rows = []
        for seed in seeds:
            run_path = runs_dir / f"{inst_id}_seed{seed}.json"
            expected = {
                "instance_id": inst_id,
                "instance": ref_problem.instance.name,
                "instance_sha256": inst_sha256,
                "algorithm": "FjspDqceMoead",
                "seed": seed,
                "evaluations": evaluations,
                "partitions": partitions,
                "neighbors": neighbors,
                "crossover_probability": op_config.crossover_probability,
                "mutation_probability": op_config.mutation_probability,
                "dqn": asdict(dqn_cfg),
                "use_compaction": use_compaction,
                "hv_reference": list(hv_reference),
            }
            payload, reused = _load_or_run(
                run_path,
                expected,
                reuse,
                lambda p=inst_path, iid=inst_id, sha=inst_sha256, b=bounds, s=seed: _run_dqce_single(
                    p, iid, sha, s, evaluations, partitions, neighbors,
                    op_config, dqn_cfg, use_compaction, b, hv_reference,
                ),
            )

            run = payload["run"]
            row = {
                "instance": inst_id,
                "seed": seed,
                "hv": run["hv"],
                "best_makespan": run["min_objectives"][0],
                "best_twt": run["min_objectives"][1],
                "archive_size": run["archive_size"],
                "parasite_size": run["parasite_size"],
                "evaluations": run["actual_evaluations"],
                "elapsed_seconds": run["elapsed_seconds"],
            }
            inst_rows.append(row)
            per_seed_rows.append(row)
            done += 1
            print(
                f"[{done}/{total_runs}] {inst_id} seed={seed}: "
                f"HV={row['hv']:.6f}, best=({row['best_makespan']}, {row['best_twt']}), "
                f"{row['elapsed_seconds']:.2f}s" + (" (reused)" if reused else ""),
                file=sys.stderr,
                flush=True,
            )

        hv_stats = _mean_std([r["hv"] for r in inst_rows])
        cmax_stats = _mean_std([r["best_makespan"] for r in inst_rows])
        twt_stats = _mean_std([r["best_twt"] for r in inst_rows])
        arch_stats = _mean_std([r["archive_size"] for r in inst_rows])
        sec_stats = _mean_std([r["elapsed_seconds"] for r in inst_rows])

        summaries.append({
            "instance": inst_id,
            "num_seeds": len(seeds),
            "evaluations": evaluations,
            "partitions": partitions,
            "neighbors": neighbors,
            "crossover": op_config.crossover_probability,
            "mutation": op_config.mutation_probability,
            "hv_mean": hv_stats["mean"],
            "hv_std": hv_stats["std"],
            "hv_min": hv_stats["min"],
            "hv_max": hv_stats["max"],
            "makespan_best": int(cmax_stats["min"]),
            "makespan_mean": cmax_stats["mean"],
            "makespan_std": cmax_stats["std"],
            "twt_best": int(twt_stats["min"]),
            "twt_mean": twt_stats["mean"],
            "twt_std": twt_stats["std"],
            "archive_size_mean": arch_stats["mean"],
            "seconds_mean": sec_stats["mean"],
            "seconds_std": sec_stats["std"],
        })

    summary = {
        "algorithm": "FjspDqceMoead",
        "hv_reference": list(hv_reference),
        "seeds": seeds,
        "num_seeds": len(seeds),
        "algorithm_config": {
            "evaluations": evaluations,
            "partitions": partitions,
            "population_size": partitions + 1,
            "neighbors": neighbors,
            "crossover_probability": op_config.crossover_probability,
            "mutation_probability": op_config.mutation_probability,
            "dqn": asdict(dqn_cfg),
            "use_compaction": use_compaction,
        },
        "inputs": inputs_meta,
        "bounds": bounds_by_id,
        "summaries": summaries,
        "per_seed": per_seed_rows,
    }

    md_lines = [
        "# Đánh giá HV của FjspDqceMoead",
        "",
        f"- Tổng số lượt chạy: {total_runs} ({len(resolved_paths)} instance × {len(seeds)} seed `{seeds[0]}..{seeds[-1]}`)",
        f"- Cấu hình (khớp baseline greywolf): `evaluations={evaluations}`, `partitions={partitions}` (N={partitions + 1}), "
        f"`neighbors={neighbors}`, `crossover={op_config.crossover_probability}`, `mutation={op_config.mutation_probability}`, "
        f"`use_compaction={use_compaction}`",
        f"- HV reference: `{list(hv_reference)}`",
        "",
        "| Instance | HV mean +/- SD | Best Cmax | Best TWT | Mean seconds |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for s in summaries:
        md_lines.append(
            f"| {s['instance']} | {s['hv_mean']:.6f} +/- {s['hv_std']:.6f} | "
            f"{s['makespan_best']} | {s['twt_best']} | {s['seconds_mean']:.3f} |"
        )

    _write_json(output_dir / "summary.json", summary)
    _write_csv(output_dir / "per_seed.csv", per_seed_rows)
    _write_csv(output_dir / "summary.csv", summaries)
    (output_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path, nargs="+", default=MODERATE_SIX_INSTANCES)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
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

    parser.add_argument("--hv-reference", type=float, nargs=2, default=[1.1, 1.1])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/eval_dqce_moderate_six")
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()

    try:
        dqn_config = DQNConfig(
            learning_rate=args.lr,
            gamma=args.gamma,
            epsilon=args.epsilon,
            batch_size=args.batch_size,
            experience_pool_size=args.buffer_size,
            epochs=args.epochs,
            target_update_freq=args.target_update,
        )
        summary = evaluate_dqce(
            instance_paths=args.instance,
            output_dir=args.output_dir,
            seeds=args.seeds,
            evaluations=args.evaluations,
            partitions=args.partitions,
            neighbors=args.neighbors,
            crossover=args.crossover,
            mutation=args.mutation,
            dqn_config=dqn_config,
            use_compaction=not args.no_compaction,
            hv_reference=tuple(args.hv_reference),
            reuse=args.reuse,
        )
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))

    print(json.dumps({
        "output_dir": str(Path(args.output_dir).resolve()),
        "algorithm_config": summary["algorithm_config"],
        "summaries": summary["summaries"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
