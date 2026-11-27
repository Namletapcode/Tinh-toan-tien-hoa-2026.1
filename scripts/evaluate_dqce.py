"""Evaluate FjspDqceMoead vs baseline Moead across seeds using normalized HV and C(A, B)."""

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

from moead_framework.algorithm.fjsp import create_fjsp_moead
from moead_framework.algorithm.fjsp_dqce import create_fjsp_dqce
from moead_framework.core.genetic_operator.fjsp import FJSPOperatorConfig
from moead_framework.core.reinforcement_learning.dqn_agent import DQNConfig
from moead_framework.problem.fjsp import FJSPProblem
from moead_framework.tool.fjsp_experiments import (
    independent_pareto_front, normalized_hypervolume, objective_bounds, set_coverage,
)
from moead_framework.tool.result import set_seed


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INSTANCE = ROOT / "instance/main/Brandimate1993/Moderate/Mk01/Mk01.fjs"
DEFAULT_SEEDS = list(range(10))


def _mean_std(values):
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _run_single(algorithm_kind, instance_path, instance_sha256, seed, evaluations, partitions,
                neighbors, op_config, dqn_config, use_compaction, bounds, hv_reference):
    problem = FJSPProblem(instance_path)
    set_seed(seed)
    torch.manual_seed(seed)

    start = perf_counter()
    if algorithm_kind == "dqce":
        algorithm = create_fjsp_dqce(
            problem,
            max_evaluation=evaluations,
            number_of_weight_neighborhood=neighbors,
            partitions=partitions,
            operator_config=op_config,
            dqn_config=dqn_config,
            use_compaction=use_compaction,
        )
    elif algorithm_kind == "baseline":
        algorithm = create_fjsp_moead(
            problem,
            max_evaluation=evaluations,
            number_of_weight_neighborhood=neighbors,
            partitions=partitions,
            operator_config=op_config,
        )
    else:
        raise ValueError(f"Unsupported algorithm kind: {algorithm_kind}")

    archive = sorted(algorithm.run(), key=lambda sol: tuple(sol.F))
    elapsed = perf_counter() - start

    if algorithm.current_eval != problem.evaluation_count or problem.evaluation_count != evaluations:
        raise RuntimeError(
            f"{algorithm_kind} seed={seed} budget mismatch: "
            f"current_eval={algorithm.current_eval}, "
            f"actual={problem.evaluation_count}, requested={evaluations}"
        )

    to_validate = archive + algorithm.population
    if algorithm_kind == "dqce":
        to_validate += algorithm.parasite_population
    for sol in to_validate:
        problem.validate_solution(sol)

    front = [list(sol.F) for sol in archive]
    hv = normalized_hypervolume(front, bounds, reference=hv_reference)

    run_meta = {
        "instance": problem.instance.name,
        "instance_sha256": instance_sha256,
        "algorithm": "FjspDqceMoead" if algorithm_kind == "dqce" else "Moead",
        "seed": seed,
        "partitions": partitions,
        "population_size": algorithm.number_of_weight,
        "neighbors": neighbors,
        "evaluations": evaluations,
        "actual_evaluations": problem.evaluation_count,
        "crossover_probability": op_config.crossover_probability,
        "mutation_probability": op_config.mutation_probability,
        "elapsed_seconds": round(elapsed, 6),
        "archive_size": len(archive),
        "hv": hv,
        "hv_reference": list(hv_reference),
        "objective_vectors": front,
        "min_objectives": [min(pt[0] for pt in front), min(pt[1] for pt in front)],
        "all_schedules_valid": True,
    }
    if algorithm_kind == "dqce":
        run_meta["dqn"] = asdict(dqn_config)
        run_meta["use_compaction"] = use_compaction
        run_meta["parasite_size"] = len(algorithm.parasite_population)
        run_meta["operator_selection_history"] = algorithm.operator_history

    export = {
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
    return export


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


def evaluate_dqce(instance_path, output_dir, seeds=None, evaluations=11000, partitions=99,
                  neighbors=10, crossover=1.0, mutation=0.2, dqn_config=None,
                  use_compaction=True, hv_reference=(1.1, 1.1), reuse=False):
    instance_path = Path(instance_path).resolve()
    output_dir = Path(output_dir).resolve()
    seeds = list(DEFAULT_SEEDS if seeds is None else seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and distinct")

    op_config = FJSPOperatorConfig(crossover, mutation)
    dqn_cfg = DQNConfig() if dqn_config is None else dqn_config
    instance_bytes = instance_path.read_bytes()
    instance_sha256 = sha256(instance_bytes).hexdigest()

    ref_problem = FJSPProblem(instance_path)
    bounds = objective_bounds(ref_problem.instance)
    instance_name = ref_problem.instance.name

    if output_dir.exists() and not reuse:
        raise ValueError(f"Output directory already exists: {output_dir} (pass --reuse to reuse runs)")

    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    common_check = {
        "instance": instance_name,
        "instance_sha256": instance_sha256,
        "evaluations": evaluations,
        "partitions": partitions,
        "neighbors": neighbors,
        "crossover_probability": op_config.crossover_probability,
        "mutation_probability": op_config.mutation_probability,
        "hv_reference": list(hv_reference),
    }

    per_seed_rows = []
    dqce_fronts = []
    baseline_fronts = []

    for seed in seeds:
        dqce_path = runs_dir / f"dqce_seed{seed}.json"
        dqce_expected = {
            **common_check,
            "algorithm": "FjspDqceMoead",
            "seed": seed,
            "dqn": asdict(dqn_cfg),
            "use_compaction": use_compaction,
        }
        dqce_payload, dqce_reused = _load_or_run(
            dqce_path,
            dqce_expected,
            reuse,
            lambda s=seed: _run_single(
                "dqce", instance_path, instance_sha256, s, evaluations, partitions, neighbors,
                op_config, dqn_cfg, use_compaction, bounds, hv_reference,
            ),
        )

        base_path = runs_dir / f"baseline_seed{seed}.json"
        base_expected = {
            **common_check,
            "algorithm": "Moead",
            "seed": seed,
        }
        base_payload, base_reused = _load_or_run(
            base_path,
            base_expected,
            reuse,
            lambda s=seed: _run_single(
                "baseline", instance_path, instance_sha256, s, evaluations, partitions, neighbors,
                op_config, dqn_cfg, use_compaction, bounds, hv_reference,
            ),
        )

        dqce_run = dqce_payload["run"]
        base_run = base_payload["run"]
        front_dqce = dqce_run["objective_vectors"]
        front_base = base_run["objective_vectors"]
        dqce_fronts.extend(front_dqce)
        baseline_fronts.extend(front_base)

        c_dqce_base = set_coverage(front_dqce, front_base)
        c_base_dqce = set_coverage(front_base, front_dqce)

        row = {
            "seed": seed,
            "hv_dqce": dqce_run["hv"],
            "hv_baseline": base_run["hv"],
            "hv_diff": dqce_run["hv"] - base_run["hv"],
            "c_dqce_over_baseline": c_dqce_base,
            "c_baseline_over_dqce": c_base_dqce,
            "dqce_best_makespan": dqce_run["min_objectives"][0],
            "dqce_best_twt": dqce_run["min_objectives"][1],
            "baseline_best_makespan": base_run["min_objectives"][0],
            "baseline_best_twt": base_run["min_objectives"][1],
            "dqce_archive_size": dqce_run["archive_size"],
            "baseline_archive_size": base_run["archive_size"],
            "dqce_seconds": dqce_run["elapsed_seconds"],
            "baseline_seconds": base_run["elapsed_seconds"],
        }
        per_seed_rows.append(row)
        print(
            f"[seed={seed}] HV_DQCE={row['hv_dqce']:.6f}"
            + (" (reused)" if dqce_reused else "")
            + f" | HV_Base={row['hv_baseline']:.6f}"
            + (" (reused)" if base_reused else "")
            + f" | C(DQCE,Base)={c_dqce_base:.4f} | C(Base,DQCE)={c_base_dqce:.4f}",
            file=sys.stderr,
            flush=True,
        )

    pooled_dqce = sorted(independent_pareto_front(dqce_fronts))
    pooled_base = sorted(independent_pareto_front(baseline_fronts))
    pooled_c_dqce_base = set_coverage(pooled_dqce, pooled_base)
    pooled_c_base_dqce = set_coverage(pooled_base, pooled_dqce)

    hv_dqce_stats = _mean_std([r["hv_dqce"] for r in per_seed_rows])
    hv_base_stats = _mean_std([r["hv_baseline"] for r in per_seed_rows])
    hv_diff_stats = _mean_std([r["hv_diff"] for r in per_seed_rows])
    c_ab_stats = _mean_std([r["c_dqce_over_baseline"] for r in per_seed_rows])
    c_ba_stats = _mean_std([r["c_baseline_over_dqce"] for r in per_seed_rows])
    time_dqce_stats = _mean_std([r["dqce_seconds"] for r in per_seed_rows])
    time_base_stats = _mean_std([r["baseline_seconds"] for r in per_seed_rows])

    summary = {
        "instance": instance_name,
        "instance_path": str(instance_path),
        "instance_sha256": instance_sha256,
        "bounds": bounds,
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
        "metrics": {
            "hv_dqce": hv_dqce_stats,
            "hv_baseline": hv_base_stats,
            "hv_diff": hv_diff_stats,
            "c_dqce_over_baseline_per_seed": c_ab_stats,
            "c_baseline_over_dqce_per_seed": c_ba_stats,
            "c_dqce_over_baseline_pooled": pooled_c_dqce_base,
            "c_baseline_over_dqce_pooled": pooled_c_base_dqce,
            "dqce_best_makespan": min(r["dqce_best_makespan"] for r in per_seed_rows),
            "dqce_best_twt": min(r["dqce_best_twt"] for r in per_seed_rows),
            "baseline_best_makespan": min(r["baseline_best_makespan"] for r in per_seed_rows),
            "baseline_best_twt": min(r["baseline_best_twt"] for r in per_seed_rows),
            "dqce_seconds": time_dqce_stats,
            "baseline_seconds": time_base_stats,
        },
        "pooled_fronts": {
            "dqce": [list(pt) for pt in pooled_dqce],
            "baseline": [list(pt) for pt in pooled_base],
        },
        "per_seed": per_seed_rows,
    }

    summary_csv_rows = [{
        "instance": instance_name,
        "num_seeds": len(seeds),
        "evaluations": evaluations,
        "partitions": partitions,
        "crossover": op_config.crossover_probability,
        "mutation": op_config.mutation_probability,
        "hv_dqce_mean": hv_dqce_stats["mean"],
        "hv_dqce_std": hv_dqce_stats["std"],
        "hv_baseline_mean": hv_base_stats["mean"],
        "hv_baseline_std": hv_base_stats["std"],
        "c_dqce_over_baseline_mean": c_ab_stats["mean"],
        "c_dqce_over_baseline_std": c_ab_stats["std"],
        "c_baseline_over_dqce_mean": c_ba_stats["mean"],
        "c_baseline_over_dqce_std": c_ba_stats["std"],
        "c_dqce_over_baseline_pooled": pooled_c_dqce_base,
        "c_baseline_over_dqce_pooled": pooled_c_base_dqce,
        "dqce_best_makespan": summary["metrics"]["dqce_best_makespan"],
        "dqce_best_twt": summary["metrics"]["dqce_best_twt"],
        "baseline_best_makespan": summary["metrics"]["baseline_best_makespan"],
        "baseline_best_twt": summary["metrics"]["baseline_best_twt"],
        "dqce_seconds_mean": time_dqce_stats["mean"],
        "dqce_seconds_std": time_dqce_stats["std"],
        "baseline_seconds_mean": time_base_stats["mean"],
        "baseline_seconds_std": time_base_stats["std"],
    }]

    md_lines = [
        f"# Đánh giá HV và C(A, B) trên {instance_name}",
        "",
        f"- Seeds: `{seeds}` ({len(seeds)} seeds)",
        f"- Cấu hình: `evaluations={evaluations}`, `partitions={partitions}` (N={partitions + 1}), "
        f"`neighbors={neighbors}`, `crossover={op_config.crossover_probability}`, "
        f"`mutation={op_config.mutation_probability}`, `use_compaction={use_compaction}`",
        f"- HV reference: `{list(hv_reference)}`",
        "",
        "## Tổng hợp (Mean ± SD)",
        "",
        "| Metric | Giá trị |",
        "| --- | ---: |",
        f"| HV (DQCE) | {hv_dqce_stats['mean']:.6f} ± {hv_dqce_stats['std']:.6f} |",
        f"| HV (Baseline) | {hv_base_stats['mean']:.6f} ± {hv_base_stats['std']:.6f} |",
        f"| C(DQCE, Baseline) per-seed | {c_ab_stats['mean']:.4f} ± {c_ab_stats['std']:.4f} |",
        f"| C(Baseline, DQCE) per-seed | {c_ba_stats['mean']:.4f} ± {c_ba_stats['std']:.4f} |",
        f"| C(DQCE, Baseline) pooled front | {pooled_c_dqce_base:.4f} |",
        f"| C(Baseline, DQCE) pooled front | {pooled_c_base_dqce:.4f} |",
        f"| Best (Cmax, TWT) DQCE | ({summary['metrics']['dqce_best_makespan']}, {summary['metrics']['dqce_best_twt']}) |",
        f"| Best (Cmax, TWT) Baseline | ({summary['metrics']['baseline_best_makespan']}, {summary['metrics']['baseline_best_twt']}) |",
        f"| Time DQCE (s) | {time_dqce_stats['mean']:.3f} ± {time_dqce_stats['std']:.3f} |",
        f"| Time Baseline (s) | {time_base_stats['mean']:.3f} ± {time_base_stats['std']:.3f} |",
        "",
        "## Chi tiết từng seed",
        "",
        "| Seed | HV DQCE | HV Baseline | C(DQCE, Base) | C(Base, DQCE) | DQCE best (Cmax, TWT) | Base best (Cmax, TWT) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in per_seed_rows:
        md_lines.append(
            f"| {r['seed']} | {r['hv_dqce']:.6f} | {r['hv_baseline']:.6f} | "
            f"{r['c_dqce_over_baseline']:.4f} | {r['c_baseline_over_dqce']:.4f} | "
            f"({r['dqce_best_makespan']}, {r['dqce_best_twt']}) | "
            f"({r['baseline_best_makespan']}, {r['baseline_best_twt']}) |"
        )

    _write_json(output_dir / "summary.json", summary)
    _write_csv(output_dir / "per_seed.csv", per_seed_rows)
    _write_csv(output_dir / "summary.csv", summary_csv_rows)
    (output_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path, default=DEFAULT_INSTANCE)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--evaluations", type=int, default=11000)
    parser.add_argument("--partitions", type=int, default=99)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--crossover", type=float, default=1.0)
    parser.add_argument("--mutation", type=float, default=0.2)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--buffer-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--target-update", type=int, default=10)
    parser.add_argument("--no-compaction", action="store_true")

    parser.add_argument("--hv-reference", type=float, nargs=2, default=[1.1, 1.1])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/eval_mk01_dqce")
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
            instance_path=args.instance,
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
        "metrics": summary["metrics"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
