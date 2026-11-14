"""Run/resume the fixed step-6 protocol, preserving raw results and provenance."""

import argparse
import csv
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import platform
import re

import numpy as np

from moead_framework.algorithm.fjsp import uniform_weights_3d
from moead_framework.core.genetic_operator.fjsp import FJSPOperatorConfig
from moead_framework.problem.fjsp import parse_instance
from moead_framework.tool.fjsp_experiments import (
    digest, objective_bounds, run_validation_case, summarize_cases,
)


ROOT = Path(__file__).resolve().parents[1]


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def validate_protocol(protocol):
    if protocol.get("schema_version") != 1:
        raise ValueError("Unsupported protocol schema")
    settings = protocol["algorithm"]
    population = len(uniform_weights_3d(settings["partitions"]))
    if not _integer(settings["evaluations"]) or settings["evaluations"] < population:
        raise ValueError("NFE must cover the initial population")
    if not _integer(settings["neighbors"]) or not 1 <= settings["neighbors"] <= population:
        raise ValueError("Neighborhood must fit the population")
    FJSPOperatorConfig(settings["crossover"], settings["mutation"])
    seeds = protocol["seeds"]
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(not _integer(seed) or not 0 <= seed < 2 ** 32 for seed in seeds)):
        raise ValueError("Seeds must be distinct integers in [0, 2**32)")
    if protocol.get("repeat_seed") is not None and protocol["repeat_seed"] not in seeds:
        raise ValueError("repeat_seed must be one of the primary seeds")
    checkpoints = protocol["checkpoints"]
    if (not checkpoints or any(not _integer(n) or not population <= n <= settings["evaluations"] for n in checkpoints)
            or sorted(set(checkpoints)) != checkpoints
            or checkpoints[0] != population or checkpoints[-1] != settings["evaluations"]):
        raise ValueError("Checkpoints must be ordered/distinct and include initialization and final NFE")
    reference = protocol["hv_reference"]
    if len(reference) != 3 or any(not isinstance(v, (int, float)) or not isfinite(v) or v <= 1 for v in reference):
        raise ValueError("HV reference needs three finite values greater than 1")
    instances = protocol["instances"]
    identifiers = [entry["id"] for entry in instances]
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("Instance IDs must be distinct and nonempty")
    if any(re.fullmatch(r"[A-Za-z0-9_-]+", identifier) is None for identifier in identifiers):
        raise ValueError("Instance IDs may contain only letters, numbers, underscores and hyphens")


def runtime_fingerprint(root):
    paths = [p for p in (root / "vendor/moead_framework").rglob("*.py")
             if "test" not in p.relative_to(root).parts]
    paths += [root / "scripts/run_fjsp.py", root / "scripts/run_baseline_validation.py", root / "requirements-lock.txt"]
    files = {path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}
    return {"digest": digest(files), "files": files}


def write_json(path, value, exclusive=False):
    with path.open("x" if exclusive else "w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")


def read_case(path, identifier, seed):
    value = json.loads(path.read_text(encoding="utf-8"))
    checksum = value.pop("integrity_sha256")
    if digest(value) != checksum or value["instance_id"] != identifier or value["seed"] != seed:
        raise ValueError(f"Result integrity/identity check failed: {path}")
    return value


def save_case(path, value):
    write_json(path, {**value, "integrity_sha256": digest(value)}, exclusive=True)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_convergence(path, all_cases, seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = min(3, len(all_cases))
    rows = (len(all_cases) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3.5 * rows), squeeze=False)
    for axis, (identifier, cases) in zip(axes.flat, all_cases.items()):
        nfe = [point["nfe"] for point in cases[0]["snapshots"]]
        values = np.array([[point["hv"] for point in case["snapshots"]] for case in cases])
        average = values.mean(axis=0)
        deviation = values.std(axis=0, ddof=1) if len(cases) > 1 else np.zeros(len(nfe))
        axis.plot(nfe, average, marker="o", color="#236b8e")
        axis.fill_between(nfe, np.maximum(0, average - deviation), np.minimum(1, average + deviation),
                          color="#236b8e", alpha=0.18)
        axis.set(title=identifier, xlabel="NFE", ylabel="Normalized HV")
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        axis.grid(alpha=0.22)
    for axis in list(axes.flat)[len(all_cases):]:
        axis.set_visible(False)
    figure.suptitle(f"MOEA/D-FJSP baseline: mean HV +/- sample SD ({len(seeds)} seeds)")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_study(config_path, output_dir, resume=False, make_plot=True, root=ROOT):
    root, config_path, output_dir = Path(root).resolve(), Path(config_path).resolve(), Path(output_dir).resolve()
    protocol_bytes = config_path.read_bytes()
    protocol = json.loads(protocol_bytes.decode("utf-8"))
    validate_protocol(protocol)
    models, inputs, bounds = {}, {}, {}
    for entry in protocol["instances"]:
        source = (root / entry["path"]).resolve()
        if not source.is_relative_to((root / "instance").resolve()):
            raise ValueError("Protocol instances must stay inside the project's instance directory")
        content = source.read_bytes()
        identifier = entry["id"]
        models[identifier] = parse_instance(content.decode("utf-8-sig"), source.stem)
        inputs[identifier] = {"path": entry["path"], "sha256": sha256(content).hexdigest()}
        bounds[identifier] = objective_bounds(models[identifier])
    code = runtime_fingerprint(root)
    identity = {"protocol_sha256": sha256(protocol_bytes).hexdigest(), "code_sha256": code["digest"],
                "inputs": inputs, "python": platform.python_version(), "numpy": np.__version__}
    manifest_path = output_dir / "manifest.json"
    if output_dir.exists():
        if not resume or not manifest_path.is_file():
            raise ValueError("Output directory already exists; choose a new directory or explicitly use --resume")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["identity"] != identity:
            raise ValueError("Resume refused: protocol, runtime code, inputs or environment changed")
    else:
        output_dir.mkdir(parents=True)
        manifest = {"identity": identity, "protocol": protocol, "bounds": bounds, "runtime_source": code,
                    "environment": {"platform": platform.platform(), "processor": platform.processor(),
                                    "logical_cpus": os.cpu_count()}}
        write_json(manifest_path, manifest, exclusive=True)
    for name in ("runs", "replays"):
        (output_dir / name).mkdir(exist_ok=True)

    all_cases, summaries, rows, replay_checks = {}, [], [], []
    total = len(protocol["instances"]) * len(protocol["seeds"])
    done = 0
    for entry in protocol["instances"]:
        identifier = entry["id"]
        cases = []
        for seed in protocol["seeds"]:
            path = output_dir / "runs" / f"{identifier}_seed{seed}.json"
            reused = path.exists()
            if reused:
                case = read_case(path, identifier, seed)
            else:
                case = run_validation_case(models[identifier], protocol["algorithm"], seed,
                                           protocol["checkpoints"], protocol["hv_reference"], entry.get("known_front"))
                case.update(instance_id=identifier, family=entry["family"])
                save_case(path, case)
            cases.append(case)
            done += 1
            print(f"[{done}/{total}] {identifier} seed={seed}: HV {case['initial_hv']:.6f} -> "
                  f"{case['final_hv']:.6f}, NFE={case['actual_evaluations']}, "
                  f"{case['elapsed_seconds']:.2f}s" + (" (reused)" if reused else ""), flush=True)
            rows.append({"instance": identifier, "family": entry["family"], "seed": seed,
                         "nfe": case["actual_evaluations"], "initial_hv": case["initial_hv"],
                         "final_hv": case["final_hv"], "hv_gain": case["hv_gain"],
                         "makespan": case["min_objectives"][0], "workload": case["min_objectives"][1],
                         "max_machine_workload": case["min_objectives"][2],
                         "archive_size": case["archive_size"], "elapsed_seconds": case["elapsed_seconds"],
                         "valid": case["all_schedules_valid"] and case["archive_matches_history"] and case["hv_monotonic"]})
        all_cases[identifier] = cases
        summaries.append(summarize_cases(identifier, entry["family"], cases, bounds[identifier]))
        repeat_seed = protocol.get("repeat_seed")
        if repeat_seed is not None:
            path = output_dir / "replays" / f"{identifier}_seed{repeat_seed}.json"
            if path.exists():
                repeated = read_case(path, identifier, repeat_seed)
            else:
                repeated = run_validation_case(models[identifier], protocol["algorithm"], repeat_seed,
                                               protocol["checkpoints"], protocol["hv_reference"], entry.get("known_front"))
                repeated.update(instance_id=identifier, family=entry["family"])
                save_case(path, repeated)
            original = next(case for case in cases if case["seed"] == repeat_seed)
            if repeated["reproducibility_signature"] != original["reproducibility_signature"]:
                raise AssertionError(f"Repeated seed differs for {identifier}")
            replay_checks.append({"instance": identifier, "seed": repeat_seed, "identical": True})
            print(f"  replay {identifier} seed={repeat_seed}: identical trace, archive and schedules", flush=True)

    if runtime_fingerprint(root)["digest"] != identity["code_sha256"]:
        raise RuntimeError("Runtime sources changed while the study was running")
    for identifier, source in inputs.items():
        if sha256((root / source["path"]).read_bytes()).hexdigest() != source["sha256"]:
            raise RuntimeError(f"Input changed during the study: {identifier}")
    write_csv(output_dir / "runs.csv", rows)
    write_csv(output_dir / "summary.csv", summaries)
    report = {"primary_runs": total, "replay_runs": len(replay_checks), "valid_runs": sum(row["valid"] for row in rows),
              "primary_nfe": sum(row["nfe"] for row in rows), "summaries": summaries,
              "replay_checks": replay_checks, "bounds": bounds, "hv_reference": protocol["hv_reference"]}
    write_json(output_dir / "summary.json", report)
    markdown = ["# Kiem chung baseline MOEA/D-FJSP", "", f"{total} lan chay chinh; {len(replay_checks)} lan lap seed.", "",
                "HV duoc chuan hoa bang can input co dinh, chia the tich hop tham chieu; chi so sanh trong cung instance.", "",
                "| Instance | HV mean +/- SD | Mean HV gain | Best Cmax | Mean seconds | Valid |",
                "| --- | ---: | ---: | ---: | ---: | ---: |"]
    markdown += [f"| {s['instance']} | {s['hv_mean']:.6f} +/- {s['hv_std']:.6f} | {s['hv_gain_mean']:.6f} | "
                 f"{s['makespan_best']} | {s['seconds_mean']:.3f} | {s['valid_runs']}/{s['runs']} |" for s in summaries]
    (output_dir / "summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    if make_plot:
        plot_convergence(output_dir / "convergence.png", all_cases, protocol["seeds"])
    if not (output_dir / "completion.json").exists():
        write_json(output_dir / "completion.json", {"identity": identity, "primary_runs": total,
                                                   "replay_runs": len(replay_checks), "valid_runs": report["valid_runs"]}, exclusive=True)
    print(f"Completed: {report['valid_runs']}/{total} valid primary runs; {len(replay_checks)} deterministic replays", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "scripts/configs/baseline_validation.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/baseline_step6")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    run_study(args.config, args.output_dir, resume=args.resume, make_plot=not args.no_plot)


if __name__ == "__main__":
    main()
