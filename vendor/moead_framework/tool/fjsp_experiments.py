"""Read-only observation and metrics for the fixed FJSP baseline experiment."""

from dataclasses import asdict
from hashlib import sha256
import json
from statistics import mean, stdev
from time import perf_counter

import numpy as np

from moead_framework.algorithm.fjsp import create_fjsp_moead
from moead_framework.core.genetic_operator.fjsp import FJSPOperatorConfig
from moead_framework.problem.fjsp import FJSPProblem
from moead_framework.tool.result import HyperVolume, set_seed


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return sha256(encoded).hexdigest()


def objective_bounds(instance):
    """Data-only lower bounds and the serial upper bound for this decoder.

    The decoder never inserts voluntary idle time: its makespan is at most
    the sum of selected durations, bounded by sum(max alternative duration).
    These bounds do not use any observed optimization result.
    """
    minima = [[min(option.duration for option in op) for op in job] for job in instance.jobs]
    total_min = sum(sum(job) for job in minima)
    serial_max = sum(max(option.duration for option in op) for job in instance.jobs for op in job)
    average_load = (total_min + instance.machine_count - 1) // instance.machine_count
    forced_loads = [0] * instance.machine_count
    possible_loads = [0] * instance.machine_count
    for job in instance.jobs:
        for options in job:
            for option in options:
                possible_loads[option.machine] += option.duration
            if len(options) == 1:
                forced_loads[options[0].machine] += options[0].duration
    lower = [max(max(map(sum, minima)), average_load, max(forced_loads)), total_min,
             max(average_load, max(v for job in minima for v in job), max(forced_loads))]
    upper = [serial_max, serial_max, max(possible_loads)]
    return {"lower": lower, "upper": upper,
            "span": [max(hi - lo, 1) for lo, hi in zip(lower, upper)],
            "constant_objectives": [i for i, (lo, hi) in enumerate(zip(lower, upper)) if lo == hi]}


def normalized_hypervolume(points, bounds, reference=(1.1, 1.1, 1.1)):
    """Postprocessing only: fixed data bounds, no clipping, reference-box fraction."""
    values = np.asarray(points, dtype=float)
    if values.size == 0:
        return 0.0
    ref = np.asarray(reference, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("HV requires finite three-objective vectors")
    if ref.shape != (3,) or not np.isfinite(ref).all() or (ref <= 1).any():
        raise ValueError("Normalized HV reference must have three finite coordinates greater than 1")
    lower, upper = np.asarray(bounds["lower"]), np.asarray(bounds["upper"])
    if (values < lower - 1e-10).any() or (values > upper + 1e-10).any():
        raise ValueError("Objective vectors are outside the fixed instance bounds")
    scaled = (values - lower) / np.asarray(bounds["span"])
    return float(HyperVolume(reference_point=list(ref)).compute(scaled) / np.prod(ref))


def independent_pareto_front(points):
    """Independent pairwise definition, blockwise to keep temporary memory bounded."""
    values = np.asarray(points, dtype=float)
    if values.size == 0:
        return set()
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("Expected finite three-objective vectors")
    values = np.unique(values, axis=0)
    kept = []
    for start in range(0, len(values), 128):
        block = values[start:start + 128]
        less_equal = np.all(values[None, :, :] <= block[:, None, :], axis=2)
        strictly_less = np.any(values[None, :, :] < block[:, None, :], axis=2)
        kept.extend(block[~np.any(less_equal & strictly_less, axis=1)])
    return {tuple(map(float, row)) for row in kept}


class _RecordingProblem(FJSPProblem):
    def __init__(self, instance):
        super().__init__(instance)
        self.trace = []

    def evaluate(self, x):
        solution = super().evaluate(x)
        self.trace.append(list(solution.F))
        return solution


def run_validation_case(instance, settings, seed, checkpoints, reference=(1.1, 1.1, 1.1), known_front=None):
    """Run the unchanged step-5 pipeline, recording objectives without new NFE/RNG calls."""
    config = FJSPOperatorConfig(settings["crossover"], settings["mutation"])
    problem = _RecordingProblem(instance)
    bounds = objective_bounds(instance)
    snapshots = {}
    wanted = set(checkpoints)

    def observe(algorithm):
        nfe = algorithm.current_eval
        if nfe in wanted and nfe not in snapshots:
            snapshots[nfe] = [list(solution.F) for solution in algorithm.ep]

    set_seed(seed)
    started = perf_counter()
    algorithm = create_fjsp_moead(
        problem, max_evaluation=settings["evaluations"],
        number_of_weight_neighborhood=settings["neighbors"],
        partitions=settings["partitions"], operator_config=config,
    )
    observe(algorithm)
    archive = sorted(algorithm.run(checkpoint=observe), key=lambda solution: tuple(solution.F))
    observe(algorithm)
    elapsed = perf_counter() - started

    budget = settings["evaluations"]
    if algorithm.current_eval != budget or problem.evaluation_count != budget or len(problem.trace) != budget:
        raise AssertionError("NFE counters and actual evaluation trace do not match the protocol")
    if set(snapshots) != wanted:
        raise AssertionError("Not all requested NFE checkpoints were recorded")
    for solution in archive + algorithm.population:
        problem.validate_solution(solution)

    curve = []
    for nfe, points in sorted(snapshots.items()):
        expected = independent_pareto_front(problem.trace[:nfe])
        if {tuple(p) for p in points} != expected or len(points) != len(expected):
            raise AssertionError(f"Archive at NFE {nfe} differs from the complete evaluation history")
        curve.append({"nfe": nfe, "front": sorted(points),
                      "hv": normalized_hypervolume(points, bounds, reference)})
    if any(a["hv"] > b["hv"] + 1e-12 for a, b in zip(curve, curve[1:])):
        raise AssertionError("Archive hypervolume decreased between checkpoints")

    front = [list(solution.F) for solution in archive]
    solutions = [{"objectives": solution.F, "decision_vector": solution.decision_vector.tolist(),
                  "machine_loads": list(solution.schedule.machine_loads),
                  "schedule": [asdict(operation) for operation in solution.schedule.operations]}
                 for solution in archive]
    result = {
        "seed": seed, "evaluations": budget, "actual_evaluations": problem.evaluation_count,
        "population_size": algorithm.number_of_weight, "elapsed_seconds": elapsed,
        "archive_size": len(archive), "objective_vectors": front, "snapshots": curve,
        "initial_hv": curve[0]["hv"], "final_hv": curve[-1]["hv"],
        "hv_gain": curve[-1]["hv"] - curve[0]["hv"],
        "min_objectives": [min(row[i] for row in front) for i in range(3)],
        "all_schedules_valid": True, "archive_matches_history": True,
        "hv_monotonic": True, "evaluation_trace": problem.trace, "solutions": solutions,
        "known_front_covered": None if known_front is None else {tuple(p) for p in front} == {tuple(p) for p in known_front},
    }
    result["reproducibility_signature"] = digest({"trace": problem.trace, "solutions": solutions, "snapshots": curve})
    return result


def summarize_cases(identifier, family, cases, bounds):
    def stats(key):
        values = [case[key] for case in cases]
        return mean(values), stdev(values) if len(values) > 1 else 0.0

    hv_mean, hv_std = stats("final_hv")
    time_mean, time_std = stats("elapsed_seconds")
    result = {
        "instance": identifier, "family": family, "runs": len(cases),
        "initial_hv_mean": mean(c["initial_hv"] for c in cases),
        "hv_mean": hv_mean, "hv_std": hv_std,
        "hv_gain_mean": mean(c["hv_gain"] for c in cases),
        "hv_improved_runs": sum(c["hv_gain"] > 1e-12 for c in cases),
        "makespan_best": min(c["min_objectives"][0] for c in cases),
        "makespan_mean": mean(c["min_objectives"][0] for c in cases),
        "makespan_std": stdev([c["min_objectives"][0] for c in cases]) if len(cases) > 1 else 0.0,
        "workload_best": min(c["min_objectives"][1] for c in cases),
        "max_machine_workload_best": min(c["min_objectives"][2] for c in cases),
        "archive_size_mean": mean(c["archive_size"] for c in cases),
        "seconds_mean": time_mean, "seconds_std": time_std,
        "valid_runs": sum(c["all_schedules_valid"] and c["archive_matches_history"] and c["hv_monotonic"] for c in cases),
        "constant_objectives": bounds["constant_objectives"],
        "known_front_covered_runs": (None if cases[0]["known_front_covered"] is None
                                      else sum(c["known_front_covered"] for c in cases)),
    }
    return result
