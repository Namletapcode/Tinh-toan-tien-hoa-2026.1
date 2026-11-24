"""Read-only observation and metrics for the fixed FJSP baseline experiment."""

from dataclasses import asdict
from hashlib import sha256
import json
from statistics import mean, stdev
from time import perf_counter

import numpy as np

from moead_framework.algorithm.fjsp_variants import (
    create_fjsp_variant, population_layout, population_solutions,
)
from moead_framework.problem.fjsp import FJSPProblem
from moead_framework.tool.result import HyperVolume, set_seed


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return sha256(encoded).hexdigest()


def objective_bounds(instance):
    """Data-only lower bounds and conservative upper bounds for Makespan and TWT.

    The decoder never inserts voluntary idle time beyond release-date waits: its
    makespan is at most the sum of selected durations after releases, bounded by
    sum(max alternative duration) plus the latest release date. These bounds do
    not use any observed optimization result.
    """
    minima = [[min(option.duration for option in op) for op in job] for job in instance.jobs]
    serial_max = sum(max(option.duration for option in op) for job in instance.jobs for op in job)
    average_load = (sum(sum(job) for job in minima) + instance.machine_count - 1) // instance.machine_count
    forced_loads = [0] * instance.machine_count
    for job in instance.jobs:
        for options in job:
            if len(options) == 1:
                forced_loads[options[0].machine] += options[0].duration
    latest_release = max(instance.release_dates, default=0)
    makespan_lower = max(
        max((release + sum(job) for release, job in zip(instance.release_dates, minima)), default=0),
        average_load,
        max(forced_loads, default=0),
    )
    makespan_upper = latest_release + serial_max
    twt_lower = 0
    twt_upper = sum(
        weight * max(0, makespan_upper - due)
        for due, weight in zip(instance.due_dates, instance.weights)
    )
    lower = [makespan_lower, twt_lower]
    upper = [makespan_upper, twt_upper]
    return {"lower": lower, "upper": upper,
            "span": [max(hi - lo, 1) for lo, hi in zip(lower, upper)],
            "constant_objectives": [i for i, (lo, hi) in enumerate(zip(lower, upper)) if lo == hi]}


def normalized_hypervolume(points, bounds, reference=(1.1, 1.1)):
    """Postprocessing only: fixed data bounds, no clipping, reference-box fraction."""
    values = np.asarray(points, dtype=float)
    if values.size == 0:
        return 0.0
    ref = np.asarray(reference, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("HV requires finite two-objective vectors")
    if ref.shape != (2,) or not np.isfinite(ref).all() or (ref <= 1).any():
        raise ValueError("Normalized HV reference must have two finite coordinates greater than 1")
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
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite two-objective vectors")
    values = np.unique(values, axis=0)
    kept = []
    for start in range(0, len(values), 128):
        block = values[start:start + 128]
        less_equal = np.all(values[None, :, :] <= block[:, None, :], axis=2)
        strictly_less = np.any(values[None, :, :] < block[:, None, :], axis=2)
        kept.extend(block[~np.any(less_equal & strictly_less, axis=1)])
    return {tuple(map(float, row)) for row in kept}


def igd_plus(front, reference, bounds):
    """IGD+ trên không gian đã chuẩn hóa bằng bounds['lower']/['span']."""
    if not reference:
        return 0.0
    if not front:
        raise ValueError("IGD+ requires a non-empty approximation front when reference is non-empty")
    span = np.asarray(bounds["span"], dtype=float)
    ref = (np.asarray(reference, dtype=float) - bounds["lower"]) / span
    apx = (np.asarray(front, dtype=float) - bounds["lower"]) / span
    diffs = np.maximum(apx[None, :, :] - ref[:, None, :], 0)
    distances = np.sqrt((diffs ** 2).sum(axis=2))
    return float(distances.min(axis=1).mean())


def set_coverage(front_a, front_b):
    """C(A,B): tỉ lệ điểm trong B bị chi phối yếu bởi ít nhất một điểm trong A."""
    if not front_b:
        return 0.0
    if not front_a:
        return 0.0
    a = np.asarray(front_a, dtype=float)
    b = np.asarray(front_b, dtype=float)
    dominated = np.any(np.all(a[:, None, :] <= b[None, :, :], axis=2), axis=0)
    return float(dominated.sum() / len(front_b))


class _RecordingProblem(FJSPProblem):
    def __init__(self, instance):
        super().__init__(instance)
        self.trace = []

    def evaluate(self, x):
        solution = super().evaluate(x)
        self.trace.append(list(solution.F))
        return solution


def run_validation_case(instance, settings, seed, checkpoints, reference=(1.1, 1.1), known_front=None):
    """Observe the selected variant without adding NFE or random draws."""
    layout = population_layout(settings)
    if (not checkpoints or sorted(set(checkpoints)) != list(checkpoints)
            or checkpoints[0] < layout["initial_evaluations"]
            or checkpoints[-1] != settings["evaluations"]):
        raise ValueError("Checkpoints must be ordered, after initialization, and end at the budget")
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
    algorithm = create_fjsp_variant(problem, settings)
    observe(algorithm)
    archive = sorted(algorithm.run(checkpoint=observe), key=lambda solution: tuple(solution.F))
    observe(algorithm)
    elapsed = perf_counter() - started

    budget = settings["evaluations"]
    if algorithm.current_eval != budget or problem.evaluation_count != budget or len(problem.trace) != budget:
        raise AssertionError("NFE counters and actual evaluation trace do not match the protocol")
    if set(snapshots) != wanted:
        raise AssertionError("Not all requested NFE checkpoints were recorded")
    for solution in archive + population_solutions(algorithm) + algorithm.population:
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
        **layout, "algorithm": type(algorithm).__name__, "elapsed_seconds": elapsed,
        "archive_size": len(archive), "objective_vectors": front, "snapshots": curve,
        "initial_hv": curve[0]["hv"], "final_hv": curve[-1]["hv"],
        "hv_gain": curve[-1]["hv"] - curve[0]["hv"],
        "min_objectives": [min(row[i] for row in front) for i in range(2)],
        "all_schedules_valid": True, "archive_matches_history": True,
        "hv_monotonic": True, "evaluation_trace": problem.trace, "solutions": solutions,
        "known_front_covered": None if known_front is None else {tuple(p) for p in front} == {tuple(p) for p in known_front},
    }
    signature_data = {"trace": problem.trace, "solutions": solutions, "snapshots": curve}
    if layout["variant"] != "baseline":
        result["micro_populations"] = [
            [{"objectives": resident.F, "decision_vector": resident.decision_vector.tolist()}
             for resident in group] for group in algorithm.micro_populations
        ]
        result["parent_selection_counts"] = dict(algorithm.parent_selector.selection_counts)
        signature_data.update(micro_populations=result["micro_populations"],
                              parent_selection_counts=result["parent_selection_counts"])
    result["reproducibility_signature"] = digest(signature_data)
    return result


def summarize_cases(identifier, family, cases, bounds):
    def stats(key):
        values = [case[key] for case in cases]
        return mean(values), stdev(values) if len(cases) > 1 else 0.0

    hv_mean, hv_std = stats("final_hv")
    time_mean, time_std = stats("elapsed_seconds")
    result = {
        "instance": identifier, "family": family, "runs": len(cases),
        **{key: cases[0][key] for key in ("variant", "number_of_subproblems",
                                        "micro_population_size", "population_size", "initial_evaluations")},
        "initial_hv_mean": mean(c["initial_hv"] for c in cases),
        "hv_mean": hv_mean, "hv_std": hv_std,
        "hv_gain_mean": mean(c["hv_gain"] for c in cases),
        "hv_improved_runs": sum(c["hv_gain"] > 1e-12 for c in cases),
        "makespan_best": min(c["min_objectives"][0] for c in cases),
        "makespan_mean": mean(c["min_objectives"][0] for c in cases),
        "makespan_std": stdev([c["min_objectives"][0] for c in cases]) if len(cases) > 1 else 0.0,
        "twt_best": min(c["min_objectives"][1] for c in cases),
        "twt_mean": mean(c["min_objectives"][1] for c in cases),
        "twt_std": stdev([c["min_objectives"][1] for c in cases]) if len(cases) > 1 else 0.0,
        "archive_size_mean": mean(c["archive_size"] for c in cases),
        "seconds_mean": time_mean, "seconds_std": time_std,
        "valid_runs": sum(c["all_schedules_valid"] and c["archive_matches_history"] and c["hv_monotonic"] for c in cases),
        "constant_objectives": bounds["constant_objectives"],
        "known_front_covered_runs": (None if cases[0]["known_front_covered"] is None
                                      else sum(c["known_front_covered"] for c in cases)),
    }
    return result
