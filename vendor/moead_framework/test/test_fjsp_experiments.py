"""Independent metric checks and a small complete/resumable experiment."""

from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from itertools import combinations
import json
from math import prod
from pathlib import Path
import random

import numpy as np
import pytest

from moead_framework.problem.fjsp import (
    Chromosome, decode, load_instance, parse_instance, random_chromosome,
)
from moead_framework.tool.fjsp_experiments import (
    independent_pareto_front, normalized_hypervolume, objective_bounds, run_validation_case,
)


ROOT = Path(__file__).resolve().parents[3]
SPEC = spec_from_file_location("baseline_validation_runner", ROOT / "scripts/run_baseline_validation.py")
RUNNER = module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def union_volume_oracle(points, reference):
    """Inclusion-exclusion of axis-aligned boxes; independent of the HV sweep."""
    total = 0.0
    for size in range(1, len(points) + 1):
        for group in combinations(points, size):
            corner = [max(p[i] for p in group) for i in range(3)]
            total += (-1) ** (size + 1) * prod(max(r - c, 0) for r, c in zip(reference, corner))
    return total


@pytest.mark.parametrize("points", [
    [[1, 1, 1]], [[1, 3, 3], [3, 1, 3], [3, 3, 1]],
    [[1, 1, 1], [2, 2, 2]], [[1, 1, 1], [1, 1, 1]],
])
def test_normalized_hv_matches_independent_box_union(points):
    bounds = {"lower": [0, 0, 0], "upper": [10, 10, 10], "span": [10, 10, 10]}
    reference = [1.1] * 3
    expected = union_volume_oracle(np.asarray(points) / 10, reference) / prod(reference)
    assert normalized_hypervolume(points, bounds, reference) == pytest.approx(expected, abs=1e-12)
    assert 0 <= expected <= 1
    assert normalized_hypervolume([], bounds) == 0


@pytest.mark.parametrize("points", [[[11, 2, 2]], [[-1, 0, 0]], [[float("nan"), 1, 1]], [[1, 2]]])
def test_hv_rejects_invalid_vectors_instead_of_clipping(points):
    bounds = {"lower": [0, 0, 0], "upper": [10, 10, 10], "span": [10, 10, 10]}
    with pytest.raises(ValueError):
        normalized_hypervolume(points, bounds)


def test_pairwise_pareto_definition_handles_ties_tradeoffs_and_dominance():
    points = [[1, 3, 3], [3, 1, 3], [3, 3, 1], [5, 5, 5], [1, 3, 3]]
    assert independent_pareto_front(points) == {(1, 3, 3), (3, 1, 3), (3, 3, 1)}
    assert independent_pareto_front([]) == set()


def test_bounds_are_data_only_and_handle_constant_objectives_and_zero_times():
    instance = parse_instance("1 1\n2 1 1 3 1 1 4\n")
    bounds = objective_bounds(instance)
    assert bounds["lower"] == bounds["upper"] == [7, 7, 7]
    assert bounds["constant_objectives"] == [0, 1, 2]
    assert normalized_hypervolume([[7, 7, 7]], bounds) == pytest.approx(1)
    zero = parse_instance("1 1\n1 1 1 0\n")
    assert normalized_hypervolume([[0, 0, 0]], objective_bounds(zero)) == pytest.approx(1)


def test_fattahi_bounds_match_processing_time_calculation():
    instance = load_instance(ROOT / "instance/FattahiMehrabadJolai2007/fattahi1.txt")
    bounds = objective_bounds(instance)
    assert bounds["lower"] == [66, 115, 58]
    assert bounds["upper"] == [199, 199, 191]  # Potential load on M2: 37 + 24 + 65 + 65.
    point = decode(instance, Chromosome((0, 1, 0, 1), (1, 1, 0, 0))).objectives
    assert 0 < normalized_hypervolume([point], bounds) <= 1


def tiny_protocol():
    return {
        "schema_version": 1, "study": "test", "study_date": "2026-11-12",
        "algorithm": {"partitions": 2, "neighbors": 3, "evaluations": 13, "crossover": 0.8, "mutation": 0.05},
        "seeds": [0, 1], "repeat_seed": 0, "checkpoints": [6, 10, 13], "hv_reference": [1.1] * 3,
        "instances": [{"id": "fattahi1", "family": "Fattahi", "path": "instance/FattahiMehrabadJolai2007/fattahi1.txt"}],
    }


def test_validation_observer_keeps_nfe_archive_and_seed_reproducibility():
    protocol = tiny_protocol()
    instance = load_instance(ROOT / protocol["instances"][0]["path"])
    first = run_validation_case(instance, protocol["algorithm"], 0, protocol["checkpoints"])
    second = run_validation_case(instance, protocol["algorithm"], 0, protocol["checkpoints"])
    assert first["actual_evaluations"] == len(first["evaluation_trace"]) == 13
    assert [p["nfe"] for p in first["snapshots"]] == [6, 10, 13]
    assert first["final_hv"] >= first["initial_hv"]
    assert first["archive_matches_history"] and first["all_schedules_valid"]
    assert first["reproducibility_signature"] == second["reproducibility_signature"]


def test_study_saves_integrity_checked_results_and_resumes_without_rerunning(tmp_path):
    config = tmp_path / "protocol.json"
    config.write_text(json.dumps(tiny_protocol()), encoding="utf-8")
    output = tmp_path / "study"
    report = RUNNER.run_study(config, output, make_plot=True)
    assert report["primary_runs"] == report["valid_runs"] == 2
    assert report["replay_runs"] == 1 and report["primary_nfe"] == 26
    assert (output / "convergence.png").read_bytes().startswith(b"\x89PNG")
    raw = output / "runs/fattahi1_seed0.json"
    original, mtime = raw.read_bytes(), raw.stat().st_mtime_ns
    repeated = RUNNER.run_study(config, output, resume=True, make_plot=False)
    assert repeated["summaries"] == report["summaries"]
    assert raw.read_bytes() == original and raw.stat().st_mtime_ns == mtime
    with pytest.raises(ValueError, match="already exists"):
        RUNNER.run_study(config, output, make_plot=False)
    protocol = tiny_protocol()
    protocol["algorithm"]["mutation"] = 0.1
    config.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        RUNNER.run_study(config, output, resume=True, make_plot=False)


def test_resume_rejects_tampering_with_raw_results(tmp_path):
    config = tmp_path / "protocol.json"
    config.write_text(json.dumps(tiny_protocol()), encoding="utf-8")
    output = tmp_path / "study"
    RUNNER.run_study(config, output, make_plot=False)
    path = output / "runs/fattahi1_seed0.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["final_hv"] = 123
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        RUNNER.run_study(config, output, resume=True, make_plot=False)


def test_interrupted_study_resumes_only_missing_runs(tmp_path, monkeypatch):
    config = tmp_path / "protocol.json"
    config.write_text(json.dumps(tiny_protocol()), encoding="utf-8")
    output = tmp_path / "study"
    original = RUNNER.run_validation_case

    def interrupted(instance, settings, seed, *args):
        if seed == 1:
            raise RuntimeError("simulated interruption")
        return original(instance, settings, seed, *args)

    monkeypatch.setattr(RUNNER, "run_validation_case", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        RUNNER.run_study(config, output, make_plot=False)
    saved = (output / "runs/fattahi1_seed0.json").read_bytes()
    assert not (output / "completion.json").exists()
    new_seeds = []

    def record(instance, settings, seed, *args):
        new_seeds.append(seed)
        return original(instance, settings, seed, *args)

    monkeypatch.setattr(RUNNER, "run_validation_case", record)
    report = RUNNER.run_study(config, output, resume=True, make_plot=False)
    assert new_seeds == [1, 0]  # Missing primary run, then the independent replay.
    assert report["valid_runs"] == 2
    assert (output / "runs/fattahi1_seed0.json").read_bytes() == saved


def test_fixed_bounds_cover_the_protocol_instances_before_optimization():
    protocol = json.loads((ROOT / "scripts/configs/baseline_validation.json").read_text(encoding="utf-8"))
    rng = random.Random(17)
    for entry in protocol["instances"]:
        instance = load_instance(ROOT / entry["path"])
        point = decode(instance, random_chromosome(instance, rng)).objectives
        assert 0 <= normalized_hypervolume([point], objective_bounds(instance), protocol["hv_reference"]) <= 1


@pytest.mark.parametrize("key,value", [("seeds", [0, 0]), ("seeds", [True]), ("repeat_seed", 99),
                                       ("checkpoints", [6, 13, 10]), ("hv_reference", [1, 1, 1])])
def test_protocol_validation_rejects_inconsistent_settings(key, value):
    protocol = deepcopy(tiny_protocol())
    protocol[key] = value
    with pytest.raises(ValueError):
        RUNNER.validate_protocol(protocol)
