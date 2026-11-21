"""FJSP adapters, feasible variation and full runs through the basic Moead loop."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest

from moead_framework.algorithm.moead import Moead
from moead_framework.algorithm.fjsp import create_fjsp_moead, uniform_weights_2d
from moead_framework.core.genetic_operator.fjsp import (
    FJSPCrossoverAndMutation, FJSPOperatorConfig,
)
from moead_framework.problem.fjsp import (
    Chromosome, FJSPInstance, FJSPProblem, MachineOption, decode, load_instance,
    parse_instance, random_chromosome,
)
from moead_framework.tool.result import set_seed


ROOT = Path(__file__).resolve().parents[3]
DATA_MAIN = ROOT / "instance" / "main"
PROBLEM_MODULE = "moead_framework.problem.fjsp.problem"

FATTAHI_TEXT = """\
2 2
2 2 1 25 2 37 2 1 32 2 24 0 50 2
2 2 1 45 2 65 2 1 21 2 65 0 60 1
"""


def make_instance(machine_count, jobs, release=None, due=None, weight=None):
    job_count = len(jobs)
    return FJSPInstance(
        machine_count, jobs,
        release_dates=tuple(release if release is not None else (0,) * job_count),
        due_dates=tuple(due if due is not None else (10 ** 9,) * job_count),
        weights=tuple(weight if weight is not None else (1,) * job_count),
    )


@pytest.fixture
def instance():
    return parse_instance(FATTAHI_TEXT, name="fattahi1")


def hand_vector():
    return np.array([0, 1, 0, 1, 1, 1, 0, 0], dtype=int)


def test_evaluate_decodes_once_for_two_objectives_and_keeps_owned_integer_genes(instance):
    problem = FJSPProblem(instance)
    vector = hand_vector()
    with patch(PROBLEM_MODULE + ".decode", wraps=decode) as calls:
        solution = problem.evaluate(vector)
        assert calls.call_count == problem.evaluation_count == 1
    assert solution.F == [66, 28]
    assert np.issubdtype(solution.decision_vector.dtype, np.integer)
    vector[0] = 99
    problem.validate_solution(solution)
    assert problem.evaluation_count == 1  # Validation uses the cached schedule, no NFE.


def test_evaluate_refreshes_existing_solution_without_using_old_cache(instance):
    problem = FJSPProblem(instance)
    solution = problem.evaluate(hand_vector())
    solution.decision_vector[4] = 0
    with pytest.raises(ValueError, match="MS"):
        problem.validate_solution(solution)
    with patch(PROBLEM_MODULE + ".decode", wraps=decode) as calls:
        assert problem.evaluate(solution) is solution
        assert calls.call_count == 1
    assert problem.evaluation_count == 2
    assert solution.F == list(decode(instance, Chromosome.from_vector(instance, solution.decision_vector)).objectives)
    problem.validate_solution(solution)
    solution.F[0] = -1
    with pytest.raises(ValueError, match="objective"):
        problem.validate_solution(solution)


@pytest.mark.parametrize("index", [0, 1])
def test_scalar_f_accessor_has_explicit_single_call_cost(instance, index):
    problem = FJSPProblem(instance)
    assert problem.f(index, hand_vector()) == [66, 28][index]
    assert problem.evaluation_count == 1


@pytest.mark.parametrize("index", [-1, 2, 1.5, True])
def test_bad_objective_index_does_not_spend_nfe(instance, index):
    problem = FJSPProblem(instance)
    with pytest.raises(ValueError):
        problem.f(index, hand_vector())
    assert problem.evaluation_count == 0


def test_invalid_or_float_genes_are_not_silently_repaired_or_counted(instance):
    problem = FJSPProblem(instance)
    for vector in (hand_vector().astype(float), np.array([0, 0]), np.array([99] * 8)):
        with pytest.raises(ValueError):
            problem.evaluate(vector)
    assert problem.evaluation_count == 0


def test_seeded_initial_solution_is_reproducible_and_decodes_once(instance):
    set_seed(17)
    first = FJSPProblem(instance)
    a = first.generate_random_solution()
    set_seed(17)
    second = FJSPProblem(instance)
    b = second.generate_random_solution()
    np.testing.assert_array_equal(a.decision_vector, b.decision_vector)
    assert a.F == b.F
    assert first.evaluation_count == second.evaluation_count == 1
    first.validate_solution(a)


@pytest.mark.parametrize("name", ["crossover_probability", "mutation_probability"])
@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_operator_config_rejects_invalid_probabilities(name, value):
    with pytest.raises(ValueError):
        FJSPOperatorConfig(**{name: value})


def test_variation_with_zero_rates_clones_without_aliasing_or_evaluation(instance):
    first = hand_vector()
    second = np.array(random_chromosome(instance, random.Random(2)).to_vector())
    original = first.copy()
    with patch(PROBLEM_MODULE + ".decode", wraps=decode) as calls:
        child = FJSPCrossoverAndMutation([first, second], instance, FJSPOperatorConfig(0, 0)).run()
        assert calls.call_count == 0
    np.testing.assert_array_equal(child, first)
    child[0] = 99
    np.testing.assert_array_equal(first, original)


def test_pox_keeps_selected_jobs_and_fills_others_in_second_parent_order(monkeypatch):
    instance = parse_instance(
        "3 1\n2 1 1 1 1 1 1 0 100 1\n2 1 1 1 1 1 1 0 100 1\n2 1 1 1 1 1 1 0 100 1\n"
    )
    first = Chromosome((0, 1, 2, 0, 1, 2), (0,) * 6)
    second = Chromosome((2, 2, 1, 1, 0, 0), (0,) * 6)
    operator = FJSPCrossoverAndMutation([first.to_vector(), second.to_vector()], instance)
    monkeypatch.setattr(random, "randint", lambda a, b: 1)
    monkeypatch.setattr(random, "sample", lambda population, k: [0])
    assert operator._pox(first.os, second.os) == [0, 2, 2, 0, 1, 1]


@pytest.mark.parametrize("crossover,mutation", [(0, 0), (1, 0), (0, 1), (1, 1), (0.8, 0.05)])
def test_operators_preserve_unequal_job_multisets_and_local_machine_choices(crossover, mutation):
    set_seed(19)
    rng = random.Random(19)
    for _ in range(35):
        jobs = tuple(tuple(tuple(MachineOption(m, rng.randrange(6)) for m in rng.sample(range(4), rng.randrange(1, 5)))
                           for _ in range(rng.randrange(1, 5))) for _ in range(rng.randrange(1, 5)))
        instance = make_instance(4, jobs)
        first, second = random_chromosome(instance, rng), random_chromosome(instance, rng)
        child = FJSPCrossoverAndMutation([first.to_vector(), second.to_vector()], instance,
                                        FJSPOperatorConfig(crossover, mutation)).run()
        chromosome = Chromosome.from_vector(instance, child)
        chromosome.validate(instance)
        assert np.issubdtype(child.dtype, np.integer)
        problem = FJSPProblem(instance)
        problem.validate_solution(problem.evaluate(child))


def test_ms_mutation_changes_the_only_flexible_choice_even_for_one_operation():
    instance = parse_instance("1 2\n1 2 1 3 2 7 0 100 1\n")
    for _ in range(5):
        child = FJSPCrossoverAndMutation([[0, 0], [0, 0]], instance, FJSPOperatorConfig(0, 1)).run()
        np.testing.assert_array_equal(child, [0, 1])


def test_single_job_single_machine_all_zero_case_runs_without_special_repair():
    instance = parse_instance("1 1\n1 1 1 0 0 100 1\n")
    problem = FJSPProblem(instance)
    algorithm = create_fjsp_moead(problem, max_evaluation=9, partitions=1,
                                  number_of_weight_neighborhood=2,
                                  operator_config=FJSPOperatorConfig(1, 1))
    archive = algorithm.run()
    assert problem.evaluation_count == algorithm.current_eval == 9
    assert len(archive) == 1 and archive[0].F == [0, 0]
    problem.validate_solution(archive[0])


@pytest.mark.parametrize("partitions", [1, 2, 4, 12, 16])
def test_uniform_weights_have_exact_count_unique_rows_and_two_boundaries(partitions):
    weights = uniform_weights_2d(partitions)
    assert len(weights) == partitions + 1
    assert len(set(weights)) == len(weights)
    assert {(1.0, 0.0), (0.0, 1.0)} <= set(weights)
    assert np.isfinite(weights).all() and (np.array(weights) >= 0).all()
    np.testing.assert_allclose(np.sum(weights, axis=1), 1)


@pytest.mark.parametrize("partitions", [0, -1, True, 2.5])
def test_invalid_partitions_are_rejected(partitions):
    with pytest.raises(ValueError):
        uniform_weights_2d(partitions)


def _pareto(points):
    unique = set(points)
    return {p for p in unique if not any(
        all(a <= b for a, b in zip(q, p)) and any(a < b for a, b in zip(q, p))
        for q in unique
    )}


@pytest.mark.parametrize("budget", [6, 7, 13, 40])
def test_basic_moead_fjsp_evaluates_each_candidate_once_and_archives_all_seen(instance, budget):
    set_seed(7)
    problem = FJSPProblem(instance)
    seen = []
    evaluate = problem.evaluate

    def record(x):
        solution = evaluate(x)
        problem.validate_solution(solution)
        seen.append((tuple(solution.decision_vector), tuple(solution.F)))
        return solution

    with patch.object(problem, "evaluate", side_effect=record) as calls, \
            patch(PROBLEM_MODULE + ".decode", wraps=decode) as decoding:
        algorithm = create_fjsp_moead(problem, max_evaluation=budget, partitions=2,
                                      number_of_weight_neighborhood=3)
        initial = deepcopy(algorithm.ep)
        archive = algorithm.run()
        assert calls.call_count == decoding.call_count == problem.evaluation_count == budget
    assert type(algorithm) is Moead
    assert algorithm.current_eval == budget
    assert {tuple(s.F) for s in archive} == _pareto(point for _, point in seen)
    assert len(archive) == len({tuple(s.F) for s in archive})
    for solution in archive + algorithm.population + initial:
        problem.validate_solution(solution)
    algorithm.run()
    assert problem.evaluation_count == budget


def test_same_seed_repeats_the_complete_archive_and_selected_config(instance):
    outcomes = []
    config = FJSPOperatorConfig(0.7, 0.2)
    for _ in range(2):
        set_seed(11)
        problem = FJSPProblem(instance)
        algorithm = create_fjsp_moead(problem, max_evaluation=40, partitions=2,
                                      number_of_weight_neighborhood=3, operator_config=config)
        assert algorithm.offspring_generator.config == config
        archive = algorithm.run()
        outcomes.append(sorted((tuple(s.F), tuple(s.decision_vector)) for s in archive))
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("kwargs", [dict(max_evaluation=2), dict(number_of_weight_neighborhood=7)])
def test_bad_run_configuration_fails_before_initialization_evaluations(instance, kwargs):
    problem = FJSPProblem(instance)
    options = dict(max_evaluation=20, partitions=2, number_of_weight_neighborhood=3)
    options.update(kwargs)
    with pytest.raises(ValueError):
        create_fjsp_moead(problem, **options)
    assert problem.evaluation_count == 0


def test_factory_rejects_reusing_a_problem_with_previous_nfe(instance):
    problem = FJSPProblem(instance)
    problem.generate_random_solution()
    with pytest.raises(ValueError, match="fresh"):
        create_fjsp_moead(problem)


@pytest.mark.parametrize("path", [
    "Brandimate1993/Loose/Mk01/Mk01.fjs",
    "Brandimate1993/Tight/Mk01/Mk01.fjs",
    "DauzerePeresPaulli1994/Loose/01a/01a.fjs",
    "DauzerePeresPaulli1994/Moderate/01a/01a.fjs",
    "DauzerePeresPaulli1994/Tight/01a/01a.fjs",
])
def test_end_to_end_small_budget_on_main_instances(path):
    set_seed(5)
    problem = FJSPProblem(DATA_MAIN / path)
    algorithm = create_fjsp_moead(problem, max_evaluation=17, partitions=2,
                                  number_of_weight_neighborhood=3,
                                  operator_config=FJSPOperatorConfig(1, 1))
    archive = algorithm.run()
    assert algorithm.current_eval == problem.evaluation_count == 17
    assert archive
    for solution in archive + algorithm.population:
        problem.validate_solution(solution)


def test_forced_variation_is_valid_on_every_main_instance():
    set_seed(3)
    rng = random.Random(3)
    paths = sorted(DATA_MAIN.rglob("*.fjs"))
    assert len(paths) == 84
    for path in paths:
        instance = load_instance(path)
        parents = [random_chromosome(instance, rng).to_vector() for _ in range(2)]
        child = FJSPCrossoverAndMutation(parents, instance, FJSPOperatorConfig(1, 1)).run()
        problem = FJSPProblem(instance)
        problem.validate_solution(problem.evaluate(child))


def test_cli_exports_replayable_solutions_and_refuses_overwrite(tmp_path, instance):
    output = tmp_path / "run.json"
    command = [sys.executable, str(ROOT / "scripts/run_fjsp.py"),
               "--instance", str(DATA_MAIN / "Brandimate1993/Loose/Mk01/Mk01.fjs"),
               "--partitions", "2", "--neighbors", "3", "--evaluations", "13", "--output", str(output)]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True)
    summary = json.loads(completed.stdout)
    saved = output.read_bytes()
    data = json.loads(saved)
    assert summary["actual_evaluations"] == data["run"]["actual_evaluations"] == 13
    assert data["run"]["objective_names"] == ["makespan", "twt"]
    assert data["index_base"] == 0 and len(data["weights"]) == 3
    assert len(data["solutions"]) == summary["archive_size"] > 0
    mk01 = load_instance(DATA_MAIN / "Brandimate1993/Loose/Mk01/Mk01.fjs")
    for entry in data["solutions"]:
        chromosome = Chromosome.from_vector(mk01, entry["decision_vector"])
        schedule = decode(mk01, chromosome)
        assert list(schedule.objectives) == entry["objectives"]
        assert [asdict(op) for op in schedule.operations] == entry["schedule"]
    repeated = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    assert repeated.returncode == 2
    assert output.read_bytes() == saved
