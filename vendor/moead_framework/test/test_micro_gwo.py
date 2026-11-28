"""Micro-populations: independent budget/archive checks and explicit mating cases."""

from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from moead_framework.algorithm.fjsp_variants import (
    create_fjsp_variant, population_layout, population_solutions,
)
from moead_framework.aggregation import Tchebycheff
from moead_framework.algorithm.moead_micro_gwo import MoeadMicroGWO
from moead_framework.core.genetic_operator.fjsp import FJSPCrossoverAndMutation
from moead_framework.core.initializer import AbstractInitializer
from moead_framework.core.offspring_generator.fjsp import FJSPOffspringGenerator
from moead_framework.problem.fjsp import FJSPProblem, decode, parse_instance, Chromosome
from moead_framework.solution import OneDimensionSolution
from moead_framework.tool.fjsp_experiments import independent_pareto_front, run_validation_case
from moead_framework.tool.result import set_seed


ROOT = Path(__file__).resolve().parents[3]
TEXT = "2 2\n2 2 1 25 2 37 2 1 32 2 24 0 50 2\n2 2 1 45 2 65 2 1 21 2 65 0 60 1\n"
INSTANCE_PATH = "instance/main/Brandimate1993/Loose/Mk01/Mk01.fjs"


class RecordingProblem(FJSPProblem):
    def __init__(self, instance):
        super().__init__(instance)
        self.seen = []

    def evaluate(self, genes):
        solution = super().evaluate(genes)
        self.seen.append(deepcopy(solution))
        return solution


def settings(variant="micro_gwo", budget=31):
    return dict(variant=variant, micro_population_size=4, partitions=2,
                neighbors=3, evaluations=budget, crossover=0.8, mutation=0.05)


def build(variant="micro_gwo", budget=31):
    set_seed(4)
    problem = RecordingProblem(parse_instance(TEXT))
    return problem, create_fjsp_variant(problem, settings(variant, budget))


def runner_module():
    spec = spec_from_file_location("micro_study_runner", ROOT / "scripts/run_baseline_validation.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("variant", ["micro_gwo", "micro_random"])
@pytest.mark.parametrize("budget", [12, 13, 17, 31])
def test_every_resident_is_counted_and_archive_matches_all_evaluations(variant, budget):
    problem, algorithm = build(variant, budget)
    assert algorithm.current_eval == 12
    assert algorithm.number_of_weight == 3 and algorithm.population_size == 12
    assert np.array_equal(algorithm.z, np.min([s.F for s in problem.seen], axis=0))
    archive = algorithm.run()
    assert algorithm.current_eval == problem.evaluation_count == len(problem.seen) == budget
    assert [len(group) for group in algorithm.micro_populations] == [4, 4, 4]
    assert {tuple(s.F) for s in archive} == independent_pareto_front([s.F for s in problem.seen])
    assert len(archive) == len({tuple(s.F) for s in archive})
    assert np.array_equal(algorithm.z, np.min([s.F for s in problem.seen], axis=0))
    for solution in archive + population_solutions(algorithm) + algorithm.population:
        problem.validate_solution(solution)
    for index, representative in enumerate(algorithm.population):
        best = min(algorithm.scalar_score(s, index) for s in algorithm.micro_populations[index])
        assert algorithm.scalar_score(representative, index) == best
        assert all(representative is not resident for resident in algorithm.micro_populations[index])
    assert sum(algorithm.parent_selector.selection_counts.values()) == budget - 12


@pytest.mark.parametrize("key,value", [
    ("micro_population_size", 3), ("micro_population_size", True),
    ("micro_population_size", 4.5), ("evaluations", 11),
    ("neighbors", 4), ("variant", "unknown"),
])
def test_bad_configuration_fails_before_spending_nfe(key, value):
    config = settings()
    config[key] = value
    problem = FJSPProblem(parse_instance(TEXT))
    with pytest.raises((ValueError, TypeError)):
        create_fjsp_variant(problem, config)
    assert problem.evaluation_count == 0


def test_leader_ranking_uses_one_target_weight_without_mutating_or_evaluating():
    problem, algorithm = build()
    # Deliberately synthetic objective vectors isolate the scalar ranking rule.
    algorithm.z = [0, 0]
    algorithm.weights[0] = np.array([0.9, 0.1])
    for group, residents in enumerate(algorithm.micro_populations):
        for slot in range(len(residents)):
            residents[slot] = OneDimensionSolution(np.array([group * 4 + slot]), f=[100, 100])
    algorithm.micro_populations[0][0].F = [1, 100]   # score 10
    algorithm.micro_populations[1][0].F = [50, 1]    # score 45 under TARGET w_0
    before = problem.evaluation_count
    selector = algorithm.parent_selector
    ranked = selector.ranked_candidates([0, 1, 2], 0)
    assert ranked[:2] == [(0, 0), (1, 0)]
    assert selector.select_leaders(ranked) == ranked[:3]
    assert problem.evaluation_count == before


@pytest.mark.parametrize("leader_index", [0, 1, 2])
def test_early_leader_branch_can_choose_alpha_beta_or_delta(monkeypatch, leader_index):
    problem, algorithm = build(budget=32)
    algorithm.current_sub_problem = 0
    selector = algorithm.parent_selector
    ranked = selector.ranked_candidates([0, 1, 2], 0)
    leaders = selector.select_leaders(ranked)
    assert len(leaders) == 3
    saved = [s.decision_vector.copy() for s in population_solutions(algorithm)]
    monkeypatch.setattr("random.random", lambda: 0.1)  # B0=.9, progress=0
    monkeypatch.setattr("random.choice", lambda choices: choices[leader_index])
    parents = selector.select([0, 1, 2])
    record = selector.last_selection
    assert record["branch"] == "leader" and record["balance"] == pytest.approx(0.9)
    assert record["parents"][0][0] == 0
    assert record["parents"][0] not in leaders
    assert record["parents"][1] == leaders[leader_index]
    assert parents[1] is selector.solution_at(leaders[leader_index])
    assert problem.evaluation_count == 12
    assert all(np.array_equal(old, s.decision_vector)
               for old, s in zip(saved, population_solutions(algorithm)))


def test_halfway_search_uses_peer_branch_and_threshold_is_unambiguous(monkeypatch):
    _, algorithm = build(budget=32)
    algorithm.current_sub_problem = 0
    selector = algorithm.parent_selector
    monkeypatch.setattr("random.random", lambda: 0.5)
    selector.select([0, 1, 2])
    assert selector.last_selection["balance"] == 0.5
    assert selector.last_selection["branch"] == "leader"
    algorithm.current_eval = 22  # (22-12)/(32-12)=.5, not 22/32
    selector.select([0, 1, 2])
    record = selector.last_selection
    assert record["progress"] == 0.5 and record["branch"] == "peer"
    assert record["parents"][1] not in record["leaders"]
    assert record["parents"][0] != record["parents"][1]


def test_constant_tiny_instance_needs_no_extra_evaluations_or_three_unique_wolves():
    instance = parse_instance("1 1\n1 1 1 0 0 0 1\n")
    case = run_validation_case(instance, settings(budget=17), 3, [12, 13, 17])
    assert case["actual_evaluations"] == len(case["evaluation_trace"]) == 17
    assert case["objective_vectors"] == [[0, 0]]
    assert case["archive_matches_history"] and case["all_schedules_valid"]


def test_replacement_changes_only_worst_slot_and_copies_between_groups():
    _, algorithm = build()
    algorithm.z = [0, 0]
    for group, residents in enumerate(algorithm.micro_populations):
        for slot in range(4):
            residents[slot] = OneDimensionSolution(np.array([group * 4 + slot]), f=[10 + slot] * 2)
    before = [[id(s) for s in group] for group in algorithm.micro_populations]
    child = OneDimensionSolution(np.array([-1]), f=[1, 1])
    algorithm.update_solutions(child, algorithm.aggregation_function, 0)
    for index, group in enumerate(algorithm.micro_populations):
        assert [id(s) for s in group[:3]] == before[index][:3]
        assert group[3].F == [1, 1]
    inserted = [group[3] for group in algorithm.micro_populations]
    assert len({id(s) for s in inserted + [child]}) == 4
    child.F[0] = 999
    inserted[0].decision_vector[0] = 999
    assert inserted[1].F == [1, 1] and inserted[1].decision_vector[0] == -1


def test_rejected_child_still_reaches_archive():
    _, algorithm = build()
    algorithm.z = [0, 0]
    algorithm.weights = [np.array([1.0, 0.0]) for _ in range(3)]
    for group, residents in enumerate(algorithm.micro_populations):
        for slot in range(4):
            residents[slot] = OneDimensionSolution(np.array([group * 4 + slot]), f=[0, 100])
    algorithm.ep = [deepcopy(algorithm.micro_populations[0][0])]
    child = OneDimensionSolution(np.array([-1]), f=[1, 0])
    algorithm.update_solutions(child, algorithm.aggregation_function, 0)
    assert all(s.F == [0, 100] for s in population_solutions(algorithm))
    algorithm.update_archive(child)
    assert {tuple(s.F) for s in algorithm.ep} == {(0, 100), (1, 0)}
    child.F[0] = 99
    assert (1, 0) in {tuple(s.F) for s in algorithm.ep}


def test_representative_copy_is_reused_but_reranked_when_ideal_point_changes():
    problem, algorithm = build()
    algorithm.weights[0] = np.array([0.5, 0.5])
    algorithm.micro_populations[0] = [
        OneDimensionSolution(np.array([slot]), f=point)
        for slot, point in enumerate([[1, 10], [10, 1], [100, 100], [101, 101]])
    ]
    algorithm.z = [1, 1]
    algorithm.refresh_representatives()
    first_copy = algorithm.population[0]
    assert first_copy.F == [1, 10]
    algorithm.refresh_representatives()
    assert algorithm.population[0] is first_copy
    algorithm.z = [1, 0]
    algorithm.refresh_representatives()
    assert algorithm.population[0].F == [10, 1]
    assert algorithm.population[0] is not algorithm.micro_populations[0][1]
    assert problem.evaluation_count == 12


@pytest.mark.parametrize("variant", ["micro_gwo", "micro_random"])
def test_validation_case_replays_all_groups_and_keeps_checkpoints(variant):
    config = settings(variant)
    first = run_validation_case(parse_instance(TEXT), config, 2, [12, 13, 25, 31])
    second = run_validation_case(parse_instance(TEXT), config, 2, [12, 13, 25, 31])
    assert first["reproducibility_signature"] == second["reproducibility_signature"]
    assert first["population_size"] == 12 and first["number_of_subproblems"] == 3
    assert [s["nfe"] for s in first["snapshots"]] == [12, 13, 25, 31]
    assert sum(first["parent_selection_counts"].values()) == 19


@pytest.mark.parametrize("variant", ["micro_gwo", "micro_random"])
def test_study_resumes_and_rejects_changed_micro_configuration(tmp_path, variant):
    runner = runner_module()
    protocol = dict(schema_version=1, algorithm=settings(variant, 17),
                    seeds=[0, 1], repeat_seed=0, checkpoints=[12, 13, 17],
                    hv_reference=[1.1, 1.1], instances=[dict(id="mk01_loose", family="Brandimate1993", path=INSTANCE_PATH)])
    config = tmp_path / "protocol.json"
    config.write_text(json.dumps(protocol), encoding="utf-8")
    report = runner.run_study(config, tmp_path / "study", make_plot=False)
    assert report["valid_runs"] == 2 and report["replay_runs"] == 1
    raw = tmp_path / "study/runs/mk01_loose_seed0.json"
    saved = raw.read_bytes()
    assert runner.run_study(config, tmp_path / "study", resume=True, make_plot=False)["valid_runs"] == 2
    assert raw.read_bytes() == saved
    protocol["algorithm"]["micro_population_size"] = 5
    protocol["checkpoints"] = [15, 17]
    config.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="Resume refused"):
        runner.run_study(config, tmp_path / "study", resume=True, make_plot=False)


def test_protocol_validates_total_population_but_neighborhood_counts_groups():
    runner = runner_module()
    protocol = dict(schema_version=1, algorithm=settings(), seeds=[0], repeat_seed=0,
                    checkpoints=[12, 31], hv_reference=[1.1, 1.1],
                    instances=[dict(id="test", family="test", path=INSTANCE_PATH)])
    runner.validate_protocol(protocol)
    protocol["algorithm"]["neighbors"] = 4  # 4 <= 12 residents, but only 3 groups!
    with pytest.raises(ValueError, match="Neighborhood"):
        runner.validate_protocol(protocol)


@pytest.mark.parametrize("variant", ["micro_gwo", "micro_random"])
def test_cli_exports_micro_groups_and_refuses_overwrite(tmp_path, variant):
    output = tmp_path / "demo.json"
    command = [sys.executable, str(ROOT / "scripts/run_fjsp.py"), "--variant", variant,
               "--partitions", "2", "--neighbors", "3", "--evaluations", "17", "--output", str(output)]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    summary = json.loads(completed.stdout)
    assert summary["population_size"] == 12 and summary["actual_evaluations"] == 17
    saved = output.read_bytes()
    data = json.loads(saved)
    instance = FJSPProblem(ROOT / INSTANCE_PATH).instance
    assert [len(group) for group in data["micro_populations"]] == [4, 4, 4]
    for group in data["micro_populations"]:
        for resident in group:
            chromosome = Chromosome.from_vector(instance, resident["decision_vector"])
            assert list(decode(instance, chromosome).objectives) == resident["objectives"]
    assert subprocess.run(command, cwd=ROOT, capture_output=True).returncode == 2
    assert output.read_bytes() == saved


class RecordingInitializer(AbstractInitializer):
    """Weight-aware stub: resident i is random but tagged with weights[i]."""
    weight_aware = True

    def __init__(self):
        self.tags = {}
        self.last_stats = {}

    def initialize(self, problem, population_size, weights, on_evaluated):
        population = []
        for weight in weights[:population_size]:
            solution = problem.generate_random_solution()
            on_evaluated(solution)
            self.tags[id(solution)] = tuple(weight)
            population.append(solution)
        return population


@pytest.mark.parametrize("variant", ["micro_gwo", "micro_random"])
@pytest.mark.parametrize("mode", ["qlearning", "hybrid_q_random"])
def test_micro_with_qinit_respects_nfe_budget_and_archive(variant, mode):
    set_seed(7)
    problem = RecordingProblem(parse_instance(TEXT))
    config = settings(variant, budget=12 + 10)
    config["initializer"] = mode
    algorithm = create_fjsp_variant(problem, config)
    assert algorithm.current_eval == problem.evaluation_count == 12
    assert algorithm.initializer.last_stats["population_size"] == 12
    assert [len(group) for group in algorithm.micro_populations] == [4, 4, 4]
    archive = algorithm.run()
    assert algorithm.current_eval == problem.evaluation_count == len(problem.seen) == 22
    assert {tuple(s.F) for s in archive} == independent_pareto_front([s.F for s in problem.seen])
    for solution in archive + population_solutions(algorithm) + algorithm.population:
        problem.validate_solution(solution)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_micro_weight_aware_initializer_follows_group_shuffle(seed):
    set_seed(seed)
    problem = FJSPProblem(parse_instance(TEXT))
    initializer = RecordingInitializer()
    algorithm = MoeadMicroGWO(
        problem=problem, max_evaluation=20, number_of_weight_neighborhood=3,
        micro_population_size=4, weight_file=_weight_file(),
        aggregation_function=Tchebycheff, genetic_operator=FJSPCrossoverAndMutation,
        offspring_generator=FJSPOffspringGenerator, initializer=initializer,
    )
    assert len(initializer.tags) == 12
    for index, group in enumerate(algorithm.micro_populations):
        expected = tuple(float(v) for v in algorithm.weights[index])
        assert len(group) == 4
        assert all(initializer.tags[id(resident)] == expected for resident in group)


def _weight_file():
    from moead_framework.algorithm.fjsp import uniform_weights_2d
    import tempfile
    handle = tempfile.NamedTemporaryFile("w", suffix=".ws", delete=False, encoding="utf-8")
    handle.write("\n".join(" ".join(map(str, row)) for row in uniform_weights_2d(2)))
    handle.close()
    return handle.name


def test_micro_rejects_invalid_initializer_without_spending_nfe():
    problem = FJSPProblem(parse_instance(TEXT))
    config = settings()
    config["initializer"] = "unknown"
    with pytest.raises(ValueError):
        create_fjsp_variant(problem, config)
    assert problem.evaluation_count == 0


def test_baseline_variant_with_initializer_is_rejected_by_variant_factory():
    problem = FJSPProblem(parse_instance(TEXT))
    config = settings("baseline")
    config["initializer"] = "hybrid_q_random"
    with pytest.raises(ValueError):
        create_fjsp_variant(problem, config)


@pytest.mark.parametrize("variant", ["micro_gwo", "micro_random"])
def test_micro_with_qinit_is_reproducible(variant):
    config = settings(variant, 22)
    config["initializer"] = "hybrid_q_random"
    first = run_validation_case(parse_instance(TEXT), config, 2, [12, 22])
    second = run_validation_case(parse_instance(TEXT), config, 2, [12, 22])
    assert first["reproducibility_signature"] == second["reproducibility_signature"]
    assert first["algorithm"] == "MoeadMicroGWO" and first["population_size"] == 12


@pytest.mark.parametrize("mode", ["qlearning", "hybrid_q_random"])
def test_cli_micro_gwo_with_qinit(tmp_path, mode):
    output = tmp_path / "demo.json"
    command = [sys.executable, str(ROOT / "scripts/run_fjsp.py"), "--variant", "micro_gwo",
               "--partitions", "2", "--neighbors", "3", "--evaluations", "62",
               "--initializer", mode, "--output", str(output)]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    summary = json.loads(completed.stdout)
    assert summary["population_size"] == 12 and summary["actual_evaluations"] == 62
    assert summary["algorithm"] == "MoeadMicroGWO" and summary["initializer"] == mode
    stats = summary["initializer_stats"]
    assert stats["n_random"] + stats["n_qlearning"] == 12
    assert "q_init" in summary
    assert [len(g) for g in json.loads(output.read_text(encoding="utf-8"))["micro_populations"]] == [4, 4, 4]
