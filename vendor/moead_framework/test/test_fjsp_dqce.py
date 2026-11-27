"""Unit and integration tests for FJSP DQCE, DQN agent, neighborhood local search and CLI."""

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
import torch

from moead_framework.algorithm.fjsp_dqce import FjspDqceMoead, create_fjsp_dqce
from moead_framework.core.genetic_operator.fjsp import FJSPOperatorConfig
from moead_framework.core.local_search.fjsp_neighborhoods import FJSPNeighborhoodOperators
from moead_framework.core.reinforcement_learning.dqn_agent import DQNAgent, DQNConfig
from moead_framework.core.termination_criteria.abstract_termination_criteria import TerminationCriteria
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


def _pareto(points):
    unique = set(points)
    return {p for p in unique if not any(
        all(a <= b for a, b in zip(q, p)) and any(a < b for a, b in zip(q, p))
        for q in unique
    )}


class StopAtEval(TerminationCriteria):
    def test(self):
        return self.algorithm.current_eval < self.algorithm.number_of_weight + 2


@pytest.fixture
def instance():
    return parse_instance(FATTAHI_TEXT, name="fattahi1")


@pytest.mark.parametrize("kwargs", [
    {"learning_rate": 0},
    {"learning_rate": -0.01},
    {"learning_rate": float("nan")},
    {"gamma": -0.1},
    {"gamma": 1.1},
    {"epsilon": -0.1},
    {"epsilon": 1.1},
    {"batch_size": 0},
    {"batch_size": True},
    {"experience_pool_size": 0},
    {"epochs": 0},
    {"target_update_freq": 0},
])
def test_dqn_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        DQNConfig(**kwargs)


def test_dqn_agent_learns_and_updates_target_when_pool_is_ready():
    torch.manual_seed(5)
    set_seed(5)
    config = DQNConfig(
        learning_rate=0.01,
        gamma=0.9,
        epsilon=0.9,
        batch_size=3,
        experience_pool_size=4,
        epochs=2,
        target_update_freq=2,
    )
    agent = DQNAgent(state_dim=8, action_dim=6, config=config)
    initial_tar = [param.detach().clone() for param in agent.q_tar.parameters()]

    for step in range(4):
        state = np.full(8, float(step), dtype=np.float32)
        next_state = np.full(8, float(step + 1), dtype=np.float32)
        action = agent.select_action(state)
        assert 0 <= action < 6
        agent.store_and_learn(state, action, reward=5.0, next_state=next_state)

    assert len(agent.memory) == 4
    assert agent.learn_step_counter == 2
    assert any(
        not torch.equal(before, after)
        for before, after in zip(initial_tar, agent.q_tar.parameters())
    )


def test_dqn_compute_reward_cases():
    assert DQNAgent.compute_reward([10, 20], [9, 20]) == 5
    assert DQNAgent.compute_reward([10, 20], [12, 18]) == 10
    assert DQNAgent.compute_reward([10, 20], [10, 20]) == 0
    assert DQNAgent.compute_reward([10, 20], [12, 22]) == 0


def test_neighborhood_operators_preserve_feasibility_and_do_not_mutate_parent():
    rng = random.Random(13)
    jobs = tuple(
        tuple(
            tuple(MachineOption(m, rng.randrange(1, 10)) for m in rng.sample(range(4), rng.randrange(1, 5)))
            for _ in range(rng.randrange(1, 5))
        )
        for _ in range(4)
    )
    inst = FJSPInstance(
        machine_count=4,
        jobs=jobs,
        release_dates=(0, 0, 0, 0),
        due_dates=(50, 50, 50, 50),
        weights=(1, 1, 1, 1),
    )
    problem = FJSPProblem(inst)
    ops = FJSPNeighborhoodOperators(problem)
    assert ops.total_ops == inst.operation_count

    set_seed(13)
    sol = problem.generate_random_solution()
    original_vec = sol.decision_vector.copy()

    for action_id in range(ops.num_operators):
        for _ in range(5):
            cand = ops.apply(action_id, sol)
            np.testing.assert_array_equal(sol.decision_vector, original_vec)
            problem.validate_solution(sol)
            Chromosome.from_vector(inst, cand).validate(inst)

    # MS operators must change flexible choices to valid local indices, not no-op.
    assert any(not np.array_equal(ops.apply(4, sol), original_vec) for _ in range(10))
    assert any(not np.array_equal(ops.apply(5, sol), original_vec) for _ in range(10))


def test_find_critical_indices_only_selects_makespan_jobs(instance):
    problem = FJSPProblem(instance)
    ops = FJSPNeighborhoodOperators(problem)
    # Schedule from hand vector [0, 1, 0, 1, 1, 1, 0, 0]: job 0 ends at 61, job 1 ends at 66 (makespan).
    sol = problem.evaluate([0, 1, 0, 1, 1, 1, 0, 0])
    assert sol.F[0] == 66
    os_part, _ = ops._split_vector(sol.decision_vector)
    assert ops._find_critical_indices(sol, os_part) == [1, 3]


def test_active_compaction_reorders_os_by_schedule_start_times():
    # Job 0 (M0:5, M1:3) dispatched before Job 1 (M1:4 on gap [0, 4] before Job 0's op 1 at [5, 8]).
    inst = parse_instance("2 2\n2 1 1 5 1 2 3 0 100 1\n1 1 2 4 0 100 1\n")
    problem = FJSPProblem(inst)
    ops = FJSPNeighborhoodOperators(problem)
    sol = problem.evaluate([0, 0, 1, 0, 0, 0])
    assert [(op.job, op.start) for op in sol.schedule.operations] == [(0, 0), (0, 5), (1, 0)]
    compacted = ops.active_compaction(sol)
    assert compacted[:3].tolist() == [0, 1, 0]
    Chromosome.from_vector(inst, compacted).validate(inst)


@pytest.mark.parametrize("budget", [6, 7, 13, 40])
@pytest.mark.parametrize("use_compaction", [True, False])
def test_dqce_respects_budget_evaluates_once_and_archives_pareto(instance, budget, use_compaction):
    set_seed(7)
    torch.manual_seed(7)
    problem = FJSPProblem(instance)
    seen = []
    evaluate = problem.evaluate

    def record(x):
        sol = evaluate(x)
        problem.validate_solution(sol)
        seen.append((tuple(sol.decision_vector), tuple(sol.F)))
        return sol

    dqn_cfg = DQNConfig(experience_pool_size=4, batch_size=2, target_update_freq=2)
    with patch.object(problem, "evaluate", side_effect=record) as calls, \
            patch(PROBLEM_MODULE + ".decode", wraps=decode) as decoding:
        algorithm = create_fjsp_dqce(
            problem,
            max_evaluation=budget,
            partitions=2,
            number_of_weight_neighborhood=3,
            dqn_config=dqn_cfg,
            use_compaction=use_compaction,
        )
        archive = algorithm.run()
        assert calls.call_count == decoding.call_count == problem.evaluation_count == budget

    assert type(algorithm) is FjspDqceMoead
    assert algorithm.current_eval == budget
    assert {tuple(s.F) for s in archive} == _pareto(point for _, point in seen)
    assert len(archive) == len({tuple(s.F) for s in archive})
    for solution in archive + algorithm.population + algorithm.parasite_population:
        problem.validate_solution(solution)

    # Re-entering run() must not spend additional evaluations.
    algorithm.run()
    assert problem.evaluation_count == budget


def test_dqce_triggers_dqn_learning_and_is_reproducible(instance):
    outcomes = []
    histories = []
    dqn_cfg = DQNConfig(experience_pool_size=5, batch_size=3, epochs=1, target_update_freq=2)
    for _ in range(2):
        set_seed(11)
        torch.manual_seed(11)
        problem = FJSPProblem(instance)
        algorithm = create_fjsp_dqce(
            problem,
            max_evaluation=60,
            partitions=2,
            number_of_weight_neighborhood=3,
            operator_config=FJSPOperatorConfig(0.8, 0.05),
            dqn_config=dqn_cfg,
            use_compaction=True,
        )
        archive = algorithm.run()
        assert algorithm._agent is not None and algorithm._agent.learn_step_counter > 0
        assert len(algorithm.operator_history) > 0
        outcomes.append(sorted((tuple(s.F), tuple(s.decision_vector)) for s in archive))
        histories.append(algorithm.operator_history)
    assert outcomes[0] == outcomes[1]
    assert histories[0] == histories[1]


def test_dqce_checkpoint_and_custom_termination(instance):
    set_seed(3)
    torch.manual_seed(3)
    problem = FJSPProblem(instance)
    observed = []
    algorithm = create_fjsp_dqce(problem, max_evaluation=20, partitions=2, number_of_weight_neighborhood=3)
    algorithm.termination_criteria = StopAtEval(algorithm)
    algorithm.run(checkpoint=lambda alg: observed.append(alg.current_eval))
    assert algorithm.current_eval == problem.evaluation_count == 5
    assert observed == [3, 4]


@pytest.mark.parametrize("kwargs", [
    {"max_evaluation": 2},
    {"number_of_weight_neighborhood": 7},
    {"dqn_config": "invalid"},
    {"use_compaction": "yes"},
])
def test_dqce_bad_config_fails_before_any_evaluation(instance, kwargs):
    problem = FJSPProblem(instance)
    options = dict(max_evaluation=20, partitions=2, number_of_weight_neighborhood=3)
    options.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        create_fjsp_dqce(problem, **options)
    assert problem.evaluation_count == 0


def test_dqce_cli_exports_replayable_solutions_and_validates_args(tmp_path):
    output = tmp_path / "dqce_run.json"
    command = [
        sys.executable, str(ROOT / "scripts/run_fjsp_dqce.py"),
        "--instance", str(DATA_MAIN / "Brandimate1993/Loose/Mk01/Mk01.fjs"),
        "--partitions", "2", "--neighbors", "3", "--evaluations", "30",
        "--buffer-size", "4", "--batch-size", "2", "--target-update", "2",
        "--output", str(output),
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True)
    summary = json.loads(completed.stdout)
    saved = output.read_bytes()
    data = json.loads(saved)
    assert summary["algorithm"] == "FjspDqceMoead"
    assert summary["actual_evaluations"] == data["run"]["actual_evaluations"] == 30
    assert summary["all_schedules_valid"] is True
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

    bad_cli = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_fjsp_dqce.py"), "--buffer-size", "0"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
    )
    assert bad_cli.returncode == 2


def test_evaluate_dqce_computes_hv_set_coverage_and_summary(tmp_path):
    out_dir = tmp_path / "eval_out"
    command = [
        sys.executable, str(ROOT / "scripts/evaluate_dqce.py"),
        "--instance", str(DATA_MAIN / "Brandimate1993/Loose/Mk01/Mk01.fjs"),
        "--seeds", "0", "1",
        "--evaluations", "20", "--partitions", "4", "--neighbors", "3",
        "--crossover", "1", "--mutation", "0.2",
        "--buffer-size", "4", "--batch-size", "2", "--target-update", "2",
        "--output-dir", str(out_dir),
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True)
    stdout_payload = json.loads(completed.stdout)
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "summary.csv").is_file()
    assert (out_dir / "summary.md").is_file()
    assert (out_dir / "per_seed.csv").is_file()
    assert (out_dir / "runs/dqce_seed0.json").is_file()
    assert (out_dir / "runs/baseline_seed1.json").is_file()

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["seeds"] == [0, 1]
    assert len(summary["per_seed"]) == 2
    metrics = summary["metrics"]
    assert metrics == stdout_payload["metrics"]
    for key in ("hv_dqce", "hv_baseline", "c_dqce_over_baseline_per_seed", "c_baseline_over_dqce_per_seed"):
        assert 0.0 <= metrics[key]["mean"] <= 1.0
        assert metrics[key]["std"] >= 0.0
    assert 0.0 <= metrics["c_dqce_over_baseline_pooled"] <= 1.0
    assert 0.0 <= metrics["c_baseline_over_dqce_pooled"] <= 1.0

    # Without --reuse, re-running on the same output-dir is refused.
    no_reuse = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    assert no_reuse.returncode == 2

    # With --reuse, existing runs are reused and identical summary is produced.
    reused = subprocess.run(command + ["--reuse"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True)
    reused_payload = json.loads(reused.stdout)
    assert reused_payload["metrics"]["hv_dqce"] == metrics["hv_dqce"]

