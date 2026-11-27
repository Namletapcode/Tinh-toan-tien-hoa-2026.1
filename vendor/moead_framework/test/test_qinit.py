"""Unit and integration tests for AbstractInitializer, rule-based Q-learning, and Hybrid initialization."""

from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import random
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest

from moead_framework.algorithm.fjsp import create_fjsp_moead
from moead_framework.algorithm.fjsp_qinit import (
    build_initializer,
    create_fjsp_qinit_moead,
    parse_q_init_config,
)
from moead_framework.algorithm.moead_qinit import MoeadQInit
from moead_framework.core.genetic_operator.fjsp import FJSPOperatorConfig
from moead_framework.core.initializer import (
    ACTIONS,
    AbstractInitializer,
    FJSPConstructionEnv,
    HybridQRandomInitializer,
    QInitConfig,
    QLearningInitializer,
)
from moead_framework.core.initializer.qlearning_initializer import spread_indices
from moead_framework.problem.fjsp import (
    FJSPInstance,
    FJSPProblem,
    MachineOption,
    decode,
    load_instance,
    parse_instance,
)
from moead_framework.tool.result import set_seed


ROOT = Path(__file__).resolve().parents[3]
DATA_MAIN = ROOT / "instance" / "main"
PROBLEM_MODULE = "moead_framework.problem.fjsp.problem"

SPEC = spec_from_file_location("baseline_validation_runner_qinit", ROOT / "scripts/run_baseline_validation.py")
RUNNER = module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)

FATTAHI_TEXT = """\
2 2
2 2 1 25 2 37 2 1 32 2 24 0 50 2
2 2 1 45 2 65 2 1 21 2 65 0 60 1
"""


@pytest.fixture
def instance():
    return parse_instance(FATTAHI_TEXT, name="fattahi1")


def _pareto(points):
    unique = set(points)
    return {
        p for p in unique
        if not any(
            all(a <= b for a, b in zip(q, p)) and any(a < b for a, b in zip(q, p))
            for q in unique
        )
    }


# --- 1. Config & Builder validation ---

@pytest.mark.parametrize("kwargs", [
    {"alpha": 0.0},
    {"alpha": 1.1},
    {"alpha": float("nan")},
    {"alpha": True},
    {"gamma": -0.1},
    {"gamma": 1.1},
    {"epsilon_start": -0.1},
    {"epsilon_start": 1.1},
    {"epsilon_start": 0.2, "epsilon_end": 0.5},
    {"random_ratio": -0.1},
    {"random_ratio": 1.2},
    {"progress_bins": 0},
    {"weight_bins": True},
    {"weight_bins": 1.5},
    {"max_resample": -1},
])
def test_q_init_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        QInitConfig(**kwargs)


def test_parse_q_init_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown"):
        parse_q_init_config({"alpha": 0.2, "nonexistent": 123})


def test_build_initializer_returns_expected_types():
    assert isinstance(build_initializer("qlearning"), QLearningInitializer)
    assert isinstance(build_initializer("hybrid_q_random"), HybridQRandomInitializer)
    for mode in ("baseline", "random", "invalid_mode"):
        with pytest.raises(ValueError):
            build_initializer(mode)


# --- 2. Construction environment, TD update and weight awareness ---

def _random_instance(rng):
    jobs = tuple(
        tuple(
            tuple(MachineOption(m, rng.randrange(0, 10)) for m in rng.sample(range(4), rng.randrange(1, 5)))
            for _ in range(rng.randrange(1, 5))
        )
        for _ in range(rng.randrange(1, 6))
    )
    return FJSPInstance(
        4, jobs,
        release_dates=tuple(rng.randrange(0, 5) for _ in jobs),
        due_dates=tuple(rng.randrange(0, 30) for _ in jobs),
        weights=tuple(rng.randrange(1, 4) for _ in jobs),
    )


def test_construction_env_reproduces_decoder_for_every_action_sequence():
    set_seed(21)
    rng = random.Random(21)
    for _ in range(60):
        inst = _random_instance(rng)
        env = FJSPConstructionEnv(inst)
        total_makespan = total_twt = 0
        while not env.finished:
            _, job_rule, machine_rule = ACTIONS[rng.randrange(len(ACTIONS))]
            job = job_rule(env)
            d_makespan, d_twt = env.step(job, machine_rule(env, job))
            assert d_makespan >= 0 and d_twt >= 0
            total_makespan += d_makespan
            total_twt += d_twt
        chromosome = env.chromosome()
        chromosome.validate(inst)
        schedule = decode(inst, chromosome)
        assert schedule.objectives == (env.makespan, env.twt) == (total_makespan, total_twt)


def test_episode_return_equals_weighted_normalized_objective():
    set_seed(3)
    inst = load_instance(DATA_MAIN / "Brandimate1993/Loose/Mk01/Mk01.fjs")
    init = QLearningInitializer(QInitConfig())
    init.reset_table()
    scales = init.scales(inst)
    weight = (0.3, 0.7)
    chromosome, transitions, simulated = init.run_episode(FJSPConstructionEnv(inst), weight, 0.5, scales)
    assert len(transitions) == inst.operation_count
    assert transitions[-1][3] is None and all(t[3] is not None for t in transitions[:-1])
    expected = -(weight[0] * simulated[0] / scales[0] + weight[1] * simulated[1] / scales[1])
    assert sum(t[2] for t in transitions) == pytest.approx(expected)
    assert list(decode(inst, chromosome).objectives) == simulated


def test_q_learning_update_matches_exact_formula_in_reverse_order():
    init = QLearningInitializer(QInitConfig(alpha=0.5, gamma=0.8))
    init.reset_table()
    assert init.q_table.shape[1] == len(ACTIONS) == 8
    init.q_table[2] = [-1.0, -0.2, -3.0, -0.4, -5.0, -5.0, -5.0, -5.0]
    transitions = [(0, 1, -0.5, 2), (2, 1, -0.25, None)]
    assert init.learn(transitions) == 2
    # Terminal step first: Q[2,1] = -0.2 + 0.5 * (-0.25 - (-0.2)) = -0.225
    assert init.q_table[2, 1] == pytest.approx(-0.225)
    # Then Q[0,1] = 0 + 0.5 * (-0.5 + 0.8 * max(Q[2]) - 0), max(Q[2]) = -0.225
    assert init.q_table[0, 1] == pytest.approx(0.5 * (-0.5 + 0.8 * -0.225))
    assert np.count_nonzero(init.q_table[0]) == 1


def test_gamma_changes_the_learned_population():
    inst = load_instance(DATA_MAIN / "Brandimate1993/Moderate/Mk05/Mk05.fjs")
    populations = {}
    for gamma in (0.0, 0.95):
        runs = []
        for seed in range(3):
            set_seed(seed)
            pop = QLearningInitializer(QInitConfig(gamma=gamma)).initialize(
                FJSPProblem(inst), 13, None, lambda s: None)
            runs.append([tuple(s.decision_vector) for s in pop])
        populations[gamma] = runs
    assert populations[0.0] != populations[0.95]


class _TaggingInitializer(AbstractInitializer):
    weight_aware = True

    def initialize(self, problem, population_size, weights, on_evaluated):
        population = []
        for weight in weights:
            solution = problem.generate_random_solution()
            solution.target_weight = tuple(weight)
            on_evaluated(solution)
            population.append(solution)
        return population


def test_weight_aware_population_follows_the_weight_shuffle(instance):
    set_seed(5)
    algo = create_fjsp_qinit_moead(
        FJSPProblem(instance), max_evaluation=13, partitions=12, number_of_weight_neighborhood=10,
        initializer=_TaggingInitializer(),
    )
    lattice = [(i / 12, 1 - i / 12) for i in range(13)]
    assert [tuple(w) for w in algo.weights] != lattice  # MOEA/D still shuffled its weights
    for weight, solution in zip(algo.weights, algo.population):
        assert solution.target_weight == tuple(float(v) for v in weight)


def test_qlearning_initializer_places_q_solutions_on_spread_subproblems(instance):
    set_seed(8)
    init = HybridQRandomInitializer(QInitConfig(random_ratio=0.3))
    pop = init.initialize(FJSPProblem(instance), 13, None, lambda s: None)
    stats = init.last_stats
    assert len(pop) == 13 and all(s is not None for s in pop)
    assert (stats["n_random"], stats["n_qlearning"]) == (4, 9)
    assert sum(stats["action_counts"].values()) == stats["td_updates"] == 9 * instance.operation_count
    assert len(stats["epsilon_schedule"]) == 9
    assert stats["epsilon_schedule"][0] == 0.5 and stats["epsilon_schedule"][-1] == 0.05
    unique = len({tuple(s.decision_vector) for s in pop})
    assert unique >= 13 - stats["accepted_duplicates"]


@pytest.mark.parametrize("ratio, expected_q_slots", [
    (0.2, [0, 1, 3, 4, 5, 7, 8, 9, 11, 12]),  # default: 3 random + 10 Q-learning, both ends included
    (0.95, [6]),                              # a single Q-learning solution sits on the middle subproblem
])
def test_hybrid_q_slots_match_documented_placement(ratio, expected_q_slots):
    n_rand, n_q = HybridQRandomInitializer(QInitConfig(random_ratio=ratio)).split_counts(13)
    assert n_q == len(expected_q_slots) and n_rand + n_q == 13
    assert spread_indices(n_q, 13) == expected_q_slots


def test_qlearning_on_single_job_single_machine():
    inst = parse_instance("1 1\n1 1 1 0 0 100 1\n")
    problem = FJSPProblem(inst)
    algo = create_fjsp_qinit_moead(
        problem, max_evaluation=9, partitions=1, number_of_weight_neighborhood=2,
        initializer=HybridQRandomInitializer(),
    )
    archive = algo.run()
    assert problem.evaluation_count == algo.current_eval == 9
    assert len(archive) == 1 and archive[0].F == [0, 0]
    problem.validate_solution(archive[0])


# --- 3. Integration of the Q-learning initializers with MoeadQInit ---

class _BaselineInitializer(AbstractInitializer):
    """Test-only: the baseline random initialization routed through the pluggable interface."""

    def initialize(self, problem, population_size, weights, on_evaluated):
        population = []
        for _ in range(population_size):
            solution = problem.generate_random_solution()
            on_evaluated(solution)
            population.append(solution)
        return population


def test_moead_qinit_framework_leaves_baseline_behaviour_unchanged(instance):
    """Only the initializer differs: with baseline initialization, MoeadQInit == Moead bit for bit."""
    set_seed(13)
    p_base = FJSPProblem(instance)
    a_base = create_fjsp_moead(p_base, max_evaluation=30, partitions=4, number_of_weight_neighborhood=3)
    ep_base = a_base.run()

    set_seed(13)
    p_rand = FJSPProblem(instance)
    a_rand = create_fjsp_qinit_moead(
        p_rand, max_evaluation=30, partitions=4, number_of_weight_neighborhood=3,
        initializer=_BaselineInitializer(),
    )
    ep_rand = a_rand.run()

    assert [s.F for s in ep_base] == [s.F for s in ep_rand]
    for sb, sr in zip(a_base.population, a_rand.population):
        np.testing.assert_array_equal(sb.decision_vector, sr.decision_vector)


@pytest.mark.parametrize("mode", ["qlearning", "hybrid_q_random"])
@pytest.mark.parametrize("budget", [13, 14, 40])
def test_moead_qinit_respects_nfe_archive_and_schedule_validity(instance, mode, budget):
    set_seed(7)
    problem = FJSPProblem(instance)
    seen = []
    evaluate = problem.evaluate

    def record(x):
        sol = evaluate(x)
        problem.validate_solution(sol)
        seen.append((tuple(sol.decision_vector), tuple(sol.F)))
        return sol

    with patch.object(problem, "evaluate", side_effect=record) as calls, \
            patch(PROBLEM_MODULE + ".decode", wraps=decode) as decoding:
        algo = create_fjsp_qinit_moead(
            problem, max_evaluation=budget, partitions=12, number_of_weight_neighborhood=10,
            initializer=mode,
        )
        assert algo.current_eval == 13  # initialization spends exactly one NFE per subproblem
        initial_ep = deepcopy(algo.ep)
        archive = algo.run()
        assert calls.call_count == decoding.call_count == problem.evaluation_count == budget

    assert isinstance(algo, MoeadQInit)
    assert algo.current_eval == budget
    assert {tuple(s.F) for s in archive} == _pareto(pt for _, pt in seen)
    assert len(archive) == len({tuple(s.F) for s in archive})
    for sol in archive + algo.population + initial_ep:
        problem.validate_solution(sol)

    if mode == "hybrid_q_random":
        stats = algo.initializer.last_stats
        assert stats["n_random"] + stats["n_qlearning"] == 13
        assert stats["n_random"] > 0 and stats["n_qlearning"] > 0
        assert stats["episodes"] == stats["n_qlearning"]
        assert stats["td_updates"] == stats["n_qlearning"] * problem.instance.operation_count
        assert np.any(algo.initializer.q_table < 0)

    # Calling run() after budget exhaustion must spend 0 NFE.
    algo.run()
    assert problem.evaluation_count == budget


@pytest.mark.parametrize("mode", ["qlearning", "hybrid_q_random"])
def test_qinit_is_strictly_reproducible_across_same_seed(instance, mode):
    outcomes = []
    for _ in range(2):
        set_seed(42)
        problem = FJSPProblem(instance)
        algo = create_fjsp_qinit_moead(
            problem, max_evaluation=50, partitions=12, number_of_weight_neighborhood=10,
            initializer=mode, operator_config=FJSPOperatorConfig(0.8, 0.05),
        )
        archive = algo.run()
        outcomes.append((
            sorted((tuple(s.F), tuple(s.decision_vector)) for s in archive),
            [tuple(s.decision_vector) for s in algo.population],
        ))
    assert outcomes[0] == outcomes[1]


def test_invalid_initializer_or_budget_fails_before_any_nfe(instance):
    problem = FJSPProblem(instance)
    with pytest.raises(TypeError):
        create_fjsp_qinit_moead(problem, initializer=123)
    with pytest.raises(ValueError):
        create_fjsp_qinit_moead(problem, max_evaluation=5, partitions=12, number_of_weight_neighborhood=10)
    assert problem.evaluation_count == 0


# --- 4. Experiment runner, protocol validation, and CLI ---

def test_qinit_validation_protocol_config_is_valid():
    raw = json.loads((ROOT / "scripts/configs/qinit_validation.json").read_text(encoding="utf-8"))
    protocol = RUNNER.resolve_protocol_instances(raw, root=ROOT)
    RUNNER.validate_protocol(protocol)
    assert len(protocol["instances"]) == 84
    assert protocol["algorithm"]["initializer"] == "hybrid_q_random"
    assert protocol["checkpoints"] == [13, 250, 500, 1000]


def test_validation_case_and_study_work_end_to_end_with_hybrid_qinit(tmp_path):
    protocol = {
        "schema_version": 1,
        "study": "test_qinit",
        "study_date": "2026-11-25",
        "algorithm": {
            "partitions": 12,
            "neighbors": 10,
            "evaluations": 40,
            "crossover": 0.8,
            "mutation": 0.05,
            "initializer": "hybrid_q_random",
            "q_init": {
                "alpha": 0.1,
                "gamma": 0.95,
                "epsilon_start": 0.5,
                "epsilon_end": 0.05,
                "random_ratio": 0.2,
                "progress_bins": 4,
                "weight_bins": 3,
                "max_resample": 5,
            },
        },
        "seeds": [0, 1],
        "repeat_seed": 0,
        "checkpoints": [13, 25, 40],
        "hv_reference": [1.1, 1.1],
        "instances": [{
            "id": "mk01_loose",
            "family": "Brandimate1993",
            "path": "instance/main/Brandimate1993/Loose/Mk01/Mk01.fjs",
        }],
    }
    config_path = tmp_path / "qinit_protocol.json"
    config_path.write_text(json.dumps(protocol), encoding="utf-8")
    output = tmp_path / "study_qinit"
    report = RUNNER.run_study(config_path, output, make_plot=False)
    assert report["primary_runs"] == report["valid_runs"] == 2
    assert report["replay_runs"] == 1
    assert report["primary_nfe"] == 80


def test_cli_run_fjsp_supports_hybrid_q_random(tmp_path):
    output = tmp_path / "qinit_run.json"
    command = [
        sys.executable, str(ROOT / "scripts/run_fjsp.py"),
        "--instance", str(DATA_MAIN / "Brandimate1993/Loose/Mk01/Mk01.fjs"),
        "--initializer", "hybrid_q_random",
        "--seed", "1",
        "--evaluations", "40",
        "--output", str(output),
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True)
    summary = json.loads(completed.stdout)
    assert summary["algorithm"] == "MoeadQInit"
    assert summary["initializer"] == "hybrid_q_random"
    assert summary["actual_evaluations"] == 40
    assert summary["all_schedules_valid"] is True
    assert summary["initializer_stats"]["n_random"] + summary["initializer_stats"]["n_qlearning"] == 13
    assert output.is_file()

    baseline_cmd = [
        sys.executable, str(ROOT / "scripts/run_fjsp.py"),
        "--instance", str(DATA_MAIN / "Brandimate1993/Loose/Mk01/Mk01.fjs"),
        "--seed", "1",
        "--evaluations", "40",
    ]
    baseline_done = subprocess.run(baseline_cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True)
    baseline_summary = json.loads(baseline_done.stdout)
    assert baseline_summary["algorithm"] == "Moead"
    assert "initializer" not in baseline_summary
    assert "initializer_stats" not in baseline_summary
    assert "q_init" not in baseline_summary


def test_cli_run_fjsp_rejects_q_flags_without_q_initializer():
    command = [
        sys.executable, str(ROOT / "scripts/run_fjsp.py"),
        "--instance", str(DATA_MAIN / "Brandimate1993/Loose/Mk01/Mk01.fjs"),
        "--q-alpha", "0.2", "--evaluations", "20",
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    assert completed.returncode == 2
    assert "--q-alpha" in completed.stderr and "--initializer" in completed.stderr
    assert completed.stdout == ""


@pytest.mark.parametrize("mode", ["baseline", "qlearning", "hybrid_q_random"])
def test_pilot6_configs_are_valid_and_share_one_protocol(mode):
    raw = json.loads((ROOT / f"scripts/configs/qinit_pilot6_{mode}.json").read_text(encoding="utf-8"))
    RUNNER.validate_protocol(RUNNER.resolve_protocol_instances(raw, root=ROOT))
    assert raw["algorithm"].get("initializer", "baseline") == mode
    assert [entry["id"] for entry in raw["instances"]] == [
        "mk02_loose", "mk06_moderate", "mk09_tight", "03a_loose", "08a_moderate", "12a_tight"]
    reference = json.loads((ROOT / "scripts/configs/baseline_validation.json").read_text(encoding="utf-8"))
    for key in ("seeds", "repeat_seed", "checkpoints", "hv_reference"):
        assert raw[key] == reference[key]
    assert {k: v for k, v in raw["algorithm"].items() if k != "initializer"} == reference["algorithm"]


def _tiny_study(tmp_path, name, evaluations, initializer=None):
    algorithm = {"partitions": 12, "neighbors": 10, "evaluations": evaluations, "crossover": 0.8, "mutation": 0.05}
    if initializer is not None:
        algorithm["initializer"] = initializer
    protocol = {
        "schema_version": 1, "study": name, "study_date": "2026-11-25", "algorithm": algorithm,
        "seeds": [0, 1], "repeat_seed": None, "checkpoints": [13, evaluations], "hv_reference": [1.1, 1.1],
        "instances": [{"id": "mk01_loose", "family": "Brandimate1993",
                       "path": "instance/main/Brandimate1993/Loose/Mk01/Mk01.fjs"}],
    }
    config_path = tmp_path / f"{name}.json"
    config_path.write_text(json.dumps(protocol), encoding="utf-8")
    RUNNER.run_study(config_path, tmp_path / name, make_plot=False)
    return tmp_path / name


def test_compare_initializers_requires_matching_protocol(tmp_path):
    spec = spec_from_file_location("compare_initializers_qinit", ROOT / "scripts/compare_initializers.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    base = _tiny_study(tmp_path, "base", 30)
    hybrid = _tiny_study(tmp_path, "hybrid", 30, initializer="hybrid_q_random")
    longer = _tiny_study(tmp_path, "longer", 40, initializer="hybrid_q_random")

    with patch.object(module, "plot"):
        report = module.compare({"base": base, "hybrid": hybrid}, "base", tmp_path / "ok")
    assert sum(report["tally"]["hybrid"]["hv"].values()) == 1
    assert "| HV@13 |" in (tmp_path / "ok" / "comparison.md").read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="different protocol.*evaluations, checkpoints"):
        module.compare({"base": base, "longer": longer}, "base", tmp_path / "bad")
