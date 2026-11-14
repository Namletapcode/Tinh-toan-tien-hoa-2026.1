"""Project regression tests for the corrected MOEA/D core (step 3)."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from moead_framework.aggregation import Tchebycheff
from moead_framework.algorithm import Moead, MoeadDeltaNr, MoeadDRA, MoeadSPSRandom
from moead_framework.core.genetic_operator import GeneticOperator
from moead_framework.core.offspring_generator.abstract_mating import OffspringGenerator
from moead_framework.core.termination_criteria.abstract_termination_criteria import TerminationCriteria
from moead_framework.problem import Problem
from moead_framework.solution import OneDimensionSolution
from moead_framework.tool.mop import generate_weight_vectors


class CountingProblem(Problem):
    """Identity objectives expose every actual evaluation and its full vector."""

    def __init__(self, initial=None, child=None):
        self.initial = initial if initial is not None else [[10, 30, 20], [20, 10, 30], [30, 20, 10]]
        super().__init__(len(self.initial[0]))
        self.child = child if child is not None else [9] * self.number_of_objective
        self.generated = 0
        self.evaluated = []
        self.objective_calls = 0

    def f(self, function_id, decision_vector):
        self.objective_calls += 1
        return float(decision_vector[function_id])

    def evaluate(self, x):
        solution = super().evaluate(x)
        self.evaluated.append(tuple(solution.F))
        return solution

    def generate_random_solution(self):
        point = self.initial[self.generated % len(self.initial)]
        self.generated += 1
        return self.evaluate(point)


class FixedOffspring(OffspringGenerator):
    def run(self, population_indexes):
        return self.algorithm.problem.evaluate(self.algorithm.problem.child)


class InPlaceOperator(GeneticOperator):
    def run(self):
        self.solutions[0][0] = 999
        return self.solutions[0]


class StopAfterOneChild(TerminationCriteria):
    def test(self):
        return self.algorithm.current_eval < self.algorithm.number_of_weight + 1


@pytest.fixture
def build(tmp_path):
    def make(problem=None, weights=None, budget=7, neighborhood=2, **kwargs):
        problem = problem if problem is not None else CountingProblem()
        weights = weights if weights is not None else np.eye(problem.number_of_objective)
        weight_file = tmp_path / "weights.ws"
        weight_file.write_text("\n".join(" ".join(map(str, row)) for row in weights), encoding="utf-8")
        options = dict(problem=problem, max_evaluation=budget,
                       number_of_weight_neighborhood=neighborhood,
                       aggregation_function=Tchebycheff, weight_file=str(weight_file),
                       offspring_generator=FixedOffspring)
        options.update(kwargs)
        return Moead(**options)
    return make


def pareto_oracle(points):
    """Independent definition: a dominating point is <= everywhere and < somewhere."""
    unique = set(points)
    return {p for p in unique if not any(
        all(a <= b for a, b in zip(q, p)) and any(a < b for a, b in zip(q, p))
        for q in unique
    )}


@pytest.mark.parametrize("budget", [3, 4, 5, 6, 7, 8, 10])
def test_budget_counts_real_evaluations_including_initial_population(build, budget):
    problem = CountingProblem()
    algorithm = build(problem=problem, budget=budget)
    assert algorithm.current_eval == problem.generated == len(problem.evaluated) == 3
    archive = algorithm.run()
    assert algorithm.current_eval == len(problem.evaluated) == budget
    assert problem.objective_calls == 3 * budget
    assert {tuple(s.F) for s in archive} == pareto_oracle(problem.evaluated)
    assert len(archive) == len({tuple(s.F) for s in archive})
    algorithm.run()
    assert len(problem.evaluated) == budget  # Re-entering run must not spend more NFE.
    assert not algorithm.termination_criteria.test()
    algorithm.optimize_sub_problem(0)
    assert len(problem.evaluated) == budget


@pytest.mark.parametrize("budget,error", [(0, ValueError), (-1, ValueError), (2, ValueError),
                                         (1.5, TypeError), (True, TypeError)])
def test_invalid_budget_fails_before_any_evaluation(build, budget, error):
    problem = CountingProblem()
    with pytest.raises(error):
        build(problem=problem, budget=budget)
    assert problem.evaluated == []


@pytest.mark.parametrize("size,error", [(0, ValueError), (-1, ValueError), (4, ValueError),
                                       (1.5, TypeError), (True, TypeError)])
def test_invalid_neighborhood_fails_before_any_evaluation(build, size, error):
    problem = CountingProblem()
    with pytest.raises(error):
        build(problem=problem, neighborhood=size)
    assert problem.evaluated == []


@pytest.mark.parametrize("weights", [[], [[1, 0]], [[1, -1, 1]], [[0, 0, 0]],
                                    [[float("nan"), 0, 1]], [[float("inf"), 0, 1]]])
def test_invalid_weights_fail_before_any_evaluation(build, weights):
    problem = CountingProblem()
    with pytest.raises(ValueError):
        build(problem=problem, weights=weights, neighborhood=1)
    assert problem.evaluated == []


def test_ideal_point_is_observed_minimum_for_positive_objectives(build):
    algorithm = build()
    np.testing.assert_array_equal(algorithm.z, [10, 10, 10])
    algorithm.update_z(OneDimensionSolution(np.zeros(3), [8, 40, 6]))
    np.testing.assert_array_equal(algorithm.z, [8, 10, 6])
    np.testing.assert_array_equal(algorithm.population[0].F, [10, 30, 20])


def test_custom_termination_is_checked_inside_each_generation(build):
    problem = CountingProblem()
    algorithm = build(problem=problem, budget=20, termination_criteria=StopAfterOneChild)
    algorithm.run()
    assert algorithm.current_eval == len(problem.evaluated) == 4


def test_archive_includes_nondominated_child_even_without_replacement(build):
    problem = CountingProblem(initial=[[1, 10, 10], [1, 11, 9], [1, 9, 11]], child=[2, 8, 8])
    algorithm = build(problem=problem, weights=[[1, 0, 0]] * 3, budget=4)
    original = [tuple(s.F) for s in algorithm.population]
    archive = algorithm.run()
    assert [tuple(s.F) for s in algorithm.population] == original
    assert (2, 8, 8) in {tuple(s.F) for s in archive}
    assert {tuple(s.F) for s in archive} == pareto_oracle(problem.evaluated)


def test_archive_filters_dominated_points_and_objective_duplicates(build):
    algorithm = build()
    algorithm.update_archive(OneDimensionSolution([0, 0, 0], [9, 9, 9]))
    algorithm.update_archive(OneDimensionSolution([1, 1, 1], [9, 9, 9]))
    algorithm.update_archive(OneDimensionSolution([2, 2, 2], [100, 100, 100]))
    assert [s.F for s in algorithm.ep] == [[9, 9, 9]]


def test_archive_owns_copies_of_initial_solutions_and_children(build):
    algorithm = build()
    stored = [deepcopy(s) for s in algorithm.ep]
    algorithm.population[0].F[0] = -100
    algorithm.population[0].decision_vector[0] = -100
    assert [s.F for s in algorithm.ep] == [s.F for s in stored]
    for actual, expected in zip(algorithm.ep, stored):
        np.testing.assert_array_equal(actual.decision_vector, expected.decision_vector)
    child = OneDimensionSolution(np.array([9, 9, 9]), [9, 9, 9])
    algorithm.update_archive(child)
    child.F[0] = -200
    child.decision_vector[0] = -200
    assert algorithm.ep[0].F == [9, 9, 9]
    np.testing.assert_array_equal(algorithm.ep[0].decision_vector, [9, 9, 9])


def test_replacement_uses_each_neighbors_weight_and_independent_copies(build):
    algorithm = build()
    algorithm.weights = list(np.eye(3))
    algorithm.z = np.zeros(3)
    algorithm.population = [OneDimensionSolution(np.array(f, dtype=float), list(f)) for f in
                            [[1, 100, 100], [100, 1, 100], [100, 100, 1]]]
    algorithm.b[1] = [2, 0]
    child = OneDimensionSolution(np.array([0.5, 200, 0.5]), [0.5, 200, 0.5])
    algorithm.update_solutions(child, algorithm.aggregation_function, sub_problem=1)
    assert algorithm.population[0].F == algorithm.population[2].F == child.F
    assert algorithm.population[1].F == [100, 1, 100]
    algorithm.population[0].F[0] = -10
    algorithm.population[0].decision_vector[0] = -10
    assert algorithm.population[2].F == child.F == [0.5, 200, 0.5]
    np.testing.assert_array_equal(algorithm.population[2].decision_vector, child.decision_vector)


def test_in_place_operator_cannot_modify_selected_parents(build, monkeypatch):
    algorithm = build(genetic_operator=InPlaceOperator, offspring_generator=None)
    monkeypatch.setattr(algorithm.parent_selector, "select", lambda indexes: algorithm.population[:2])
    original = [deepcopy(s) for s in algorithm.population]
    child = algorithm.generate_offspring([0, 1])
    assert child.F[0] == 999
    for actual, expected in zip(algorithm.population, original):
        assert actual.F == expected.F
        np.testing.assert_array_equal(actual.decision_vector, expected.decision_vector)


def test_neighborhood_uses_all_three_weight_coordinates(build):
    weights = [[0.25, 0.25, 0.5], [0.5, 0, 0.5], [0.45, 0.45, 0.1]]
    algorithm = build(weights=weights)
    algorithm.weights = [np.array(w) for w in weights]
    neighbors = algorithm.generate_closest_weight_vectors()
    assert set(neighbors[0]) == {0, 1}  # Two-coordinate distance would choose {0, 2}.
    assert len({id(row) for row in neighbors}) == len(weights)
    for row in neighbors:
        assert len(row) == len(set(row)) == 2


def test_weight_reader_supports_tabs_repeated_spaces_and_blank_lines(tmp_path):
    path = tmp_path / "weights.ws"
    path.write_text("1\t0\t0\n\n0  1  0\n0 0 1\n", encoding="utf-8")
    np.testing.assert_array_equal(generate_weight_vectors(str(path), shuffle=False), np.eye(3))


@pytest.mark.parametrize("variant", [MoeadDeltaNr, MoeadDRA, MoeadSPSRandom])
@pytest.mark.parametrize("budget", [11, 14])
def test_shared_fixes_preserve_exact_budget_in_existing_variants(variant, budget):
    problem = CountingProblem(initial=[[10, 30], [30, 10]], child=[9, 9])
    options = dict(problem=problem, max_evaluation=budget, aggregation_function=Tchebycheff,
                   number_of_weight_neighborhood=2,
                   weight_file=str(Path(__file__).parent / "data/weights/SOBOL-2objs-10wei.ws"))
    if variant is MoeadSPSRandom:
        options["number_of_subproblem_to_visit"] = 1
    else:
        options.update(number_of_objective=2, delta=0.9, number_of_replacement=2)
    algorithm = variant(**options)
    # Replace only the offspring generator, leaving the variant's run loop and selector intact.
    algorithm.offspring_generator = FixedOffspring(algorithm_instance=algorithm)
    assert algorithm.current_eval == len(problem.evaluated) == 10
    archive = algorithm.run()
    assert algorithm.current_eval == len(problem.evaluated) == budget
    assert {tuple(s.F) for s in archive} == pareto_oracle(problem.evaluated)
