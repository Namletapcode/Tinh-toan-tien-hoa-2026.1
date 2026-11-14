import unittest
from unittest.mock import patch
import os
from moead_framework.aggregation.tchebycheff import Tchebycheff
from moead_framework.algorithm import Moead, MoeadDeltaNr, MoeadSPSRandom, MoeadDRA
from moead_framework.core.offspring_generator.abstract_mating import OffspringGenerator
from moead_framework.test.fixtures import BinaryTestProblem
from moead_framework.tool.result import compute_hypervolume, set_seed


class AlgorithmsTest(unittest.TestCase):
    """Test implemented algorithms."""

    def setUp(self):
        """Init"""
        set_seed(1)
        self.number_of_evaluations = 100

        project_path = os.path.dirname(os.path.abspath(__file__))
        self.problem = BinaryTestProblem()

        self.number_of_objective = self.problem.number_of_objective
        self.number_of_weight = 10
        self.number_of_weight_neighborhood = 2
        self.number_of_crossover_points = 4
        self.weight_file = project_path + "/data/weights/SOBOL-" \
                           + str(self.number_of_objective) \
                           + "objs-" + str(self.number_of_weight) \
                           + "wei.ws"
        self.weight_file50 = project_path + "/data/weights/SOBOL-" \
                           + str(self.number_of_objective) \
                           + "objs-" + str(50) \
                           + "wei.ws"

    def assert_valid_run(self, algorithm):
        """Validate the corrected contract, rather than the old extra-budget HV.

        Upstream golden fronts used a different NFE convention and archive
        policy. Compare against every actual evaluation using independent
        Pareto dominance, and require that initial archive coverage is retained.
        """
        seen = [tuple(solution.F) for solution in algorithm.population]
        initial_hv = compute_hypervolume(algorithm.ep, [25, 25])
        evaluate = algorithm.problem.evaluate

        def record(x):
            solution = evaluate(x)
            seen.append(tuple(solution.F))
            return solution

        with patch.object(algorithm.problem, 'evaluate', side_effect=record) as calls:
            archive = algorithm.run()

        self.assertEqual(algorithm.current_eval, algorithm.max_evaluation)
        self.assertEqual(len(seen), algorithm.max_evaluation)
        self.assertEqual(calls.call_count, algorithm.max_evaluation - algorithm.number_of_weight)
        unique = set(seen)
        expected = {p for p in unique if not any(
            all(a <= b for a, b in zip(q, p)) and any(a < b for a, b in zip(q, p))
            for q in unique
        )}
        self.assertEqual({tuple(solution.F) for solution in archive}, expected)
        self.assertEqual(len(archive), len(expected))
        self.assertGreaterEqual(compute_hypervolume(archive, [25, 25]), initial_hv)

    def test_moead(self):
        """Test MOEA/D algorithm"""

        moead = Moead(problem=self.problem,
                      max_evaluation=self.number_of_evaluations,
                      number_of_weight=self.number_of_weight,
                      aggregation_function=Tchebycheff,
                      number_of_weight_neighborhood=self.number_of_weight_neighborhood,
                      number_of_crossover_points=self.number_of_crossover_points,
                      weight_file=self.weight_file,
                      )

        self.assert_valid_run(moead)

    def test_moead_delta_nr(self):
        """Test MOEA/D algorithm with parameters delta & nr"""
        delta = 0.9
        nr = 2

        moead = MoeadDeltaNr(problem=self.problem,
                             max_evaluation=self.number_of_evaluations,
                             number_of_objective=self.number_of_objective,
                             number_of_weight_neighborhood=self.number_of_weight_neighborhood,
                             number_of_crossover_points=self.number_of_crossover_points,
                             aggregation_function=Tchebycheff,
                             weight_file=self.weight_file,
                             delta=delta,
                             number_of_replacement=nr
                             )

        self.assert_valid_run(moead)

    def test_moead_sps_random(self):
        """Test MOEA/D algorithm with the random sps strategy"""
        number_of_subproblem = 2
        moead = MoeadSPSRandom(problem=self.problem,
                               max_evaluation=self.number_of_evaluations,
                               number_of_objective=self.number_of_objective,
                               aggregation_function=Tchebycheff,
                               number_of_weight_neighborhood=self.number_of_weight_neighborhood,
                               number_of_crossover_points=self.number_of_crossover_points,
                               number_of_subproblem_to_visit=number_of_subproblem,
                               weight_file=self.weight_file,
                               )

        self.assert_valid_run(moead)

    def test_moead_dra(self):
        """Test MOEA/D algorithm with the random sps strategy"""
        delta = 0.9
        nr = 2

        moead = MoeadDRA(problem=self.problem,
                         max_evaluation=500,
                         number_of_objective=self.number_of_objective,
                         number_of_weight_neighborhood=self.number_of_weight_neighborhood,
                         aggregation_function=Tchebycheff,
                         number_of_crossover_points=self.number_of_crossover_points,
                         weight_file=self.weight_file50,
                         delta=delta,
                         number_of_replacement=nr
                         )

        self.assert_valid_run(moead)

    def test_sad_path_moead(self):
        """Test MOEA/D algorithm"""

        with self.assertRaises(TypeError):
            moead = Moead(problem="bad_parameter",
                          max_evaluation=self.number_of_evaluations,
                          number_of_weight=self.number_of_weight,
                          aggregation_function=Tchebycheff,
                          number_of_weight_neighborhood=self.number_of_weight_neighborhood,
                          number_of_crossover_points=self.number_of_crossover_points,
                          weight_file=self.weight_file,
                          )
        with self.assertRaises(TypeError):
            moead = Moead(problem=self.problem,
                          max_evaluation="bad_parameter",
                          number_of_weight=self.number_of_weight,
                          aggregation_function=Tchebycheff,
                          number_of_weight_neighborhood=self.number_of_weight_neighborhood,
                          number_of_crossover_points=self.number_of_crossover_points,
                          weight_file=self.weight_file,
                          )
        with self.assertRaises(TypeError):
            moead = Moead(problem=self.problem,
                          max_evaluation=self.number_of_evaluations,
                          number_of_weight=self.number_of_weight,
                          aggregation_function="bad_parameter",
                          number_of_weight_neighborhood=self.number_of_weight_neighborhood,
                          number_of_crossover_points=self.number_of_crossover_points,
                          weight_file=self.weight_file,
                          )
        with self.assertRaises(TypeError):
            moead = Moead(problem=self.problem,
                          max_evaluation=self.number_of_evaluations,
                          number_of_weight=self.number_of_weight,
                          aggregation_function=Tchebycheff,
                          number_of_weight_neighborhood="bad_parameter",
                          number_of_crossover_points=self.number_of_crossover_points,
                          weight_file=self.weight_file,
                          )
        with self.assertRaises(TypeError):
            moead = Moead(problem=self.problem,
                          max_evaluation=self.number_of_evaluations,
                          number_of_weight=self.number_of_weight,
                          aggregation_function=Tchebycheff,
                          number_of_weight_neighborhood=self.number_of_weight_neighborhood,
                          number_of_crossover_points="bad_parameter",
                          weight_file=self.weight_file,
                          )
        with self.assertRaises(TypeError):
            moead = Moead(problem=self.problem,
                          max_evaluation=self.number_of_evaluations,
                          number_of_weight=self.number_of_weight,
                          aggregation_function=Tchebycheff,
                          number_of_weight_neighborhood=self.number_of_weight_neighborhood,
                          number_of_crossover_points=self.number_of_crossover_points,
                          weight_file=10,
                          )
        with self.assertRaises(FileNotFoundError):
            moead = Moead(problem=self.problem,
                          max_evaluation=self.number_of_evaluations,
                          number_of_weight=self.number_of_weight,
                          aggregation_function=Tchebycheff,
                          number_of_weight_neighborhood=self.number_of_weight_neighborhood,
                          number_of_crossover_points=self.number_of_crossover_points,
                          weight_file="false_path",
                          )
