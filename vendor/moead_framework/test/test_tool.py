"""Pareto, volume, distances and file export using explicit objective vectors."""

import ast
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from moead_framework.test.fixtures import make_solutions
from moead_framework.tool.mop import (
    get_non_dominated, population_size_without_duplicate, compute_crowding_distance,
)
from moead_framework.tool.result import (
    save_population, save_population_full, compute_hypervolume,
)


class ToolsTest(unittest.TestCase):
    def setUp(self):
        self.population = make_solutions([[1, 4], [2, 2], [4, 1], [3, 4], [4, 4], [2, 2]])
        self.volume_population = make_solutions([
            [1, 2, 2], [2, 1, 2], [2, 2, 1], [2, 2, 2], [1, 2, 2],
        ])
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = str(Path(self.directory.name) / "population.txt")

    def test_get_non_dominated(self):
        with self.assertRaises(TypeError):
            get_non_dominated(population=1)
        with self.assertRaises(TypeError):
            get_non_dominated(population=[1, 0])

    def test_PF(self):
        front = get_non_dominated(population=self.population)
        self.assertEqual({tuple(solution.F) for solution in front},
                         {(1, 4), (2, 2), (4, 1)})
        self.assertEqual(len(front), 3)

    def test_population_size_without_duplicate(self):
        population = make_solutions([[i, 10 - i] for i in range(10)])
        duplicate = deepcopy(population[4])
        duplicate.F = [99, 99]  # Uniqueness here concerns decision vectors.
        population.extend([duplicate, deepcopy(population[2])])
        self.assertEqual(len(population), 12)
        self.assertEqual(population_size_without_duplicate(population), 10)

    def test_population_size_without_duplicate2(self):
        with self.assertRaises(TypeError):
            population_size_without_duplicate(population=1)
        with self.assertRaises(TypeError):
            population_size_without_duplicate(population=[1, 2, 3])

    def test_hypervolume3D(self):
        # Three boxes of volume 2; pairwise and triple intersections have volume 1.
        # Inclusion-exclusion: 2 + 2 + 2 - 1 - 1 - 1 + 1 = 4.
        self.assertEqual(compute_hypervolume(self.volume_population, [3, 3, 3]), 4)

    def test_hypervolume3D_2(self):
        with self.assertRaises(TypeError):
            compute_hypervolume(2, [3, 3, 3])
        with self.assertRaises(TypeError):
            compute_hypervolume([9, 8], [3, 3, 3])
        with self.assertRaises(TypeError):
            compute_hypervolume(self.volume_population, 1)
        with self.assertRaises(TypeError):
            compute_hypervolume(self.volume_population, [3, 3])

    def test_compute_crowding_distance(self):
        population = make_solutions([[1, 10], [2, 6], [4, 3], [8, 1]])
        self.assertTrue(all(solution.distance == 0 for solution in population))
        # Interior gaps: (4-1)+(10-3)=10; (8-2)+(6-1)=11.
        # The upstream routine assigns the largest gap to the boundary points.
        expected = {(1, 10): 11, (2, 6): 10, (4, 3): 11, (8, 1): 11}
        for _ in range(2):  # A second call must reset, not accumulate distances.
            result = compute_crowding_distance(population)
            self.assertEqual({tuple(s.F): s.distance for s in result}, expected)

    def test_compute_crowding_distance2(self):
        with self.assertRaises(TypeError):
            compute_crowding_distance(1)
        with self.assertRaises(TypeError):
            compute_crowding_distance([1, 2, 3])

    def test_save_population(self):
        save_population(self.output, self.population)
        rows = [list(map(float, line.split()))
                for line in Path(self.output).read_text().splitlines()]
        self.assertEqual(rows, [solution.F for solution in self.population])

    def test_save_population2(self):
        with self.assertRaises(TypeError):
            save_population(self.output, 1)
        with self.assertRaises(TypeError):
            save_population(self.output, [1, 2])

    def test_save_population_full(self):
        save_population_full(self.output, self.population)
        rows = Path(self.output).read_text().splitlines()
        self.assertEqual(len(rows), len(self.population))
        for line, solution in zip(rows, self.population):
            first, second, vector = line.split(" ", 2)
            self.assertEqual([float(first), float(second)], solution.F)
            self.assertEqual(ast.literal_eval(vector), solution.decision_vector.tolist())

    def test_save_population_full2(self):
        with self.assertRaises(TypeError):
            save_population_full(self.output, 1)
        with self.assertRaises(TypeError):
            save_population_full(self.output, [1, 2])
