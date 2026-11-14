import unittest

from moead_framework.core.genetic_operator import Crossover, BinaryMutation
from moead_framework.tool.result import set_seed


class GeneticOperatorsTest(unittest.TestCase):
    """Test genetic operators."""

    def setUp(self):
        """Init"""
        set_seed(1)

    def test_combinatorial_binary_mutation(self):
        """Test combinatorial binary mutation"""
        array_solution = [1, 0, 0, 1, 1, 0]
        new_solution = BinaryMutation(solutions=[array_solution]).run()

        self.assertEqual(new_solution, [0, 0, 0, 1, 1, 0])

    def test_combinatorial_crossover(self):
        """Test combinatorial crossover multi-points"""
        array_solution1 = [1, 2, 3, 4, 5]
        array_solution2 = [6, 7, 8, 9, 10]
        new_solution_one_point = Crossover(solutions=[array_solution1, array_solution2],
                                           crossover_points=1).run().tolist()
        new_solution_three_point = Crossover(solutions=[array_solution1, array_solution2],
                                             crossover_points=3).run().tolist()

        self.assertEqual(new_solution_one_point, [1, 2, 8, 9, 10])
        self.assertEqual(new_solution_three_point, [1, 7, 8, 4, 10])

    def test_binary_mutation_example1(self):
        array_solution = ["a", 0]
        with self.assertRaises(AttributeError):
            new_solution = BinaryMutation(solutions=[array_solution]).run()

    # Tried to pass a single array
    # Result: TypeError: object of type 'int' has no len()
    def test_binary_mutation_example3(self):
        array_solution = [0, 1]
        with self.assertRaises(AttributeError):
            new_solution = BinaryMutation(solutions=array_solution).run()

    # Tried to pass an integer
    # Result: TypeError: 'int' object is not subscriptable
    def test_binary_mutation_example4(self):
        array_solution = [0, 1]
        with self.assertRaises(AttributeError):
            new_solution = BinaryMutation(solutions=array_solution).run()

    # Tried to pass empty arrays
    # Result: ZeroDivisionError: division by zero
    def test_binary_mutation_example5(self):
        array_solution = [0, 1]
        with self.assertRaises(AttributeError):
            new_solution = BinaryMutation(solutions=[[]]).run()

    # Tried to ask more crossover points than the length
    # Result: Infinite loop
    def test_crossover_example(self):
        array_solution1 = [1, 2, 3, 4, 5]
        array_solution2 = [6, 7, 8, 9, 10]

        with self.assertRaises(ValueError):
            new_solution_one_point = Crossover(solutions=[array_solution1, array_solution2],
                                               crossover_points=10).run().tolist()

    # Tried to ask a zero crossover points than the length
    # Result: IndexError: list index out of range
    def test_crossover_example1(self):
        array_solution1 = [1, 2, 3, 4, 5]
        array_solution2 = [6, 7, 8, 9, 10]

        with self.assertRaises(ValueError):
            new_solution_one_point = Crossover(solutions=[array_solution1, array_solution2],
                                               crossover_points=0).run().tolist()

    def test_number_of_solution_is_correct(self):
        array_solution1 = [1, 2, 3, 4, 5]
        array_solution2 = [6, 7, 8, 9, 10]
        crossover = Crossover(solutions=[array_solution1, array_solution2], crossover_points=1)

        crossover.number_of_solution_is_correct(n=2)

        with self.assertRaises(ValueError):
            crossover.number_of_solution_is_correct(n=3)

