"""MOEA/D with several FJSP solutions per weight, instead of one.

The inherited loop still evaluates exactly one child per visit. Only storage,
initialization, parent selection and neighborhood replacement change.
"""

from copy import deepcopy
from numbers import Integral

import numpy as np

from moead_framework.algorithm.moead import Moead
from moead_framework.core.initializer import AbstractInitializer
from moead_framework.core.parent_selector.micro_gwo_selector import MicroGWOParentSelector


class MoeadMicroGWO(Moead):
    def __init__(self, *, micro_population_size=4, parent_selection="grey_wolf",
                 initializer=None, **kwargs):
        if (isinstance(micro_population_size, bool)
                or not isinstance(micro_population_size, Integral)
                or micro_population_size < 4):
            raise ValueError("micro_population_size must be an integer >= 4")
        if parent_selection not in ("random", "grey_wolf"):
            raise ValueError("parent_selection must be 'random' or 'grey_wolf'")
        self.micro_population_size = int(micro_population_size)
        self.parent_selection = parent_selection
        if initializer is not None and not isinstance(initializer, AbstractInitializer):
            raise TypeError("The expected type of `initializer` is `AbstractInitializer` or None")
        self.initializer = initializer
        self._weight_order = None
        self.micro_populations = []
        super().__init__(parent_selector=MicroGWOParentSelector, **kwargs)
        if initializer is not None and getattr(initializer, "weight_aware", False):
            # random.shuffle moved the weight objects themselves; follow them by identity
            # so group i keeps the residents that were built for its weight.
            original_index = {id(weight): index for index, weight in enumerate(self._weight_order)}
            self.micro_populations = [self.micro_populations[original_index[id(weight)]]
                                      for weight in self.weights]
        self._weight_order = None
        # AbstractMoead shuffles weights after initialization. Representatives
        # must be selected after that shuffle, with the final assigned weights.
        self.refresh_representatives()

    @property
    def population_size(self):
        """Total resident individuals; number_of_weight remains the group count."""
        return self.number_of_weight * self.micro_population_size

    @property
    def initial_evaluations(self):
        return self.population_size

    def all_population_solutions(self):
        """Visit all residents, not just the compatibility representatives."""
        return [solution for group in self.micro_populations for solution in group]

    def initial_population(self):
        # Called by AbstractMoead after validating weights and neighborhood size,
        # before consuming any evaluation. Check the larger budget here.
        if self.max_evaluation < self.initial_evaluations:
            raise ValueError("max_evaluation must cover all micro-population individuals")
        if self.initializer is None:
            for _ in range(self.number_of_weight):
                group = []
                for _ in range(self.micro_population_size):
                    solution = self.problem.generate_random_solution()
                    self.current_eval += 1
                    group.append(solution)
                    self.update_archive(solution)
                self.micro_populations.append(group)
        else:
            self._initialize_with_initializer()
        # These are only placeholders until weights and z are ready.
        return [deepcopy(group[0]) for group in self.micro_populations]

    def _initialize_with_initializer(self):
        """Fill every group via the initializer; weight w_i is repeated for group i's residents."""
        k = self.micro_population_size
        # Weights are still unshuffled here; remember their order to follow the shuffle later.
        self._weight_order = list(self.weights)
        expanded_weights = [tuple(float(v) for v in self.weights[i])
                            for i in range(self.number_of_weight) for _ in range(k)]

        def on_evaluated(solution):
            self.current_eval += 1
            self.update_archive(solution=solution)

        flat = self.initializer.initialize(
            problem=self.problem, population_size=self.initial_evaluations,
            weights=expanded_weights, on_evaluated=on_evaluated,
        )
        if len(flat) != self.initial_evaluations:
            raise RuntimeError(
                f"Initializer returned {len(flat)} solutions; expected {self.initial_evaluations}")
        self.micro_populations = [list(flat[i * k:(i + 1) * k]) for i in range(self.number_of_weight)]

    def init_z(self):
        # A nonrepresentative resident may contain the best value of an objective.
        return np.min([solution.F for solution in self.all_population_solutions()], axis=0)

    def scalar_score(self, solution, group_index):
        """Score a cached F under exactly one target group's weight; no NFE."""
        return self.aggregation_function.run(
            solution=solution, number_of_objective=self.number_of_objective,
            weights=self.weights, sub_problem=group_index, z=self.z,
        )

    def refresh_representatives(self):
        """population[i] is a copy of the best resident under current w_i and z."""
        previous_sources = getattr(self, "_representative_sources", [])
        representatives, sources = [], []
        for index, group in enumerate(self.micro_populations):
            best = min(group, key=lambda solution: self.scalar_score(solution, index))
            # Residents are replaced with new objects, never mutated in place.
            # Reuse the existing independent copy while its source is unchanged.
            if index < len(previous_sources) and best is previous_sources[index]:
                representatives.append(self.population[index])
            else:
                representatives.append(deepcopy(best))
            sources.append(best)
        self.population = representatives
        self._representative_sources = sources

    def update_solutions(self, solution, aggregation_function, sub_problem):
        """Replace at most one worst resident in each neighboring group.

        All comparisons within group j use w_j and the already updated z.
        Reject equal scores and duplicate genes; the inherited loop still
        considers every evaluated child for the external archive afterwards.
        """
        for group_index in self.b[sub_problem]:
            group = self.micro_populations[group_index]
            if any(np.array_equal(solution.decision_vector, resident.decision_vector)
                   for resident in group):
                continue
            worst_index = max(
                range(len(group)), key=lambda index: self.scalar_score(group[index], group_index),
            )
            old_score = self.scalar_score(group[worst_index], group_index)
            new_score = self.scalar_score(solution, group_index)
            if aggregation_function.is_better(old_score, new_score):
                group[worst_index] = deepcopy(solution)
        # z may change rankings even in groups that did not accept this child.
        self.refresh_representatives()
