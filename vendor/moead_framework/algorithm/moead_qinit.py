"""MOEA/D subclass whose only change is a pluggable population initializer (Q-learning / Hybrid)."""

from moead_framework.algorithm.moead import Moead
from moead_framework.core.initializer import (
    AbstractInitializer,
    HybridQRandomInitializer,
)


class MoeadQInit(Moead):
    """MOEA/D with a pluggable `AbstractInitializer` for Phase-0 population generation.

    Overrides ONLY `initial_population()`; keeps weight shuffling, ideal point z,
    neighborhoods B(i), evolutionary loop, NFE accounting, and external archive ep
    identical to `Moead`.

    `AbstractMoead` builds the population before it shuffles `self.weights`, so
    the baseline pairs initial solutions with random subproblems. For an
    initializer with `weight_aware = True` (solution i was built for weight i),
    the population is permuted with the same shuffle so every solution stays on
    the subproblem it was built for. The shuffle itself (and its RNG draws) is
    unchanged; initializers without `weight_aware` keep the baseline pairing.
    """

    def __init__(
        self,
        problem,
        max_evaluation,
        number_of_weight_neighborhood,
        aggregation_function,
        weight_file,
        initializer=None,
        **kwargs,
    ):
        self.initializer = HybridQRandomInitializer() if initializer is None else initializer
        if not isinstance(self.initializer, AbstractInitializer):
            raise TypeError("The expected type of `initializer` is `AbstractInitializer`")
        self._weight_order = None

        super().__init__(
            problem=problem,
            max_evaluation=max_evaluation,
            number_of_weight_neighborhood=number_of_weight_neighborhood,
            aggregation_function=aggregation_function,
            weight_file=weight_file,
            **kwargs,
        )

        if getattr(self.initializer, "weight_aware", False):
            # random.shuffle moves the weight objects themselves; follow them by identity.
            original_index = {id(weight): index for index, weight in enumerate(self._weight_order)}
            self.population = [self.population[original_index[id(weight)]] for weight in self.weights]
        self._weight_order = None

    def initial_population(self):
        def on_evaluated(solution):
            self.current_eval += 1
            self.update_archive(solution=solution)

        self._weight_order = list(self.weights)
        population = self.initializer.initialize(
            problem=self.problem,
            population_size=self.number_of_weight,
            weights=[tuple(float(v) for v in weight) for weight in self.weights],
            on_evaluated=on_evaluated,
        )
        if len(population) != self.number_of_weight:
            raise RuntimeError(
                f"Initializer returned {len(population)} solutions; expected {self.number_of_weight}"
            )
        return population
