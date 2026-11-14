"""Wire parent selection and OS/MS variation into exactly one FJSP evaluation."""

from copy import deepcopy

from moead_framework.core.offspring_generator.abstract_mating import OffspringGenerator
from moead_framework.core.genetic_operator.fjsp import FJSPOperatorConfig
from moead_framework.problem.fjsp import FJSPProblem


class FJSPOffspringGenerator(OffspringGenerator):
    def __init__(self, algorithm_instance):
        super().__init__(algorithm_instance)
        if not isinstance(self.algorithm.problem, FJSPProblem):
            raise TypeError("FJSP offspring generation requires FJSPProblem")
        self.config = FJSPOperatorConfig()

    def run(self, population_indexes):
        parents = self.algorithm.parent_selector.select(indexes=population_indexes)
        vectors = [deepcopy(parent.decision_vector) for parent in parents]
        operator = self.algorithm.genetic_operator(
            solutions=vectors, instance=self.algorithm.problem.instance, config=self.config,
        )
        return self.algorithm.problem.evaluate(operator.run())
