"""Hybrid Random + Q-learning population initializer."""

from .qlearning_initializer import QLearningInitializer


class HybridQRandomInitializer(QLearningInitializer):
    """Q-learning initializer that keeps a share of uniformly random solutions.

    - `random_ratio` of the population is generated uniformly at random and
      placed on the subproblems in between the Q-learning targets, preserving
      genotype diversity that rule-based construction tends to lose.
    - The remaining subproblems are built by Q-learning episodes (see
      `QLearningInitializer`). They include both extreme weights whenever at
      least two solutions are built by Q-learning; a single Q-learning
      solution goes to the middle subproblem.
    - Total initial evaluations equal `population_size`, preserving the
      baseline NFE split and checkpoint schedule.
    """

    mode_name = "hybrid_q_random"

    def split_counts(self, size):
        if size == 1:
            return (1, 0) if self.config.random_ratio >= 0.5 else (0, 1)
        ratio = self.config.random_ratio
        if ratio in (0.0, 1.0):
            return (0, size) if ratio == 0.0 else (size, 0)
        n_rand = max(1, min(size - 1, int(round(size * ratio))))
        return n_rand, size - n_rand
