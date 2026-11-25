"""Tabular Q-learning population initializer for bi-objective FJSP (Makespan, TWT).

Markov decision process (one episode = one constructed chromosome = one NFE):

- State  s = (weight bin of the episode's subproblem, progress bin, tardiness-pressure bin).
  The weight bin lets one table learn different policies for makespan- and
  TWT-oriented subproblems; progress and tardiness pressure describe the
  partial schedule (see `FJSPConstructionEnv`).
- Action a = composite dispatching rule (job rule x machine rule), e.g. MWKR+ECT.
  Every action dispatches the next operation of an eligible job on one of its
  eligible machines, so every episode yields a feasible OS/MS chromosome.
- Reward r_t = -(l_1 * dCmax_t / s_C + l_2 * dTWT_t / s_T), where (l_1, l_2) is the
  subproblem weight vector, s_C is a data-only makespan lower bound and
  s_T = sum(w_j) * s_C. Increments telescope, so the undiscounted return equals
  minus the weighted normalized objective of the final schedule.
- Update: off-policy Q-learning
  Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a)),
  applied (in reverse step order) only after the candidate has been evaluated,
  so learning never uses information that was not paid for with an NFE.
- Policy: epsilon-greedy, epsilon decaying linearly over episodes. Q starts at 0
  while rewards are <= 0, which is optimistic and drives early exploration.
"""

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
import random

import numpy as np

from moead_framework.problem.fjsp import FJSPProblem

from .abstract_initializer import AbstractInitializer, validate_population_size
from .fjsp_construction import ACTIONS, FJSPConstructionEnv, makespan_lower_bound

TARDY_BINS = 3


@dataclass(frozen=True)
class QInitConfig:
    """Hyperparameters of the Q-learning (and hybrid Random + Q-learning) initializer."""

    alpha: float = 0.1
    gamma: float = 0.95
    epsilon_start: float = 0.5
    epsilon_end: float = 0.05
    random_ratio: float = 0.2
    progress_bins: int = 4
    weight_bins: int = 3
    max_resample: int = 5

    def __post_init__(self):
        self._validate_real("alpha", 0.0, 1.0, low_inclusive=False, high_inclusive=True)
        self._validate_real("gamma", 0.0, 1.0, low_inclusive=True, high_inclusive=True)
        self._validate_real("epsilon_start", 0.0, 1.0, low_inclusive=True, high_inclusive=True)
        self._validate_real("epsilon_end", 0.0, 1.0, low_inclusive=True, high_inclusive=True)
        if self.epsilon_end > self.epsilon_start:
            raise ValueError("epsilon_end must be <= epsilon_start")
        self._validate_real("random_ratio", 0.0, 1.0, low_inclusive=True, high_inclusive=True)
        for name, minimum in (("progress_bins", 1), ("weight_bins", 1), ("max_resample", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
            object.__setattr__(self, name, int(value))

    def _validate_real(self, name, low, high, low_inclusive, high_inclusive):
        value = getattr(self, name)
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
            raise ValueError(f"{name} must be a finite real number")
        val = float(value)
        low_ok = val >= low if low_inclusive else val > low
        high_ok = val <= high if high_inclusive else val < high
        if not (low_ok and high_ok):
            lb = "[" if low_inclusive else "("
            rb = "]" if high_inclusive else ")"
            raise ValueError(f"{name} must be in {lb}{low}, {high}{rb}")
        object.__setattr__(self, name, val)


def epsilon_for_episode(index, total, eps_start, eps_end):
    """Linear decay from `eps_start` (first episode) to `eps_end` (last episode)."""
    if total <= 1:
        return eps_start
    return eps_start + (eps_end - eps_start) * index / (total - 1)


def spread_indices(count, size):
    """`count` distinct indices in range(size), evenly spread; both ends when count >= 2, else the middle."""
    if count <= 0:
        return []
    if count == 1:
        return [size // 2]
    return sorted({int(round(x)) for x in np.linspace(0, size - 1, count)})


def _normalized_weights(weights, size):
    if weights is None:
        return [(1.0, 0.0)] if size == 1 else [(i / (size - 1), 1 - i / (size - 1)) for i in range(size)]
    rows = [tuple(float(v) for v in row) for row in weights]
    if len(rows) != size or any(len(row) != 2 for row in rows):
        raise ValueError("weights must contain one 2-objective vector per solution")
    if any(not isfinite(v) or v < 0 for row in rows for v in row) or any(sum(row) <= 0 for row in rows):
        raise ValueError("weights must be finite, nonnegative and have positive sums")
    return [(row[0] / sum(row), row[1] / sum(row)) for row in rows]


class QLearningInitializer(AbstractInitializer):
    """Q-learning over composite dispatching rules; every solution targets its subproblem weight.

    `initialize()` returns solution i built for `weights[i]` (`weight_aware`), so
    `MoeadQInit` keeps that pairing after MOEA/D shuffles its weight vectors.
    """

    weight_aware = True
    mode_name = "qlearning"
    action_names = tuple(name for name, _, _ in ACTIONS)

    def __init__(self, config=None):
        self.config = QInitConfig() if config is None else config
        if not isinstance(self.config, QInitConfig):
            raise TypeError("config must be QInitConfig")
        self.q_table = None
        self.last_stats = {}

    # --- Q-table and state ---

    @property
    def state_count(self):
        return self.config.weight_bins * self.config.progress_bins * TARDY_BINS

    def reset_table(self):
        self.q_table = np.zeros((self.state_count, len(ACTIONS)), dtype=float)

    def weight_bin(self, weight):
        return min(self.config.weight_bins - 1, int(weight[0] * self.config.weight_bins))

    def state(self, env, weight_bin):
        bins = self.config.progress_bins
        progress = min(bins - 1, env.scheduled * bins // env.operation_count)
        pressure = env.tardy_pressure()
        tardy = 0 if pressure == 0 else (1 if pressure <= 0.5 else 2)
        return (weight_bin * bins + progress) * TARDY_BINS + tardy

    def choose_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(len(ACTIONS))
        row = self.q_table[state]
        best = np.flatnonzero(row >= row.max() - 1e-12)
        return int(best[0]) if len(best) == 1 else int(random.choice(best.tolist()))

    # --- Episodes and learning ---

    def run_episode(self, env, weight, epsilon, scales):
        """Construct one chromosome; return it with its transitions and simulated objectives."""
        if self.q_table is None:
            self.reset_table()
        env.reset()
        scale_c, scale_t = scales
        weight_bin = self.weight_bin(weight)
        transitions = []
        state = self.state(env, weight_bin)
        while not env.finished:
            action = self.choose_action(state, epsilon)
            _, job_rule, machine_rule = ACTIONS[action]
            job = job_rule(env)
            d_makespan, d_twt = env.step(job, machine_rule(env, job))
            reward = -(weight[0] * d_makespan / scale_c + weight[1] * d_twt / scale_t)
            next_state = None if env.finished else self.state(env, weight_bin)
            transitions.append((state, action, reward, next_state))
            state = next_state
        return env.chromosome(), transitions, [env.makespan, env.twt]

    def learn(self, transitions):
        """Q-learning TD updates, replayed from the last step back to the first."""
        alpha, gamma = self.config.alpha, self.config.gamma
        for state, action, reward, next_state in reversed(transitions):
            target = reward if next_state is None else reward + gamma * float(self.q_table[next_state].max())
            self.q_table[state, action] += alpha * (target - self.q_table[state, action])
        return len(transitions)

    @staticmethod
    def scales(instance):
        scale_c = float(max(1, makespan_lower_bound(instance)))
        return scale_c, float(max(1.0, sum(instance.weights) * scale_c))

    # --- AbstractInitializer ---

    def split_counts(self, size):
        """(n_random, n_qlearning); pure Q-learning uses no random solutions."""
        return 0, size

    def initialize(self, problem, population_size, weights, on_evaluated):
        if not isinstance(problem, FJSPProblem):
            raise TypeError(f"{type(self).__name__} requires FJSPProblem")
        size = validate_population_size(population_size)
        targets = _normalized_weights(weights, size)
        instance = problem.instance
        self.reset_table()

        n_rand, n_q = self.split_counts(size)
        q_slots = spread_indices(n_q, size)
        q_set = set(q_slots)
        random_slots = [i for i in range(size) if i not in q_set]
        population = [None] * size
        seen = set()

        # Random solutions (diversity), placed on the subproblems Q-learning does not target.
        for slot in random_slots:
            solution = problem.generate_random_solution()
            on_evaluated(solution)
            seen.add(tuple(int(v) for v in solution.decision_vector))
            population[slot] = solution

        # Q-learning episodes in random subproblem order, so every weight bin learns early.
        order = list(q_slots)
        random.shuffle(order)
        env = FJSPConstructionEnv(instance)
        scales = self.scales(instance)
        action_counts = [0] * len(ACTIONS)
        eps_schedule, returns = [], []
        td_updates = resampled = duplicates = 0
        for episode, slot in enumerate(order):
            eps = epsilon_for_episode(episode, len(order), self.config.epsilon_start, self.config.epsilon_end)
            eps_schedule.append(round(eps, 6))
            for attempt in range(self.config.max_resample + 1):
                # A duplicate is rebuilt with more exploration; it was never evaluated, so it is not learned.
                attempt_eps = eps + (1.0 - eps) * attempt / max(1, self.config.max_resample)
                chromosome, transitions, simulated = self.run_episode(env, targets[slot], attempt_eps, scales)
                key = chromosome.to_vector()
                if key not in seen:
                    break
                if attempt < self.config.max_resample:
                    resampled += 1
            else:
                duplicates += 1
            seen.add(key)
            solution = problem.evaluate(key)
            on_evaluated(solution)
            if list(solution.F) != simulated:
                raise RuntimeError("Construction environment disagrees with the FJSP decoder")
            population[slot] = solution
            td_updates += self.learn(transitions)
            returns.append(round(sum(t[2] for t in transitions), 6))
            for _, action, _, _ in transitions:
                action_counts[action] += 1

        self.last_stats = {
            "mode": self.mode_name,
            "population_size": size,
            "n_random": n_rand,
            "n_qlearning": n_q,
            "episodes": len(order),
            "td_updates": td_updates,
            "resampled_duplicates": resampled,
            "accepted_duplicates": duplicates,
            "epsilon_schedule": eps_schedule,
            "episode_returns": returns,
            "action_counts": dict(zip(self.action_names, action_counts)),
            "greedy_policy": self.greedy_policy(),
        }
        return population

    def greedy_policy(self):
        """Greedy action name per visited (weight bin, progress bin, tardy bin) state."""
        if self.q_table is None:
            return {}
        policy = {}
        bins = self.config.progress_bins
        for state in range(self.state_count):
            row = self.q_table[state]
            if not np.any(row):
                continue
            weight_bin, rest = divmod(state, bins * TARDY_BINS)
            progress, tardy = divmod(rest, TARDY_BINS)
            policy[f"w{weight_bin}-p{progress}-t{tardy}"] = self.action_names[int(row.argmax())]
        return policy
