import numpy as np
from pathlib import Path
from copy import deepcopy
from tempfile import TemporaryDirectory
from moead_framework.aggregation import *
from moead_framework.algorithm.fjsp import *
from moead_framework.algorithm.moead import *
from moead_framework.core.genetic_operator import *
from moead_framework.core.local_search.fjsp_neighborhoods import *
from moead_framework.core.offspring_generator.fjsp import *
from moead_framework.core.reinforcement_learning.dqn_agent import *
from moead_framework.problem.fjsp import *


def _dominates(f_a, f_b):
    a = np.asarray(f_a, dtype=float)
    b = np.asarray(f_b, dtype=float)
    return bool(np.all(a <= b)) and bool(np.any(a < b))


def _none_dominated_filter(solutions):
    unique = []
    seen = set()
    for sol in solutions:
        key = tuple(np.asarray(sol.decision_vector).tolist())
        if key not in seen:
            seen.add(key)
            unique.append(sol)

    front = []
    for i, sol_i in enumerate(unique):
        dominated = False
        for j, sol_j in enumerate(unique):
            if i != j and _dominates(sol_j.F, sol_i.F):
                dominated = True
                break
        if not dominated:
            front.append(sol_i)

    return front


class FjspDqceMoead(Moead):
    def __init__(self, dqn_config=None, use_compaction=True, **kwargs):
        self.dqn_config = DQNConfig() if dqn_config is None else dqn_config
        if not isinstance(self.dqn_config, DQNConfig):
            raise TypeError("dqn_config must be DQNConfig")
        if not isinstance(use_compaction, bool):
            raise TypeError("use_compaction must be bool")
        super().__init__(**kwargs)
        self.use_compaction = use_compaction
        self.parasite_population = _none_dominated_filter([deepcopy(sol) for sol in self.population])
        self.operator_history = []
        self._ls_ops = None
        self._agent = None

    def _init_rl_components(self):
        if self._ls_ops is None:
            self._ls_ops = FJSPNeighborhoodOperators(self.problem)

        if self._agent is None and len(self.population) > 0:
            state_dim = len(self.population[0].decision_vector)
            self._agent = DQNAgent(
                state_dim=state_dim,
                action_dim=self._ls_ops.num_operators,
                config=self.dqn_config
            )

    def _evaluate_candidate(self, decision_vector, checkpoint=None):
        if self.current_eval >= self.max_evaluation or not self.termination_criteria.test():
            return None

        if checkpoint is not None:
            checkpoint(self)

        sol = self.problem.evaluate(decision_vector)
        self.current_eval += 1
        self.update_z(sol)
        self.update_archive(sol)
        return sol

    def _feedback_to_host(self, solution):
        for idx in range(self.number_of_weight):
            f_new = self.aggregation_function.run(
                solution=solution,
                number_of_objective=self.number_of_objective,
                weights=self.weights,
                sub_problem=idx,
                z=self.z,
            )
            f_old = self.aggregation_function.run(
                solution=self.population[idx],
                number_of_objective=self.number_of_objective,
                weights=self.weights,
                sub_problem=idx,
                z=self.z,
            )
            if self.aggregation_function.is_better(f_old, f_new):
                self.population[idx] = deepcopy(solution)

    def _evolve_parasite(self, checkpoint=None):
        self._init_rl_components()
        if not self.parasite_population or self._agent is None:
            return

        action_counts = [0] * self._ls_ops.num_operators
        updated_parasites = []
        for sol in list(self.parasite_population):
            if self.current_eval >= self.max_evaluation or not self.termination_criteria.test():
                updated_parasites.append(sol)
                continue

            state_t = np.asarray(sol.decision_vector, dtype=np.float32)
            action_t = self._agent.select_action(state_t)
            action_counts[action_t] += 1

            cand_vec = self._ls_ops.apply(action_t, sol)
            cand_sol = self._evaluate_candidate(cand_vec, checkpoint=checkpoint)
            if cand_sol is None:
                updated_parasites.append(sol)
                continue

            reward_t = self._agent.compute_reward(sol.F, cand_sol.F)
            next_state = np.asarray(cand_sol.decision_vector, dtype=np.float32)
            self._agent.store_and_learn(state_t, action_t, reward_t, next_state)

            if _dominates(cand_sol.F, sol.F):
                updated_parasites.append(cand_sol)
                self._feedback_to_host(cand_sol)
            elif not _dominates(sol.F, cand_sol.F):
                updated_parasites.append(sol)
                updated_parasites.append(cand_sol)
                self._feedback_to_host(cand_sol)
            else:
                updated_parasites.append(sol)

        if self.use_compaction:
            compacted_list = []
            for sol in updated_parasites:
                if self.current_eval >= self.max_evaluation or not self.termination_criteria.test():
                    compacted_list.append(sol)
                    continue

                comp_vec = self._ls_ops.active_compaction(sol)
                if np.array_equal(comp_vec, sol.decision_vector):
                    compacted_list.append(sol)
                    continue

                comp_sol = self._evaluate_candidate(comp_vec, checkpoint=checkpoint)
                if comp_sol is None:
                    compacted_list.append(sol)
                elif _dominates(comp_sol.F, sol.F):
                    compacted_list.append(comp_sol)
                    self._feedback_to_host(comp_sol)
                elif not _dominates(sol.F, comp_sol.F):
                    compacted_list.append(sol)
                    compacted_list.append(comp_sol)
                    self._feedback_to_host(comp_sol)
                else:
                    compacted_list.append(sol)

            updated_parasites = compacted_list

        self.parasite_population = _none_dominated_filter(updated_parasites)
        total_actions = max(1, sum(action_counts))
        self.operator_history.append([c / total_actions for c in action_counts])

    def run(self, checkpoint=None):
        while self.current_eval < self.max_evaluation and self.termination_criteria.test():
            for i in self.get_sub_problems_to_visit():
                if self.current_eval >= self.max_evaluation or not self.termination_criteria.test():
                    break

                if checkpoint is not None:
                    checkpoint(self)

                self.optimize_sub_problem(sub_problem_index=i)

            host_pf = _none_dominated_filter([deepcopy(sol) for sol in self.population])
            self.parasite_population = _none_dominated_filter(self.parasite_population + host_pf)

            if self.current_eval < self.max_evaluation and self.termination_criteria.test():
                self._evolve_parasite(checkpoint=checkpoint)

        return self.ep


def create_fjsp_dqce(problem, max_evaluation=1000, number_of_weight_neighborhood=10,
                     partitions=12, operator_config=None, dqn_config=None, use_compaction=True):
    if not isinstance(problem, FJSPProblem):
        raise TypeError("problem must be FJSPProblem")
    if problem.evaluation_count:
        raise ValueError("Use a fresh FJSPProblem for each DQCE run")

    config = FJSPOperatorConfig() if operator_config is None else operator_config
    if not isinstance(config, FJSPOperatorConfig):
        raise TypeError("operator_config must be FJSPOperatorConfig")

    dqn_cfg = DQNConfig() if dqn_config is None else dqn_config
    if not isinstance(dqn_cfg, DQNConfig):
        raise TypeError("dqn_config must be DQNConfig")

    weights = uniform_weights_2d(partitions)
    with TemporaryDirectory(prefix='moead-fjsp-dqce-') as directory:
        path = Path(directory) / 'weights.ws'
        path.write_text('\n'.join(' '.join(map(str, row)) for row in weights) + '\n', encoding='utf-8')
        algorithm = FjspDqceMoead(
            dqn_config=dqn_cfg,
            use_compaction=use_compaction,
            problem=problem,
            max_evaluation=max_evaluation,
            number_of_weight_neighborhood=number_of_weight_neighborhood,
            weight_file=str(path),
            aggregation_function=Tchebycheff,
            genetic_operator=FJSPCrossoverAndMutation,
            offspring_generator=FJSPOffspringGenerator
        )
    algorithm.offspring_generator.config = config

    return algorithm
