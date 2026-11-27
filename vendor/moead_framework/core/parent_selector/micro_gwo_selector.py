"""Discrete Grey Wolf mating: choose parents, never evaluate or mutate genes."""

import random

from moead_framework.core.parent_selector.abstract_parent_selector import ParentSelector


class MicroGWOParentSelector(ParentSelector):
    def __init__(self, algorithm):
        super().__init__(algorithm)
        self._omega_cursors = [0] * algorithm.number_of_weight
        self.selection_counts = {"leader": 0, "peer": 0, "random": 0}
        self.last_selection = None

    def solution_at(self, position):
        group_index, resident_index = position
        return self.algorithm.micro_populations[group_index][resident_index]

    def gene_key(self, position):
        return tuple(self.solution_at(position).decision_vector)

    def ranked_candidates(self, indexes, target_group):
        """Xếp hạng ứng viên theo trọng số nhóm đang tối ưu"""
        positions = [(group, slot) for group in sorted(set(indexes))
                     for slot in range(len(self.algorithm.micro_populations[group]))]
        return sorted(positions, key=lambda position: (
            self.algorithm.scalar_score(self.solution_at(position), target_group), position,
        ))

    def select_leaders(self, ranked):
        """Chọn tối đa ba chromosome khác nhau làm α/β/δ"""
        leaders, seen = [], set()
        for position in ranked:
            key = self.gene_key(position)
            if key not in seen:
                leaders.append(position)
                seen.add(key)
            if len(leaders) == 3:
                break
        return leaders

    def search_progress(self):
        """tính Bf"""
        initial = self.algorithm.initial_evaluations
        search_budget = self.algorithm.max_evaluation - initial
        if search_budget == 0:
            return 1.0
        return min(1.0, max(0.0, (self.algorithm.current_eval - initial) / search_budget))

    def select(self, indexes):
        #hàm thực hiện toàn bộ quy trình chọn hai cha mẹ
        super().select(indexes)
        target = self.algorithm.current_sub_problem
        groups = sorted(set(indexes) | {target})
        positions = [(group, slot) for group in groups
                     for slot in range(len(self.algorithm.micro_populations[group]))]
        if self.algorithm.parent_selection == "random":
            first, second = random.sample(positions, 2)
            return self.record_selection("random", first, second)

        leaders = self.select_leaders(self.ranked_candidates(groups, target))
        # Followers come from the group being optimized. With mu >= 4, at
        # least one slot remains outside the three leader positions.
        own_followers = [position for position in positions
                         if position[0] == target and position not in leaders]
        cursor = self._omega_cursors[target]
        follower = own_followers[cursor % len(own_followers)]
        self._omega_cursors[target] += 1

        progress = self.search_progress()
        # A fresh B0 for every child is an explicit adaptation of paper Eq. 23.
        b0 = 1.0 - random.random()
        balance = (1.0 - progress) * b0
        if balance >= 0.5:
            partner = random.choice(leaders)  # alpha/beta/delta, equal probability
            branch = "leader"
        else:
            peers = [position for position in positions
                     if position not in leaders and position != follower]

            different = [position for position in peers
                         if self.gene_key(position) != self.gene_key(follower)]
            candidates = different or peers or [p for p in positions if p != follower]
            partner = random.choice(candidates)
            branch = "peer"
        return self.record_selection(
            branch, follower, partner, leaders=leaders, progress=progress,
            b0=b0, balance=balance,
        )

    def record_selection(self, branch, first, second, **details):
        self.selection_counts[branch] += 1
        self.last_selection = {"branch": branch, "parents": [first, second], **details}
        return [self.solution_at(first), self.solution_at(second)]
