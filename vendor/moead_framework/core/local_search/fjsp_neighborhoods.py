import random
import numpy as np


class FJSPNeighborhoodOperators:
    def __init__(self, problem):
        self.problem = problem
        self.total_ops = self._infer_total_operations()
        self._flat_ops = self._build_flat_operations()
        self.num_operators = 6

    def _infer_total_operations(self):
        if hasattr(self.problem, "total_operations"):
            return int(self.problem.total_operations)

        instance = getattr(self.problem, "instance", None)
        if instance is not None:
            if hasattr(instance, "operation_count"):
                return int(instance.operation_count)
            if hasattr(instance, "total_operations"):
                return int(instance.total_operations)
            if hasattr(instance, "jobs"):
                return sum(len(job) for job in instance.jobs)

        raise AttributeError("Unable to infer total_operations from FJSPProblem")

    def _build_flat_operations(self):
        instance = getattr(self.problem, "instance", None)
        if instance is not None and hasattr(instance, "jobs"):
            return tuple(options for job in instance.jobs for options in job)
        return ()

    def _split_vector(self, decision_vector):
        vec = np.array(decision_vector, copy=True)
        os_part = vec[:self.total_ops].copy()
        ms_part = vec[self.total_ops:].copy()
        return os_part, ms_part

    def _merge_vector(self, os_part, ms_part):
        return np.concatenate([os_part, ms_part])

    def _find_critical_indices(self, solution, os_part):
        schedule = getattr(solution, "schedule", None)
        if schedule is None or not hasattr(schedule, "operations"):
            return list(range(len(os_part)))

        makespan = float(solution.F[0])
        critical_jobs = set()
        for op in schedule.operations:
            end_t = getattr(op, "end_time", getattr(op, "end", None))
            if end_t is not None and abs(float(end_t) - makespan) < 1e-6:
                job_id = getattr(op, "job_id", getattr(op, "job", None))
                if job_id is not None:
                    critical_jobs.add(job_id)

        if not critical_jobs:
            return list(range(len(os_part)))

        indices = [idx for idx, gene in enumerate(os_part) if int(gene) in critical_jobs]
        return indices if indices else list(range(len(os_part)))

    def _op_n6_critical_block(self, solution, os_part):
        crit_idx = self._find_critical_indices(solution, os_part)
        if len(crit_idx) < 2:
            return self._op_insert_os(os_part)

        i, j = sorted(random.sample(crit_idx, 2))
        os_list = os_part.tolist()
        val = os_list.pop(j)
        os_list.insert(i, val)

        return np.array(os_list, dtype=os_part.dtype)

    def _op_swap_os(self, os_part):
        if len(os_part) < 2:
            return os_part

        i, j = random.sample(range(len(os_part)), 2)
        os_part[i], os_part[j] = os_part[j], os_part[i]

        return os_part

    def _op_insert_os(self, os_part):
        if len(os_part) < 2:
            return os_part

        i, j = random.sample(range(len(os_part)), 2)
        os_list = os_part.tolist()
        val = os_list.pop(j)
        os_list.insert(i, val)

        return np.array(os_list, dtype=os_part.dtype)

    def _op_critical_insert_os(self, solution, os_part):
        if len(os_part) < 2:
            return os_part

        crit_idx = self._find_critical_indices(solution, os_part)
        if not crit_idx:
            return self._op_swap_os(os_part)

        src = random.choice(crit_idx)
        dst = random.randrange(src) if src > 0 else random.randrange(1, len(os_part))
        os_list = os_part.tolist()
        val = os_list.pop(src)
        os_list.insert(dst, val)
        return np.array(os_list, dtype=os_part.dtype)

    def _op_random_ms(self, ms_part):
        if len(ms_part) == 0:
            return ms_part

        if self._flat_ops:
            mutable = [idx for idx, options in enumerate(self._flat_ops) if len(options) > 1]
            if not mutable:
                return ms_part
            op_idx = random.choice(mutable)
            n_choices = len(self._flat_ops[op_idx])
            current = int(ms_part[op_idx])
            choice = random.randrange(n_choices - 1)
            ms_part[op_idx] = choice + (choice >= current)
            return ms_part

        if hasattr(self.problem, "upper_bounds") and hasattr(self.problem, "lower_bounds"):
            op_idx = random.randrange(len(ms_part))
            ub = int(self.problem.upper_bounds[self.total_ops + op_idx])
            lb = int(self.problem.lower_bounds[self.total_ops + op_idx])
            if ub > lb:
                choices = [v for v in range(lb, ub + 1) if v != int(ms_part[op_idx])]
                if choices:
                    ms_part[op_idx] = random.choice(choices)

        return ms_part

    def _op_ranking_ms(self, ms_part):
        if len(ms_part) == 0:
            return ms_part

        if not self._flat_ops:
            return self._op_random_ms(ms_part)

        mutable = [idx for idx, options in enumerate(self._flat_ops) if len(options) > 1]
        if not mutable:
            return ms_part

        op_idx = random.choice(mutable)
        options = self._flat_ops[op_idx]
        inv_times = np.array([1.0 / max(1e-6, float(opt.duration)) for opt in options], dtype=float)
        probs = inv_times / inv_times.sum()
        chosen = random.choices(range(len(options)), weights=probs.tolist(), k=1)[0]
        ms_part[op_idx] = chosen

        return ms_part

    def apply(self, action_id, solution):
        os_part, ms_part = self._split_vector(solution.decision_vector)

        if action_id == 0:
            os_part = self._op_n6_critical_block(solution, os_part)
        elif action_id == 1:
            os_part = self._op_swap_os(os_part)
        elif action_id == 2:
            os_part = self._op_insert_os(os_part)
        elif action_id == 3:
            os_part = self._op_critical_insert_os(solution, os_part)
        elif action_id == 4:
            ms_part = self._op_random_ms(ms_part)
        elif action_id == 5:
            ms_part = self._op_ranking_ms(ms_part)
        else:
            raise ValueError(f"Unknown operator index: {action_id}")

        return self._merge_vector(os_part, ms_part)

    def active_compaction(self, solution):
        os_part, ms_part = self._split_vector(solution.decision_vector)
        schedule = getattr(solution, "schedule", None)
        if schedule is None or not hasattr(schedule, "operations"):
            return np.array(solution.decision_vector, copy=True)

        indexed_ops = list(enumerate(schedule.operations))
        indexed_ops.sort(key=lambda item: (
            getattr(item[1], "start_time", getattr(item[1], "start", 0)),
            getattr(item[1], "op_index", getattr(item[1], "operation_id", item[0])),
            item[0]
        ))
        compacted_os = np.array(
            [getattr(op, "job_id", getattr(op, "job", os_part[idx])) for idx, op in indexed_ops],
            dtype=os_part.dtype,
        )
        if len(compacted_os) != len(os_part):
            return np.array(solution.decision_vector, copy=True)

        return self._merge_vector(compacted_os, ms_part)
