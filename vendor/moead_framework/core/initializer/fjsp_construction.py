"""Constructive FJSP environment and dispatching rules used by the Q-learning initializer.

The environment builds one OS/MS chromosome operation by operation and keeps
the partial schedule with exactly the decoder's semantics (release dates,
earliest-feasible gap insertion). Because of that, the per-step objective
increments telescope to the final (Makespan, TWT) that `FJSPProblem.evaluate`
returns for the finished chromosome. The environment never calls `evaluate`
and never counts NFE: it is the construction procedure of one candidate.
"""

import random

from moead_framework.problem.fjsp import Chromosome
from moead_framework.problem.fjsp.decoder import _earliest_slot


class FJSPConstructionEnv:
    """Incremental partial schedule over an `FJSPInstance`."""

    def __init__(self, instance):
        self.instance = instance
        self.job_count = instance.job_count
        self.operation_count = instance.operation_count
        self.offsets = instance.operation_offsets
        self.min_durations = [[min(o.duration for o in ops) for ops in job] for job in instance.jobs]
        self.reset()

    def reset(self):
        inst = self.instance
        self.next_op = [0] * self.job_count
        self.job_ready = list(inst.release_dates)
        self.remaining_work = [sum(durations) for durations in self.min_durations]
        self.busy = [[] for _ in range(inst.machine_count)]
        self.os_genes = []
        self.ms_genes = [0] * self.operation_count
        self.makespan = 0
        self.twt = 0

    @property
    def scheduled(self):
        return len(self.os_genes)

    @property
    def finished(self):
        return self.scheduled == self.operation_count

    def available_jobs(self):
        return [j for j in range(self.job_count) if self.next_op[j] < len(self.instance.jobs[j])]

    def options(self, job):
        return self.instance.jobs[job][self.next_op[job]]

    def slot(self, job, choice):
        """(start, insert_position, option) for the next operation of `job` on `choice`."""
        option = self.options(job)[choice]
        start, position = _earliest_slot(self.busy[option.machine], self.job_ready[job], option.duration)
        return start, position, option

    def slack(self, job):
        return self.instance.due_dates[job] - self.job_ready[job] - self.remaining_work[job]

    def tardy_pressure(self):
        """Fraction of unfinished jobs that are late even if all remaining work ran back to back."""
        unfinished = self.available_jobs()
        if not unfinished:
            return 0.0
        return sum(self.slack(j) < 0 for j in unfinished) / len(unfinished)

    def step(self, job, choice):
        """Dispatch the next operation of `job` on local alternative `choice`.

        :return: (makespan increase, weighted tardiness incurred) of this step.
        """
        operation = self.next_op[job]
        start, position, option = self.slot(job, choice)
        end = start + option.duration
        self.busy[option.machine].insert(position, _Interval(start, end))
        self.os_genes.append(job)
        self.ms_genes[self.offsets[job] + operation] = choice
        self.job_ready[job] = end
        self.remaining_work[job] -= self.min_durations[job][operation]
        self.next_op[job] += 1

        delta_makespan = max(0, end - self.makespan)
        self.makespan += delta_makespan
        delta_twt = 0
        if self.next_op[job] == len(self.instance.jobs[job]):
            delta_twt = self.instance.weights[job] * max(0, end - self.instance.due_dates[job])
            self.twt += delta_twt
        return delta_makespan, delta_twt

    def chromosome(self):
        if not self.finished:
            raise RuntimeError("Chromosome requested before every operation was dispatched")
        return Chromosome(tuple(self.os_genes), tuple(self.ms_genes))


class _Interval:
    """Minimal busy interval compatible with decoder._earliest_slot (start/end only)."""

    __slots__ = ("start", "end")

    def __init__(self, start, end):
        self.start = start
        self.end = end


def _argbest(candidates, key):
    """Candidate with the smallest key; ties broken uniformly at random."""
    best_key = None
    best = []
    for candidate in candidates:
        value = key(candidate)
        if best_key is None or value < best_key:
            best_key, best = value, [candidate]
        elif value == best_key:
            best.append(candidate)
    return best[0] if len(best) == 1 else random.choice(best)


def _earliest_completion(env, job):
    return min(start + option.duration
               for start, _, option in (env.slot(job, c) for c in range(len(env.options(job)))))


# --- Job-selection rules: env -> job ---

def job_mwkr(env):
    """Most Work Remaining (min-duration estimate): makespan oriented."""
    return _argbest(env.available_jobs(), lambda j: -env.remaining_work[j])


def job_weighted_slack(env):
    """Weighted minimum slack: late heavy jobs first, then tight heavy jobs (TWT oriented)."""
    weights = env.instance.weights

    def key(j):
        slack = env.slack(j)
        return slack * weights[j] if slack < 0 else slack / weights[j]

    return _argbest(env.available_jobs(), key)


def job_earliest_completion(env):
    """Job whose next operation can finish earliest (compact, active-like schedules)."""
    return _argbest(env.available_jobs(), lambda j: _earliest_completion(env, j))


def job_random(env):
    """Uniform random eligible job (diversity)."""
    return random.choice(env.available_jobs())


# --- Machine-selection rules: (env, job) -> local alternative index ---

def machine_earliest_completion(env, job):
    """Alternative that finishes the operation earliest (load aware); ties by shorter duration."""
    def key(choice):
        start, _, option = env.slot(job, choice)
        return start + option.duration, option.duration

    return _argbest(range(len(env.options(job))), key)


def machine_shortest_processing(env, job):
    """Alternative with the shortest processing time; ties by earliest completion."""
    def key(choice):
        start, _, option = env.slot(job, choice)
        return option.duration, start + option.duration

    return _argbest(range(len(env.options(job))), key)


JOB_RULES = (
    ("MWKR", job_mwkr),
    ("WSLACK", job_weighted_slack),
    ("EEC", job_earliest_completion),
    ("RANDOM", job_random),
)
MACHINE_RULES = (
    ("ECT", machine_earliest_completion),
    ("SPT", machine_shortest_processing),
)
ACTIONS = tuple(
    (f"{job_name}+{machine_name}", job_rule, machine_rule)
    for job_name, job_rule in JOB_RULES
    for machine_name, machine_rule in MACHINE_RULES
)


def makespan_lower_bound(instance):
    """Data-only makespan lower bound (job chains, average load, forced single-machine load)."""
    minima = [[min(o.duration for o in ops) for ops in job] for job in instance.jobs]
    average_load = -(-sum(map(sum, minima)) // instance.machine_count)
    forced = [0] * instance.machine_count
    for job in instance.jobs:
        for options in job:
            if len(options) == 1:
                forced[options[0].machine] += options[0].duration
    chains = max(release + sum(job) for release, job in zip(instance.release_dates, minima))
    return max(chains, average_load, max(forced))
