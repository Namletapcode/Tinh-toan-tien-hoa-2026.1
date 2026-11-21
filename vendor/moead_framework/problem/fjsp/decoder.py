"""Serial, earliest-feasible gap insertion and independent schedule validation."""

from __future__ import annotations

from dataclasses import dataclass

from .encoding import Chromosome
from .instance import FJSPInstance, _integer


@dataclass(frozen=True)
class ScheduledOperation:
    job: int
    operation: int
    machine: int
    start: int
    end: int

    @property
    def duration(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Schedule:
    operations: tuple[ScheduledOperation, ...]  # OS dispatch order, not start-time order.
    machine_loads: tuple[int, ...]
    completion_times: tuple[int, ...]
    twt: int

    def __post_init__(self):
        object.__setattr__(self, "operations", tuple(self.operations))
        object.__setattr__(self, "machine_loads", tuple(self.machine_loads))
        object.__setattr__(self, "completion_times", tuple(_integer(value, "Completion time") for value in self.completion_times))
        object.__setattr__(self, "twt", _integer(self.twt, "TWT"))
        if self.twt < 0:
            raise ValueError("TWT must be nonnegative")

    @property
    def makespan(self) -> int:
        return max((operation.end for operation in self.operations), default=0)

    @property
    def total_workload(self) -> int:
        return sum(self.machine_loads)

    @property
    def max_machine_workload(self) -> int:
        return max(self.machine_loads, default=0)

    @property
    def objectives(self) -> tuple[int, int]:
        return self.makespan, self.twt


def _earliest_slot(busy: list[ScheduledOperation], ready: int, duration: int) -> tuple[int, int]:
    start = ready
    for index, operation in enumerate(busy):
        if start + duration <= operation.start:
            return start, index
        start = max(start, operation.end)
    return start, len(busy)


def decode(instance: FJSPInstance, chromosome: Chromosome) -> Schedule:
    """Decode OS/MS without changing its genes or any instance state.

    Preemption, setup/transport times and machine downtime are not part of this
    static FJSP model. Job release dates are respected. Equality at interval
    endpoints is valid, including zero-duration operations.
    """
    chromosome.validate(instance)
    next_operation = [0] * instance.job_count
    job_ready = list(instance.release_dates)
    busy = [[] for _ in range(instance.machine_count)]
    loads = [0] * instance.machine_count
    scheduled = []
    offsets = instance.operation_offsets
    for job in chromosome.os:
        operation = next_operation[job]
        choice = chromosome.ms[offsets[job] + operation]
        option = instance.jobs[job][operation][choice]
        start, position = _earliest_slot(busy[option.machine], job_ready[job], option.duration)
        record = ScheduledOperation(job, operation, option.machine, start, start + option.duration)
        busy[option.machine].insert(position, record)
        scheduled.append(record)
        loads[option.machine] += option.duration
        job_ready[job] = record.end
        next_operation[job] += 1
    completion_times = tuple(job_ready)
    twt = sum(
        weight * max(0, completion - due)
        for completion, due, weight in zip(completion_times, instance.due_dates, instance.weights)
    )
    return Schedule(tuple(scheduled), tuple(loads), completion_times, twt)


def validate_schedule(instance: FJSPInstance, schedule: Schedule) -> None:
    """Check the resulting schedule independently of the insertion algorithm.

    Every operation must appear exactly once on an eligible machine, with its
    original processing time. Precedence and chronological machine intervals
    require previous.end <= next.start, also for zero-duration operations.
    Release dates, recorded completion times and TWT must also match.
    """
    if len(schedule.operations) != instance.operation_count:
        raise ValueError("Schedule must contain every operation exactly once")
    if len(schedule.machine_loads) != instance.machine_count:
        raise ValueError("Machine-load vector has the wrong length")
    if len(schedule.completion_times) != instance.job_count:
        raise ValueError("Completion-time vector has the wrong length")
    records = {}
    busy = [[] for _ in range(instance.machine_count)]
    loads = [0] * instance.machine_count
    for record in schedule.operations:
        if not isinstance(record, ScheduledOperation):
            raise ValueError("Schedule entries must be ScheduledOperation objects")
        for field in ("job", "operation", "machine", "start", "end"):
            _integer(getattr(record, field), f"Scheduled {field}")
        if not 0 <= record.job < instance.job_count:
            raise ValueError("Scheduled job is out of range")
        if not 0 <= record.operation < len(instance.jobs[record.job]):
            raise ValueError("Scheduled operation is out of range")
        if not 0 <= record.machine < instance.machine_count:
            raise ValueError("Scheduled machine is out of range")
        if record.start < 0 or record.end < record.start:
            raise ValueError("Schedule has invalid start/end times")
        if record.operation == 0 and record.start < instance.release_dates[record.job]:
            raise ValueError("Job starts before its release date")
        key = record.job, record.operation
        if key in records:
            raise ValueError("Schedule contains a duplicate operation")
        options = instance.jobs[record.job][record.operation]
        allowed = {option.machine: option.duration for option in options}
        if record.machine not in allowed:
            raise ValueError("Operation assigned to an ineligible machine")
        if record.duration != allowed[record.machine]:
            raise ValueError("Scheduled duration differs from the processing time")
        records[key] = record
        busy[record.machine].append(record)
        loads[record.machine] += record.duration

    for job, operations in enumerate(instance.jobs):
        for operation in range(len(operations)):
            if (job, operation) not in records:
                raise ValueError("Schedule is missing an operation")
            if operation and records[job, operation].start < records[job, operation - 1].end:
                raise ValueError("Job precedence is violated")
    for operations in busy:
        ordered = sorted(operations, key=lambda op: (op.start, op.end, op.job, op.operation))
        if any(left.end > right.start for left, right in zip(ordered, ordered[1:])):
            raise ValueError("Operations overlap on a machine")
    recorded_loads = tuple(_integer(v, "Machine load") for v in schedule.machine_loads)
    if recorded_loads != tuple(loads):
        raise ValueError("Recorded machine loads disagree with the scheduled operations")
    expected_completion = tuple(
        records[job, len(operations) - 1].end for job, operations in enumerate(instance.jobs)
    )
    if schedule.completion_times != expected_completion:
        raise ValueError("Recorded completion times disagree with the scheduled operations")
    expected_twt = sum(
        weight * max(0, completion - due)
        for completion, due, weight in zip(
            expected_completion, instance.due_dates, instance.weights
        )
    )
    if schedule.twt != expected_twt:
        raise ValueError("Recorded TWT disagrees with completion times and due dates")
