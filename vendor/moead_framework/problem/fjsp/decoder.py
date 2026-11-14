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

    def __post_init__(self):
        object.__setattr__(self, "operations", tuple(self.operations))
        object.__setattr__(self, "machine_loads", tuple(self.machine_loads))

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
    def objectives(self) -> tuple[int, int, int]:
        return self.makespan, self.total_workload, self.max_machine_workload


def _earliest_slot(busy: list[ScheduledOperation], ready: int, duration: int) -> tuple[int, int]:
    start = ready
    for index, operation in enumerate(busy):
        if start + duration <= operation.start:
            return start, index
        start = max(start, operation.end)
    return start, len(busy)


def decode(instance: FJSPInstance, chromosome: Chromosome) -> Schedule:
    """Decode OS/MS without changing its genes or any instance state.

    Preemption, setup/transport times, release dates and machine downtime are
    not part of this static FJSP model. Equality at interval endpoints is valid,
    including zero-duration operations.
    """
    chromosome.validate(instance)
    next_operation = [0] * instance.job_count
    job_ready = [0] * instance.job_count
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
    return Schedule(tuple(scheduled), tuple(loads))


def validate_schedule(instance: FJSPInstance, schedule: Schedule) -> None:
    """Check the resulting schedule independently of the insertion algorithm.

    Every operation must appear exactly once on an eligible machine, with its
    original processing time. Precedence and chronological machine intervals
    require previous.end <= next.start, also for zero-duration operations.
    """
    if len(schedule.operations) != instance.operation_count:
        raise ValueError("Schedule must contain every operation exactly once")
    if len(schedule.machine_loads) != instance.machine_count:
        raise ValueError("Machine-load vector has the wrong length")
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
