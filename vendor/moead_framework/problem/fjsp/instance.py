"""Immutable FJSP input data and the text format used by this project's instances."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from pathlib import Path


def _integer(value, label: str) -> int:
    """Accept integer scalars (including NumPy integers), without rounding floats."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer, received {value!r}")
    return int(value)


@dataclass(frozen=True)
class MachineOption:
    machine: int  # Zero-based physical machine ID, not an MS option index.
    duration: int

    def __post_init__(self):
        object.__setattr__(self, "machine", _integer(self.machine, "Machine ID"))
        object.__setattr__(self, "duration", _integer(self.duration, "Processing time"))
        if self.machine < 0 or self.duration < 0:
            raise ValueError("Machine IDs and processing times must be nonnegative")


@dataclass(frozen=True)
class FJSPInstance:
    machine_count: int
    jobs: tuple[tuple[tuple[MachineOption, ...], ...], ...]
    name: str = "<memory>"
    metadata: float | None = None

    def __post_init__(self):
        count = _integer(self.machine_count, "Machine count")
        if count <= 0:
            raise ValueError("An instance must have at least one machine")
        jobs = tuple(tuple(tuple(options) for options in job) for job in self.jobs)
        if not jobs or any(not job for job in jobs):
            raise ValueError("An instance must have jobs, each with at least one operation")
        for j, job in enumerate(jobs):
            for o, options in enumerate(job):
                if not options:
                    raise ValueError(f"Job {j}, operation {o} has no eligible machines")
                if any(not isinstance(option, MachineOption) for option in options):
                    raise ValueError("Operation alternatives must be MachineOption objects")
                machines = [option.machine for option in options]
                if any(machine >= count for machine in machines):
                    raise ValueError(f"Job {j}, operation {o} has an out-of-range machine")
                if len(set(machines)) != len(machines):
                    raise ValueError(f"Job {j}, operation {o} repeats an eligible machine")
        if self.metadata is not None and not isfinite(self.metadata):
            raise ValueError("Instance metadata must be finite")
        object.__setattr__(self, "machine_count", count)
        object.__setattr__(self, "jobs", jobs)

    @property
    def job_count(self) -> int:
        return len(self.jobs)

    @property
    def operations_per_job(self) -> tuple[int, ...]:
        return tuple(len(job) for job in self.jobs)

    @property
    def operation_count(self) -> int:
        return sum(self.operations_per_job)

    @property
    def operation_offsets(self) -> tuple[int, ...]:
        """Starting MS position for each job, in fixed job/operation order."""
        offsets = []
        offset = 0
        for job in self.jobs:
            offsets.append(offset)
            offset += len(job)
        return tuple(offsets)


def parse_instance(text: str, name: str = "<memory>") -> FJSPInstance:
    """Read n_jobs n_machines [metadata], followed by one nonblank line per job.

    Machine IDs in the file are one-based. Alternative order is preserved, so
    MS choice k consistently means the kth listed alternative for that operation.
    """
    lines = [(n, line.split()) for n, line in enumerate(text.splitlines(), 1) if line.strip()]
    if not lines:
        raise ValueError(f"{name}: empty instance")
    line_number, header = lines[0]
    if len(header) not in (2, 3):
        raise ValueError(f"{name}:{line_number}: expected n_jobs n_machines [metadata]")
    try:
        job_count, machine_count = map(int, header[:2])
        metadata = float(header[2]) if len(header) == 3 else None
    except ValueError as error:
        raise ValueError(f"{name}:{line_number}: invalid header") from error
    if job_count <= 0 or machine_count <= 0:
        raise ValueError(f"{name}:{line_number}: job and machine counts must be positive")
    if len(lines) != job_count + 1:
        raise ValueError(f"{name}: expected {job_count} job lines, found {len(lines) - 1}")

    jobs = []
    for j, (line_number, tokens) in enumerate(lines[1:]):
        context = f"{name}:{line_number}: job {j}"
        try:
            values = [int(token) for token in tokens]
        except ValueError as error:
            raise ValueError(f"{context}: operation data must contain integers") from error
        operation_count = values[0]
        if operation_count <= 0:
            raise ValueError(f"{context}: operation count must be positive")
        cursor = 1
        operations = []
        for o in range(operation_count):
            if cursor >= len(values):
                raise ValueError(f"{context}: missing operation {o}")
            alternatives = values[cursor]
            cursor += 1
            if alternatives <= 0 or cursor + 2 * alternatives > len(values):
                raise ValueError(f"{context}: invalid or incomplete alternatives for operation {o}")
            options = []
            for _ in range(alternatives):
                machine, duration = values[cursor:cursor + 2]
                cursor += 2
                if not 1 <= machine <= machine_count or duration < 0:
                    raise ValueError(f"{context}: invalid machine or processing time for operation {o}")
                options.append(MachineOption(machine - 1, duration))
            if len({option.machine for option in options}) != len(options):
                raise ValueError(f"{context}: duplicate machine in operation {o}")
            operations.append(tuple(options))
        if cursor != len(values):
            raise ValueError(f"{context}: unexpected trailing operation data")
        jobs.append(tuple(operations))
    return FJSPInstance(machine_count, tuple(jobs), name=name, metadata=metadata)


def load_instance(path: str | Path) -> FJSPInstance:
    path = Path(path)
    return parse_instance(path.read_text(encoding="utf-8-sig"), name=path.stem)
