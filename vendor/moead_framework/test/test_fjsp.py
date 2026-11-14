"""Independent checks for FJSP data, representation and schedule decoding."""

from dataclasses import FrozenInstanceError, replace
from itertools import permutations, product
from pathlib import Path
import random

import numpy as np
import pytest

from moead_framework.problem.fjsp import (
    Chromosome, FJSPInstance, MachineOption, Schedule, ScheduledOperation,
    decode, load_instance, parse_instance, random_chromosome, validate_schedule,
)


ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def fattahi():
    return load_instance(ROOT / "instance/FattahiMehrabadJolai2007/fattahi1.txt")


def test_parse_preserves_option_order_and_converts_physical_machine_ids():
    instance = parse_instance("\n1\t5\t1.5\n\n1  2  5 4  2 7\n", name="unsorted")
    assert instance.name == "unsorted"
    assert instance.metadata == 1.5
    assert (instance.job_count, instance.machine_count, instance.operation_count) == (1, 5, 1)
    assert instance.jobs[0][0] == (MachineOption(4, 4), MachineOption(1, 7))
    first = decode(instance, Chromosome((0,), (0,)))
    second = decode(instance, Chromosome((0,), (1,)))
    assert first.operations[0].machine == 4  # MS=0 means the first option, not machine 0.
    assert second.operations[0].machine == 1
    assert first.machine_loads == (0, 0, 0, 0, 4)


def test_load_supports_utf8_bom_and_optional_header_metadata(tmp_path):
    path = tmp_path / "bom.txt"
    path.write_text("1 1\r\n1 1 1 0\r\n", encoding="utf-8-sig")
    instance = load_instance(path)
    assert instance.name == "bom"
    assert instance.metadata is None
    assert instance.jobs[0][0][0].duration == 0


@pytest.mark.parametrize("text", [
    "", "1", "1 1 2 3\n1 1 1 2", "x 1\n1 1 1 2",
    "0 1", "1 0\n1 1 1 2", "1 1 nan\n1 1 1 2", "1 1 nope\n1 1 1 2",
    "2 1\n1 1 1 2", "1 1\n1 1 1 2\n1 1 1 2", "1 1\n0",
    "1 1\n2 1 1 2", "1 1\n1 0", "1 1\n1 -1", "1 1\n1 2 1 2",
    "1 1\n1 1 0 2", "1 1\n1 1 2 2", "1 1\n1 1 1 -2",
    "1 1\n1 1 1 2 extra", "1 1\n1 1 1 2 7", "1 1\n1 2 1 2 1 3",
    "1 1\n1 1 1 2.5",
])
def test_parser_rejects_malformed_or_inconsistent_input(text):
    with pytest.raises(ValueError):
        parse_instance(text)


def test_models_do_not_alias_mutable_input_lists():
    options = [MachineOption(0, 2)]
    jobs = [[options]]
    instance = FJSPInstance(1, jobs)
    options.append(MachineOption(0, 9))
    jobs.clear()
    assert instance.jobs == (((MachineOption(0, 2),),),)
    sequence, choices = [0], [0]
    chromosome = Chromosome(sequence, choices)
    sequence[0] = 99
    choices[0] = 99
    assert chromosome.to_vector() == (0, 0)
    with pytest.raises(FrozenInstanceError):
        chromosome.os = (99,)


@pytest.mark.parametrize("machine,duration", [(-1, 2), (0, -1), (0.0, 1), (0, True)])
def test_machine_options_require_nonnegative_integers(machine, duration):
    with pytest.raises(ValueError):
        MachineOption(machine, duration)


def test_encoding_round_trip_accepts_numpy_integer_vector(fattahi):
    chromosome = Chromosome((0, 1, 0, 1), (1, 1, 0, 0))
    vector = np.array(chromosome.to_vector(), dtype=np.int64)
    assert Chromosome.from_vector(fattahi, vector) == chromosome
    with pytest.raises(ValueError):
        Chromosome.from_vector(fattahi, vector.astype(float))
    with pytest.raises(ValueError):
        Chromosome.from_vector(fattahi, vector.reshape(4, 2))


@pytest.mark.parametrize("os,ms", [
    ((0, 1), (0, 0, 0, 0)), ((0, 0, 1, 1), (0, 0)),
    ((0, 0, 0, 1), (0, 0, 0, 0)), ((0, 0, 1, 2), (0, 0, 0, 0)),
    ((0, -1, 0, 1), (0, 0, 0, 0)), ((0, 1, 0, 1), (2, 0, 0, 0)),
    ((0, 1, 0, 1), (-1, 0, 0, 0)), ((False, 1, 0, 1), (0, 0, 0, 0)),
    ((0, 1, 0, 1), (0.5, 0, 0, 0)),
])
def test_invalid_chromosomes_are_rejected_before_decoding(fattahi, os, ms):
    with pytest.raises(ValueError):
        decode(fattahi, Chromosome(os, ms))


def test_seeded_random_encoding_is_valid_and_reproducible(fattahi):
    first, second = random.Random(17), random.Random(17)
    for _ in range(10):
        chromosome = random_chromosome(fattahi, first)
        assert chromosome == random_chromosome(fattahi, second)
        chromosome.validate(fattahi)
        assert Chromosome.from_vector(fattahi, chromosome.to_vector()) == chromosome


def test_hand_calculated_fattahi_schedule_and_all_three_objectives(fattahi):
    chromosome = Chromosome((0, 1, 0, 1), (1, 1, 0, 0))
    schedule = decode(fattahi, chromosome)
    assert schedule.operations == (
        ScheduledOperation(0, 0, 1, 0, 37),
        ScheduledOperation(1, 0, 0, 0, 45),
        ScheduledOperation(0, 1, 1, 37, 61),
        ScheduledOperation(1, 1, 0, 45, 66),
    )
    assert schedule.machine_loads == (66, 61)
    assert schedule.objectives == (66, 127, 66)
    validate_schedule(fattahi, schedule)
    decode(fattahi, random_chromosome(fattahi, random.Random(42)))
    assert decode(fattahi, chromosome) == schedule  # No hidden decoder state.


def test_decoder_inserts_before_the_first_existing_machine_operation():
    instance = parse_instance("2 2\n2 1 1 5 1 2 3\n1 1 2 4\n")
    schedule = decode(instance, Chromosome((0, 0, 1), (0, 0, 0)))
    assert schedule.operations[-1] == ScheduledOperation(1, 0, 1, 0, 4)
    assert schedule.objectives == (8, 12, 7)
    validate_schedule(instance, schedule)


def test_decoder_fills_an_internal_gap_at_exact_endpoint_boundaries():
    instance = parse_instance("3 3\n2 1 1 3 1 2 2\n2 1 1 8 1 2 2\n2 1 3 5 1 2 6\n")
    schedule = decode(instance, Chromosome((0, 0, 1, 1, 2, 2), (0,) * 6))
    assert schedule.operations[-1] == ScheduledOperation(2, 1, 1, 5, 11)
    assert schedule.objectives == (13, 26, 11)
    validate_schedule(instance, schedule)


def test_ms_offsets_use_job_operation_order_with_unequal_job_lengths():
    instance = parse_instance("2 3\n1 2 3 4 1 8\n2 2 2 2 3 5 1 1 3\n")
    assert instance.operations_per_job == (1, 2)
    assert instance.operation_offsets == (0, 1)
    schedule = decode(instance, Chromosome((1, 0, 1), (1, 0, 0)))
    assert [op.machine for op in schedule.operations] == [1, 0, 0]
    assert [op.duration for op in schedule.operations] == [2, 8, 3]
    assert schedule.objectives == (11, 13, 11)
    validate_schedule(instance, schedule)


def test_zero_duration_operations_can_share_endpoints_but_respect_precedence():
    instance = parse_instance("1 1\n3 1 1 0 1 1 3 1 1 0\n")
    schedule = decode(instance, Chromosome((0, 0, 0), (0, 0, 0)))
    assert [(op.start, op.end) for op in schedule.operations] == [(0, 0), (0, 3), (3, 3)]
    assert schedule.objectives == (3, 3, 3)
    validate_schedule(instance, schedule)


def test_zero_duration_operation_is_not_inserted_inside_a_busy_interval():
    instance = parse_instance("2 2\n1 1 1 10\n2 1 2 5 1 1 0\n")
    schedule = decode(instance, Chromosome((0, 1, 1), (0, 0, 0)))
    assert schedule.operations[-1] == ScheduledOperation(1, 1, 0, 10, 10)
    assert schedule.objectives == (10, 15, 10)
    validate_schedule(instance, schedule)


def test_earliest_start_matches_integer_time_search_on_small_random_instances():
    rng = random.Random(23)
    for _ in range(30):
        jobs = tuple(tuple(tuple(MachineOption(machine, rng.randrange(6))
                                 for machine in rng.sample(range(3), 2))
                           for _ in range(3)) for _ in range(3))
        instance = FJSPInstance(3, jobs)
        schedule = decode(instance, random_chromosome(instance, rng))
        validate_schedule(instance, schedule)
        previous = [[] for _ in range(3)]
        job_end = [0] * 3
        for operation in schedule.operations:
            earliest = next(t for t in range(job_end[operation.job], operation.start + 1) if all(
                t + operation.duration <= busy.start or busy.end <= t
                for busy in previous[operation.machine]
            ))
            assert operation.start == earliest
            previous[operation.machine].append(operation)
            job_end[operation.job] = operation.end


def test_validator_checks_structure_duration_precedence_and_loads(fattahi):
    valid = decode(fattahi, Chromosome((0, 1, 0, 1), (1, 1, 0, 0)))
    operations = valid.operations
    invalid = [
        replace(valid, operations=operations[:-1]),
        replace(valid, operations=(operations[0],) + operations[1:3] + (operations[0],)),
        replace(valid, machine_loads=(66,)),
        replace(valid, machine_loads=(66, 99)),
        replace(valid, operations=(replace(operations[0], start=-1, end=36),) + operations[1:]),
        replace(valid, operations=(replace(operations[0], end=38),) + operations[1:]),
        replace(valid, operations=(replace(operations[0], job=99),) + operations[1:]),
        replace(valid, operations=(replace(operations[0], operation=99),) + operations[1:]),
        replace(valid, operations=(replace(operations[0], machine=99),) + operations[1:]),
        replace(valid, operations=operations[:2] + (replace(operations[2], start=0, end=24),) + operations[3:]),
    ]
    for schedule in invalid:
        with pytest.raises(ValueError):
            validate_schedule(fattahi, schedule)
    validate_schedule(fattahi, replace(valid, operations=tuple(reversed(operations))))


def test_validator_rejects_capacity_overlap_and_ineligible_machine():
    instance = parse_instance("2 2\n1 1 1 2\n1 1 1 2\n")
    valid = decode(instance, Chromosome((0, 1), (0, 0)))
    with pytest.raises(ValueError, match="overlap"):
        validate_schedule(instance, replace(valid, operations=(valid.operations[0], ScheduledOperation(1, 0, 0, 1, 3))))
    with pytest.raises(ValueError, match="ineligible"):
        validate_schedule(instance, replace(valid, operations=(replace(valid.operations[0], machine=1), valid.operations[1])))


def _pareto(points):
    unique = set(points)
    return {p for p in unique if not any(
        all(a <= b for a, b in zip(q, p)) and any(a < b for a, b in zip(q, p))
        for q in unique
    )}


def _exhaustive_machine_order_front(instance):
    """Independent tiny-instance oracle: machine permutations + longest paths.

    This does not use decode(), gap insertion, Schedule or its objective fields.
    The reference instance has strictly positive processing times.
    """
    options = [operation for job in instance.jobs for operation in job]
    offsets = instance.operation_offsets
    job_edges = [(offsets[j] + o - 1, offsets[j] + o)
                 for j, job in enumerate(instance.jobs) for o in range(1, len(job))]
    points = set()
    for choices in product(*(range(len(op)) for op in options)):
        selected = [op[choice] for op, choice in zip(options, choices)]
        per_machine = [[i for i, op in enumerate(selected) if op.machine == m]
                       for m in range(instance.machine_count)]
        loads = [sum(selected[i].duration for i in group) for group in per_machine]
        for orders in product(*(permutations(group) for group in per_machine)):
            predecessors = [set() for _ in options]
            for before, after in job_edges + [(a, b) for order in orders for a, b in zip(order, order[1:])]:
                predecessors[after].add(before)
            pending, ends = set(range(len(options))), {}
            while pending:
                ready = [i for i in pending if not (predecessors[i] & pending)]
                if not ready:
                    break  # Cyclic machine/job order is infeasible for positive durations.
                for i in ready:
                    ends[i] = max((ends[p] for p in predecessors[i]), default=0) + selected[i].duration
                    pending.remove(i)
            if not pending:
                points.add((max(ends.values()), sum(loads), max(loads)))
    return _pareto(points)


def test_exhaustive_tiny_instance_matches_independent_three_objective_oracle(fattahi):
    sequences = set(permutations((0, 0, 1, 1)))
    points = set()
    count = 0
    for sequence in sequences:
        for choices in product(range(2), repeat=4):
            schedule = decode(fattahi, Chromosome(sequence, choices))
            validate_schedule(fattahi, schedule)
            points.add(schedule.objectives)
            count += 1
    assert count == 96
    assert min(point[0] for point in points) == 66
    assert _pareto(points) == _exhaustive_machine_order_front(fattahi)


@pytest.fixture(scope="module")
def repository_instances():
    paths = sorted((ROOT / "instance").rglob("*.txt"))
    assert len(paths) >= 400
    return [(path, load_instance(path)) for path in paths]


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_all_repository_instances_decode_to_valid_schedules(repository_instances, seed):
    rng = random.Random(seed)
    for path, instance in repository_instances:
        chromosome = random_chromosome(instance, rng)
        schedule = decode(instance, chromosome)
        try:
            validate_schedule(instance, schedule)
        except ValueError as error:
            pytest.fail(f"{path.name}, seed {seed}: {error}")
        assert schedule.total_workload == sum(op.duration for op in schedule.operations), path.name
        assert schedule.makespan == max(op.end for op in schedule.operations), path.name


def test_repository_zero_processing_times_are_preserved(repository_instances):
    orb7 = [instance for path, instance in repository_instances if path.name == "orb7.txt"]
    assert len(orb7) == 4
    for instance in orb7:
        assert any(option.duration == 0 for job in instance.jobs for op in job for option in op)
