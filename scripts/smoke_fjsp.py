"""Decode and validate one FJSP chromosome; no optimization algorithm is run."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random

from moead_framework.problem.fjsp import (
    Chromosome, decode, load_instance, random_chromosome, validate_schedule,
)


def integer_list(value):
    try:
        return tuple(int(token.strip()) for token in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated integer genes") from error


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path,
                        default=root / "instance/FattahiMehrabadJolai2007/fattahi1.txt")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--os", type=integer_list, help="Zero-based job IDs, comma-separated")
    parser.add_argument("--ms", type=integer_list, help="Zero-based local choice indices, comma-separated")
    parser.add_argument("--show-operations", action="store_true")
    args = parser.parse_args()
    if (args.os is None) != (args.ms is None):
        parser.error("Provide both --os and --ms, or omit both for a seeded random chromosome")
    instance = load_instance(args.instance)
    chromosome = (Chromosome(args.os, args.ms) if args.os is not None
                  else random_chromosome(instance, random.Random(args.seed)))
    schedule = decode(instance, chromosome)
    validate_schedule(instance, schedule)
    result = {
        "instance": instance.name,
        "mode": "FJSP decoding only (not MOEA/D)",
        "index_base": 0,
        "jobs": instance.job_count,
        "machines": instance.machine_count,
        "operations": instance.operation_count,
        "objectives": dict(zip(("makespan", "total_workload", "max_machine_workload"), schedule.objectives)),
        "machine_loads": schedule.machine_loads,
        "feasible": True,
    }
    if args.show_operations:
        result["os"] = chromosome.os
        result["ms"] = chromosome.ms
        result["schedule"] = [dict(asdict(op), duration=op.duration) for op in schedule.operations]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
