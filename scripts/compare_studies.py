"""So sánh 2 lần chạy study: tính IGD+ và C(A,B) dùng empirical reference front chung."""

import argparse
import csv
from collections import defaultdict
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import re
from statistics import mean, stdev
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))

from moead_framework.tool.fjsp_experiments import (  # noqa: E402
    igd_plus, independent_pareto_front, set_coverage,
)

SPEC = spec_from_file_location("baseline_validation_runner", ROOT / "scripts/run_baseline_validation.py")
RUNNER = module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)

RUN_NAME = re.compile(r"(.+)_seed(\d+)\.json")


def load_study(study_dir):
    """Đọc manifest.json và toàn bộ runs/*.json từ một thư mục study."""
    study_dir = Path(study_dir).resolve()
    manifest_path = study_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing manifest.json in {study_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bounds = manifest["bounds"]
    cases_by_instance = defaultdict(list)
    runs_dir = study_dir / "runs"
    if not runs_dir.is_dir():
        raise ValueError(f"Missing runs/ directory in {study_dir}")
    for path in sorted(runs_dir.glob("*.json")):
        match = RUN_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected run filename: {path.name}")
        identifier, seed = match.group(1), int(match.group(2))
        case = RUNNER.read_case(path, identifier, seed)
        cases_by_instance[identifier].append(case)
    for identifier in cases_by_instance:
        cases_by_instance[identifier].sort(key=lambda case: case["seed"])
    return dict(cases_by_instance), bounds, manifest


def build_reference_front(cases_a, cases_b):
    """Gộp objective_vectors từ cả 2 bên, lọc Pareto → P*_ref chung."""
    all_pts = [pt for cases in (cases_a, cases_b) for case in cases for pt in case["objective_vectors"]]
    return list(independent_pareto_front(all_pts))


def pooled_pareto_front(cases):
    points = [pt for case in cases for pt in case["objective_vectors"]]
    return list(independent_pareto_front(points))


def compare_instance(identifier, cases_a, cases_b, bounds):
    """Tính IGD+ per-seed và C(A,B)/C(B,A) trên front gộp cho một instance."""
    ref_front = build_reference_front(cases_a, cases_b)
    igd_a = [igd_plus(case["objective_vectors"], ref_front, bounds) for case in cases_a]
    igd_b = [igd_plus(case["objective_vectors"], ref_front, bounds) for case in cases_b]
    pf_a = pooled_pareto_front(cases_a)
    pf_b = pooled_pareto_front(cases_b)
    return {
        "instance": identifier,
        "ref_front_size": len(ref_front),
        "runs_a": len(cases_a),
        "runs_b": len(cases_b),
        "igd_plus_a_mean": mean(igd_a) if igd_a else 0.0,
        "igd_plus_a_std": stdev(igd_a) if len(igd_a) > 1 else 0.0,
        "igd_plus_b_mean": mean(igd_b) if igd_b else 0.0,
        "igd_plus_b_std": stdev(igd_b) if len(igd_b) > 1 else 0.0,
        "coverage_a_over_b": set_coverage(pf_a, pf_b),
        "coverage_b_over_a": set_coverage(pf_b, pf_a),
    }, [
        {"instance": identifier, "study": "a", "seed": case["seed"], "igd_plus": value}
        for case, value in zip(cases_a, igd_a)
    ] + [
        {"instance": identifier, "study": "b", "seed": case["seed"], "igd_plus": value}
        for case, value in zip(cases_b, igd_b)
    ]


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows, study_a, study_b):
    lines = [
        "# So sánh Study A vs Study B (IGD+ và C(A,B))",
        "",
        f"- Study A: `{study_a}`",
        f"- Study B: `{study_b}`",
        "",
        "IGD+ càng nhỏ càng tốt. C(A,B) = tỉ lệ điểm của B bị A chi phối yếu.",
        "",
        "| Instance | Ref | IGD+ A mean +/- SD | IGD+ B mean +/- SD | C(A,B) | C(B,A) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['instance']} | {row['ref_front_size']} | "
            f"{row['igd_plus_a_mean']:.6f} +/- {row['igd_plus_a_std']:.6f} | "
            f"{row['igd_plus_b_mean']:.6f} +/- {row['igd_plus_b_std']:.6f} | "
            f"{row['coverage_a_over_b']:.4f} | {row['coverage_b_over_a']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_studies(study_a_dir, study_b_dir, output_dir):
    cases_a, bounds_a, _ = load_study(study_a_dir)
    cases_b, bounds_b, _ = load_study(study_b_dir)

    instances_a = set(cases_a)
    instances_b = set(cases_b)
    if instances_a != instances_b:
        missing_a = sorted(instances_b - instances_a)
        missing_b = sorted(instances_a - instances_b)
        raise ValueError(
            "Study A and Study B must cover the same instances; "
            f"only in A: {missing_b}; only in B: {missing_a}"
        )

    for identifier in sorted(instances_a):
        if bounds_a.get(identifier) != bounds_b.get(identifier):
            raise ValueError(f"Bounds differ between studies for instance {identifier}")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    per_seed_rows = []
    for identifier in sorted(instances_a):
        row, seed_rows = compare_instance(
            identifier, cases_a[identifier], cases_b[identifier], bounds_a[identifier],
        )
        summary_rows.append(row)
        per_seed_rows.extend(seed_rows)

    write_csv(output_dir / "comparison.csv", summary_rows)
    write_csv(output_dir / "per_seed_igd_plus.csv", per_seed_rows)
    write_markdown(
        output_dir / "comparison.md", summary_rows,
        Path(study_a_dir).resolve(), Path(study_b_dir).resolve(),
    )
    print(f"Wrote comparison for {len(summary_rows)} instances to {output_dir}", flush=True)
    return summary_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-a", type=Path, required=True, help="Output dir of Study A (baseline)")
    parser.add_argument("--study-b", type=Path, required=True, help="Output dir of Study B (improved)")
    parser.add_argument("--output", type=Path, required=True, help="Directory for comparison reports")
    args = parser.parse_args()
    compare_studies(args.study_a, args.study_b, args.output)


if __name__ == "__main__":
    main()
