"""So sánh nhiều study (baseline và các biến thể khởi tạo) trên cùng protocol.

Mỗi instance: HV cuối và theo checkpoint, IGD+ trên reference front gộp từ TẤT CẢ
study, C-metric gộp, Cmax/TWT tốt nhất, kích thước archive và thời gian. Mỗi biến
thể được kiểm định với baseline bằng Mann–Whitney U hai phía (p chính xác qua
hoán vị), ghi +/=/- ở mức alpha.

Ví dụ (một dòng):
    python scripts/compare_initializers.py --baseline baseline=results/baseline_main
        --study hybrid=results/qinit_main --output results/compare_qinit
"""

import argparse
import csv
import json
from math import comb
from pathlib import Path
from statistics import mean, stdev
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vendor"))

from compare_studies import load_study  # noqa: E402
from moead_framework.tool.fjsp_experiments import (  # noqa: E402
    igd_plus, independent_pareto_front, set_coverage,
)


def rank_sum_p_value(a, b):
    """Two-sided exact permutation p-value of the Mann–Whitney rank-sum statistic (midranks)."""
    pooled = list(a) + list(b)
    if len(set(pooled)) == 1:
        return 1.0
    order = sorted(range(len(pooled)), key=lambda i: pooled[i])
    ranks = [0.0] * len(pooled)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and pooled[order[j + 1]] == pooled[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    # Exact null distribution of the rank sum of any n of the N items, by dynamic
    # programming over doubled (integer) midranks: counts[k][s] = #subsets of size k, sum s.
    n, size = len(a), len(pooled)
    doubled = [int(round(2 * r)) for r in ranks]
    counts = [dict() for _ in range(n + 1)]
    counts[0][0] = 1
    for value in doubled:
        for k in range(min(n, size) - 1, -1, -1):
            for s, c in counts[k].items():
                counts[k + 1][s + value] = counts[k + 1].get(s + value, 0) + c
    expected = n * (size + 1)  # doubled
    observed = abs(sum(doubled[:n]) - expected)
    total = sum(counts[n].values())
    extreme = sum(c for s, c in counts[n].items() if abs(s - expected) >= observed)
    return extreme / total


def _sd(values):
    return stdev(values) if len(values) > 1 else 0.0


def verdict(variant, baseline, higher_is_better, alpha):
    p = rank_sum_p_value(variant, baseline)
    if p >= alpha:
        return "=", p
    better = mean(variant) > mean(baseline) if higher_is_better else mean(variant) < mean(baseline)
    return ("+" if better else "-"), p


def pooled_front(cases):
    return list(independent_pareto_front([pt for case in cases for pt in case["objective_vectors"]]))


# Settings allowed to differ between compared studies: only the initialization strategy.
INITIALIZER_KEYS = ("initializer", "q_init")


def shared_protocol(manifest):
    """Protocol fields that must match for a fair comparison (budget, operators, seeds, HV)."""
    protocol = manifest["protocol"]
    algorithm = {k: v for k, v in protocol["algorithm"].items() if k not in INITIALIZER_KEYS}
    return {"algorithm": algorithm, "seeds": protocol["seeds"],
            "checkpoints": protocol["checkpoints"], "hv_reference": protocol["hv_reference"]}


def compare(studies, baseline_name, output_dir, alpha=0.05):
    loaded = {name: load_study(path) for name, path in studies.items()}
    names = list(studies)
    instances = sorted(loaded[baseline_name][0])
    reference_protocol = shared_protocol(loaded[baseline_name][2])
    for name in names:
        cases, bounds, manifest = loaded[name]
        protocol = shared_protocol(manifest)
        if protocol != reference_protocol:
            differing = sorted(key for key in set(protocol["algorithm"]) | set(reference_protocol["algorithm"])
                               if protocol["algorithm"].get(key) != reference_protocol["algorithm"].get(key))
            differing += [key for key in ("seeds", "checkpoints", "hv_reference")
                          if protocol[key] != reference_protocol[key]]
            raise ValueError(f"Study {name} uses a different protocol than {baseline_name}: {', '.join(differing)}")
        if sorted(cases) != instances:
            raise ValueError(f"Study {name} covers different instances than {baseline_name}")
        for identifier in instances:
            if bounds[identifier] != loaded[baseline_name][1][identifier]:
                raise ValueError(f"Bounds differ for {identifier} in study {name}")
            seeds = [c["seed"] for c in cases[identifier]]
            if seeds != [c["seed"] for c in loaded[baseline_name][0][identifier]]:
                raise ValueError(f"Seeds differ for {identifier} in study {name}")

    rows, plot_data = [], {}
    tally = {name: {"hv": {"+": 0, "=": 0, "-": 0}, "igd": {"+": 0, "=": 0, "-": 0}}
             for name in names if name != baseline_name}
    for identifier in instances:
        bounds = loaded[baseline_name][1][identifier]
        by_study = {name: loaded[name][0][identifier] for name in names}
        reference = list(independent_pareto_front(
            [pt for cases in by_study.values() for case in cases for pt in case["objective_vectors"]]))
        fronts = {name: pooled_front(cases) for name, cases in by_study.items()}
        metrics = {}
        for name, cases in by_study.items():
            metrics[name] = {
                "hv": [c["final_hv"] for c in cases],
                "hv0": [c["initial_hv"] for c in cases],
                "igd": [igd_plus(c["objective_vectors"], reference, bounds) for c in cases],
                "cmax": [c["min_objectives"][0] for c in cases],
                "twt": [c["min_objectives"][1] for c in cases],
                "archive": [c["archive_size"] for c in cases],
                "seconds": [c["elapsed_seconds"] for c in cases],
            }
            plot_data.setdefault(identifier, {})[name] = [
                (snap["nfe"], mean(c["snapshots"][k]["hv"] for c in cases))
                for k, snap in enumerate(cases[0]["snapshots"])]
        for name in names:
            m = metrics[name]
            row = {
                "instance": identifier, "study": name, "runs": len(m["hv"]), "ref_front_size": len(reference),
                "hv_initial_mean": mean(m["hv0"]),
                "hv_mean": mean(m["hv"]), "hv_std": _sd(m["hv"]),
                "igd_plus_mean": mean(m["igd"]), "igd_plus_std": _sd(m["igd"]),
                "cmax_best": min(m["cmax"]), "cmax_mean": mean(m["cmax"]),
                "twt_best": min(m["twt"]), "twt_mean": mean(m["twt"]),
                "archive_mean": mean(m["archive"]), "seconds_mean": mean(m["seconds"]),
                "hv_vs_baseline": "", "hv_p": "", "igd_vs_baseline": "", "igd_p": "",
                "c_variant_over_baseline": "", "c_baseline_over_variant": "",
            }
            if name != baseline_name:
                base = metrics[baseline_name]
                row["hv_vs_baseline"], row["hv_p"] = verdict(m["hv"], base["hv"], True, alpha)
                row["igd_vs_baseline"], row["igd_p"] = verdict(m["igd"], base["igd"], False, alpha)
                row["c_variant_over_baseline"] = set_coverage(fronts[name], fronts[baseline_name])
                row["c_baseline_over_variant"] = set_coverage(fronts[baseline_name], fronts[name])
                tally[name]["hv"][row["hv_vs_baseline"]] += 1
                tally[name]["igd"][row["igd_vs_baseline"]] += 1
            rows.append(row)

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"baseline": baseline_name, "studies": {n: str(Path(p).resolve()) for n, p in studies.items()},
              "alpha": alpha, "tally": tally, "rows": rows}
    (output_dir / "comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    initial_nfe = loaded[baseline_name][0][instances[0]][0]["snapshots"][0]["nfe"]
    write_markdown(output_dir / "comparison.md", rows, names, baseline_name, tally, alpha, studies, initial_nfe)
    plot(output_dir / "convergence.png", plot_data, names)
    shown = output_dir.relative_to(ROOT).as_posix() if output_dir.is_relative_to(ROOT) else output_dir.name
    print(f"Wrote comparison of {len(names)} studies on {len(instances)} instances to {shown}", flush=True)
    return report


def write_markdown(path, rows, names, baseline_name, tally, alpha, studies, initial_nfe):
    runs = rows[0]["runs"]
    lines = [
        "# So sánh baseline và khởi tạo cải tiến", "",
        *[f"- `{name}`: `{Path(p).as_posix()}`" for name, p in studies.items()], "",
        f"Mỗi ô là trung bình trên {runs} seed. HV càng lớn càng tốt; IGD+ (reference front gộp "
        "từ mọi study) càng nhỏ càng tốt. `+`/`-`: tốt hơn/kém hơn baseline có ý nghĩa "
        f"(Mann–Whitney U hai phía, p < {alpha}); `=`: không khác biệt có ý nghĩa.", "",
    ]
    # Smallest two-sided p-value an exact rank-sum test can reach with `runs` seeds per study.
    if 2 / comb(2 * runs, runs) >= alpha:
        lines += [f"**Lưu ý:** với {runs} seed mỗi study, p nhỏ nhất đạt được là "
                  f"{2 / comb(2 * runs, runs):.3f} ≥ {alpha}, nên mọi so sánh đều là `=`. "
                  "Cần ít nhất 4 seed.", ""]
    lines += [
        "## Tổng hợp thắng/hòa/thua so với baseline", "",
        "| Biến thể | HV (+/=/-) | IGD+ (+/=/-) |", "| --- | :---: | :---: |",
    ]
    for name, t in tally.items():
        lines.append(f"| {name} | {t['hv']['+']}/{t['hv']['=']}/{t['hv']['-']} | "
                     f"{t['igd']['+']}/{t['igd']['=']}/{t['igd']['-']} |")
    lines += ["", "## Chi tiết theo instance", "",
              f"| Instance | Study | HV@{initial_nfe} | HV cuối ± SD | IGD+ ± SD | Cmax tốt nhất / TB | TWT TB | C(biến thể, base) | C(base, biến thể) |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        hv_mark = f" {row['hv_vs_baseline']}" if row["hv_vs_baseline"] else ""
        igd_mark = f" {row['igd_vs_baseline']}" if row["igd_vs_baseline"] else ""
        cov = ("" if row["study"] == baseline_name else
               f"{row['c_variant_over_baseline']:.2f} | {row['c_baseline_over_variant']:.2f}")
        lines.append(
            f"| {row['instance']} | {row['study']} | {row['hv_initial_mean']:.4f} | "
            f"{row['hv_mean']:.4f} ± {row['hv_std']:.4f}{hv_mark} | "
            f"{row['igd_plus_mean']:.4f} ± {row['igd_plus_std']:.4f}{igd_mark} | "
            f"{row['cmax_best']} / {row['cmax_mean']:.1f} | {row['twt_mean']:.0f} | "
            + (cov if cov else " | ") + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(path, plot_data, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    instances = list(plot_data)
    cols = min(3, len(instances))
    rows = -(-len(instances) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), squeeze=False)
    for ax, identifier in zip(axes.flat, instances):
        for name in names:
            points = plot_data[identifier][name]
            ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", markersize=3, label=name)
        ax.set_title(identifier, fontsize=10)
        ax.set_xlabel("NFE")
        ax.set_ylabel("HV (mean)")
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(instances):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _named_path(text):
    name, sep, path = text.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError("Expected NAME=DIR")
    return name, Path(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=_named_path, required=True, help="NAME=DIR of the baseline study")
    parser.add_argument("--study", type=_named_path, action="append", required=True, help="NAME=DIR (repeatable)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    studies = dict([args.baseline] + args.study)
    if len(studies) != 1 + len(args.study):
        parser.error("Study names must be unique")
    compare(studies, args.baseline[0], args.output, args.alpha)


if __name__ == "__main__":
    main()
