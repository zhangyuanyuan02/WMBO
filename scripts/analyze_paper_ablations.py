"""Paired analysis for the V5.1 RC3 paper ablation experiments.

Example:
    python scripts/analyze_paper_ablations.py \
      --baseline results/synthetic_bbob_portfolio_v51_rc3_formal \
      --ablations results/paper_ablations/v51_all_ablations \
      --output results/paper_ablations/analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import wilcoxon

LANDSCAPE_METHODS = (
    "v51_ablate_smoothness",
    "v51_ablate_modality",
    "v51_ablate_curvature",
    "v51_ablate_geometry",
    "v51_ablate_identifiability",
)
EXPECTED_METHODS = LANDSCAPE_METHODS + (
    "v51_no_landscape",
    "v51_route_context_only",
    "v51_route_static_balanced",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--ablations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser.parse_args()


def _iter_run_summaries(root: Path) -> Iterable[dict[str, object]]:
    for path in sorted(root.rglob("summary.json")):
        if "seed_" not in str(path):
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict) and "benchmark_name" in value and "method" in value:
            yield value


def _key(row: dict[str, object]) -> tuple[str, int]:
    return str(row["benchmark_name"]), int(row["seed"])


def _function_and_dim(name: str) -> tuple[int, int]:
    match = re.search(r"bbob_f(\d+)_d(\d+)_i(\d+)", name)
    if not match:
        raise ValueError(f"Cannot parse BBOB name: {name}")
    return int(match.group(1)), int(match.group(2))


def _family(function: int) -> str:
    if 1 <= function <= 5:
        return "separable"
    if 6 <= function <= 9:
        return "moderate_conditioning"
    if 10 <= function <= 14:
        return "high_conditioning"
    if 15 <= function <= 19:
        return "multimodal_global_structure"
    return "multimodal_weak_structure"


def _log_regret_at_fraction(row: dict[str, object], fraction: float) -> float:
    curve = np.asarray(row.get("regret_curve", []), dtype=float)
    if curve.size == 0:
        return float("nan")
    index = max(0, min(curve.size - 1, int(math.ceil(fraction * curve.size)) - 1))
    return float(math.log10(max(float(curve[index]), 1.0e-12)))


def _bootstrap_median_ci(
    differences: np.ndarray,
    *,
    samples: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if differences.size == 0:
        return float("nan"), float("nan")
    medians = np.empty(samples, dtype=float)
    for index in range(samples):
        draw = rng.choice(differences, size=differences.size, replace=True)
        medians[index] = float(np.median(draw))
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(medians, alpha)), float(np.quantile(medians, 1.0 - alpha))


def _paired_stats(
    baseline_rows: list[dict[str, object]],
    ablation_rows: list[dict[str, object]],
    *,
    metric: str,
    bootstrap_samples: int,
    confidence: float,
    rng: np.random.Generator,
) -> dict[str, object]:
    base = {_key(row): row for row in baseline_rows}
    ablate = {_key(row): row for row in ablation_rows}
    keys = sorted(set(base).intersection(ablate))
    if not keys:
        raise ValueError("No paired runs found.")

    def value(row: dict[str, object]) -> float:
        if metric.startswith("checkpoint_"):
            fraction = float(metric.split("_", 1)[1])
            return _log_regret_at_fraction(row, fraction)
        return float(row[metric])

    base_values = np.asarray([value(base[key]) for key in keys], dtype=float)
    ablation_values = np.asarray([value(ablate[key]) for key in keys], dtype=float)
    valid = np.isfinite(base_values) & np.isfinite(ablation_values)
    base_values = base_values[valid]
    ablation_values = ablation_values[valid]
    differences = ablation_values - base_values
    tolerance = 1.0e-12
    wins = int(np.sum(differences < -tolerance))
    ties = int(np.sum(np.abs(differences) <= tolerance))
    losses = int(np.sum(differences > tolerance))
    nonzero = differences[np.abs(differences) > tolerance]
    p_value = (
        float(wilcoxon(nonzero, alternative="two-sided").pvalue)
        if nonzero.size
        else 1.0
    )
    ci_low, ci_high = _bootstrap_median_ci(
        differences,
        samples=bootstrap_samples,
        confidence=confidence,
        rng=rng,
    )
    return {
        "n_pairs": int(differences.size),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "median_delta": float(np.median(differences)),
        "mean_delta": float(np.mean(differences)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "wilcoxon_p": p_value,
        "baseline_median": float(np.median(base_values)),
        "ablation_median": float(np.median(ablation_values)),
    }


def _holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p_value) in enumerate(ordered):
        candidate = min(1.0, (m - rank) * float(p_value))
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    baseline_rows = list(_iter_run_summaries(args.baseline))
    ablation_rows = list(_iter_run_summaries(args.ablations))
    if not baseline_rows:
        raise SystemExit(f"No baseline run summaries under {args.baseline}")
    if not ablation_rows:
        raise SystemExit(f"No ablation run summaries under {args.ablations}")

    by_method: dict[str, list[dict[str, object]]] = {}
    for row in ablation_rows:
        by_method.setdefault(str(row["method"]), []).append(row)

    rng = np.random.default_rng(args.seed)
    metrics = (
        "log10_final_regret",
        "relative_regret_auc",
        "target_auc",
        "checkpoint_0.25",
        "checkpoint_0.50",
        "checkpoint_0.75",
        "checkpoint_1.00",
    )
    overall_rows: list[dict[str, object]] = []
    for method in EXPECTED_METHODS:
        if method not in by_method:
            continue
        for metric in metrics:
            result = _paired_stats(
                baseline_rows,
                by_method[method],
                metric=metric,
                bootstrap_samples=args.bootstrap_samples,
                confidence=args.confidence,
                rng=rng,
            )
            overall_rows.append({"method": method, "metric": metric, **result})

    family_rows: list[dict[str, object]] = []
    for method in EXPECTED_METHODS:
        if method not in by_method:
            continue
        functions = sorted({_function_and_dim(str(row["benchmark_name"]))[0] for row in baseline_rows})
        for family in sorted({_family(function) for function in functions}):
            base_subset = [
                row for row in baseline_rows
                if _family(_function_and_dim(str(row["benchmark_name"]))[0]) == family
            ]
            ablate_subset = [
                row for row in by_method[method]
                if _family(_function_and_dim(str(row["benchmark_name"]))[0]) == family
            ]
            for metric in ("log10_final_regret", "relative_regret_auc"):
                result = _paired_stats(
                    base_subset,
                    ablate_subset,
                    metric=metric,
                    bootstrap_samples=args.bootstrap_samples,
                    confidence=args.confidence,
                    rng=rng,
                )
                family_rows.append(
                    {"method": method, "family": family, "metric": metric, **result}
                )

    holm_rows: list[dict[str, object]] = []
    for metric in ("log10_final_regret", "relative_regret_auc"):
        raw = {
            row["method"]: float(row["wilcoxon_p"])
            for row in overall_rows
            if row["metric"] == metric and row["method"] in LANDSCAPE_METHODS
        }
        adjusted = _holm_adjust(raw)
        for method in LANDSCAPE_METHODS:
            if method in raw:
                holm_rows.append(
                    {
                        "metric": metric,
                        "method": method,
                        "raw_p": raw[method],
                        "holm_p": adjusted[method],
                    }
                )

    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "overall_paired.csv", overall_rows)
    _write_csv(args.output / "family_paired.csv", family_rows)
    _write_csv(args.output / "landscape_holm.csv", holm_rows)
    manifest = {
        "baseline": str(args.baseline),
        "ablations": str(args.ablations),
        "methods_found": sorted(by_method),
        "baseline_runs": len(baseline_rows),
        "ablation_runs": len(ablation_rows),
        "bootstrap_samples": args.bootstrap_samples,
        "confidence": args.confidence,
    }
    (args.output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"wrote {len(overall_rows)} overall rows to {args.output / 'overall_paired.csv'}")
    print(f"wrote {len(family_rows)} family rows to {args.output / 'family_paired.csv'}")
    print(f"wrote {len(holm_rows)} Holm rows to {args.output / 'landscape_holm.csv'}")


if __name__ == "__main__":
    main()
