"""Paper-facing analysis for synthetic BBOB benchmark results."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

from .plotting import find_observation_files, load_observations_csv
from .utils import ensure_dir, write_json


_BBOB_RE = re.compile(r"^bbob_f(?P<function>\d+)_d(?P<dim>\d+)_i(?P<instance>\d+)$")


def bootstrap_median_ci(
    values: Sequence[float],
    *,
    samples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    data = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=float)
    if data.size == 0:
        return math.nan, math.nan, math.nan
    if data.size == 1:
        value = float(data[0])
        return value, value, value
    rng = np.random.default_rng(seed)
    draws = rng.choice(data, size=(int(samples), data.size), replace=True)
    medians = np.median(draws, axis=1)
    return (
        float(np.median(data)),
        float(np.quantile(medians, 0.025)),
        float(np.quantile(medians, 0.975)),
    )


def vargha_delaney_a12(left: Sequence[float], right: Sequence[float]) -> float:
    """Probability that a lower-is-better draw from left beats right."""

    wins = ties = 0
    for left_value, right_value in zip(left, right):
        if left_value < right_value:
            wins += 1
        elif left_value == right_value:
            ties += 1
    return (wins + 0.5 * ties) / max(1, min(len(left), len(right)))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    count = len(p_values)
    order = sorted(range(count), key=lambda index: float(p_values[index]))
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(p_values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def generate_synthetic_report(results_dir: str | Path) -> list[Path]:
    root = Path(results_dir)
    with (root / "summary.json").open("r", encoding="utf-8") as file:
        rows = json.load(file)
    rows = [dict(row) for row in rows if str(row.get("benchmark_name", "")).startswith("bbob_")]
    if not rows:
        raise ValueError(f"No BBOB rows found under {root}")

    output = ensure_dir(root / "paper_report")
    generated: list[Path] = []
    expected_methods: list[str] = []
    expected_per_method: int | None = None
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        manifest_config = manifest.get("config", {}) if isinstance(manifest, Mapping) else {}
        if isinstance(manifest_config, Mapping):
            expected_methods = [str(value) for value in manifest_config.get("methods", [])]
            benchmarks = manifest_config.get("benchmarks", [])
            seeds = manifest_config.get("seeds", [])
            if isinstance(benchmarks, list) and isinstance(seeds, list):
                expected_per_method = len(benchmarks) * len(seeds)
    aggregate_rows = _aggregate_methods(
        rows,
        expected_methods=expected_methods,
        expected_per_method=expected_per_method,
    )
    aggregate_path = output / "aggregate_metrics.csv"
    _write_csv(aggregate_path, aggregate_rows)
    generated.append(aggregate_path)

    stats = _statistical_tests(rows)
    stats_path = output / "statistical_tests.json"
    write_json(stats_path, stats)
    generated.append(stats_path)
    pairwise_path = output / "pairwise_wilcoxon.csv"
    _write_csv(pairwise_path, stats["pairwise"])
    generated.append(pairwise_path)

    generated.extend(
        [
            _plot_final_distribution(rows, output / "final_log_regret.png"),
            _plot_average_ranks(stats["average_ranks"], output / "average_ranks.png"),
            _plot_heatmap(rows, output / "function_method_heatmap.png"),
            _plot_target_ecdf(rows, output / "target_ecdf_1e-4.png"),
            _plot_convergence(root, output / "convergence_log_regret.png"),
            _plot_llm_cost(rows, output / "llm_cost.png"),
        ]
    )
    return generated


def _aggregate_methods(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_methods: Sequence[str] = (),
    expected_per_method: int | None = None,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    methods = sorted({str(row["method"]) for row in rows} | {str(method) for method in expected_methods})
    for method in methods:
        group = [row for row in rows if str(row["method"]) == method]
        item: dict[str, object] = {
            "method": method,
            "runs": len(group),
            "expected_runs": expected_per_method,
            "completion_rate": (
                len(group) / expected_per_method
                if expected_per_method
                else math.nan
            ),
        }
        for metric in ("log10_final_regret", "relative_regret_auc", "target_auc"):
            values = [_number(row.get(metric)) for row in group]
            finite = [value for value in values if value is not None]
            median, low, high = bootstrap_median_ci(finite)
            item[f"{metric}_median"] = median
            item[f"{metric}_q1"] = float(np.quantile(finite, 0.25)) if finite else math.nan
            item[f"{metric}_q3"] = float(np.quantile(finite, 0.75)) if finite else math.nan
            item[f"{metric}_iqr"] = item[f"{metric}_q3"] - item[f"{metric}_q1"]
            item[f"{metric}_ci_low"] = low
            item[f"{metric}_ci_high"] = high
        item["llm_calls"] = sum(_number(row.get("llm_calls")) or 0.0 for row in group)
        item["llm_total_tokens"] = sum(_number(row.get("llm_total_tokens")) or 0.0 for row in group)
        item["llm_latency_seconds"] = sum(_number(row.get("llm_latency_seconds")) or 0.0 for row in group)
        result.append(item)
    return result


def _statistical_tests(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    blocks: dict[tuple[str, int], dict[str, float]] = {}
    methods = sorted({str(row["method"]) for row in rows})
    for row in rows:
        value = _number(row.get("log10_final_regret"))
        if value is None:
            continue
        key = (str(row["benchmark_name"]), int(row.get("seed", 0)))
        blocks.setdefault(key, {})[str(row["method"])] = value
    complete = [block for block in blocks.values() if all(method in block for method in methods)]
    matrix = np.asarray([[block[method] for method in methods] for block in complete], dtype=float)
    if len(complete) >= 2 and len(methods) >= 3:
        statistic, p_value = friedmanchisquare(*(matrix[:, index] for index in range(len(methods))))
    else:
        statistic, p_value = math.nan, math.nan
    ranks = np.asarray([rankdata(row, method="average") for row in matrix], dtype=float)
    average_ranks = {
        method: float(np.mean(ranks[:, index])) if ranks.size else math.nan
        for index, method in enumerate(methods)
    }

    pairwise: list[dict[str, object]] = []
    reference_name = "wmbo_llm" if "wmbo_llm" in methods else "wmbo_rule" if "wmbo_rule" in methods else None
    if reference_name is not None:
        reference = methods.index(reference_name)
        raw_p: list[float] = []
        comparisons: list[tuple[str, np.ndarray]] = []
        for index, method in enumerate(methods):
            if method == reference_name or not matrix.size:
                continue
            try:
                test = wilcoxon(matrix[:, reference], matrix[:, index], zero_method="zsplit")
                p = float(test.pvalue)
            except ValueError:
                p = 1.0
            raw_p.append(p)
            comparisons.append((method, matrix[:, index]))
        adjusted = holm_adjust(raw_p)
        for (method, baseline), raw, corrected in zip(comparisons, raw_p, adjusted):
            pairwise.append(
                {
                    "reference": reference_name,
                    "baseline": method,
                    "paired_blocks": len(complete),
                    "p_value": raw,
                    "holm_p_value": corrected,
                    "a12_lower_is_better": vargha_delaney_a12(matrix[:, reference], baseline),
                }
            )
    return {
        "complete_blocks": len(complete),
        "methods": methods,
        "friedman_statistic": float(statistic),
        "friedman_p_value": float(p_value),
        "average_ranks": average_ranks,
        "pairwise": pairwise,
    }


def _plot_final_distribution(rows: Sequence[Mapping[str, object]], path: Path) -> Path:
    methods = sorted({str(row["method"]) for row in rows})
    values = [
        [float(row["log10_final_regret"]) for row in rows if row["method"] == method]
        for method in methods
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(values, tick_labels=methods, showfliers=False)
    ax.set_ylabel("Final log10 simple regret")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_average_ranks(ranks: Mapping[str, object], path: Path) -> Path:
    ordered = sorted(((str(key), float(value)) for key, value in ranks.items()), key=lambda item: item[1])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([item[0] for item in ordered], [item[1] for item in ordered])
    ax.set_ylabel("Average rank (lower is better)")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_heatmap(rows: Sequence[Mapping[str, object]], path: Path) -> Path:
    methods = sorted({str(row["method"]) for row in rows})
    functions = list(range(1, 25))
    matrix = np.full((len(functions), len(methods)), np.nan)
    for function_id in functions:
        for method_index, method in enumerate(methods):
            values = []
            for row in rows:
                match = _BBOB_RE.fullmatch(str(row["benchmark_name"]))
                if match and int(match.group("function")) == function_id and str(row["method"]) == method:
                    value = _number(row.get("log10_final_regret"))
                    if value is not None:
                        values.append(value)
            if values:
                matrix[function_id - 1, method_index] = float(np.median(values))
    fig, ax = plt.subplots(figsize=(13, 9))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(methods)), methods, rotation=35, ha="right")
    ax.set_yticks(range(24), [f"F{value}" for value in functions])
    fig.colorbar(image, ax=ax, label="Median final log10 regret")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_target_ecdf(rows: Sequence[Mapping[str, object]], path: Path) -> Path:
    methods = sorted({str(row["method"]) for row in rows})
    fig, ax = plt.subplots(figsize=(9, 6))
    for method in methods:
        fractions = []
        for row in rows:
            budget = int(row["num_evaluations"])
            hit = row.get("evals_to_target_1em4")
            fractions.append(float(hit) / budget if hit is not None else 1.05)
        x = np.sort(np.asarray(fractions, dtype=float))
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where="post", label=method)
    ax.set_xlim(0, 1.06)
    ax.set_xlabel("Fraction of evaluation budget to regret <= 1e-4")
    ax.set_ylabel("ECDF")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_convergence(root: Path, path: Path) -> Path:
    curves: dict[str, list[np.ndarray]] = {}
    grid = np.linspace(0.0, 1.0, 101)
    for observation_file in find_observation_files(root):
        rows = load_observations_csv(observation_file)
        if not rows or not str(rows[0].get("benchmark", "")).startswith("bbob_"):
            continue
        method = str(rows[0].get("method", "unknown"))
        regrets = np.asarray([max(_number(row.get("simple_regret")) or 0.0, 1.0e-12) for row in rows])
        source = np.linspace(0.0, 1.0, len(regrets))
        curves.setdefault(method, []).append(np.interp(grid, source, np.log10(regrets)))
    fig, ax = plt.subplots(figsize=(10, 6))
    for method, method_curves in sorted(curves.items()):
        matrix = np.asarray(method_curves)
        median = np.median(matrix, axis=0)
        low, high = _bootstrap_curve_ci(matrix, samples=10_000, seed=0)
        ax.plot(grid, median, label=method)
        ax.fill_between(grid, low, high, alpha=0.12)
    ax.set_xlabel("Fraction of evaluation budget")
    ax.set_ylabel("log10 simple regret")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _bootstrap_curve_ci(
    matrix: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if matrix.shape[0] <= 1:
        curve = matrix[0].copy()
        return curve, curve
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, matrix.shape[0], size=(int(samples), matrix.shape[0]))
    low = np.empty(matrix.shape[1], dtype=float)
    high = np.empty(matrix.shape[1], dtype=float)
    for column in range(matrix.shape[1]):
        medians = np.median(matrix[indices, column], axis=1)
        low[column], high[column] = np.quantile(medians, [0.025, 0.975])
    return low, high


def _plot_llm_cost(rows: Sequence[Mapping[str, object]], path: Path) -> Path:
    methods = sorted({str(row["method"]) for row in rows})
    calls = [sum(_number(row.get("llm_calls")) or 0.0 for row in rows if row["method"] == method) for method in methods]
    tokens = [sum(_number(row.get("llm_total_tokens")) or 0.0 for row in rows if row["method"] == method) for method in methods]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(methods, calls)
    axes[0].set_ylabel("LLM calls")
    axes[1].bar(methods, tokens)
    axes[1].set_ylabel("LLM total tokens")
    for axis in axes:
        axis.tick_params(axis="x", rotation=35)
        axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    ensure_dir(path.parent)
    fieldnames = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _number(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


__all__ = [
    "bootstrap_median_ci",
    "generate_synthetic_report",
    "holm_adjust",
    "vargha_delaney_a12",
]
