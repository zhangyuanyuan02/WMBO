"""Plotting utilities for saved benchmark outputs."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .utils import ensure_dir


ObservationRow = dict[str, object]
SummaryRow = dict[str, object]


def plot_best_so_far(
    steps: Sequence[int],
    best_values: Sequence[float],
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    """Plot a best-so-far optimisation curve.

    Inputs:
        steps: Evaluation indices.
        best_values: Best objective value after each step.
        output_path: File path for the generated figure.
        title: Optional plot title.

    Output:
        Path to the saved figure.
    """

    return _plot_curve(
        steps=steps,
        values=best_values,
        output_path=output_path,
        ylabel="Best objective value so far",
        title=title or "Best-so-far curve",
    )


def plot_simple_regret(
    steps: Sequence[int],
    regret_values: Sequence[float],
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    """Plot a simple-regret curve.

    Inputs:
        steps: Evaluation indices.
        regret_values: Simple regret after each step.
        output_path: File path for the generated figure.
        title: Optional plot title.

    Output:
        Path to the saved figure.
    """

    return _plot_curve(
        steps=steps,
        values=regret_values,
        output_path=output_path,
        ylabel="Simple regret",
        title=title or "Simple-regret curve",
        log_y=True,
    )


def plot_summary_table(summary: Sequence[Mapping[str, object]], output_path: str | Path) -> Path:
    """Create a compact visual summary of run-level metrics.

    Inputs:
        summary: Sequence of run summary dictionaries.
        output_path: File path for the generated figure.

    Output:
        Path to the saved figure.
    """

    rows = [dict(row) for row in summary]
    if not rows:
        raise ValueError("summary must contain at least one row.")

    output = Path(output_path)
    ensure_dir(output.parent)

    headers = ["benchmark", "method", "seed", "final_best", "final_regret"]
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                str(row.get("benchmark_name", row.get("benchmark", ""))),
                str(row.get("method", "")),
                str(row.get("seed", "")),
                _format_float(row.get("final_best")),
                _format_float(row.get("final_regret")),
            ]
        )

    height = max(2.0, 0.38 * (len(table_rows) + 1))
    fig, ax = plt.subplots(figsize=(10, height))
    ax.axis("off")
    table = ax.table(cellText=table_rows, colLabels=headers, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


def load_summary_csv(path: str | Path) -> list[SummaryRow]:
    """Load the suite-level summary CSV written by the runner.

    Input:
        path: Path to ``summary.csv``.

    Output:
        List of summary dictionaries with numeric fields converted where possible.
    """

    with Path(path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return [_coerce_row(row) for row in reader]


def load_observations_csv(path: str | Path) -> list[ObservationRow]:
    """Load one run's observation CSV.

    Input:
        path: Path to ``observations.csv``.

    Output:
        List of observation dictionaries with numeric fields converted where possible.
    """

    with Path(path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return [_coerce_row(row) for row in reader]


def find_observation_files(results_dir: str | Path) -> list[Path]:
    """Find all ``observations.csv`` files under a runner output directory.

    Input:
        results_dir: Root output directory produced by ``run_benchmark.py``.

    Output:
        Sorted list of observation CSV paths.
    """

    root = Path(results_dir)
    return sorted(root.glob("*/*/seed_*/observations.csv"))


def plot_convergence_from_observations(
    observations: Sequence[Mapping[str, object]],
    output_path: str | Path,
    metric: str = "best_y",
    title: str | None = None,
) -> Path:
    """Plot mean convergence curves from loaded observation rows.

    Inputs:
        observations: Rows loaded from one or more ``observations.csv`` files.
        output_path: Destination image path.
        metric: Column to aggregate, usually ``best_y`` or ``simple_regret``.
        title: Optional plot title.

    Output:
        Path to the saved figure.
    """

    rows = [dict(row) for row in observations]
    if not rows:
        raise ValueError("observations must contain at least one row.")

    grouped = _aggregate_by_method_and_step(rows, metric)
    output = Path(output_path)
    ensure_dir(output.parent)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method in sorted(grouped):
        points = grouped[method]
        steps = [step for step, _ in points]
        values = [value for _, value in points]
        ax.plot(steps, values, marker="o", linewidth=1.5, label=method)

    ax.set_xlabel("Evaluation step")
    ax.set_ylabel(_metric_label(metric))
    ax.set_title(title or f"Convergence by method ({metric})")
    if metric == "simple_regret" and all(value > 0 for points in grouped.values() for _, value in points):
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_final_metric_from_summary(
    summary: Sequence[Mapping[str, object]],
    output_path: str | Path,
    metric: str = "final_best",
    title: str | None = None,
) -> Path:
    """Plot mean final metric by method from suite summary rows.

    Inputs:
        summary: Rows loaded from ``summary.csv``.
        output_path: Destination image path.
        metric: Summary column to aggregate, usually ``final_best`` or ``final_regret``.
        title: Optional plot title.

    Output:
        Path to the saved figure.
    """

    rows = [dict(row) for row in summary]
    if not rows:
        raise ValueError("summary must contain at least one row.")

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        method = str(row.get("method", "unknown"))
        value = _optional_float(row.get(metric))
        if value is not None:
            grouped[method].append(value)
    if not grouped:
        raise ValueError(f"summary does not contain numeric values for {metric!r}.")

    methods = sorted(grouped)
    values = [sum(grouped[method]) / len(grouped[method]) for method in methods]

    output = Path(output_path)
    ensure_dir(output.parent)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(methods, values)
    ax.set_xlabel("Method")
    ax.set_ylabel(_metric_label(metric))
    ax.set_title(title or f"Mean {metric} by method")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_results_directory(results_dir: str | Path, output_dir: str | Path | None = None) -> list[Path]:
    """Generate standard plots for a runner output directory.

    Inputs:
        results_dir: Directory containing ``summary.csv`` and per-run observation files.
        output_dir: Optional directory for figures. Defaults to ``results_dir / 'figures'``.

    Output:
        List of generated figure paths.
    """

    root = Path(results_dir)
    figure_dir = ensure_dir(output_dir or root / "figures")
    generated: list[Path] = []

    summary_path = root / "summary.csv"
    if summary_path.exists():
        summary = load_summary_csv(summary_path)
        generated.append(plot_final_metric_from_summary(summary, figure_dir / "final_best.png", metric="final_best"))
        if any(_optional_float(row.get("final_regret")) is not None for row in summary):
            generated.append(plot_final_metric_from_summary(summary, figure_dir / "final_regret.png", metric="final_regret"))
        generated.append(plot_summary_table(summary, figure_dir / "summary_table.png"))

    benchmark_rows: dict[str, list[ObservationRow]] = defaultdict(list)
    for observation_file in find_observation_files(root):
        for row in load_observations_csv(observation_file):
            benchmark = str(row.get("benchmark", observation_file.parents[2].name))
            benchmark_rows[benchmark].append(row)

    for benchmark, rows in sorted(benchmark_rows.items()):
        generated.append(
            plot_convergence_from_observations(
                rows,
                figure_dir / f"{benchmark}_best_y.png",
                metric="best_y",
                title=f"{benchmark}: best objective value so far",
            )
        )
        if any(_optional_float(row.get("simple_regret")) is not None for row in rows):
            generated.append(
                plot_convergence_from_observations(
                    rows,
                    figure_dir / f"{benchmark}_simple_regret.png",
                    metric="simple_regret",
                    title=f"{benchmark}: simple regret",
                )
            )
    return generated


def _plot_curve(
    steps: Sequence[int],
    values: Sequence[float],
    output_path: str | Path,
    ylabel: str,
    title: str,
    log_y: bool = False,
) -> Path:
    if len(steps) != len(values):
        raise ValueError("steps and values must have the same length.")
    if not steps:
        raise ValueError("steps and values must not be empty.")

    output = Path(output_path)
    ensure_dir(output.parent)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([int(step) for step in steps], [float(value) for value in values], marker="o", linewidth=1.5)
    ax.set_xlabel("Evaluation step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_y and all(float(value) > 0 for value in values):
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def _aggregate_by_method_and_step(rows: Sequence[Mapping[str, object]], metric: str) -> dict[str, list[tuple[int, float]]]:
    buckets: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        method = str(row.get("method", "unknown"))
        step = _optional_int(row.get("step"))
        value = _optional_float(row.get(metric))
        if step is None or value is None:
            continue
        buckets[method][step].append(value)

    aggregated: dict[str, list[tuple[int, float]]] = {}
    for method, by_step in buckets.items():
        aggregated[method] = [
            (step, sum(values) / len(values))
            for step, values in sorted(by_step.items())
            if values
        ]
    if not aggregated:
        raise ValueError(f"observations do not contain numeric values for {metric!r}.")
    return aggregated


def _coerce_row(row: Mapping[str, object]) -> dict[str, object]:
    coerced: dict[str, object] = {}
    for key, value in row.items():
        if value is None:
            coerced[str(key)] = None
            continue
        text = str(value).strip()
        if text == "" or text.lower() == "none":
            coerced[str(key)] = None
        elif key in {"step", "seed", "num_evaluations"}:
            coerced[str(key)] = _optional_int(text)
        elif key in {"y", "best_y", "simple_regret", "final_best", "final_regret"}:
            coerced[str(key)] = _optional_float(text)
        else:
            coerced[str(key)] = value
    return coerced


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _format_float(value: object) -> str:
    numeric = _optional_float(value)
    if numeric is None:
        return ""
    return f"{numeric:.6g}"


def _metric_label(metric: str) -> str:
    labels = {
        "best_y": "Best objective value so far",
        "simple_regret": "Simple regret",
        "final_best": "Mean final best objective value",
        "final_regret": "Mean final regret",
    }
    return labels.get(metric, metric.replace("_", " ").title())


__all__ = [
    "load_observations_csv",
    "load_summary_csv",
    "find_observation_files",
    "plot_best_so_far",
    "plot_convergence_from_observations",
    "plot_final_metric_from_summary",
    "plot_results_directory",
    "plot_simple_regret",
    "plot_summary_table",
]
