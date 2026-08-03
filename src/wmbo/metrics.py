"""Metric utilities for optimisation runs."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence


REGRET_FLOOR = 1.0e-12
BBOB_TARGETS = tuple(10.0**exponent for exponent in range(2, -9, -1))


@dataclass(frozen=True)
class RunSummary:
    """Summary metrics for one optimisation run.

    Inputs:
        benchmark_name: Name of the benchmark.
        method: Optimisation method.
        seed: Random seed.
        final_best: Best objective value at the end of the run.
        final_regret: Simple regret at the end of the run, if known.
        num_evaluations: Number of objective evaluations used.
        metadata: Optional extra metrics.

    Output:
        Stored or aggregated by the runner.
    """

    benchmark_name: str
    method: str
    seed: int
    final_best: float | None
    final_regret: float | None
    num_evaluations: int
    metadata: Mapping[str, object] = field(default_factory=dict)


def cumulative_best(values: Sequence[float], minimise: bool = True) -> list[float]:
    """Compute the best objective value after each evaluation.

    Inputs:
        values: Objective values in evaluation order.
        minimise: Whether lower values are better.

    Output:
        Best-so-far curve.
    """

    best_values: list[float] = []
    best: float | None = None
    for value in values:
        current = float(value)
        if best is None:
            best = current
        elif minimise:
            best = min(best, current)
        else:
            best = max(best, current)
        best_values.append(float(best))
    return best_values


def simple_regret(best_values: Sequence[float], optimum_value: float | None) -> list[float | None]:
    """Compute simple regret against a known optimum.

    Inputs:
        best_values: Best-so-far objective values.
        optimum_value: Known global optimum value, if available.

    Output:
        Simple-regret curve. Values are ``None`` when optimum is unknown.
    """

    if optimum_value is None:
        return [None for _ in best_values]
    optimum = float(optimum_value)
    return [max(0.0, float(best) - optimum) for best in best_values]


def synthetic_run_metrics(
    best_values: Sequence[float],
    optimum_value: float,
    initial_samples: int,
    targets: Sequence[float] = BBOB_TARGETS,
) -> dict[str, object]:
    """Compute paper-facing anytime and target metrics for a synthetic run."""

    regrets = [max(0.0, float(value) - float(optimum_value)) for value in best_values]
    if not regrets:
        return {}
    initial_index = min(max(int(initial_samples), 1), len(regrets)) - 1
    denominator = max(regrets[initial_index], REGRET_FLOOR)
    post_initial = regrets[initial_index:]
    relative = [min(1.0, max(0.0, value / denominator)) for value in post_initial]
    final_regret = regrets[-1]
    metrics: dict[str, object] = {
        "log10_final_regret": math.log10(max(final_regret, REGRET_FLOOR)),
        "relative_regret_auc": sum(relative) / len(relative),
        "initial_design_regret": regrets[initial_index],
        "post_initial_evaluations": len(post_initial),
    }
    target_hits = 0
    target_progress = 0.0
    for target in targets:
        label = _target_label(float(target))
        first = next((index for index, regret in enumerate(regrets, start=1) if regret <= target), None)
        metrics[f"evals_to_target_{label}"] = first
        metrics[f"success_target_{label}"] = first is not None
        target_hits += int(first is not None)
        if first is not None:
            target_progress += 1.0 - (first - 1) / max(1, len(regrets))
    metrics["target_success_rate"] = target_hits / len(tuple(targets))
    metrics["target_auc"] = target_progress / len(tuple(targets))
    return metrics


def _target_label(target: float) -> str:
    exponent = int(round(math.log10(target)))
    return f"1e{exponent:+d}".replace("+", "p").replace("-", "m")


def summarise_run(
    benchmark_name: str,
    method: str,
    seed: int,
    values: Sequence[float],
    optimum_value: float | None = None,
    metadata: Mapping[str, object] | None = None,
    initial_samples: int = 1,
) -> RunSummary:
    """Summarise one optimisation run.

    Inputs:
        benchmark_name: Name of the benchmark.
        method: Optimisation method.
        seed: Random seed.
        values: Objective values in evaluation order.
        optimum_value: Known global optimum, if available.
        metadata: Optional domain-specific run metrics.

    Output:
        ``RunSummary`` containing final best value and regret.
    """

    objective_values = [float(value) for value in values]
    best_curve = cumulative_best(objective_values, minimise=True)
    regret_curve = simple_regret(best_curve, optimum_value)
    final_best = best_curve[-1] if best_curve else None
    final_regret = regret_curve[-1] if regret_curve else None
    extra = dict(metadata or {})
    if optimum_value is not None:
        extra.update(
            synthetic_run_metrics(
                best_curve,
                optimum_value=float(optimum_value),
                initial_samples=initial_samples,
            )
        )
    return RunSummary(
        benchmark_name=benchmark_name,
        method=method,
        seed=int(seed),
        final_best=final_best,
        final_regret=final_regret,
        num_evaluations=len(objective_values),
        metadata={
            "best_curve": best_curve,
            "regret_curve": regret_curve,
            "first_value": objective_values[0] if objective_values else None,
            "last_value": objective_values[-1] if objective_values else None,
            **extra,
        },
    )


__all__ = [
    "RunSummary",
    "cumulative_best",
    "simple_regret",
    "synthetic_run_metrics",
    "BBOB_TARGETS",
    "REGRET_FLOOR",
    "summarise_run",
]
