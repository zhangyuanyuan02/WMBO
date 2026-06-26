"""Metric interfaces for optimisation runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


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
        best_values.append(best)
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
    return [float(best) - optimum for best in best_values]


def summarise_run(
    benchmark_name: str,
    method: str,
    seed: int,
    values: Sequence[float],
    optimum_value: float | None = None,
) -> RunSummary:
    """Summarise one optimisation run.

    Inputs:
        benchmark_name: Name of the benchmark.
        method: Optimisation method.
        seed: Random seed.
        values: Objective values in evaluation order.
        optimum_value: Known global optimum, if available.

    Output:
        ``RunSummary`` containing final best value and regret.
    """

    best_values = cumulative_best(values, minimise=True)
    regrets = simple_regret(best_values, optimum_value)
    return RunSummary(
        benchmark_name=benchmark_name,
        method=method,
        seed=seed,
        final_best=best_values[-1] if best_values else None,
        final_regret=regrets[-1] if regrets else None,
        num_evaluations=len(values),
        metadata={
            "cumulative_best": best_values,
            "simple_regret": regrets,
        },
    )
