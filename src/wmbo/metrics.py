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

    raise NotImplementedError("Cumulative-best metric is not implemented yet.")


def simple_regret(best_values: Sequence[float], optimum_value: float | None) -> list[float | None]:
    """Compute simple regret against a known optimum.

    Inputs:
        best_values: Best-so-far objective values.
        optimum_value: Known global optimum value, if available.

    Output:
        Simple-regret curve. Values are ``None`` when optimum is unknown.
    """

    raise NotImplementedError("Simple regret metric is not implemented yet.")


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

    raise NotImplementedError("Run summary metric is not implemented yet.")
