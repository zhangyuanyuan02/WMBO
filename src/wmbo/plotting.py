"""Plotting interfaces for future experiment analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence


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

    raise NotImplementedError("Best-so-far plotting is not implemented yet.")


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

    raise NotImplementedError("Simple-regret plotting is not implemented yet.")


def plot_summary_table(summary: Sequence[Mapping[str, object]], output_path: str | Path) -> Path:
    """Create a compact visual summary of run-level metrics.

    Inputs:
        summary: Sequence of run summary dictionaries.
        output_path: File path for the generated figure.

    Output:
        Path to the saved figure.
    """

    raise NotImplementedError("Summary plotting is not implemented yet.")
