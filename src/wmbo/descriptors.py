"""Landscape descriptor interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

Vector = Sequence[float]
Matrix = Sequence[Vector]


@dataclass(frozen=True)
class LandscapeDescriptor:
    """Structured summary of the observed objective landscape.

    Inputs:
        dim: Search-space dimensionality.
        num_observations: Number of evaluated points.
        best_y: Best objective value observed so far.
        y_range: Difference between worst and best observed values.
        smoothness: Estimated smoothness score.
        modality: Estimated modality score.
        uncertainty: Surrogate uncertainty summary.
        labels: Human-readable descriptor labels.

    Output:
        Used by the reasoning agent and optimiser.
    """

    dim: int
    num_observations: int
    best_y: float | None
    y_range: float | None
    smoothness: float | None = None
    modality: float | None = None
    uncertainty: float | None = None
    labels: Mapping[str, str] = field(default_factory=dict)


def describe_landscape(
    observed_x: Matrix,
    observed_y: Sequence[float],
    surrogate_metadata: Mapping[str, object] | None = None,
) -> LandscapeDescriptor:
    """Build a landscape descriptor from current observations.

    Inputs:
        observed_x: Evaluated points in normalised coordinates.
        observed_y: Objective values for each point.
        surrogate_metadata: Optional information from the surrogate model.

    Output:
        ``LandscapeDescriptor`` for downstream reasoning.
    """

    y = np.asarray(observed_y, dtype=float)
    dim = len(observed_x[0]) if observed_x else 0
    best_y = float(np.min(y)) if len(y) else None
    y_range = float(np.max(y) - np.min(y)) if len(y) else None
    metadata = dict(surrogate_metadata or {})
    std_values = metadata.get("std")
    uncertainty = None
    if isinstance(std_values, Sequence) and not isinstance(std_values, (str, bytes)) and len(std_values):
        uncertainty = float(np.mean(np.asarray(std_values, dtype=float)))
    descriptor = LandscapeDescriptor(
        dim=dim,
        num_observations=len(y),
        best_y=best_y,
        y_range=y_range,
        smoothness=estimate_smoothness(observed_x, observed_y) if len(y) >= 2 else None,
        modality=estimate_modality(observed_x, observed_y) if len(y) >= 3 else None,
        uncertainty=uncertainty,
    )
    return LandscapeDescriptor(
        dim=descriptor.dim,
        num_observations=descriptor.num_observations,
        best_y=descriptor.best_y,
        y_range=descriptor.y_range,
        smoothness=descriptor.smoothness,
        modality=descriptor.modality,
        uncertainty=descriptor.uncertainty,
        labels=label_descriptor(descriptor),
    )


def estimate_smoothness(observed_x: Matrix, observed_y: Sequence[float]) -> float:
    """Estimate local smoothness from observations.

    Inputs:
        observed_x: Evaluated points.
        observed_y: Objective values.

    Output:
        Smoothness score as a float.
    """

    x = np.asarray(observed_x, dtype=float)
    y = np.asarray(observed_y, dtype=float)
    if len(x) < 2:
        return 1.0
    distances = np.sqrt(np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2))
    deltas = np.abs(y[:, None] - y[None, :])
    mask = distances > 1e-12
    if not np.any(mask):
        return 1.0
    slopes = deltas[mask] / distances[mask]
    median_slope = float(np.median(slopes))
    return 1.0 / (1.0 + median_slope)


def estimate_modality(observed_x: Matrix, observed_y: Sequence[float]) -> float:
    """Estimate whether the landscape appears multimodal.

    Inputs:
        observed_x: Evaluated points.
        observed_y: Objective values.

    Output:
        Modality score as a float.
    """

    y = np.asarray(observed_y, dtype=float)
    if len(y) < 3:
        return 0.0
    ranks = np.argsort(np.argsort(y)).astype(float)
    spread = float(np.std(ranks) / max(1.0, len(y) - 1))
    return min(1.0, max(0.0, spread * 2.0))


def label_descriptor(descriptor: LandscapeDescriptor) -> dict[str, str]:
    """Convert numeric descriptor scores into text labels.

    Input:
        descriptor: Numeric landscape descriptor.

    Output:
        Dictionary of label names and label values.
    """

    labels: dict[str, str] = {}
    if descriptor.smoothness is not None:
        labels["smoothness"] = "smooth" if descriptor.smoothness >= 0.5 else "rugged"
    if descriptor.modality is not None:
        labels["modality"] = "multimodal" if descriptor.modality >= 0.6 else "simple"
    if descriptor.uncertainty is not None:
        labels["uncertainty"] = "high" if descriptor.uncertainty >= 0.35 else "low"
    return labels
