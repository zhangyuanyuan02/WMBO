"""Landscape descriptor utilities for world-model-guided optimisation."""

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
        smoothness: Ruggedness-like score in ``[0, 1]``; larger means less smooth.
        modality: Multimodality score in ``[0, 1]``.
        uncertainty: Surrogate uncertainty score in ``[0, 1]``.
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

    def to_dict(self) -> dict[str, object]:
        """Convert the descriptor to a plain dictionary.

        Input:
            None.

        Output:
            Dictionary representation suitable for ``AgentState`` metadata.
        """

        return {
            "dim": self.dim,
            "num_observations": self.num_observations,
            "best_y": self.best_y,
            "y_range": self.y_range,
            "smoothness": self.smoothness,
            "modality": self.modality,
            "uncertainty": self.uncertainty,
            "labels": dict(self.labels),
        }


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

    x, y = _as_observation_arrays(observed_x, observed_y)
    dim = int(x.shape[1]) if x.ndim == 2 and x.shape[1] else 0

    if len(y) == 0:
        descriptor = LandscapeDescriptor(
            dim=dim,
            num_observations=0,
            best_y=None,
            y_range=None,
            smoothness=None,
            modality=None,
            uncertainty=1.0,
        )
        return LandscapeDescriptor(**{**descriptor.__dict__, "labels": label_descriptor(descriptor)})

    smoothness = estimate_smoothness(x.tolist(), y.tolist())
    modality = estimate_modality(x.tolist(), y.tolist())
    uncertainty = _extract_uncertainty(surrogate_metadata, y)
    descriptor = LandscapeDescriptor(
        dim=dim,
        num_observations=int(len(y)),
        best_y=float(np.min(y)),
        y_range=float(np.max(y) - np.min(y)),
        smoothness=smoothness,
        modality=modality,
        uncertainty=uncertainty,
    )
    return LandscapeDescriptor(**{**descriptor.__dict__, "labels": label_descriptor(descriptor)})


def estimate_smoothness(observed_x: Matrix, observed_y: Sequence[float]) -> float:
    """Estimate local ruggedness from observations.

    Inputs:
        observed_x: Evaluated points.
        observed_y: Objective values.

    Output:
        Smoothness score in ``[0, 1]`` where larger means more rugged.
    """

    x, y = _as_observation_arrays(observed_x, observed_y)
    if len(y) < 3:
        return 0.0

    slopes: list[float] = []
    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            distance = float(np.linalg.norm(x[i] - x[j]))
            if distance > 1e-12:
                slopes.append(abs(float(y[i] - y[j])) / distance)

    if len(slopes) < 2:
        return 0.0

    slope_array = np.asarray(slopes, dtype=float)
    variation = float(np.std(slope_array) / (np.mean(slope_array) + 1e-12))
    return float(np.clip(variation / 2.0, 0.0, 1.0))


def estimate_modality(observed_x: Matrix, observed_y: Sequence[float]) -> float:
    """Estimate whether the landscape appears multimodal.

    Inputs:
        observed_x: Evaluated points.
        observed_y: Objective values.

    Output:
        Modality score in ``[0, 1]``.
    """

    x, y = _as_observation_arrays(observed_x, observed_y)
    n = len(y)
    if n < 4:
        return 0.0

    neighbours = min(4, n - 1)
    local_minima = 0
    for i in range(n):
        distances = np.linalg.norm(x - x[i], axis=1)
        nearest = np.argsort(distances)[1 : neighbours + 1]
        if np.all(y[i] <= y[nearest]):
            local_minima += 1

    fraction = local_minima / max(1, n)
    return float(np.clip(4.0 * fraction, 0.0, 1.0))


def label_descriptor(descriptor: LandscapeDescriptor) -> dict[str, str]:
    """Convert numeric descriptor scores into text labels.

    Input:
        descriptor: Numeric landscape descriptor.

    Output:
        Dictionary of label names and label values.
    """

    smoothness = descriptor.smoothness
    modality = descriptor.modality
    uncertainty = descriptor.uncertainty

    if smoothness is None:
        smoothness_label = "unknown"
    elif smoothness >= 0.55:
        smoothness_label = "rugged"
    elif smoothness >= 0.25:
        smoothness_label = "mixed"
    else:
        smoothness_label = "smooth"

    if modality is None:
        modality_label = "unknown"
    elif modality >= 0.65:
        modality_label = "highly_multimodal"
    elif modality >= 0.25:
        modality_label = "multimodal"
    else:
        modality_label = "mostly_unimodal"

    if uncertainty is None:
        uncertainty_label = "unknown"
    elif uncertainty >= 0.55:
        uncertainty_label = "high"
    elif uncertainty >= 0.25:
        uncertainty_label = "moderate"
    else:
        uncertainty_label = "low"

    return {
        "smoothness": smoothness_label,
        "modality": modality_label,
        "uncertainty": uncertainty_label,
        "sample_size": "small" if descriptor.num_observations < max(5, 2 * max(1, descriptor.dim)) else "usable",
    }


def _extract_uncertainty(surrogate_metadata: Mapping[str, object] | None, y: np.ndarray) -> float:
    if not surrogate_metadata:
        return 1.0

    for key in ("mean_std", "avg_std", "uncertainty"):
        if key in surrogate_metadata:
            try:
                value = float(surrogate_metadata[key])
            except (TypeError, ValueError):
                continue
            scale = max(float(np.std(y)), 1.0)
            return float(np.clip(value / scale, 0.0, 1.0))

    if "std" in surrogate_metadata:
        try:
            std_values = np.asarray(surrogate_metadata["std"], dtype=float)
            scale = max(float(np.std(y)), 1.0)
            return float(np.clip(float(np.mean(std_values)) / scale, 0.0, 1.0))
        except (TypeError, ValueError):
            return 1.0

    return 1.0


def _as_observation_arrays(observed_x: Matrix, observed_y: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(observed_x, dtype=float)
    y = np.asarray(observed_y, dtype=float)

    if x.size == 0:
        x = x.reshape(0, 0)
    elif x.ndim == 1:
        x = x.reshape(1, -1)
    elif x.ndim != 2:
        raise ValueError("observed_x must be a two-dimensional array-like value.")

    if y.ndim != 1:
        raise ValueError("observed_y must be one-dimensional.")
    if len(x) != len(y):
        raise ValueError(f"observed_x and observed_y length mismatch: {len(x)} != {len(y)}.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Observed data must contain only finite values.")
    return x, y


__all__ = [
    "LandscapeDescriptor",
    "describe_landscape",
    "estimate_smoothness",
    "estimate_modality",
    "label_descriptor",
]
