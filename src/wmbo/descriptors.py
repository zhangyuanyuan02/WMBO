"""Landscape descriptor interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

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

    raise NotImplementedError("Landscape description is not implemented yet.")


def estimate_smoothness(observed_x: Matrix, observed_y: Sequence[float]) -> float:
    """Estimate local smoothness from observations.

    Inputs:
        observed_x: Evaluated points.
        observed_y: Objective values.

    Output:
        Smoothness score as a float.
    """

    raise NotImplementedError("Smoothness estimation is not implemented yet.")


def estimate_modality(observed_x: Matrix, observed_y: Sequence[float]) -> float:
    """Estimate whether the landscape appears multimodal.

    Inputs:
        observed_x: Evaluated points.
        observed_y: Objective values.

    Output:
        Modality score as a float.
    """

    raise NotImplementedError("Modality estimation is not implemented yet.")


def label_descriptor(descriptor: LandscapeDescriptor) -> dict[str, str]:
    """Convert numeric descriptor scores into text labels.

    Input:
        descriptor: Numeric landscape descriptor.

    Output:
        Dictionary of label names and label values.
    """

    raise NotImplementedError("Descriptor labelling is not implemented yet.")
