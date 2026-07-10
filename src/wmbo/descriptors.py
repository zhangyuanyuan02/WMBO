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
        curvature: Nonlinearity score in ``[0, 1]``; larger means a linear
            trend explains less of the observed response.
        anisotropy: Dimension imbalance score in ``[0, 1]``.
        coverage: Space-filling score in ``[0, 1]``.
        boundary_bias: Fraction-like score measuring concentration near the
            normalised domain boundary.
        stagnation: Recent lack-of-progress score in ``[0, 1]``.
        improvement_rate: Normalised best-value improvement from the first
            observation.
        dimension_sensitivity: Normalised per-dimension sensitivity weights.
        sensitive_dims: Dimension indexes that appear most influential.
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
    curvature: float | None = None
    anisotropy: float | None = None
    coverage: float | None = None
    boundary_bias: float | None = None
    stagnation: float | None = None
    improvement_rate: float | None = None
    dimension_sensitivity: Sequence[float] = field(default_factory=tuple)
    sensitive_dims: Sequence[int] = field(default_factory=tuple)

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
            "curvature": self.curvature,
            "anisotropy": self.anisotropy,
            "uncertainty": self.uncertainty,
            "coverage": self.coverage,
            "boundary_bias": self.boundary_bias,
            "stagnation": self.stagnation,
            "improvement_rate": self.improvement_rate,
            "dimension_sensitivity": [float(value) for value in self.dimension_sensitivity],
            "sensitive_dims": [int(value) for value in self.sensitive_dims],
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
            curvature=None,
            anisotropy=None,
            uncertainty=1.0,
            coverage=0.0,
            boundary_bias=None,
            stagnation=1.0,
            improvement_rate=0.0,
            dimension_sensitivity=tuple(),
            sensitive_dims=tuple(),
        )
        return LandscapeDescriptor(**{**descriptor.__dict__, "labels": label_descriptor(descriptor)})

    smoothness = estimate_smoothness(x.tolist(), y.tolist())
    modality = estimate_modality(x.tolist(), y.tolist())
    curvature = estimate_curvature(x.tolist(), y.tolist())
    anisotropy, dimension_sensitivity, sensitive_dims = estimate_anisotropy(x.tolist(), y.tolist())
    uncertainty = _extract_uncertainty(surrogate_metadata, y)
    coverage = estimate_coverage(x.tolist(), y.tolist())
    boundary_bias = estimate_boundary_bias(x.tolist())
    stagnation, improvement_rate = estimate_progress(y.tolist())
    descriptor = LandscapeDescriptor(
        dim=dim,
        num_observations=int(len(y)),
        best_y=float(np.min(y)),
        y_range=float(np.max(y) - np.min(y)),
        smoothness=smoothness,
        modality=modality,
        curvature=curvature,
        anisotropy=anisotropy,
        uncertainty=uncertainty,
        coverage=coverage,
        boundary_bias=boundary_bias,
        stagnation=stagnation,
        improvement_rate=improvement_rate,
        dimension_sensitivity=dimension_sensitivity,
        sensitive_dims=sensitive_dims,
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


def estimate_curvature(observed_x: Matrix, observed_y: Sequence[float]) -> float:
    """Estimate nonlinear curvature from linear-model residuals.

    Inputs:
        observed_x: Evaluated points.
        observed_y: Objective values.

    Output:
        Curvature score in ``[0, 1]``.
    """

    x, y = _as_observation_arrays(observed_x, observed_y)
    n, dim = x.shape
    if n < max(4, dim + 2) or _value_scale(y) <= 1e-12:
        return 0.0

    y_scaled = _standardise_y(y)
    design = np.column_stack([np.ones(n), x])
    try:
        coefficients, *_ = np.linalg.lstsq(design, y_scaled, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0

    residual = y_scaled - design @ coefficients
    rmse = float(np.sqrt(np.mean(residual**2)))
    return float(np.clip(rmse, 0.0, 1.0))


def estimate_anisotropy(observed_x: Matrix, observed_y: Sequence[float]) -> tuple[float, tuple[float, ...], tuple[int, ...]]:
    """Estimate dimension-wise sensitivity imbalance.

    Inputs:
        observed_x: Evaluated points.
        observed_y: Objective values.

    Output:
        Tuple ``(anisotropy, dimension_sensitivity, sensitive_dims)``.
    """

    x, y = _as_observation_arrays(observed_x, observed_y)
    n, dim = x.shape
    if dim == 0:
        return 0.0, tuple(), tuple()
    if n < 3 or _value_scale(y) <= 1e-12:
        return 0.0, tuple(0.0 for _ in range(dim)), tuple(range(dim))

    pairwise_contributions = np.zeros(dim, dtype=float)
    scale = _value_scale(y)
    for i in range(n):
        for j in range(i + 1, n):
            delta_x = np.abs(x[i] - x[j])
            distance = float(np.linalg.norm(delta_x))
            if distance <= 1e-12:
                continue
            response_delta = abs(float(y[i] - y[j])) / scale
            pairwise_contributions += response_delta * (delta_x / distance)

    contributions = _combine_dimension_scores(pairwise_contributions, _univariate_dimension_scores(x, y))
    total = float(np.sum(contributions))
    if total <= 1e-12:
        return 0.0, tuple(0.0 for _ in range(dim)), tuple(range(dim))

    sensitivity = contributions / total
    if dim == 1:
        return 0.0, (1.0,), (0,)

    effective_dim = 1.0 / float(np.sum(sensitivity**2))
    anisotropy = 1.0 - (effective_dim - 1.0) / max(dim - 1.0, 1.0)
    sensitive_dims = _select_sensitive_dims(sensitivity, anisotropy)
    return (
        float(np.clip(anisotropy, 0.0, 1.0)),
        tuple(float(value) for value in sensitivity),
        sensitive_dims,
    )


def estimate_coverage(observed_x: Matrix, observed_y: Sequence[float] | None = None) -> float:
    """Estimate how broadly observations cover the normalised unit domain.

    Inputs:
        observed_x: Evaluated points in normalised coordinates.
        observed_y: Unused objective values, accepted for call-site symmetry.

    Output:
        Coverage score in ``[0, 1]`` where larger is more space-filling.
    """

    placeholder_y = [0.0] * len(observed_x) if observed_y is None else observed_y
    x, _y = _as_observation_arrays(observed_x, placeholder_y)
    n, dim = x.shape
    if n == 0 or dim == 0:
        return 0.0
    if n == 1:
        return 0.0

    axis_span = float(np.mean(np.ptp(x, axis=0)))
    centroid = np.mean(x, axis=0)
    max_centroid_distance = max(np.sqrt(dim) * 0.5, 1e-12)
    dispersion = float(np.mean(np.linalg.norm(x - centroid, axis=1)) / max_centroid_distance)
    nearest = _nearest_distances(x)
    target_spacing = max(0.5 * (n ** (-1.0 / max(dim, 1))), 1e-12)
    nearest_score = float(np.mean(nearest) / target_spacing) if nearest.size else 0.0
    coverage = 0.55 * axis_span + 0.25 * dispersion + 0.20 * nearest_score
    return float(np.clip(coverage, 0.0, 1.0))


def estimate_boundary_bias(observed_x: Matrix, boundary_width: float = 0.08) -> float:
    """Estimate how often observations sit near the unit-domain boundary.

    Inputs:
        observed_x: Evaluated points in normalised coordinates.
        boundary_width: Distance from 0 or 1 treated as boundary-adjacent.

    Output:
        Boundary-bias score in ``[0, 1]``.
    """

    x = np.asarray(observed_x, dtype=float)
    if x.size == 0:
        return 0.0
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2:
        raise ValueError("observed_x must be a two-dimensional array-like value.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Observed data must contain only finite values.")

    width = float(np.clip(boundary_width, 0.0, 0.5))
    if width <= 0.0:
        return 0.0
    near_boundary = (x <= width) | (x >= 1.0 - width)
    return float(np.clip(np.mean(near_boundary), 0.0, 1.0))


def estimate_progress(observed_y: Sequence[float], window: int = 5) -> tuple[float, float]:
    """Estimate recent stagnation and overall best-value improvement.

    Inputs:
        observed_y: Objective values for a minimisation run.
        window: Recent horizon used for stagnation scoring.

    Output:
        Tuple ``(stagnation, improvement_rate)`` in ``[0, 1]``.
    """

    y = np.asarray(observed_y, dtype=float)
    if y.ndim != 1:
        raise ValueError("observed_y must be one-dimensional.")
    if len(y) == 0:
        return 1.0, 0.0
    if not np.all(np.isfinite(y)):
        raise ValueError("Observed data must contain only finite values.")
    if len(y) == 1:
        return 1.0, 0.0

    best_curve = np.minimum.accumulate(y)
    improvement_events = np.where(np.diff(best_curve) < -1e-12)[0] + 1
    if improvement_events.size:
        trials_since_improvement = int(len(y) - 1 - improvement_events[-1])
    else:
        trials_since_improvement = len(y) - 1
    horizon = max(1, min(int(window), len(y) - 1))
    stagnation = float(np.clip(trials_since_improvement / horizon, 0.0, 1.0))

    value_range = _value_scale(y)
    improvement = max(0.0, float(y[0] - np.min(y)))
    improvement_rate = 0.0 if value_range <= 1e-12 else improvement / value_range
    return float(np.clip(stagnation, 0.0, 1.0)), float(np.clip(improvement_rate, 0.0, 1.0))


def label_descriptor(descriptor: LandscapeDescriptor) -> dict[str, str]:
    """Convert numeric descriptor scores into text labels.

    Input:
        descriptor: Numeric landscape descriptor.

    Output:
        Dictionary of label names and label values.
    """

    smoothness = descriptor.smoothness
    modality = descriptor.modality
    curvature = descriptor.curvature
    anisotropy = descriptor.anisotropy
    uncertainty = descriptor.uncertainty
    coverage = descriptor.coverage
    boundary_bias = descriptor.boundary_bias
    stagnation = descriptor.stagnation

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

    if curvature is None:
        curvature_label = "unknown"
    elif curvature >= 0.65:
        curvature_label = "high"
    elif curvature >= 0.25:
        curvature_label = "moderate"
    else:
        curvature_label = "low"

    if anisotropy is None:
        anisotropy_label = "unknown"
    elif anisotropy >= 0.65:
        anisotropy_label = "high"
    elif anisotropy >= 0.25:
        anisotropy_label = "moderate"
    else:
        anisotropy_label = "low"

    if uncertainty is None:
        uncertainty_label = "unknown"
    elif uncertainty >= 0.55:
        uncertainty_label = "high"
    elif uncertainty >= 0.25:
        uncertainty_label = "moderate"
    else:
        uncertainty_label = "low"

    if coverage is None:
        coverage_label = "unknown"
    elif coverage >= 0.65:
        coverage_label = "high"
    elif coverage >= 0.35:
        coverage_label = "moderate"
    else:
        coverage_label = "low"

    if boundary_bias is None:
        boundary_label = "unknown"
    elif boundary_bias >= 0.25:
        boundary_label = "high"
    elif boundary_bias >= 0.12:
        boundary_label = "moderate"
    else:
        boundary_label = "low"

    if stagnation is None:
        progress_label = "unknown"
    elif stagnation >= 0.75:
        progress_label = "stalled"
    elif stagnation >= 0.40:
        progress_label = "slow"
    else:
        progress_label = "active"

    return {
        "smoothness": smoothness_label,
        "modality": modality_label,
        "curvature": curvature_label,
        "anisotropy": anisotropy_label,
        "uncertainty": uncertainty_label,
        "coverage": coverage_label,
        "boundary_bias": boundary_label,
        "progress": progress_label,
        "dimension_profile": _dimension_profile_label(anisotropy_label),
        "sample_size": "small" if descriptor.num_observations < max(5, 2 * max(1, descriptor.dim)) else "usable",
    }


def _select_sensitive_dims(sensitivity: np.ndarray, anisotropy: float) -> tuple[int, ...]:
    dim = int(sensitivity.size)
    if dim == 0:
        return tuple()
    if dim == 1 or float(anisotropy) < 0.20:
        return tuple(range(dim))

    order = np.argsort(-sensitivity)
    selected: list[int] = []
    cumulative = 0.0
    for index in order:
        selected.append(int(index))
        cumulative += float(sensitivity[index])
        if cumulative >= 0.75:
            break
    return tuple(sorted(selected))


def _dimension_profile_label(anisotropy_label: str) -> str:
    if anisotropy_label == "high":
        return "focused"
    if anisotropy_label == "moderate":
        return "mixed"
    if anisotropy_label == "low":
        return "distributed"
    return "unknown"


def _combine_dimension_scores(pairwise_scores: np.ndarray, univariate_scores: np.ndarray) -> np.ndarray:
    pairwise_total = float(np.sum(pairwise_scores))
    univariate_total = float(np.sum(univariate_scores))
    if pairwise_total <= 1e-12:
        return univariate_scores
    if univariate_total <= 1e-12:
        return pairwise_scores
    pairwise = pairwise_scores / pairwise_total
    univariate = univariate_scores / univariate_total
    return 0.5 * pairwise + 0.5 * univariate


def _univariate_dimension_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dim = x.shape[1]
    scores = np.zeros(dim, dtype=float)
    y_centered = y - float(np.mean(y))
    total_variance = float(np.mean(y_centered**2))
    if total_variance <= 1e-12:
        return scores

    for index in range(dim):
        column = x[:, index]
        if float(np.ptp(column)) <= 1e-12:
            continue
        design = np.column_stack([np.ones(len(column)), column, column**2])
        try:
            coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        residual = y - design @ coefficients
        residual_variance = float(np.mean(residual**2))
        scores[index] = max(0.0, 1.0 - residual_variance / total_variance)
    return scores


def _standardise_y(y: np.ndarray) -> np.ndarray:
    return (y - float(np.mean(y))) / (_value_scale(y) or 1.0)


def _value_scale(y: np.ndarray) -> float:
    return max(float(np.std(y)), float(np.max(y) - np.min(y)), 1e-12)


def _nearest_distances(x: np.ndarray) -> np.ndarray:
    n = len(x)
    if n < 2:
        return np.empty((0,), dtype=float)
    distances = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    return nearest[np.isfinite(nearest)]


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
    "estimate_curvature",
    "estimate_anisotropy",
    "estimate_coverage",
    "estimate_boundary_bias",
    "estimate_progress",
    "label_descriptor",
]
