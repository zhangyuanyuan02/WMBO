"""Acquisition functions for candidate selection."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Mapping, Sequence

import numpy as np

Vector = Sequence[float]
Matrix = Sequence[Vector]


@dataclass(frozen=True)
class AcquisitionInput:
    """Inputs required to score and select candidate points.

    Inputs:
        candidates: Candidate points in the normalised search space.
        observed_x: Previously evaluated input points.
        observed_y: Objective values for ``observed_x``.
        surrogate_mean: Predicted objective mean for each candidate.
        surrogate_std: Predicted uncertainty for each candidate.
        strategy: Acquisition strategy name.

    Output:
        Passed to an acquisition function to produce ``AcquisitionResult``.
    """

    candidates: Matrix
    observed_x: Matrix
    observed_y: Sequence[float]
    surrogate_mean: Sequence[float]
    surrogate_std: Sequence[float]
    strategy: str = "expected_improvement"


@dataclass(frozen=True)
class AcquisitionResult:
    """Output returned after candidate scoring.

    Inputs:
        selected_x: Selected candidate point.
        selected_index: Position of the selected point in the candidate list.
        score: Acquisition score assigned to the selected point.
        scores: Acquisition score for every candidate.
        metadata: Optional diagnostic values.

    Output:
        Used by the optimiser as the next point to evaluate.
    """

    selected_x: Vector
    selected_index: int
    score: float
    scores: Sequence[float]
    metadata: Mapping[str, object] = field(default_factory=dict)


def expected_improvement(
    mean: Sequence[float],
    std: Sequence[float],
    best_y: float,
    xi: float = 0.01,
) -> list[float]:
    """Compute expected-improvement scores for minimisation.

    Inputs:
        mean: Predicted objective mean for each candidate.
        std: Predicted uncertainty for each candidate.
        best_y: Best observed objective value so far.
        xi: Exploration offset.

    Output:
        A list of acquisition scores, one per candidate. Higher is better.
    """

    mu = _as_1d_array(mean, name="mean")
    sigma = np.maximum(_as_1d_array(std, name="std"), 1e-12)
    if len(mu) != len(sigma):
        raise ValueError(f"mean and std length mismatch: {len(mu)} != {len(sigma)}.")

    improvement = float(best_y) - mu - float(xi)
    z = improvement / sigma
    scores = improvement * _normal_cdf(z) + sigma * _normal_pdf(z)
    return np.maximum(scores, 0.0).tolist()


def probability_improvement(
    mean: Sequence[float],
    std: Sequence[float],
    best_y: float,
    xi: float = 0.01,
) -> list[float]:
    """Compute probability-of-improvement scores for minimisation."""

    mu = _as_1d_array(mean, name="mean")
    sigma = np.maximum(_as_1d_array(std, name="std"), 1e-12)
    if len(mu) != len(sigma):
        raise ValueError(f"mean and std length mismatch: {len(mu)} != {len(sigma)}.")
    z = (float(best_y) - mu - float(xi)) / sigma
    return _normal_cdf(z).tolist()


def lower_confidence_bound(
    mean: Sequence[float],
    std: Sequence[float],
    kappa: float = 2.0,
) -> list[float]:
    """Compute lower-confidence-bound scores for minimisation.

    Inputs:
        mean: Predicted objective mean for each candidate.
        std: Predicted uncertainty for each candidate.
        kappa: Exploration weight.

    Output:
        A list of acquisition scores, one per candidate. Higher is better.
    """

    mu = _as_1d_array(mean, name="mean")
    sigma = _as_1d_array(std, name="std")
    if len(mu) != len(sigma):
        raise ValueError(f"mean and std length mismatch: {len(mu)} != {len(sigma)}.")
    lcb = mu - float(kappa) * sigma
    return (-lcb).tolist()


def diversity_score(candidates: Matrix, observed_x: Matrix) -> list[float]:
    """Score candidates by distance from previous observations.

    Inputs:
        candidates: Candidate points to score.
        observed_x: Previously evaluated points.

    Output:
        A list of diversity scores, one per candidate. Higher is better.
    """

    x = _as_2d_array(candidates, name="candidates")
    if len(x) == 0:
        return []
    if len(observed_x) == 0:
        return [1.0] * len(x)

    observed = _as_2d_array(observed_x, name="observed_x")
    if observed.shape[1] != x.shape[1]:
        raise ValueError(f"observed_x dimension mismatch: {observed.shape[1]} != {x.shape[1]}.")

    diff = x[:, None, :] - observed[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=2))
    return np.min(distances, axis=1).tolist()


def score_candidates(acquisition_input: AcquisitionInput) -> list[float]:
    """Dispatch candidate scoring according to the requested strategy.

    Input:
        acquisition_input: Candidate pool, observations, surrogate predictions, and strategy.

    Output:
        A score for each candidate in ``acquisition_input.candidates``. Higher is better.
    """

    candidates = _as_2d_array(acquisition_input.candidates, name="candidates")
    mean = _as_1d_array(acquisition_input.surrogate_mean, name="surrogate_mean")
    std = _as_1d_array(acquisition_input.surrogate_std, name="surrogate_std")
    if len(mean) != len(candidates) or len(std) != len(candidates):
        raise ValueError("Candidate, mean, and std arrays must have the same length.")

    observed_y = _as_1d_array(acquisition_input.observed_y, name="observed_y") if acquisition_input.observed_y else np.array([], dtype=float)
    best_y = float(np.min(observed_y)) if len(observed_y) else 0.0
    strategy = _normalise_strategy(acquisition_input.strategy)

    if strategy == "expected_improvement":
        return expected_improvement(mean, std, best_y=best_y, xi=0.01)
    if strategy == "exploit_ei":
        return expected_improvement(mean, std, best_y=best_y, xi=0.001)
    if strategy == "probability_improvement":
        return probability_improvement(mean, std, best_y=best_y, xi=0.01)
    if strategy == "lower_confidence_bound":
        return lower_confidence_bound(mean, std, kappa=2.0)
    if strategy == "explore_ucb":
        return lower_confidence_bound(mean, std, kappa=3.0)
    if strategy == "uncertainty":
        return std.tolist()
    if strategy == "diversity":
        return diversity_score(candidates, acquisition_input.observed_x)
    if strategy == "global_diverse":
        uncertainty = _scale_01(std)
        diversity = _scale_01(np.asarray(diversity_score(candidates, acquisition_input.observed_x), dtype=float))
        return (0.55 * uncertainty + 0.45 * diversity).tolist()
    if strategy == "random":
        rng = random.Random(0)
        return [rng.random() for _ in range(len(candidates))]

    raise ValueError(f"Unknown acquisition strategy: {acquisition_input.strategy}")


def select_next_candidate(acquisition_input: AcquisitionInput) -> AcquisitionResult:
    """Select the next candidate from a scored candidate pool.

    Input:
        acquisition_input: Candidate pool and all values needed for acquisition scoring.

    Output:
        ``AcquisitionResult`` containing the selected candidate and diagnostics.
    """

    candidates = _as_2d_array(acquisition_input.candidates, name="candidates")
    if len(candidates) == 0:
        raise ValueError("At least one candidate is required.")

    scores = np.asarray(score_candidates(acquisition_input), dtype=float)
    if len(scores) != len(candidates):
        raise ValueError("The number of acquisition scores must match the candidate pool.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Acquisition scores must be finite.")

    max_score = float(np.max(scores))
    tied = np.flatnonzero(np.isclose(scores, max_score))
    selected_index = int(tied[0]) if len(tied) else int(np.argmax(scores))
    selected_x = candidates[selected_index].astype(float).tolist()

    return AcquisitionResult(
        selected_x=selected_x,
        selected_index=selected_index,
        score=float(scores[selected_index]),
        scores=scores.tolist(),
        metadata={
            "strategy": acquisition_input.strategy,
            "num_candidates": int(len(candidates)),
            "num_tied_best": int(len(tied)),
        },
    )


def _normal_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    erf = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf(z / math.sqrt(2.0)))


def _scale_01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    spread = max_value - min_value
    if spread <= 1e-12:
        return np.ones_like(values)
    return (values - min_value) / spread


def _normalise_strategy(strategy: str) -> str:
    key = strategy.strip().lower().replace("-", "_")
    aliases = {
        "ei": "expected_improvement",
        "balanced_ei": "expected_improvement",
        "expected_improvement": "expected_improvement",
        "exploit_ei": "exploit_ei",
        "pi": "probability_improvement",
        "probability_improvement": "probability_improvement",
        "lcb": "lower_confidence_bound",
        "ucb": "lower_confidence_bound",
        "lower_confidence_bound": "lower_confidence_bound",
        "explore_ucb": "explore_ucb",
        "uncertainty": "uncertainty",
        "exploration": "uncertainty",
        "diversity": "diversity",
        "global_diverse": "global_diverse",
        "random": "random",
    }
    return aliases.get(key, key)


def _as_1d_array(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _as_2d_array(values: Matrix, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array
