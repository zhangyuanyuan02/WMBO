"""Acquisition function interfaces for candidate selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

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
        A list of acquisition scores, one per candidate.
    """

    raise NotImplementedError("Expected improvement is not implemented yet.")


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
        A list of acquisition scores, one per candidate.
    """

    raise NotImplementedError("Lower confidence bound is not implemented yet.")


def diversity_score(candidates: Matrix, observed_x: Matrix) -> list[float]:
    """Score candidates by distance from previous observations.

    Inputs:
        candidates: Candidate points to score.
        observed_x: Previously evaluated points.

    Output:
        A list of diversity scores, one per candidate.
    """

    raise NotImplementedError("Diversity scoring is not implemented yet.")


def score_candidates(acquisition_input: AcquisitionInput) -> list[float]:
    """Dispatch candidate scoring according to the requested strategy.

    Input:
        acquisition_input: Candidate pool, observations, surrogate predictions, and strategy.

    Output:
        A score for each candidate in ``acquisition_input.candidates``.
    """

    raise NotImplementedError("Candidate scoring is not implemented yet.")


def select_next_candidate(acquisition_input: AcquisitionInput) -> AcquisitionResult:
    """Select the next candidate from a scored candidate pool.

    Input:
        acquisition_input: Candidate pool and all values needed for acquisition scoring.

    Output:
        ``AcquisitionResult`` containing the selected candidate and diagnostics.
    """

    raise NotImplementedError("Candidate selection is not implemented yet.")
