"""Agent interfaces for world-model-guided optimisation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

Vector = Sequence[float]
Matrix = Sequence[Vector]


@dataclass(frozen=True)
class AgentState:
    """Input state visible to the reasoning agent.

    Inputs:
        observed_x: Evaluated points in the normalised search space.
        observed_y: Objective values for the evaluated points.
        descriptor: Current landscape summary.
        budget_used: Number of evaluations already used.
        budget_total: Total evaluation budget.

    Output:
        Passed to ``WorldModelAgent.decide`` to produce a ``ReasoningDecision``.
    """

    observed_x: Matrix
    observed_y: Sequence[float]
    descriptor: Mapping[str, object]
    budget_used: int
    budget_total: int


@dataclass(frozen=True)
class ReasoningDecision:
    """Output of a world-model reasoning step.

    Inputs:
        strategy: Name of the next optimisation strategy.
        hypothesis: Agent's current explanation of the search landscape.
        confidence: Confidence score in the range ``[0, 1]``.
        rationale: Short text explaining the decision.
        metadata: Optional structured reasoning details.

    Output:
        Consumed by the optimiser to choose an acquisition rule or candidate generator.
    """

    strategy: str
    hypothesis: str
    confidence: float
    rationale: str
    metadata: Mapping[str, object] = field(default_factory=dict)


class CandidateValidator:
    """Validate candidate points before objective evaluation."""

    def __init__(self, dim: int, lower: float = 0.0, upper: float = 1.0) -> None:
        """Create a validator for normalised continuous inputs.

        Inputs:
            dim: Expected candidate dimensionality.
            lower: Lower bound for each coordinate.
            upper: Upper bound for each coordinate.

        Output:
            A validator instance.
        """

        self.dim = dim
        self.lower = lower
        self.upper = upper

    def validate(self, candidate: Vector) -> tuple[bool, list[str]]:
        """Check whether a candidate has valid shape and bounds.

        Input:
            candidate: Candidate point to validate.

        Output:
            Tuple ``(is_valid, issues)``.
        """

        raise NotImplementedError("Candidate validation is not implemented yet.")

    def repair(self, candidate: Vector) -> list[float]:
        """Repair an invalid candidate into the allowed search domain.

        Input:
            candidate: Candidate point to repair.

        Output:
            Repaired candidate as a list of floats.
        """

        raise NotImplementedError("Candidate repair is not implemented yet.")


class WorldModelAgent:
    """Reasoning interface for selecting high-level search behaviour."""

    def decide(self, state: AgentState) -> ReasoningDecision:
        """Choose the next optimisation strategy from the current search state.

        Input:
            state: Observations, landscape descriptor, and budget information.

        Output:
            A ``ReasoningDecision`` describing the selected strategy.
        """

        raise NotImplementedError("World-model reasoning is not implemented yet.")
