"""Agent interfaces for world-model-guided optimisation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
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

        issues: list[str] = []
        if len(candidate) != self.dim:
            issues.append(f"Expected {self.dim} dimensions, got {len(candidate)}.")
            return False, issues
        for index, value in enumerate(candidate):
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                issues.append(f"Coordinate {index} is not numeric.")
                continue
            if not math.isfinite(numeric):
                issues.append(f"Coordinate {index} is not finite.")
            elif numeric < self.lower or numeric > self.upper:
                issues.append(f"Coordinate {index} is outside [{self.lower}, {self.upper}].")
        return not issues, issues

    def repair(self, candidate: Vector) -> list[float]:
        """Repair an invalid candidate into the allowed search domain.

        Input:
            candidate: Candidate point to repair.

        Output:
            Repaired candidate as a list of floats.
        """

        repaired = [float(value) for value in candidate[: self.dim]]
        if len(repaired) < self.dim:
            repaired.extend([self.lower] * (self.dim - len(repaired)))
        return [min(self.upper, max(self.lower, value if math.isfinite(value) else self.lower)) for value in repaired]


class WorldModelAgent:
    """Reasoning interface for selecting high-level search behaviour."""

    def decide(self, state: AgentState) -> ReasoningDecision:
        """Choose the next optimisation strategy from the current search state.

        Input:
            state: Observations, landscape descriptor, and budget information.

        Output:
            A ``ReasoningDecision`` describing the selected strategy.
        """

        descriptor = dict(state.descriptor)
        remaining = max(0, state.budget_total - state.budget_used)
        uncertainty = descriptor.get("uncertainty")
        modality = descriptor.get("modality")
        progress = state.budget_used / state.budget_total if state.budget_total else 1.0

        if state.budget_used < max(2, len(state.observed_x[0]) + 1 if state.observed_x else 2):
            strategy = "global_diverse"
            rationale = "Collecting an initial space-filling design."
        elif isinstance(uncertainty, (int, float)) and uncertainty > 0.35 and progress < 0.75:
            strategy = "explore_ucb"
            rationale = "Surrogate uncertainty is still high."
        elif isinstance(modality, (int, float)) and modality > 0.6 and remaining > 2:
            strategy = "global_diverse"
            rationale = "Observed values suggest a rugged or multimodal landscape."
        else:
            strategy = "expected_improvement"
            rationale = "Balancing predicted improvement and model uncertainty."

        return ReasoningDecision(
            strategy=strategy,
            hypothesis=f"Best observed value is {min(state.observed_y) if state.observed_y else None}.",
            confidence=min(1.0, 0.4 + 0.6 * progress),
            rationale=rationale,
            metadata={"remaining_budget": remaining},
        )
