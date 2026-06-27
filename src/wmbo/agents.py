"""Rule-based agents for world-model-guided optimisation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

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
    world_model: Mapping[str, str] = field(default_factory=dict)
    selected_candidate_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Convert the decision to a JSON-friendly dictionary."""

        return {
            "strategy": self.strategy,
            "hypothesis": self.hypothesis,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "world_model": dict(self.world_model),
            "selected_candidate_id": self.selected_candidate_id,
            "metadata": dict(self.metadata),
        }


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

        if dim <= 0:
            raise ValueError("dim must be positive.")
        if lower >= upper:
            raise ValueError("lower must be smaller than upper.")
        self.dim = int(dim)
        self.lower = float(lower)
        self.upper = float(upper)

    def validate(self, candidate: Vector) -> tuple[bool, list[str]]:
        """Check whether a candidate has valid shape and bounds.

        Input:
            candidate: Candidate point to validate.

        Output:
            Tuple ``(is_valid, issues)``.
        """

        issues: list[str] = []
        values = np.asarray(candidate, dtype=float)
        if values.shape != (self.dim,):
            issues.append(f"candidate shape {values.shape} != {(self.dim,)}")
            values = values.reshape(-1)
        if not np.all(np.isfinite(values)):
            issues.append("candidate contains non-finite values")
        if values.size and (np.any(values < self.lower) or np.any(values > self.upper)):
            issues.append(f"candidate is outside [{self.lower}, {self.upper}]")
        return len(issues) == 0, issues

    def repair(self, candidate: Vector) -> list[float]:
        """Repair an invalid candidate into the allowed search domain.

        Input:
            candidate: Candidate point to repair.

        Output:
            Repaired candidate as a list of floats.
        """

        values = np.asarray(candidate, dtype=float).reshape(-1)
        if values.size != self.dim:
            fixed = np.full(self.dim, 0.5 * (self.lower + self.upper), dtype=float)
            n_copy = min(self.dim, values.size)
            fixed[:n_copy] = values[:n_copy]
            values = fixed
        values = np.nan_to_num(values, nan=0.5 * (self.lower + self.upper), posinf=self.upper, neginf=self.lower)
        return np.clip(values, self.lower, self.upper).astype(float).tolist()


class WorldModelAgent:
    """Deterministic reasoning agent for early WMBO experiments.

    The first implementation is deliberately rule based. It keeps the WMBO
    interface reproducible and lightweight before adding an external LLM backend.
    """

    def decide(self, state: AgentState) -> ReasoningDecision:
        """Choose the next optimisation strategy from the current search state.

        Input:
            state: Observations, landscape descriptor, and budget information.

        Output:
            A ``ReasoningDecision`` describing the selected strategy.
        """

        if state.budget_total <= 0:
            raise ValueError("budget_total must be positive.")
        if state.budget_used < 0:
            raise ValueError("budget_used must be non-negative.")

        descriptor = dict(state.descriptor)
        labels = dict(descriptor.get("labels", {}) or {})
        dim = int(descriptor.get("dim") or _infer_dim(state.observed_x))
        n_obs = int(descriptor.get("num_observations") or len(state.observed_y))
        remaining_ratio = max(0.0, (state.budget_total - state.budget_used) / state.budget_total)
        recent_improvement = _recent_improvement(state.observed_y)

        smoothness = labels.get("smoothness", "unknown")
        modality = labels.get("modality", "unknown")
        uncertainty = labels.get("uncertainty", "unknown")
        sample_size = labels.get("sample_size", "small")

        if n_obs < max(5, 2 * max(1, dim)):
            strategy = "global_diverse"
            confidence = 0.45
            hypothesis = "The objective is still under-observed, so diverse global samples should improve the world model."
            rationale = "Observation count is low relative to dimensionality."
        elif modality in {"highly_multimodal", "multimodal"} and recent_improvement <= 1e-8:
            strategy = "global_diverse"
            confidence = 0.68
            hypothesis = "The search may be trapped in one basin of a multimodal landscape."
            rationale = "Progress has stalled while the descriptor suggests multiple local basins."
        elif uncertainty == "high" and remaining_ratio > 0.25:
            strategy = "explore_ucb"
            confidence = 0.62
            hypothesis = "The surrogate is still uncertain, and there is enough budget left for exploration."
            rationale = "Uncertainty-driven acquisition should reduce model error before exploitation."
        elif smoothness == "rugged" and remaining_ratio > 0.15:
            strategy = "global_diverse"
            confidence = 0.60
            hypothesis = "A rugged landscape makes purely local refinement risky."
            rationale = "Global diversity is preferred while budget remains."
        elif sample_size == "usable" and remaining_ratio <= 0.35:
            strategy = "exploit_ei"
            confidence = 0.70
            hypothesis = "The run is in its later budget phase, so expected improvement should refine the best region."
            rationale = "Exploitative search is favoured near the end of the budget."
        else:
            strategy = "trust_region"
            confidence = 0.64
            hypothesis = "The current observations are sufficient for local refinement around the best point."
            rationale = "No strong signal requires broad exploration, so a local candidate pool is appropriate."

        world_model = {
            "smoothness": str(labels.get("smoothness", "unknown")),
            "modality": str(labels.get("modality", "unknown")),
            "curvature": str(labels.get("curvature", "unknown")),
            "anisotropy": str(labels.get("anisotropy", "unknown")),
        }

        return ReasoningDecision(
            strategy=strategy,
            hypothesis=hypothesis,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            rationale=rationale,
            world_model=world_model,
            metadata={
                "labels": labels,
                "budget_used": state.budget_used,
                "budget_total": state.budget_total,
                "remaining_ratio": remaining_ratio,
                "recent_improvement": recent_improvement,
                "source": "rule",
            },
        )


def _infer_dim(observed_x: Matrix) -> int:
    values = np.asarray(observed_x, dtype=float)
    if values.size == 0:
        return 0
    if values.ndim == 1:
        return int(values.size)
    return int(values.shape[1])


def _recent_improvement(observed_y: Sequence[float], window: int = 5) -> float:
    values = np.asarray(observed_y, dtype=float)
    if len(values) < 2:
        return 0.0
    best_curve = np.minimum.accumulate(values)
    start = max(0, len(best_curve) - int(window) - 1)
    return float(max(0.0, best_curve[start] - best_curve[-1]))


__all__ = [
    "AgentState",
    "ReasoningDecision",
    "CandidateValidator",
    "WorldModelAgent",
]
