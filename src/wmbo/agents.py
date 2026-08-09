"""Rule-based agents for world-model-guided optimisation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .control import STRATEGIES

Vector = Sequence[float]
Matrix = Sequence[Vector]
_FINAL_REGRET_PHASE_PRIORS: dict[str, dict[str, float]] = {
    "early": {
        "global_diverse": -0.03,
        "explore_ucb": 0.00,
        "exploit_ei": 0.00,
        "trust_region": 0.04,
    },
    "middle": {
        "global_diverse": -0.06,
        "explore_ucb": -0.10,
        "exploit_ei": 0.00,
        "trust_region": 0.06,
    },
    "late": {
        "global_diverse": -0.30,
        "explore_ucb": -0.12,
        "exploit_ei": 0.00,
        "trust_region": 0.07,
    },
}
_FINAL_REGRET_PHASE_PRIORS_V4: dict[str, dict[str, float]] = {
    "early": {
        "global_diverse": -0.03,
        "explore_ucb": 0.00,
        "exploit_ei": 0.00,
        "trust_region": 0.065,
    },
    "middle": {
        "global_diverse": -0.06,
        "explore_ucb": -0.11,
        "exploit_ei": 0.00,
        "trust_region": 0.10,
    },
    "late": {
        "global_diverse": -0.30,
        "explore_ucb": -0.14,
        "exploit_ei": 0.00,
        "trust_region": 0.115,
    },
}




@dataclass(frozen=True)
class RulePolicyConfig:
    """Configuration for the deterministic rule proposal policy."""

    mode: str = "legacy_v1"
    landscape_weight: float = 0.45
    candidate_weight: float = 0.35
    history_weight: float = 0.20
    temperature: float = 0.20

    @classmethod
    def from_mapping(cls, value: Mapping[str, object] | None) -> RulePolicyConfig:
        raw = dict(value or {})
        config = cls(
            mode=str(raw.get("mode", cls.mode)).strip().lower(),
            landscape_weight=float(raw.get("landscape_weight", cls.landscape_weight)),
            candidate_weight=float(raw.get("candidate_weight", cls.candidate_weight)),
            history_weight=float(raw.get("history_weight", cls.history_weight)),
            temperature=float(raw.get("temperature", cls.temperature)),
        )
        if config.mode not in {
            "legacy_v1", "continuous_v2", "continuous_v3", "continuous_v4"
        }:
            raise ValueError(
                "rule_policy.mode must be legacy_v1, continuous_v2, continuous_v3, "
                "or continuous_v4."
            )
        weights = (config.landscape_weight, config.candidate_weight, config.history_weight)
        if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
            raise ValueError("rule policy weights must be finite and non-negative.")
        if sum(weights) <= 0.0:
            raise ValueError("at least one rule policy weight must be positive.")
        if not math.isfinite(config.temperature) or config.temperature <= 0.0:
            raise ValueError("rule_policy.temperature must be positive.")
        return config


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
    candidate_options: Sequence[Mapping[str, object]] = field(default_factory=tuple)
    decision_context: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ReasoningDecision:
    """Output of a world-model reasoning step.

    Inputs:
        strategy: Name of the next optimisation strategy.
        hypothesis: Agent's current explanation of the search landscape.
        confidence: Confidence score in the range ``[0, 1]``.
        rationale: Short text explaining the decision.
        hypothesis_region_center: Optional centre of the hypothesis region in
            normalised coordinates.
        hypothesis_region_radius: Optional normalised radius of the hypothesis
            region.
        hypothesis_sensitive_dims: Dimensions most relevant to the hypothesis.
        falsification_rule: Short rule describing when the hypothesis should be
            treated as unsupported.
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
    hypothesis_region_center: Sequence[float] | None = None
    hypothesis_region_radius: float | None = None
    hypothesis_sensitive_dims: Sequence[int] = field(default_factory=tuple)
    falsification_rule: str | None = None
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
            "hypothesis_region_center": (
                [float(value) for value in self.hypothesis_region_center]
                if self.hypothesis_region_center is not None
                else None
            ),
            "hypothesis_region_radius": (
                float(self.hypothesis_region_radius)
                if self.hypothesis_region_radius is not None
                else None
            ),
            "hypothesis_sensitive_dims": [int(dim) for dim in self.hypothesis_sensitive_dims],
            "falsification_rule": self.falsification_rule,
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

    def __init__(self, config: RulePolicyConfig | Mapping[str, object] | None = None) -> None:
        self.config = config if isinstance(config, RulePolicyConfig) else RulePolicyConfig.from_mapping(config)

    def decide(self, state: AgentState) -> ReasoningDecision:
        """Choose a strategy with the configured deterministic policy."""

        if state.budget_total <= 0:
            raise ValueError("budget_total must be positive.")
        if state.budget_used < 0:
            raise ValueError("budget_used must be non-negative.")
        if self.config.mode in {"continuous_v2", "continuous_v3", "continuous_v4"}:
            return self._decide_continuous(state)
        return self._decide_legacy(state)

    def _decide_legacy(self, state: AgentState) -> ReasoningDecision:
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
        curvature = labels.get("curvature", "unknown")
        anisotropy = labels.get("anisotropy", "unknown")
        uncertainty = labels.get("uncertainty", "unknown")
        coverage = labels.get("coverage", "unknown")
        progress = labels.get("progress", "unknown")
        sample_size = labels.get("sample_size", "small")

        if n_obs < max(5, 2 * max(1, dim)):
            strategy = "global_diverse"
            confidence = 0.45
            hypothesis = "The objective is still under-observed, so diverse global samples should improve the world model."
            rationale = "Observation count is low relative to dimensionality."
        elif progress == "stalled" and uncertainty == "high" and coverage in {"low", "moderate"} and remaining_ratio > 0.20:
            strategy = "global_diverse"
            confidence = 0.72
            hypothesis = "The search has stopped improving while the observed domain coverage is still incomplete."
            rationale = "A broader sample can test whether the current world model is over-focused on one region."
        elif modality in {"highly_multimodal", "multimodal"} and recent_improvement <= 1e-8:
            strategy = "global_diverse"
            confidence = 0.68
            hypothesis = "The search may be trapped in one basin of a multimodal landscape."
            rationale = "Progress has stalled while the descriptor suggests multiple local basins."
        elif curvature == "high" and uncertainty != "low" and remaining_ratio > 0.25:
            strategy = "explore_ucb"
            confidence = 0.66
            hypothesis = "The response appears nonlinear enough that the surrogate should reduce uncertainty before local refinement."
            rationale = "High curvature with non-low uncertainty favours uncertainty-aware exploration."
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
        elif anisotropy == "high" and sample_size == "usable":
            strategy = "trust_region"
            confidence = 0.67
            hypothesis = "Only a subset of dimensions appears to dominate the response, so local refinement can focus the search."
            rationale = "High anisotropy gives the world model a plausible sensitive subspace."
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
            "coverage": str(labels.get("coverage", "unknown")),
            "progress": str(labels.get("progress", "unknown")),
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

    def _decide_continuous(self, state: AgentState) -> ReasoningDecision:
        descriptor = dict(state.descriptor)
        labels = dict(descriptor.get("labels", {}) or {})
        context = dict(state.decision_context)
        progress = float(np.clip(state.budget_used / state.budget_total, 0.0, 1.0))
        remaining_ratio = 1.0 - progress
        uncertainty = _bounded(descriptor.get("uncertainty"), 1.0)
        coverage = _bounded(descriptor.get("coverage"), 0.0)
        stagnation = _bounded(descriptor.get("stagnation"), 1.0)
        recent_gain = _normalised_recent_improvement(state.observed_y)
        entropy = _bounded(
            dict(descriptor.get("calibration", {}) or {}).get("world_model_entropy"),
            1.0,
        )
        modality = _posterior_severity(descriptor, "modality", descriptor.get("modality"))
        ruggedness = _posterior_severity(descriptor, "smoothness", descriptor.get("smoothness"))
        curvature = _posterior_severity(descriptor, "curvature", descriptor.get("curvature"))
        anisotropy = _posterior_severity(descriptor, "anisotropy", descriptor.get("anisotropy"))

        landscape_scores = {
            "global_diverse": float(np.mean([
                (1.0 - progress) ** 2,
                1.0 - coverage,
                uncertainty,
                modality * stagnation,
                ruggedness,
            ])),
            "explore_ucb": float(np.mean([
                uncertainty,
                curvature,
                entropy,
                stagnation,
                1.0 - 0.5 * progress,
            ])),
            "exploit_ei": float(np.mean([
                progress,
                coverage,
                1.0 - uncertainty,
                recent_gain,
                1.0 - stagnation,
            ])),
            "trust_region": float(np.mean([
                coverage,
                1.0 - uncertainty,
                anisotropy,
                recent_gain,
                1.0 - min(1.0, abs(progress - 0.55) / 0.55),
            ])),
        }
        candidate_scores, best_options, candidate_components = _candidate_evidence(
            state.candidate_options
        )
        trusts = dict(context.get("strategy_trust", {}) or {})
        success_rates = dict(context.get("strategy_success_rates", {}) or {})
        history_scores = {
            strategy: (
                0.60 * _bounded(trusts.get(strategy), 0.5)
                + 0.40 * _bounded(success_rates.get(strategy), 0.0)
            )
            for strategy in STRATEGIES
        }

        weights = np.asarray(
            [
                self.config.landscape_weight,
                self.config.candidate_weight,
                self.config.history_weight,
            ],
            dtype=float,
        )
        weights = weights / float(np.sum(weights))
        base_scores = {
            strategy: float(
                weights[0] * landscape_scores[strategy]
                + weights[1] * candidate_scores[strategy]
                + weights[2] * history_scores[strategy]
            )
            for strategy in STRATEGIES
        }
        phase = str(context.get("budget_phase") or _phase_from_progress(progress))
        phase_prior_table = {
            "continuous_v3": _FINAL_REGRET_PHASE_PRIORS,
            "continuous_v4": _FINAL_REGRET_PHASE_PRIORS_V4,
        }.get(self.config.mode, {})
        phase_priors = dict(phase_prior_table.get(phase, {}))
        raw_scores = {
            strategy: float(base_scores[strategy] + phase_priors.get(strategy, 0.0))
            for strategy in STRATEGIES
        }

        configured_allowed = context.get("allowed_strategies")
        allowed = {
            str(strategy)
            for strategy in configured_allowed
            if str(strategy) in STRATEGIES
        } if isinstance(configured_allowed, Sequence) and not isinstance(
            configured_allowed, (str, bytes, bytearray)
        ) else set(STRATEGIES)
        if not allowed:
            allowed = {"exploit_ei", "trust_region"}
        forced = str(context.get("forced_strategy") or "")
        if forced not in allowed:
            forced = ""
        strategy = forced or max(
            (item for item in STRATEGIES if item in allowed),
            key=lambda item: raw_scores[item],
        )
        confidence = _score_confidence(
            scores=[raw_scores[item] for item in STRATEGIES if item in allowed],
            descriptor=descriptor,
            temperature=self.config.temperature,
            forced=bool(forced),
        )
        selected = best_options.get(strategy)
        selected_candidate_id = (
            str(selected.get("candidate_id"))
            if selected is not None and selected.get("candidate_id") is not None
            else None
        )
        hypotheses = {
            "global_diverse": "The world model still benefits from a broad sample outside the current coverage.",
            "explore_ucb": "Reducing surrogate and landscape uncertainty is more valuable than immediate local refinement.",
            "exploit_ei": "The current model is reliable enough to target expected improvement near promising regions.",
            "trust_region": "The observed basin is sufficiently defined for focused local refinement.",
        }
        rationales = {
            "global_diverse": "Early budget, incomplete coverage, and global candidate evidence favour diversity.",
            "explore_ucb": "Uncertainty, curvature, and information gain favour uncertainty-aware exploration.",
            "exploit_ei": "Budget progress, model certainty, and expected improvement favour exploitation.",
            "trust_region": "Coverage, local evidence, and anisotropy favour trust-region refinement.",
        }
        world_model = {
            name: str(labels.get(name, "unknown"))
            for name in ("smoothness", "modality", "curvature", "anisotropy", "coverage", "progress")
        }
        return ReasoningDecision(
            strategy=strategy,
            hypothesis=hypotheses[strategy],
            confidence=confidence,
            rationale=rationales[strategy],
            world_model=world_model,
            selected_candidate_id=selected_candidate_id,
            metadata={
                "labels": labels,
                "budget_used": state.budget_used,
                "budget_total": state.budget_total,
                "remaining_ratio": remaining_ratio,
                "recent_improvement": _recent_improvement(state.observed_y),
                "normalised_recent_improvement": recent_gain,
                "strategy_scores": {
                    item: raw_scores[item] if item in allowed else None for item in STRATEGIES
                },
                "score_components": {
                    item: {
                        "landscape": landscape_scores[item],
                        "candidate": candidate_scores[item],
                        "history": history_scores[item],
                        "base_score": base_scores[item],
                        "phase_prior": phase_priors.get(item, 0.0),
                        "candidate_evidence": candidate_components[item],
                    }
                    for item in STRATEGIES
                },
                "allowed_strategies": [item for item in STRATEGIES if item in allowed],
                "masked_strategies": [item for item in STRATEGIES if item not in allowed],
                "forced_strategy": forced or None,
                "forced_reason": context.get("forced_reason") if forced else None,
                "gate_reasons": list(context.get("gate_reasons", []) or []),
                "selected_candidate_evidence": dict(selected) if selected is not None else None,
                "source": "rule",
                "rule_policy_mode": self.config.mode,
                "budget_phase": phase,
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


def _phase_from_progress(progress: float) -> str:
    if progress < 0.35:
        return "early"
    if progress < 0.70:
        return "middle"
    return "late"


def _bounded(value: object, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    if not math.isfinite(parsed):
        parsed = float(default)
    return float(np.clip(parsed, 0.0, 1.0))


def _normalised_recent_improvement(observed_y: Sequence[float], window: int = 5) -> float:
    values = np.asarray(observed_y, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        return 0.0
    scale = float(np.max(finite) - np.min(finite))
    if scale <= 1e-15:
        return 0.0
    return float(np.clip(_recent_improvement(finite, window=window) / scale, 0.0, 1.0))


def _posterior_severity(
    descriptor: Mapping[str, object],
    name: str,
    numeric_value: object,
) -> float:
    numeric = _bounded(numeric_value, 0.5)
    all_posteriors = descriptor.get("property_posteriors", {})
    posterior = (
        dict(all_posteriors.get(name, {}) or {})
        if isinstance(all_posteriors, Mapping)
        else {}
    )
    categories = {
        "smoothness": ("smooth", "mixed", "rugged"),
        "modality": ("mostly_unimodal", "multimodal", "highly_multimodal"),
        "curvature": ("low", "moderate", "high"),
        "anisotropy": ("low", "moderate", "high"),
    }.get(name)
    if categories is None:
        return numeric
    probabilities = np.asarray(
        [_bounded(posterior.get(category), 0.0) for category in categories],
        dtype=float,
    )
    total = float(np.sum(probabilities))
    if total <= 1e-12:
        return numeric
    posterior_score = float(np.dot(probabilities / total, np.asarray([0.0, 0.5, 1.0])))
    return 0.5 * numeric + 0.5 * posterior_score


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def _normalise_feature(values: Mapping[str, float]) -> dict[str, float]:
    finite = [float(values.get(strategy, 0.0)) for strategy in STRATEGIES]
    lower, upper = min(finite), max(finite)
    if upper - lower <= 1e-12:
        return {strategy: 0.5 for strategy in STRATEGIES}
    return {
        strategy: float((float(values.get(strategy, lower)) - lower) / (upper - lower))
        for strategy in STRATEGIES
    }


def _candidate_evidence(
    candidates: Sequence[Mapping[str, object]],
) -> tuple[
    dict[str, float],
    dict[str, Mapping[str, object]],
    dict[str, dict[str, float]],
]:
    best: dict[str, Mapping[str, object]] = {}
    for strategy in STRATEGIES:
        matching = [
            candidate for candidate in candidates
            if str(candidate.get("strategy", "")) == strategy
        ]
        if matching:
            best[strategy] = max(
                matching,
                key=lambda candidate: (
                    _finite_float(
                        candidate.get("selection_score", candidate.get("acquisition_score")),
                        float("-inf"),
                    ),
                    str(candidate.get("candidate_id", "")),
                ),
            )

    raw_features: dict[str, dict[str, float]] = {
        "ei": {},
        "information": {},
        "novelty": {},
        "std": {},
        "confirmation": {},
    }
    for strategy in STRATEGIES:
        option = best.get(strategy, {})
        raw_features["ei"][strategy] = _finite_float(
            option.get(
                "constrained_expected_improvement",
                option.get("expected_improvement", 0.0),
            )
        )
        raw_features["information"][strategy] = _finite_float(option.get("information_gain"))
        raw_features["novelty"][strategy] = _finite_float(
            option.get("distance_to_nearest_observation")
        )
        raw_features["std"][strategy] = _finite_float(option.get("surrogate_std"))
        raw_features["confirmation"][strategy] = _finite_float(option.get("confirmation_value"))

    scaled = {name: _normalise_feature(values) for name, values in raw_features.items()}
    components = {
        strategy: {name: scaled[name][strategy] for name in scaled}
        for strategy in STRATEGIES
    }
    scores = {
        "global_diverse": (
            0.40 * scaled["novelty"]["global_diverse"]
            + 0.35 * scaled["information"]["global_diverse"]
            + 0.25 * scaled["std"]["global_diverse"]
        ),
        "explore_ucb": (
            0.45 * scaled["information"]["explore_ucb"]
            + 0.35 * scaled["std"]["explore_ucb"]
            + 0.20 * scaled["ei"]["explore_ucb"]
        ),
        "exploit_ei": (
            0.75 * scaled["ei"]["exploit_ei"]
            + 0.15 * scaled["confirmation"]["exploit_ei"]
            + 0.10 * (1.0 - scaled["std"]["exploit_ei"])
        ),
        "trust_region": (
            0.55 * scaled["ei"]["trust_region"]
            + 0.20 * scaled["information"]["trust_region"]
            + 0.15 * scaled["novelty"]["trust_region"]
            + 0.10 * scaled["std"]["trust_region"]
        ),
    }
    return {key: float(value) for key, value in scores.items()}, best, components


def _score_confidence(
    *,
    scores: Sequence[float],
    descriptor: Mapping[str, object],
    temperature: float,
    forced: bool,
) -> float:
    if forced:
        return 0.95
    values = np.asarray(scores, dtype=float)
    if not len(values):
        return 0.0
    shifted = (values - float(np.max(values))) / max(float(temperature), 1e-6)
    probabilities = np.exp(np.clip(shifted, -60.0, 0.0))
    probabilities /= max(float(np.sum(probabilities)), 1e-12)
    ordered = np.sort(probabilities)[::-1]
    margin = float(ordered[0] - ordered[1]) if len(ordered) > 1 else float(ordered[0])
    calibration = descriptor.get("calibration", {})
    posterior_confidence = _bounded(
        calibration.get("posterior_confidence") if isinstance(calibration, Mapping) else None,
        0.5,
    )
    return float(np.clip(0.5 * margin + 0.5 * posterior_confidence, 0.0, 1.0))


__all__ = [
    "RulePolicyConfig",
    "AgentState",
    "ReasoningDecision",
    "CandidateValidator",
    "WorldModelAgent",
]
