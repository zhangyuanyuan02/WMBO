"""Control-flow data structures for benchmark and WMBO runs."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .portfolio import (
    PORTFOLIO_EXPLORATION_STRATEGIES,
    PORTFOLIO_LOCAL_STRATEGIES,
    PORTFOLIO_STRATEGIES,
    V51_EXPLORATION_FAILURE_WINDOW,
    V51_EXPLORATION_GATES,
    V51_REWARD_EMA_ALPHA,
    V51_REWARD_REFERENCE,
    V51_TRUST_ALPHA,
    V51_RC3_LOCAL_DOMINANCE_WINDOW,
    V51_RC3_LOCAL_DOMINANCE_SHARE,
    V51_RC3_RECENT_FAILURE_MACROS,
    V51_RC3_UNDERPERFORM_MIN_MACROS,
    V51_RC3_UNDERPERFORM_RATIO,
    V51_RC3_ROUTING_PENALTY,
)

STRATEGIES = ("global_diverse", "explore_ucb", "exploit_ei", "trust_region")
EXPLORATION_STRATEGIES = {"global_diverse", "explore_ucb"}
LOCAL_STRATEGIES = {"exploit_ei", "trust_region"}


@dataclass(frozen=True)
class OptimizerConfig:
    """Inputs controlling a single optimiser instance.

    Inputs:
        method: Optimisation method name.
        budget: Maximum number of objective evaluations.
        initial_samples: Number of initial design points.
        candidate_pool_size: Number of candidates considered per step.
        seed: Random seed.
        options: Method-specific options.

    Output:
        Passed to optimiser construction.
    """

    method: str
    budget: int
    initial_samples: int
    candidate_pool_size: int
    seed: int
    options: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RunConfig:
    """Inputs controlling a benchmark suite run.

    Inputs:
        benchmarks: Benchmark names to run.
        methods: Optimiser names to compare.
        seeds: Random seeds for repeated runs.
        output_dir: Directory for future results.
        optimizer: Shared optimiser configuration.
        evaluation: Objective-evaluation backend options.

    Output:
        Passed to ``runner.run_benchmark_suite``.
    """

    benchmarks: Sequence[str]
    methods: Sequence[str]
    seeds: Sequence[int]
    output_dir: str
    optimizer: OptimizerConfig
    evaluation: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RunState:
    """Mutable-style state snapshot for one optimisation run.

    Inputs:
        step: Current optimisation step.
        best_y: Best objective value found so far.
        evaluations_used: Number of evaluations consumed.
        evaluations_remaining: Number of evaluations still available.

    Output:
        Used by stopping and logging utilities.
    """

    step: int
    best_y: float | None
    evaluations_used: int
    evaluations_remaining: int


@dataclass(frozen=True)
class WMBOControlConfig:
    """Policy knobs for the WMBO strategy controller.

    Inputs:
        early_fraction: Budget fraction below which the run is in the early phase.
        late_fraction: Budget fraction above which the run is in the late phase.
        global_max_consecutive_early: Maximum consecutive global-diverse steps early.
        global_max_consecutive_middle: Maximum consecutive global-diverse steps in the middle phase.
        trust_initial: Initial trust assigned to each strategy.
        trust_alpha: Exponential update rate for strategy trust.
        trust_window: Number of recent outcomes used for success-rate estimates.
        failure_cooldown_trials: Cooldown length after repeated failed exploration.
        hypothesis_window: Number of trials before an active hypothesis expires.
        middle_global_uncertainty_threshold: Minimum uncertainty for middle-phase global exploration.
        late_explore_uncertainty_threshold: Minimum uncertainty for late exploration.
        multimodal_explore_interval: Maximum non-exploratory gap tolerated on multimodal landscapes.
        multimodal_explore_uncertainty_threshold: Minimum uncertainty for multimodal exploration in early/middle phases.
        multimodal_late_explore_uncertainty_threshold: Minimum uncertainty for late multimodal exploration.
        hypothesis_alignment_weight: Soft candidate-selection weight for active hypothesis regions.
        gp_verifier_enabled: Whether GP acquisition/uncertainty can refine LLM-selected candidates.
        gp_verifier_min_score_ratio: Minimum acceptable candidate score ratio against the best same-strategy option.
        gp_verifier_duplicate_distance: Minimum distance from observed points before a candidate is considered duplicate-like.

    Output:
        Passed to ``WMBOState``.
    """

    early_fraction: float = 0.35
    late_fraction: float = 0.70
    candidate_options_per_strategy: int = 3
    global_max_consecutive_early: int = 2
    global_max_consecutive_middle: int = 1
    trust_initial: float = 0.5
    trust_alpha: float = 0.3
    trust_window: int = 5
    failure_cooldown_trials: int = 2
    hypothesis_window: int = 3
    hypothesis_support_probability: float = 0.80
    hypothesis_rejection_probability: float = 0.20
    hypothesis_support_likelihood_ratio: float = 3.0
    hypothesis_failure_likelihood_ratio: float = 0.50
    hypothesis_min_relevant_evidence: int = 1
    middle_global_uncertainty_threshold: float = 0.55
    late_explore_uncertainty_threshold: float = 0.65
    multimodal_explore_interval: int = 4
    multimodal_explore_uncertainty_threshold: float = 0.30
    multimodal_late_explore_uncertainty_threshold: float = 0.45
    hypothesis_alignment_weight: float = 0.15
    information_gain_weight: float = 0.35
    gp_verifier_enabled: bool = True
    gp_verifier_min_score_ratio: float = 0.75
    gp_verifier_duplicate_distance: float = 1e-4
    gp_verifier_min_feasibility_probability: float = 0.10


@dataclass(frozen=True)
class StrategyDecisionContext:
    """Immutable controller snapshot shared with a strategy proposer."""

    phase: str
    trial_number: int
    remaining_budget: int
    remaining_ratio: float
    allowed_strategies: tuple[str, ...]
    forced_strategy: str | None
    forced_reason: str | None
    gate_reasons: tuple[str, ...]
    cooldown_until: Mapping[str, int]
    consecutive_no_improvement: int
    steps_since_exploration: int
    strategy_trust: Mapping[str, float]
    strategy_success_rates: Mapping[str, float]
    strategy_routing_penalties: Mapping[str, float] = field(default_factory=dict)
    strategy_penalty_reasons: Mapping[str, Sequence[str]] = field(default_factory=dict)
    local_operator_recent_shares: Mapping[str, float] = field(default_factory=dict)
    weakly_identified_top2: bool = False
    regime_entropy: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation for agents and logs."""

        return {
            "budget_phase": self.phase,
            "trial_number": self.trial_number,
            "remaining_budget": self.remaining_budget,
            "remaining_ratio": self.remaining_ratio,
            "allowed_strategies": list(self.allowed_strategies),
            "forced_strategy": self.forced_strategy,
            "forced_reason": self.forced_reason,
            "gate_reasons": list(self.gate_reasons),
            "cooldown_until": dict(self.cooldown_until),
            "consecutive_no_improvement": self.consecutive_no_improvement,
            "steps_since_exploration": self.steps_since_exploration,
            "strategy_trust": dict(self.strategy_trust),
            "strategy_success_rates": dict(self.strategy_success_rates),
            "strategy_routing_penalties": dict(self.strategy_routing_penalties),
            "strategy_penalty_reasons": {
                str(strategy): list(reasons)
                for strategy, reasons in self.strategy_penalty_reasons.items()
            },
            "local_operator_recent_shares": dict(self.local_operator_recent_shares),
            "weakly_identified_top2": bool(self.weakly_identified_top2),
            "regime_entropy": float(self.regime_entropy),
        }


@dataclass
class HypothesisRecord:
    """A world-model hypothesis tracked across future trials."""

    hypothesis_id: str
    text: str
    strategy: str
    created_trial: int
    expires_trial: int
    baseline_best: float
    confidence: float = 0.0
    region_center: Sequence[float] | None = None
    region_radius: float | None = None
    sensitive_dims: Sequence[int] = field(default_factory=tuple)
    falsification_rule: str | None = None
    posterior_probability: float = 0.5
    log_likelihood_ratio: float = 0.0
    evidence_count: int = 0
    relevant_evidence_count: int = 0
    supporting_evidence: int = 0
    contradicting_evidence: int = 0
    last_evidence: Mapping[str, Any] | None = None
    status: str = "active"

    def to_dict(self) -> dict[str, Any]:
        """Convert the hypothesis record to a JSON-friendly dictionary."""

        return {
            "hypothesis_id": self.hypothesis_id,
            "text": self.text,
            "strategy": self.strategy,
            "created_trial": self.created_trial,
            "expires_trial": self.expires_trial,
            "baseline_best": self.baseline_best,
            "confidence": self.confidence,
            "region_center": [float(value) for value in self.region_center] if self.region_center is not None else None,
            "region_radius": self.region_radius,
            "sensitive_dims": [int(dim) for dim in self.sensitive_dims],
            "falsification_rule": self.falsification_rule,
            "posterior_probability": self.posterior_probability,
            "log_likelihood_ratio": self.log_likelihood_ratio,
            "evidence_count": self.evidence_count,
            "relevant_evidence_count": self.relevant_evidence_count,
            "supporting_evidence": self.supporting_evidence,
            "contradicting_evidence": self.contradicting_evidence,
            "last_evidence": dict(self.last_evidence) if self.last_evidence is not None else None,
            "status": self.status,
        }


@dataclass
class StrategyRecord:
    """One executed WMBO strategy outcome."""

    strategy: str
    trial_number: int
    improved: bool
    y: float
    best_y: float
    trust_after: float
    cooldown_until: int | None = None
    hypothesis_updates: Sequence[Mapping[str, Any]] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Convert the outcome record to a JSON-friendly dictionary."""

        return {
            "strategy": self.strategy,
            "trial_number": self.trial_number,
            "improved": self.improved,
            "y": self.y,
            "best_y": self.best_y,
            "trust_after": self.trust_after,
            "cooldown_until": self.cooldown_until,
            "hypothesis_updates": [dict(update) for update in self.hypothesis_updates],
        }


@dataclass
class WMBOState:
    """Mutable WMBO controller state.

    Inputs:
        config: Control-policy parameters.

    Output:
        Tracks strategy trust, cooldowns, active hypotheses, and budget phase.
    """

    config: WMBOControlConfig
    trusts: dict[str, float] = field(init=False)
    outcomes: dict[str, deque[bool]] = field(init=False)
    cooldown_until: dict[str, int] = field(init=False)
    hypotheses: list[HypothesisRecord] = field(default_factory=list)
    strategy_history: list[StrategyRecord] = field(default_factory=list)
    executed_strategies: list[str] = field(default_factory=list)
    consecutive_no_improvement: int = 0
    follow_up_local: bool = False
    hypothesis_counter: int = 0

    def __post_init__(self) -> None:
        self.trusts = {strategy: float(self.config.trust_initial) for strategy in STRATEGIES}
        self.outcomes = {
            strategy: deque(maxlen=max(1, int(self.config.trust_window))) for strategy in STRATEGIES
        }
        self.cooldown_until = defaultdict(int)

    def budget_phase(self, completed_trials: int, budget: int) -> str:
        """Return ``early``, ``middle``, or ``late`` for a budget position."""

        fraction = int(completed_trials) / max(int(budget), 1)
        if fraction < self.config.early_fraction:
            return "early"
        if fraction < self.config.late_fraction:
            return "middle"
        return "late"

    def remaining_budget(self, completed_trials: int, budget: int) -> int:
        """Return remaining objective evaluations."""

        return max(0, int(budget) - int(completed_trials))

    def recent_success_rates(self) -> dict[str, float]:
        """Return recent success rates for each strategy."""

        return {
            strategy: (sum(values) / len(values) if values else 0.0)
            for strategy, values in self.outcomes.items()
        }

    def hypothesis_summary(self, limit: int = 8) -> list[dict[str, Any]]:
        """Return the latest tracked hypotheses."""

        return [record.to_dict() for record in self.hypotheses[-max(1, int(limit)) :]]

    def hypothesis_status_counts(self) -> dict[str, int]:
        """Count hypotheses by status."""

        counts: dict[str, int] = {}
        for record in self.hypotheses:
            counts[record.status] = counts.get(record.status, 0) + 1
        return counts

    def allowed_strategies(
        self,
        phase: str,
        trial_number: int,
        uncertainty: float,
        modality_label: str = "unknown",
    ) -> tuple[set[str], list[str]]:
        """Return strategies currently allowed by budget, trust, and cooldown gates."""

        allowed = set(STRATEGIES)
        reasons: list[str] = []
        trial = int(trial_number)
        uncertainty_value = float(uncertainty)

        for strategy in EXPLORATION_STRATEGIES:
            if trial < int(self.cooldown_until[strategy]):
                allowed.discard(strategy)
                reasons.append(f"{strategy}_cooldown")

        global_run = self._consecutive_strategy_count("global_diverse")
        if phase == "early" and global_run >= self.config.global_max_consecutive_early:
            allowed.discard("global_diverse")
            reasons.append("early_global_consecutive_limit")
        elif phase == "middle":
            if global_run >= self.config.global_max_consecutive_middle:
                allowed.discard("global_diverse")
                reasons.append("middle_global_consecutive_limit")
            if uncertainty_value < self.config.middle_global_uncertainty_threshold:
                allowed.discard("global_diverse")
                reasons.append("middle_global_uncertainty_too_low")
            if self.trusts["global_diverse"] < self.config.trust_initial:
                allowed.discard("global_diverse")
                reasons.append("middle_global_trust_too_low")
        elif phase == "late":
            allowed.discard("global_diverse")
            reasons.append("late_global_forbidden")
            multimodal_pressure = (
                modality_label in {"multimodal", "highly_multimodal"}
                and uncertainty_value >= self.config.multimodal_late_explore_uncertainty_threshold
                and self._steps_since_exploration() >= max(1, int(self.config.multimodal_explore_interval))
            )
            allow_explore = (
                (
                    uncertainty_value >= self.config.late_explore_uncertainty_threshold
                    and self.consecutive_no_improvement >= 2
                    and self.trusts["explore_ucb"] >= self.config.trust_initial
                )
                or multimodal_pressure
            )
            if not allow_explore:
                allowed.discard("explore_ucb")
                reasons.append("late_exploration_gate")

        if not allowed:
            allowed.update(LOCAL_STRATEGIES)
            reasons.append("fallback_local_strategies_enabled")
        return allowed, reasons

    def decision_context(
        self,
        *,
        phase: str,
        trial_number: int,
        completed_trials: int,
        budget: int,
        uncertainty: float,
        smoothness_label: str = "unknown",
        modality_label: str = "unknown",
        flexible_local_follow_up: bool = False,
    ) -> StrategyDecisionContext:
        """Build the single controller snapshot used by proposers and execution."""

        allowed, gate_reasons = self.allowed_strategies(
            phase, trial_number, uncertainty, modality_label
        )
        forced_strategy = self._multimodal_exploration_guard(
            allowed=allowed,
            phase=phase,
            uncertainty=uncertainty,
            modality_label=modality_label,
        )
        forced_reason = "multimodal_exploration_guard" if forced_strategy is not None else None
        if forced_strategy is None and self.follow_up_local and flexible_local_follow_up:
            allowed.intersection_update(LOCAL_STRATEGIES)
            gate_reasons.append("new_best_local_follow_up_local_only")
        elif forced_strategy is None and self.follow_up_local:
            local = (
                "trust_region"
                if smoothness_label in {"rugged", "mixed"}
                or modality_label in {"multimodal", "highly_multimodal"}
                else "exploit_ei"
            )
            if local in allowed:
                forced_strategy = local
                forced_reason = "new_best_local_follow_up"

        total_budget = max(1, int(budget))
        return StrategyDecisionContext(
            phase=str(phase),
            trial_number=int(trial_number),
            remaining_budget=self.remaining_budget(completed_trials, total_budget),
            remaining_ratio=max(0.0, (total_budget - int(completed_trials)) / total_budget),
            allowed_strategies=tuple(strategy for strategy in self.trusts if strategy in allowed),
            forced_strategy=forced_strategy,
            forced_reason=forced_reason,
            gate_reasons=tuple(gate_reasons),
            cooldown_until={strategy: int(self.cooldown_until[strategy]) for strategy in self.trusts},
            consecutive_no_improvement=int(self.consecutive_no_improvement),
            steps_since_exploration=int(self._steps_since_exploration()),
            strategy_trust=dict(self.trusts),
            strategy_success_rates=self.recent_success_rates(),
        )

    def choose_strategy(
        self,
        proposed_strategy: str,
        phase: str,
        trial_number: int,
        uncertainty: float,
        smoothness_label: str = "unknown",
        modality_label: str = "unknown",
        decision_context: StrategyDecisionContext | None = None,
    ) -> tuple[str, str | None, set[str]]:
        """Accept or repair an agent-proposed strategy.

        Output:
            Tuple ``(executed_strategy, override_reason, allowed_strategies)``.
        """

        if decision_context is None:
            allowed, gate_reasons = self.allowed_strategies(
                phase, trial_number, uncertainty, modality_label
            )
        else:
            allowed = set(decision_context.allowed_strategies)
            gate_reasons = list(decision_context.gate_reasons)
        proposed = str(proposed_strategy).strip().lower().replace("-", "_")

        if decision_context is not None and decision_context.forced_strategy is not None:
            forced = decision_context.forced_strategy
            reason = None if proposed == forced else decision_context.forced_reason
            return forced, reason, allowed

        guard_strategy = self._multimodal_exploration_guard(
            allowed=allowed,
            phase=phase,
            uncertainty=uncertainty,
            modality_label=modality_label,
        )
        if guard_strategy is not None:
            if proposed in EXPLORATION_STRATEGIES and proposed in allowed:
                return proposed, None, allowed
            return guard_strategy, "multimodal_exploration_guard", allowed

        flexible_follow_up = (
            "new_best_local_follow_up_local_only" in gate_reasons
        )
        if self.follow_up_local and not flexible_follow_up:
            strategy = "trust_region" if smoothness_label in {"rugged", "mixed"} or modality_label in {"multimodal", "highly_multimodal"} else "exploit_ei"
            if strategy in allowed:
                return strategy, "new_best_local_follow_up", allowed

        if proposed in allowed:
            return proposed, None, allowed

        local_allowed = [strategy for strategy in ("trust_region", "exploit_ei") if strategy in allowed]
        alternatives = local_allowed or sorted(allowed)
        strategy = max(alternatives, key=lambda item: (self.trusts.get(item, 0.0), item))
        reason = "strategy_not_allowed" if proposed in STRATEGIES else "unknown_strategy"
        if gate_reasons:
            reason += ":" + ",".join(gate_reasons)
        return strategy, reason, allowed

    def create_hypothesis(
        self,
        text: str,
        strategy: str,
        trial_number: int,
        baseline_best: float,
        confidence: float = 0.0,
        region_center: Sequence[float] | None = None,
        region_radius: float | None = None,
        sensitive_dims: Sequence[int] | None = None,
        falsification_rule: str | None = None,
    ) -> HypothesisRecord | None:
        """Create a hypothesis record and expire older active records for that strategy."""

        hypothesis_text = str(text).strip()
        if not hypothesis_text:
            return None
        trial = int(trial_number)
        for record in self.hypotheses:
            if record.status == "active" and trial > record.expires_trial:
                record.status = "inconclusive"
        prior_probability = _clamp(confidence, 0.05, 0.95)
        self.hypothesis_counter += 1
        record = HypothesisRecord(
            hypothesis_id=f"h{self.hypothesis_counter}",
            text=hypothesis_text,
            strategy=str(strategy),
            created_trial=int(trial_number),
            expires_trial=int(trial_number) + max(1, int(self.config.hypothesis_window)) - 1,
            baseline_best=float(baseline_best),
            confidence=prior_probability,
            region_center=_normalise_region_center(region_center),
            region_radius=_normalise_region_radius(region_radius),
            sensitive_dims=_normalise_sensitive_dims(sensitive_dims),
            falsification_rule=_normalise_text(falsification_rule),
            posterior_probability=prior_probability,
        )
        self.hypotheses.append(record)
        return record

    def record_outcome(
        self,
        strategy: str,
        trial_number: int,
        improved: bool,
        y: float,
        best_y: float,
        *,
        candidate: Sequence[float] | None = None,
        predicted_mean: float | None = None,
        predicted_std: float | None = None,
        evidence_role: str | None = None,
        target_hypothesis_id: str | None = None,
        update_trust: bool = True,
    ) -> StrategyRecord:
        """Update trust and only apply evidence relevant to each hypothesis."""

        key = str(strategy).strip().lower().replace("-", "_")
        if key not in self.trusts:
            key = "exploit_ei" if "exploit_ei" in self.trusts else next(iter(self.trusts))
        reward = 1.0 if bool(improved) else 0.0
        if update_trust:
            alpha = float(self.config.trust_alpha)
            self.trusts[key] = (1.0 - alpha) * self.trusts[key] + alpha * reward
            self.outcomes[key].append(bool(improved))
        self.executed_strategies.append(key)
        self.follow_up_local = bool(improved)
        self.consecutive_no_improvement = 0 if improved else self.consecutive_no_improvement + 1

        cooldown: int | None = None
        recent = self.outcomes[key]
        if (
            update_trust
            and key in self._exploration_strategy_names()
            and len(recent) == recent.maxlen
            and not any(recent)
        ):
            cooldown = int(trial_number) + int(self.config.failure_cooldown_trials) + 1
            self.cooldown_until[key] = cooldown

        hypothesis_updates = self._update_hypotheses(
            trial_number=int(trial_number),
            improved=bool(improved),
            y=float(y),
            best_y=float(best_y),
            candidate=candidate,
            predicted_mean=predicted_mean,
            predicted_std=predicted_std,
            evidence_role=evidence_role,
            target_hypothesis_id=target_hypothesis_id,
        )

        outcome = StrategyRecord(
            strategy=key,
            trial_number=int(trial_number),
            improved=bool(improved),
            y=float(y),
            best_y=float(best_y),
            trust_after=float(self.trusts[key]),
            cooldown_until=cooldown,
            hypothesis_updates=hypothesis_updates,
        )
        self.strategy_history.append(outcome)
        return outcome

    def record_delayed_reward(
        self, strategy: str, reward: float, trial_number: int
    ) -> int | None:
        """Update strategy trust once after a completed macro action."""

        key = str(strategy).strip().lower().replace("-", "_")
        if key not in self.trusts:
            raise ValueError(f"Unknown strategy for delayed reward: {key}")
        bounded = float(min(max(float(reward), 0.0), 1.0))
        alpha = float(self.config.trust_alpha)
        self.trusts[key] = (1.0 - alpha) * self.trusts[key] + alpha * bounded
        self.outcomes[key].append(bounded)
        recent = self.outcomes[key]
        if (
            key in self._exploration_strategy_names()
            and len(recent) == recent.maxlen
            and not any(value > 0.0 for value in recent)
        ):
            cooldown = int(trial_number) + int(self.config.failure_cooldown_trials) + 1
            self.cooldown_until[key] = cooldown
            return cooldown
        return None

    def _update_hypotheses(
        self,
        *,
        trial_number: int,
        improved: bool,
        y: float,
        best_y: float,
        candidate: Sequence[float] | None,
        predicted_mean: float | None,
        predicted_std: float | None,
        evidence_role: str | None,
        target_hypothesis_id: str | None,
    ) -> list[dict[str, Any]]:
        """Apply one evaluation only to hypotheses for which it is evidence."""

        updates: list[dict[str, Any]] = []
        for record in self.hypotheses:
            if record.status != "active":
                continue
            record.evidence_count += 1
            relevant, distance_ratio = _hypothesis_relevance(
                record,
                candidate=candidate,
                evidence_role=evidence_role,
                target_hypothesis_id=target_hypothesis_id,
            )
            update: dict[str, Any] = {
                "hypothesis_id": record.hypothesis_id,
                "relevant": relevant,
                "distance_ratio": distance_ratio,
                "evidence_role": evidence_role,
                "prior_probability": record.posterior_probability,
            }
            if relevant:
                record.relevant_evidence_count += 1
                tolerance = max(1e-8, 0.001 * max(abs(record.baseline_best), 1.0))
                supports = bool(improved and float(best_y) < record.baseline_best - tolerance)
                likelihood_ratio = (
                    float(self.config.hypothesis_support_likelihood_ratio)
                    if supports
                    else float(self.config.hypothesis_failure_likelihood_ratio)
                )
                likelihood_ratio = max(likelihood_ratio, 1e-6)
                surprise_z: float | None = None
                if predicted_mean is not None and predicted_std is not None:
                    scale = max(abs(float(predicted_std)), 1e-8)
                    surprise_z = (float(y) - float(predicted_mean)) / scale
                evidence_strength = 1.0
                if surprise_z is not None and math.isfinite(surprise_z):
                    evidence_strength = _clamp(0.5 + 0.25 * abs(surprise_z), 0.5, 2.0)
                log_update = math.log(likelihood_ratio) * evidence_strength
                record.log_likelihood_ratio += log_update
                record.posterior_probability = _sigmoid(
                    _logit(record.confidence) + record.log_likelihood_ratio
                )
                if supports:
                    record.supporting_evidence += 1
                else:
                    record.contradicting_evidence += 1
                minimum = max(1, int(self.config.hypothesis_min_relevant_evidence))
                if record.relevant_evidence_count >= minimum:
                    if record.posterior_probability >= float(self.config.hypothesis_support_probability):
                        record.status = "supported"
                    elif record.posterior_probability <= float(self.config.hypothesis_rejection_probability):
                        record.status = "rejected"
                record.last_evidence = {
                    "trial_number": int(trial_number),
                    "candidate": [float(value) for value in candidate] if candidate is not None else None,
                    "y": float(y),
                    "best_y": float(best_y),
                    "improved": bool(improved),
                    "supports": supports,
                    "likelihood_ratio": likelihood_ratio,
                    "log_likelihood_update": log_update,
                    "surprise_z": surprise_z,
                    "evidence_role": evidence_role,
                    "distance_ratio": distance_ratio,
                }
                update.update(record.last_evidence)
            if record.status == "active" and int(trial_number) >= record.expires_trial:
                record.status = "inconclusive"
            update.update(
                {
                    "posterior_probability": record.posterior_probability,
                    "relevant_evidence_count": record.relevant_evidence_count,
                    "status": record.status,
                }
            )
            updates.append(update)
        return updates

    def _exploration_strategy_names(self) -> set[str]:
        return set(EXPLORATION_STRATEGIES)

    def to_dict(self) -> dict[str, Any]:
        """Return the controller state in a JSON-friendly form."""

        return {
            "trusts": dict(self.trusts),
            "success_rates": self.recent_success_rates(),
            "cooldown_until": {strategy: int(step) for strategy, step in self.cooldown_until.items()},
            "consecutive_no_improvement": self.consecutive_no_improvement,
            "follow_up_local": self.follow_up_local,
            "hypotheses": self.hypothesis_summary(),
            "hypothesis_status_counts": self.hypothesis_status_counts(),
            "strategy_history": [record.to_dict() for record in self.strategy_history[-8:]],
        }

    def _consecutive_strategy_count(self, strategy: str) -> int:
        count = 0
        for executed in reversed(self.executed_strategies):
            if executed != strategy:
                break
            count += 1
        return count

    def _steps_since_exploration(self) -> int:
        if not self.executed_strategies:
            return max(1, int(self.config.multimodal_explore_interval))
        for offset, strategy in enumerate(reversed(self.executed_strategies), start=1):
            if strategy in self._exploration_strategy_names():
                return offset - 1
        return len(self.executed_strategies)

    def _multimodal_exploration_guard(
        self,
        *,
        allowed: set[str],
        phase: str,
        uncertainty: float,
        modality_label: str,
    ) -> str | None:
        if modality_label not in {"multimodal", "highly_multimodal"}:
            return None
        exploratory = [strategy for strategy in ("explore_ucb", "global_diverse") if strategy in allowed]
        if not exploratory:
            return None
        interval = max(1, int(self.config.multimodal_explore_interval))
        if self._steps_since_exploration() < interval:
            return None
        threshold = (
            self.config.multimodal_late_explore_uncertainty_threshold
            if phase == "late"
            else self.config.multimodal_explore_uncertainty_threshold
        )
        if float(uncertainty) < float(threshold):
            return None
        return exploratory[0]


def _hypothesis_relevance(
    record: HypothesisRecord,
    *,
    candidate: Sequence[float] | None,
    evidence_role: str | None,
    target_hypothesis_id: str | None,
) -> tuple[bool, float | None]:
    """Return whether a candidate is an in-region or explicit hypothesis test."""

    role = str(evidence_role or "").strip().lower()
    targeted = bool(
        target_hypothesis_id == record.hypothesis_id and role in {"confirm", "falsify"}
    )
    if candidate is None or record.region_center is None or record.region_radius is None:
        return targeted, None
    try:
        point = [float(value) for value in candidate]
        center = [float(value) for value in record.region_center]
        radius = float(record.region_radius)
    except (TypeError, ValueError):
        return targeted, None
    if len(point) != len(center) or radius <= 0.0:
        return targeted, None
    dims = [int(dim) for dim in record.sensitive_dims if 0 <= int(dim) < len(point)]
    if not dims:
        dims = list(range(len(point)))
    distance = math.sqrt(sum((point[index] - center[index]) ** 2 for index in dims))
    distance_ratio = float(distance / max(radius, 1e-12))
    return bool(distance_ratio <= 1.0 or targeted), distance_ratio


def _logit(probability: float) -> float:
    bounded = _clamp(probability, 1e-6, 1.0 - 1e-6)
    return math.log(bounded / (1.0 - bounded))


def _sigmoid(value: float) -> float:
    bounded = _clamp(value, -60.0, 60.0)
    if bounded >= 0.0:
        scale = math.exp(-bounded)
        return 1.0 / (1.0 + scale)
    scale = math.exp(bounded)
    return scale / (1.0 + scale)


def _clamp(value: object, lower: float, upper: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(lower)
    if not math.isfinite(parsed):
        return float(lower)
    return float(min(max(parsed, lower), upper))


def _normalise_region_center(value: Sequence[float] | None) -> list[float] | None:
    if value is None:
        return None
    result: list[float] = []
    for item in value:
        try:
            parsed = float(item)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(parsed):
            return None
        result.append(float(min(max(parsed, 0.0), 1.0)))
    return result or None


def _normalise_region_radius(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0.0:
        return None
    return float(min(max(parsed, 1e-6), 1.0))


def _normalise_sensitive_dims(value: Sequence[int] | None) -> list[int]:
    if value is None:
        return []
    result: list[int] = []
    for item in value:
        try:
            parsed = int(item)
        except (TypeError, ValueError):
            continue
        if parsed >= 0 and parsed not in result:
            result.append(parsed)
    return result

@dataclass
class PortfolioWMBOState(WMBOState):
    """Controller profile for the five operators under the formal V5.1 policy."""

    has_feasible_observation: bool = True
    reward_history: dict[str, deque[float]] = field(init=False)

    def __post_init__(self) -> None:
        self.trusts = {
            strategy: float(self.config.trust_initial) for strategy in PORTFOLIO_STRATEGIES
        }
        self.outcomes = {
            strategy: deque(maxlen=max(1, int(self.config.trust_window)))
            for strategy in PORTFOLIO_STRATEGIES
        }
        self.reward_history = {
            strategy: deque(maxlen=max(1, int(self.config.trust_window)))
            for strategy in PORTFOLIO_STRATEGIES
        }
        self.cooldown_until = defaultdict(int)

    def recent_success_rates(self) -> dict[str, float]:
        """Return Bayesian-smoothed objective-success rates.

        Untried operators stay neutral at 0.5 instead of looking either perfect or
        failed.  The Beta(1, 1) prior also prevents one macro from dominating.
        """

        return {
            strategy: (
                (sum(bool(value) for value in values) + 1.0) / (len(values) + 2.0)
                if values else 0.5
            )
            for strategy, values in self.outcomes.items()
        }

    def recent_reward_quality(self) -> dict[str, float]:
        """Return scale-normalised recent macro reward quality."""

        result: dict[str, float] = {}
        for strategy, values in self.reward_history.items():
            if not values:
                result[strategy] = 0.5
                continue
            iterator = iter(values)
            ema = float(next(iterator))
            for value in iterator:
                ema = (1.0 - V51_REWARD_EMA_ALPHA) * ema + V51_REWARD_EMA_ALPHA * float(value)
            result[strategy] = float(np.clip(ema / V51_REWARD_REFERENCE, 0.0, 1.0))
        return result

    def recent_local_operator_shares(self) -> dict[str, float]:
        """Return local-operator shares over RC3's rolling evaluation window."""

        local = tuple(PORTFOLIO_LOCAL_STRATEGIES)
        window = list(self.executed_strategies)[-V51_RC3_LOCAL_DOMINANCE_WINDOW:]
        if len(window) < V51_RC3_LOCAL_DOMINANCE_WINDOW:
            return {strategy: 0.0 for strategy in local}
        denominator = float(len(window))
        return {
            strategy: float(sum(item == strategy for item in window) / denominator)
            for strategy in local
        }

    def local_routing_penalties(
        self,
    ) -> tuple[dict[str, float], dict[str, tuple[str, ...]], dict[str, float]]:
        """Return RC3 soft penalties for stale local-operator lock-in.

        Dominance suppression uses evaluation share but requires four recent failed
        macros. Underperformance suppression requires at least five completed
        macros and compares Bayesian-smoothed objective success with the best local
        arm. Both are soft 0.65 multipliers and deliberately do not stack.
        """

        local = tuple(PORTFOLIO_LOCAL_STRATEGIES)
        shares = self.recent_local_operator_shares()
        success_rates = self.recent_success_rates()
        penalties = {strategy: 1.0 for strategy in PORTFOLIO_STRATEGIES}
        reasons: dict[str, list[str]] = {strategy: [] for strategy in PORTFOLIO_STRATEGIES}
        best_local_success = max((success_rates[strategy] for strategy in local), default=0.5)

        for strategy in local:
            macro_outcomes = list(self.outcomes[strategy])
            recent_failures = (
                len(macro_outcomes) >= V51_RC3_RECENT_FAILURE_MACROS
                and not any(macro_outcomes[-V51_RC3_RECENT_FAILURE_MACROS:])
            )
            if shares[strategy] >= V51_RC3_LOCAL_DOMINANCE_SHARE and recent_failures:
                penalties[strategy] = min(penalties[strategy], V51_RC3_ROUTING_PENALTY)
                reasons[strategy].append("stale_local_dominance")

            if (
                len(macro_outcomes) >= V51_RC3_UNDERPERFORM_MIN_MACROS
                and recent_failures
                and best_local_success > 0.0
                and success_rates[strategy] < V51_RC3_UNDERPERFORM_RATIO * best_local_success
            ):
                penalties[strategy] = min(penalties[strategy], V51_RC3_ROUTING_PENALTY)
                reasons[strategy].append("evidence_gated_underperformance")

        return (
            penalties,
            {strategy: tuple(items) for strategy, items in reasons.items() if items},
            shares,
        )

    def decision_context(
        self,
        *,
        phase: str,
        trial_number: int,
        completed_trials: int,
        budget: int,
        uncertainty: float,
        smoothness_label: str = "unknown",
        modality_label: str = "unknown",
        flexible_local_follow_up: bool = False,
        has_feasible_observation: bool = True,
        available_strategies: Sequence[str] | None = None,
        coverage: float = 0.0,
        weakly_identified: float = 1.0,
        weakly_identified_top2: bool = False,
        regime_entropy: float = 0.0,
    ) -> StrategyDecisionContext:
        del smoothness_label, modality_label, flexible_local_follow_up
        self.has_feasible_observation = bool(has_feasible_observation)
        allowed, gate_reasons = self.allowed_strategies(
            phase,
            trial_number,
            uncertainty,
            coverage=coverage,
            weakly_identified=weakly_identified,
            weakly_identified_top2=weakly_identified_top2,
            regime_entropy=regime_entropy,
        )

        # A newly improved incumbent must be exploited locally on the next routing
        # boundary.  V5 used legacy names (trust_region/exploit_ei), so this gate
        # silently failed for portfolio operators.
        if self.follow_up_local:
            allowed.intersection_update(PORTFOLIO_LOCAL_STRATEGIES)
            gate_reasons.append("new_best_local_follow_up_local_only")
            if not allowed:
                allowed.update({"gp_ei", "anisotropic_turbo"})
                gate_reasons.append("new_best_local_follow_up_fallback")

        if available_strategies is not None:
            available = {
                str(strategy) for strategy in available_strategies
                if str(strategy) in PORTFOLIO_STRATEGIES
            }
            allowed.intersection_update(available)
            gate_reasons.extend(
                f"{strategy}_candidate_unavailable"
                for strategy in PORTFOLIO_STRATEGIES
                if strategy not in available
            )
            if not allowed:
                local_available = PORTFOLIO_LOCAL_STRATEGIES.intersection(available)
                if local_available:
                    allowed.update(local_available)
                    gate_reasons.append("fallback_available_local_strategies_enabled")
                else:
                    raise ValueError("No portfolio strategy produced a valid candidate.")

        forced_strategy = None
        forced_reason = None
        global_actions = sum(
            1 for strategy in self.executed_strategies if strategy == "global_sobol"
        )
        if (
            not self.follow_up_local
            and str(phase) == "early"
            and "global_sobol" in allowed
            and global_actions == 0
            and bool(weakly_identified_top2)
            and self.consecutive_no_improvement >= 3
        ):
            # Guarantee at most one genuinely global emergency probe when the
            # weak-identification regime remains prominent after sustained local
            # stagnation.  Later global actions must win the portfolio score.
            forced_strategy = "global_sobol"
            forced_reason = "weak_identification_emergency_probe"
            gate_reasons.append("early_global_emergency_probe")

        routing_penalties, penalty_reasons, local_shares = self.local_routing_penalties()
        total_budget = max(1, int(budget))
        return StrategyDecisionContext(
            phase=str(phase),
            trial_number=int(trial_number),
            remaining_budget=self.remaining_budget(completed_trials, total_budget),
            remaining_ratio=max(0.0, (total_budget - int(completed_trials)) / total_budget),
            allowed_strategies=tuple(strategy for strategy in self.trusts if strategy in allowed),
            forced_strategy=forced_strategy,
            forced_reason=forced_reason,
            gate_reasons=tuple(gate_reasons),
            cooldown_until={strategy: int(self.cooldown_until[strategy]) for strategy in self.trusts},
            consecutive_no_improvement=int(self.consecutive_no_improvement),
            steps_since_exploration=int(self._steps_since_exploration()),
            strategy_trust=dict(self.trusts),
            strategy_success_rates=self.recent_success_rates(),
            strategy_routing_penalties=routing_penalties,
            strategy_penalty_reasons=penalty_reasons,
            local_operator_recent_shares=local_shares,
            weakly_identified_top2=bool(weakly_identified_top2),
            regime_entropy=float(regime_entropy),
        )

    def allowed_strategies(
        self,
        phase: str,
        trial_number: int,
        uncertainty: float,
        modality_label: str = "unknown",
        *,
        coverage: float = 0.0,
        weakly_identified: float = 1.0,
        weakly_identified_top2: bool = False,
        regime_entropy: float = 0.0,
    ) -> tuple[set[str], list[str]]:
        del modality_label, coverage, weakly_identified
        allowed = set(PORTFOLIO_STRATEGIES)
        reasons: list[str] = []
        trial = int(trial_number)
        uncertainty_value = float(np.clip(float(uncertainty), 0.0, 1.0))
        entropy_value = float(np.clip(float(regime_entropy), 0.0, 1.0))
        steps_since_exploration = self._steps_since_exploration()
        global_actions = sum(
            1 for strategy in self.executed_strategies if strategy == "global_sobol"
        )

        for strategy in PORTFOLIO_EXPLORATION_STRATEGIES:
            if trial < int(self.cooldown_until[strategy]):
                allowed.discard(strategy)
                reasons.append(f"{strategy}_cooldown")

        # V5.1 formal sparse exploration.  Weak identification is evaluated
        # relatively (top-two regime) or through regime entropy instead of an
        # absolute posterior threshold.  This avoids the Stage-1 scale bug where
        # Sobol and early UCB were practically impossible to trigger.
        if phase == "early":
            weak_or_ambiguous = bool(weakly_identified_top2) or (
                entropy_value >= V51_EXPLORATION_GATES["early_global_regime_entropy"]
            )
            allow_global = (
                weak_or_ambiguous
                and uncertainty_value >= V51_EXPLORATION_GATES["early_global_uncertainty"]
                and self.consecutive_no_improvement
                >= int(V51_EXPLORATION_GATES["early_global_no_improvement"])
                and steps_since_exploration
                >= int(V51_EXPLORATION_GATES["early_global_interval"])
                and global_actions
                < int(V51_EXPLORATION_GATES["early_global_max_actions"])
            )
            if not allow_global:
                allowed.discard("global_sobol")
                reasons.append("early_global_sparse_gate")
            if global_actions >= int(V51_EXPLORATION_GATES["early_global_max_actions"]):
                reasons.append("early_global_action_cap")

            ucb_identification_evidence = bool(weakly_identified_top2) or (
                entropy_value >= V51_EXPLORATION_GATES["early_ucb_regime_entropy"]
            )
            allow_ucb = (
                ucb_identification_evidence
                and uncertainty_value >= V51_EXPLORATION_GATES["early_ucb_uncertainty"]
                and self.consecutive_no_improvement
                >= int(V51_EXPLORATION_GATES["early_ucb_no_improvement"])
                and steps_since_exploration
                >= int(V51_EXPLORATION_GATES["early_ucb_interval"])
            )
            if not allow_ucb:
                allowed.discard("gp_ucb")
                reasons.append("early_ucb_sparse_gate")
        elif phase == "middle":
            allowed.discard("global_sobol")
            reasons.append("middle_global_forbidden")
            allow_ucb = (
                uncertainty_value >= V51_EXPLORATION_GATES["middle_ucb_uncertainty"]
                and self.consecutive_no_improvement
                >= int(V51_EXPLORATION_GATES["middle_ucb_no_improvement"])
                and steps_since_exploration
                >= int(V51_EXPLORATION_GATES["middle_ucb_interval"])
            )
            if not allow_ucb:
                allowed.discard("gp_ucb")
                reasons.append("middle_ucb_sparse_gate")
        else:
            allowed.discard("global_sobol")
            reasons.append("late_global_forbidden")
            allow_ucb = (
                uncertainty_value >= V51_EXPLORATION_GATES["late_ucb_uncertainty"]
                and self.consecutive_no_improvement
                >= int(V51_EXPLORATION_GATES["late_ucb_no_improvement"])
                and steps_since_exploration
                >= int(V51_EXPLORATION_GATES["late_ucb_interval"])
            )
            if not allow_ucb:
                allowed.discard("gp_ucb")
                reasons.append("late_ucb_sparse_gate")

        if not self.has_feasible_observation:
            allowed.discard("cma_local")
            reasons.append("cma_local_requires_feasible_observation")
        if not allowed:
            allowed.update({"gp_ei", "anisotropic_turbo"})
            reasons.append("fallback_local_strategies_enabled")
        return allowed, reasons

    def _exploration_strategy_names(self) -> set[str]:
        return set(PORTFOLIO_EXPLORATION_STRATEGIES)

    def record_delayed_reward(
        self,
        strategy: str,
        reward: float,
        trial_number: int,
        *,
        improved: bool | None = None,
    ) -> int | None:
        """Update V5.1 trust from objective success plus scale-normalised reward."""

        key = str(strategy).strip().lower().replace("-", "_")
        if key not in self.trusts:
            raise ValueError(f"Unknown strategy for delayed reward: {key}")
        bounded = float(np.clip(float(reward), 0.0, 1.0))
        success = bool(improved) if improved is not None else bounded >= 0.55
        self.reward_history[key].append(bounded)
        self.outcomes[key].append(success)

        reward_quality = self.recent_reward_quality()[key]
        success_rate = self.recent_success_rates()[key]
        target = 0.60 * reward_quality + 0.40 * success_rate
        alpha = V51_TRUST_ALPHA
        self.trusts[key] = float(
            np.clip((1.0 - alpha) * self.trusts[key] + alpha * target, 0.0, 1.0)
        )

        # Cooldown is now based on repeated *objective* failures, not reward==0.
        # Information-only rewards therefore cannot keep a failed exploration arm
        # permanently alive.
        recent = self.outcomes[key]
        failure_window = min(V51_EXPLORATION_FAILURE_WINDOW, recent.maxlen or 1)
        if (
            key in self._exploration_strategy_names()
            and len(recent) >= failure_window
            and not any(list(recent)[-failure_window:])
        ):
            cooldown = int(trial_number) + int(self.config.failure_cooldown_trials) + 1
            self.cooldown_until[key] = cooldown
            return cooldown
        return None

    def _steps_since_exploration(self) -> int:
        # The initial design is itself a global exploration block.  Treat the first
        # portfolio decision as zero steps since exploration instead of immediately
        # satisfying the old multimodal interval.
        if not self.executed_strategies:
            return 0
        return super()._steps_since_exploration()

    def _multimodal_exploration_guard(
        self,
        *,
        allowed: set[str],
        phase: str,
        uncertainty: float,
        modality_label: str,
    ) -> str | None:
        del allowed, phase, uncertainty, modality_label
        return None

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["reward_quality"] = self.recent_reward_quality()
        result["policy_revision"] = "v5.1-rc3"
        return result



def _normalise_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def build_default_optimizer_config(method: str, budget: int, seed: int) -> OptimizerConfig:
    """Create a minimal optimiser configuration.

    Inputs:
        method: Optimiser name.
        budget: Evaluation budget.
        seed: Random seed.

    Output:
        ``OptimizerConfig`` with project defaults.
    """

    if budget <= 0:
        raise ValueError("budget must be positive.")
    return OptimizerConfig(
        method=method,
        budget=int(budget),
        initial_samples=min(5, int(budget)),
        candidate_pool_size=256,
        seed=int(seed),
        options={},
    )


def should_stop(state: RunState) -> bool:
    """Decide whether an optimisation run should stop.

    Input:
        state: Current run state.

    Output:
        ``True`` when no more evaluations should be performed.
    """

    return state.evaluations_remaining <= 0


def update_run_state(state: RunState, new_y: float) -> RunState:
    """Update run state after one new objective value.

    Inputs:
        state: Previous run state.
        new_y: Newly observed objective value.

    Output:
        Updated ``RunState``.
    """

    best_y = float(new_y) if state.best_y is None else min(float(state.best_y), float(new_y))
    used = state.evaluations_used + 1
    remaining = max(0, state.evaluations_remaining - 1)
    return RunState(step=state.step + 1, best_y=best_y, evaluations_used=used, evaluations_remaining=remaining)


__all__ = [
    "STRATEGIES",
    "EXPLORATION_STRATEGIES",
    "LOCAL_STRATEGIES",
    "OptimizerConfig",
    "RunConfig",
    "RunState",
    "WMBOControlConfig",
    "StrategyDecisionContext",
    "WMBOState",
    "HypothesisRecord",
    "StrategyRecord",
    "build_default_optimizer_config",
    "should_stop",
    "update_run_state",
    "PortfolioWMBOState",
]
