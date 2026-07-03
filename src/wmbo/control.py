"""Control-flow data structures for benchmark and WMBO runs."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence


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

    Output:
        Passed to ``runner.run_benchmark_suite``.
    """

    benchmarks: Sequence[str]
    methods: Sequence[str]
    seeds: Sequence[int]
    output_dir: str
    optimizer: OptimizerConfig


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
    candidate_options_per_strategy: int = 2
    global_max_consecutive_early: int = 2
    global_max_consecutive_middle: int = 1
    trust_initial: float = 0.5
    trust_alpha: float = 0.3
    trust_window: int = 5
    failure_cooldown_trials: int = 2
    hypothesis_window: int = 3
    middle_global_uncertainty_threshold: float = 0.55
    late_explore_uncertainty_threshold: float = 0.65
    multimodal_explore_interval: int = 4
    multimodal_explore_uncertainty_threshold: float = 0.30
    multimodal_late_explore_uncertainty_threshold: float = 0.45
    hypothesis_alignment_weight: float = 0.15
    gp_verifier_enabled: bool = True
    gp_verifier_min_score_ratio: float = 0.75
    gp_verifier_duplicate_distance: float = 1e-4


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

    def choose_strategy(
        self,
        proposed_strategy: str,
        phase: str,
        trial_number: int,
        uncertainty: float,
        smoothness_label: str = "unknown",
        modality_label: str = "unknown",
    ) -> tuple[str, str | None, set[str]]:
        """Accept or repair an agent-proposed strategy.

        Output:
            Tuple ``(executed_strategy, override_reason, allowed_strategies)``.
        """

        allowed, gate_reasons = self.allowed_strategies(phase, trial_number, uncertainty, modality_label)
        proposed = str(proposed_strategy).strip().lower().replace("-", "_")

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

        if self.follow_up_local:
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
        for record in self.hypotheses:
            if record.strategy == strategy and record.status == "active":
                record.status = "expired"
        self.hypothesis_counter += 1
        record = HypothesisRecord(
            hypothesis_id=f"h{self.hypothesis_counter}",
            text=hypothesis_text,
            strategy=str(strategy),
            created_trial=int(trial_number),
            expires_trial=int(trial_number) + max(1, int(self.config.hypothesis_window)) - 1,
            baseline_best=float(baseline_best),
            confidence=_clamp(confidence, 0.0, 1.0),
            region_center=_normalise_region_center(region_center),
            region_radius=_normalise_region_radius(region_radius),
            sensitive_dims=_normalise_sensitive_dims(sensitive_dims),
            falsification_rule=_normalise_text(falsification_rule),
        )
        self.hypotheses.append(record)
        return record

    def record_outcome(self, strategy: str, trial_number: int, improved: bool, y: float, best_y: float) -> StrategyRecord:
        """Update trust, cooldowns, and hypothesis statuses after one trial."""

        key = str(strategy).strip().lower().replace("-", "_")
        if key not in STRATEGIES:
            key = "exploit_ei"
        reward = 1.0 if bool(improved) else 0.0
        alpha = float(self.config.trust_alpha)
        self.trusts[key] = (1.0 - alpha) * self.trusts[key] + alpha * reward
        self.outcomes[key].append(bool(improved))
        self.executed_strategies.append(key)
        self.follow_up_local = bool(improved)
        self.consecutive_no_improvement = 0 if improved else self.consecutive_no_improvement + 1

        cooldown: int | None = None
        recent = self.outcomes[key]
        if key in EXPLORATION_STRATEGIES and len(recent) == recent.maxlen and not any(recent):
            cooldown = int(trial_number) + int(self.config.failure_cooldown_trials) + 1
            self.cooldown_until[key] = cooldown

        for record in self.hypotheses:
            if record.status != "active":
                continue
            if improved and float(best_y) < record.baseline_best:
                record.status = "supported"
            elif int(trial_number) >= record.expires_trial:
                record.status = "rejected"

        outcome = StrategyRecord(
            strategy=key,
            trial_number=int(trial_number),
            improved=bool(improved),
            y=float(y),
            best_y=float(best_y),
            trust_after=float(self.trusts[key]),
            cooldown_until=cooldown,
        )
        self.strategy_history.append(outcome)
        return outcome

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
            if strategy in EXPLORATION_STRATEGIES:
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
    "WMBOState",
    "HypothesisRecord",
    "StrategyRecord",
    "build_default_optimizer_config",
    "should_stop",
    "update_run_state",
]
