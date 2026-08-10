"""Baseline optimiser implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
from scipy.stats import qmc

from .acquisition import AcquisitionInput, expected_improvement, score_candidates, select_next_candidate
from .agents import AgentState, CandidateValidator, ReasoningDecision, WorldModelAgent
from .portfolio import (
    MACRO_DURATIONS,
    V51_SUCCESSFUL_MACRO_MAX_STEPS,
    MacroActionState,
    OnePlusOneCMAState,
    PORTFOLIO_POLICY_VERSION,
    PORTFOLIO_STRATEGIES,
    V51_CANDIDATE_INFORMATION_MULTIPLIERS,
    TurboState,
    objective_scale,
    portfolio_shape,
)
from .benchmarks import BenchmarkSpec, EvaluationResult, sample_unit_points
from .control import STRATEGIES, OptimizerConfig, PortfolioWMBOState, WMBOControlConfig, WMBOState
from .descriptors import LandscapeDescriptor, describe_landscape
from .surrogate import SurrogateDataset, make_surrogate
from .llm_api import LLMAPIError, OpenAIStyleClient, decide_with_llm

Vector = Sequence[float]
Matrix = Sequence[Vector]


@dataclass(frozen=True)
class Observation:
    """One observed input-output pair.

    Inputs:
        x: Candidate point in normalised coordinates.
        y: Objective value.
        metadata: Optional evaluation metadata.

    Output:
        Stored inside ``OptimizerState``.
    """

    x: Vector
    y: float
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class OptimizerState:
    """State snapshot for an optimiser.

    Inputs:
        benchmark: Benchmark specification.
        observations: Evaluated input-output pairs.
        step: Current optimisation step.
        metadata: Optional method-specific state.

    Output:
        Passed between optimiser methods during a run.
    """

    benchmark: BenchmarkSpec
    observations: Sequence[Observation]
    step: int
    metadata: Mapping[str, object] = field(default_factory=dict)


class Optimizer(Protocol):
    """Protocol for optimisation algorithms."""

    def ask(self, state: OptimizerState) -> Vector:
        """Propose the next candidate point.

        Input:
            state: Current optimiser state.

        Output:
            Candidate point in normalised coordinates.
        """

        ...

    def tell(self, state: OptimizerState, result: EvaluationResult) -> OptimizerState:
        """Update optimiser state with a new evaluation result.

        Inputs:
            state: Current optimiser state.
            result: New objective evaluation result.

        Output:
            Updated optimiser state.
        """

        ...


class RandomSearchOptimizer:
    """Uniform random-search baseline."""

    def __init__(self, config: OptimizerConfig) -> None:
        """Create a random-search optimiser.

        Input:
            config: Optimiser configuration.

        Output:
            Optimiser instance.
        """

        self.config = config
        self._rng = random.Random(config.seed)

    def ask(self, state: OptimizerState) -> Vector:
        """Propose a random candidate.

        Input:
            state: Current optimiser state.

        Output:
            Candidate point in normalised coordinates.
        """

        _validate_state(state)
        return [self._rng.random() for _ in range(state.benchmark.dim)]

    def tell(self, state: OptimizerState, result: EvaluationResult) -> OptimizerState:
        """Update random-search state after an evaluation.

        Inputs:
            state: Current optimiser state.
            result: New evaluation result.

        Output:
            Updated optimiser state.
        """

        return append_observation(state, result)


class SobolSearchOptimizer:
    """Low-discrepancy Sobol search baseline."""

    def __init__(self, config: OptimizerConfig) -> None:
        """Create a Sobol-sequence optimiser.

        Input:
            config: Optimiser configuration.

        Output:
            Optimiser instance.
        """

        self.config = config
        self._sampler_by_dim: dict[int, qmc.Sobol] = {}
        self._cache_by_dim: dict[int, np.ndarray] = {}

    def ask(self, state: OptimizerState) -> Vector:
        """Propose the next Sobol candidate.

        Input:
            state: Current optimiser state.

        Output:
            Candidate point in normalised coordinates.
        """

        _validate_state(state)
        cache = self._get_cache(state.benchmark.dim)
        if state.step < len(cache):
            return cache[state.step].astype(float).tolist()

        rng = random.Random(self.config.seed + state.step)
        return [rng.random() for _ in range(state.benchmark.dim)]

    def tell(self, state: OptimizerState, result: EvaluationResult) -> OptimizerState:
        """Update Sobol-search state after an evaluation."""

        return append_observation(state, result)

    def _get_cache(self, dim: int) -> np.ndarray:
        if dim not in self._cache_by_dim:
            n_points = max(self.config.budget, self.config.initial_samples, 2)
            power = int(math.ceil(math.log2(n_points)))
            sampler = qmc.Sobol(d=dim, scramble=True, seed=self.config.seed)
            self._sampler_by_dim[dim] = sampler
            self._cache_by_dim[dim] = sampler.random_base2(m=power)
        return self._cache_by_dim[dim]


class BayesianOptimizationOptimizer:
    """Bayesian-optimisation baseline using a GP surrogate and acquisition search."""

    def __init__(self, config: OptimizerConfig, acquisition_strategy: str = "expected_improvement") -> None:
        """Create a Bayesian-optimisation baseline.

        Inputs:
            config: Optimiser configuration.
            acquisition_strategy: Acquisition strategy used after the initial design.

        Output:
            Optimiser instance.
        """

        self.config = config
        self.acquisition_strategy = acquisition_strategy
        self._initial = RandomSearchOptimizer(config)

    def ask(self, state: OptimizerState) -> Vector:
        """Propose a candidate using initial design or surrogate-guided search.

        Input:
            state: Current optimiser state.

        Output:
            Candidate point in normalised coordinates.
        """

        _validate_state(state)
        if len(state.observations) < max(1, self.config.initial_samples):
            return self._initial.ask(state)

        constrained = _constraint_handling_enabled(state, self.config)
        if constrained:
            failure_ratio = float(self.config.options.get("constraint_failure_ratio", 1.0e6))
            observed_x, observed_y, constraint_log_ratio, feasible = constrained_observations_to_arrays(
                state.observations,
                dim=state.benchmark.dim,
                failure_ratio=failure_ratio,
            )
            constraint_surrogate = _constraint_surrogate(
                config=self.config,
                dim=state.benchmark.dim,
                step=state.step,
                observed_x=observed_x,
                constraint_log_ratio=constraint_log_ratio,
            )
        else:
            observed_x, observed_y = observations_to_arrays(
                state.observations,
                dim=state.benchmark.dim,
            )
            feasible = np.ones(len(observed_y), dtype=bool)
            constraint_surrogate = None
        candidate_pool = sample_unit_points(
            n_points=max(1, self.config.candidate_pool_size),
            dim=state.benchmark.dim,
            seed=self.config.seed + 10_000 + state.step,
        )
        if constraint_surrogate is not None and np.any(feasible):
            candidate_pool = _augment_pool_with_feasible_neighbourhood(
                candidate_pool,
                feasible_x=observed_x[feasible],
                center=np.asarray(
                    best_observation(state.observations, constrained=True).x,
                    dtype=float,
                ),
                seed=self.config.seed + 11_000 + state.step,
            )
        surrogate = make_surrogate(
            kind=str(self.config.options.get("surrogate", "gaussian_process")),
            dim=state.benchmark.dim,
            options={
                "seed": self.config.seed + state.step,
                "noise_level": float(self.config.options.get("noise_level", 1e-6)),
            },
        )
        surrogate.fit(SurrogateDataset(x=observed_x.tolist(), y=observed_y.tolist()))
        prediction = surrogate.predict(candidate_pool)
        feasibility_probability = None
        constraint_std = None
        if constraint_surrogate is not None:
            feasibility_probability, _constraint_mean, constraint_std = _predict_feasibility(
                constraint_surrogate,
                candidate_pool,
                feasible_x=observed_x[feasible] if np.any(feasible) else None,
            )
        acquisition = select_next_candidate(
            AcquisitionInput(
                candidates=candidate_pool,
                observed_x=observed_x.tolist(),
                observed_y=observed_y.tolist(),
                surrogate_mean=prediction.mean,
                surrogate_std=prediction.std,
                strategy=self.acquisition_strategy,
                feasibility_probability=feasibility_probability,
                constraint_std=constraint_std,
                has_feasible_observation=bool(np.any(feasible)),
                feasibility_floor=float(self.config.options.get("constraint_pof_floor", 0.05)),
            )
        )
        return list(acquisition.selected_x)

    def tell(self, state: OptimizerState, result: EvaluationResult) -> OptimizerState:
        """Update BO state after an evaluation."""

        return append_observation(state, result)


class EvolutionStrategyOptimizer:
    """Small derivative-free baseline based on elite Gaussian mutation."""

    def __init__(self, config: OptimizerConfig) -> None:
        """Create a simple evolutionary baseline.

        Input:
            config: Optimiser configuration.

        Output:
            Optimiser instance.
        """

        self.config = config
        self._initial = RandomSearchOptimizer(config)
        self._rng = np.random.default_rng(config.seed)

    def ask(self, state: OptimizerState) -> Vector:
        """Propose a mutated point around the best observed candidates.

        Input:
            state: Current optimiser state.

        Output:
            Candidate point in normalised coordinates.
        """

        _validate_state(state)
        if len(state.observations) < max(1, self.config.initial_samples):
            return self._initial.ask(state)

        x_obs, y_obs = observations_to_arrays(state.observations, dim=state.benchmark.dim)
        elite_count = min(max(2, state.benchmark.dim), len(y_obs))
        constrained = _constraint_handling_enabled(state, self.config)
        elite_indices = np.asarray(
            sorted(
                range(len(state.observations)),
                key=lambda index: observation_rank_key(
                    state.observations[index],
                    constrained=constrained,
                ),
            )[:elite_count],
            dtype=int,
        )
        elite = x_obs[elite_indices]
        center = np.mean(elite, axis=0)
        scale = np.maximum(np.std(elite, axis=0), float(self.config.options.get("mutation_scale", 0.08)))
        candidate = center + self._rng.normal(0.0, scale, size=state.benchmark.dim)
        return np.clip(candidate, 0.0, 1.0).astype(float).tolist()

    def tell(self, state: OptimizerState, result: EvaluationResult) -> OptimizerState:
        """Update evolutionary state after an evaluation."""

        return append_observation(state, result)


class TPESearchOptimizer:
    """Optuna TPE baseline with a sequential ask/tell adapter."""

    def __init__(self, config: OptimizerConfig) -> None:
        self.config = config
        self._study: Any | None = None
        self._pending: Any | None = None
        self._synced = 0

    def _ensure_study(self, state: OptimizerState) -> None:
        if self._study is not None:
            return
        try:
            import optuna
        except ImportError as exc:
            raise RuntimeError("TPE requires optuna==4.9.0") from exc
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(
            seed=self.config.seed,
            multivariate=True,
            n_startup_trials=0,
        )
        self._study = optuna.create_study(direction="minimize", sampler=sampler)
        distributions = {
            f"x_{index}": optuna.distributions.FloatDistribution(0.0, 1.0)
            for index in range(state.benchmark.dim)
        }
        for observation in state.observations:
            trial = optuna.trial.create_trial(
                params={f"x_{index}": float(value) for index, value in enumerate(observation.x)},
                distributions=distributions,
                value=float(observation.y),
            )
            self._study.add_trial(trial)
            self._synced += 1

    def ask(self, state: OptimizerState) -> Vector:
        _validate_state(state)
        self._ensure_study(state)
        self._pending = self._study.ask()
        return [
            float(self._pending.suggest_float(f"x_{index}", 0.0, 1.0))
            for index in range(state.benchmark.dim)
        ]

    def tell(self, state: OptimizerState, result: EvaluationResult) -> OptimizerState:
        if self._study is not None and self._pending is not None:
            self._study.tell(self._pending, float(result.y))
            self._pending = None
            self._synced += 1
        return append_observation(state, result)


class CMAESOptimizer:
    """pycma baseline adapted to the repository's one-point ask/tell loop."""

    def __init__(self, config: OptimizerConfig) -> None:
        self.config = config
        self._strategy: Any | None = None
        self._generation: list[list[float]] = []
        self._values: list[float] = []
        self._index = 0

    def _ensure_strategy(self, state: OptimizerState) -> None:
        if self._strategy is not None:
            return
        try:
            import cma
        except ImportError as exc:
            raise RuntimeError("CMA-ES requires cma==4.4.4") from exc
        best = best_observation(state.observations, constrained=False)
        center = list(best.x) if best is not None else [0.5] * state.benchmark.dim
        self._strategy = cma.CMAEvolutionStrategy(
            center,
            float(self.config.options.get("cma_sigma0", 0.3)),
            {
                "bounds": [0.0, 1.0],
                "seed": int(self.config.seed),
                "verbose": -9,
                "verb_disp": 0,
            },
        )

    def ask(self, state: OptimizerState) -> Vector:
        _validate_state(state)
        self._ensure_strategy(state)
        if not self._generation or self._index >= len(self._generation):
            self._generation = [
                np.clip(np.asarray(candidate, dtype=float), 0.0, 1.0).tolist()
                for candidate in self._strategy.ask()
            ]
            self._values = []
            self._index = 0
        return list(self._generation[self._index])

    def tell(self, state: OptimizerState, result: EvaluationResult) -> OptimizerState:
        if self._generation:
            self._values.append(float(result.y))
            self._index += 1
            if self._index == len(self._generation):
                self._strategy.tell(self._generation, self._values)
        return append_observation(state, result)


class HEBOOptimizer:
    """Official HEBO baseline with warm-start ingestion."""

    def __init__(self, config: OptimizerConfig) -> None:
        self.config = config
        self._optimizer: Any | None = None
        self._pending: Any | None = None

    def _ensure_optimizer(self, state: OptimizerState) -> None:
        if self._optimizer is not None:
            return
        try:
            import pandas as pd
            from hebo.design_space.design_space import DesignSpace
            from hebo.optimizers.hebo import HEBO
        except ImportError as exc:
            raise RuntimeError("HEBO requires HEBO==0.3.6 and its dependencies") from exc
        space = DesignSpace().parse(
            [
                {"name": f"x_{index}", "type": "num", "lb": 0.0, "ub": 1.0}
                for index in range(state.benchmark.dim)
            ]
        )
        self._optimizer = HEBO(space, rand_sample=0)
        if state.observations:
            frame = pd.DataFrame(
                [
                    {f"x_{index}": float(value) for index, value in enumerate(observation.x)}
                    for observation in state.observations
                ]
            )
            values = np.asarray([observation.y for observation in state.observations], dtype=float).reshape(-1, 1)
            self._optimizer.observe(frame, values)

    def ask(self, state: OptimizerState) -> Vector:
        _validate_state(state)
        self._ensure_optimizer(state)
        self._pending = self._optimizer.suggest(n_suggestions=1)
        return [
            float(self._pending.iloc[0][f"x_{index}"])
            for index in range(state.benchmark.dim)
        ]

    def tell(self, state: OptimizerState, result: EvaluationResult) -> OptimizerState:
        if self._optimizer is not None and self._pending is not None:
            self._optimizer.observe(
                self._pending,
                np.asarray([[float(result.y)]], dtype=float),
            )
            self._pending = None
        return append_observation(state, result)


class WMBOOptimizer:
    """Rule-based world-model black-box optimiser.

    This first WMBO implementation uses deterministic landscape descriptors and
    reasoning rules. It intentionally avoids external LLM calls so that the
    optimisation path is reproducible at this stage of the project.
    """

    def __init__(self, config: OptimizerConfig) -> None:
        """Create a WMBO optimiser.

        Input:
            config: Optimiser configuration.

        Output:
            Optimiser instance.
        """

        self.config = config
        self._initial = RandomSearchOptimizer(config)
        raw_rule_policy = self.config.options.get("rule_policy", {})
        if not isinstance(raw_rule_policy, Mapping):
            raise ValueError("optimizer.options.rule_policy must be a mapping.")
        self._agent = WorldModelAgent(raw_rule_policy)
        control_config = _wmbo_control_config_from_options(self.config.options)
        self._portfolio_v5 = self._agent.config.mode == "portfolio_v5"
        self._control = (
            PortfolioWMBOState(control_config)
            if self._portfolio_v5
            else WMBOState(control_config)
        )
        self._llm_client: OpenAIStyleClient | None = None
        self._last_decision: Mapping[str, object] | None = None
        self._last_acquisition: Mapping[str, object] | None = None
        self._pending_trial: Mapping[str, object] | None = None
        self._macro_action: MacroActionState | None = None
        self._macro_counter = 0
        self._turbo_state: TurboState | None = None
        self._cma_state: OnePlusOneCMAState | None = None

    def _finish_macro(self, reason: str, trial_number: int) -> dict[str, object] | None:
        if self._macro_action is None:
            return None
        macro = self._macro_action
        macro.termination_reason = str(reason)
        macro_phase = self._control.budget_phase(
            max(0, macro.start_trial - 1), self.config.budget
        )
        reward, components = macro.reward(phase=macro_phase)
        cooldown_until = self._control.record_delayed_reward(
            macro.strategy,
            reward=reward,
            trial_number=trial_number,
            improved=bool(components.get("objective_success", 0.0) > 0.5),
        )
        summary = {
            **macro.to_dict(phase=macro_phase),
            "macro_continued": False,
            "macro_reward": reward,
            "macro_reward_components": components,
            "macro_cooldown_until": cooldown_until,
        }
        self._macro_action = None
        return summary
    def ask(self, state: OptimizerState) -> Vector:
        """Propose a candidate using rule-based world-model reasoning.

        Input:
            state: Current optimiser state.

        Output:
            Candidate point in normalised coordinates.
        """

        _validate_state(state)
        if len(state.observations) < max(1, self.config.initial_samples):
            candidate = self._initial.ask(state)
            phase = self._control.budget_phase(state.step, self.config.budget)
            self._pending_trial = None
            self._last_decision = {
                "strategy": "initial_design",
                "executed_strategy": "initial_design",
                "rationale": "Collecting initial random design points before fitting a world model.",
                "budget_phase": phase,
                "remaining_budget": self._control.remaining_budget(state.step, self.config.budget),
                "wmbo_control": self._control.to_dict(),
            }
            self._last_acquisition = {"strategy": "random"}
            return candidate

        constrained = _constraint_handling_enabled(state, self.config)
        if constrained:
            failure_ratio = float(self.config.options.get("constraint_failure_ratio", 1.0e6))
            observed_x, observed_y, constraint_log_ratio, feasible = constrained_observations_to_arrays(
                state.observations,
                dim=state.benchmark.dim,
                failure_ratio=failure_ratio,
            )
            constraint_surrogate = _constraint_surrogate(
                config=self.config,
                dim=state.benchmark.dim,
                step=state.step,
                observed_x=observed_x,
                constraint_log_ratio=constraint_log_ratio,
            )
        else:
            observed_x, observed_y = observations_to_arrays(
                state.observations,
                dim=state.benchmark.dim,
            )
            feasible = np.ones(len(observed_y), dtype=bool)
            constraint_surrogate = None
        ranked_best = best_observation(state.observations, constrained=constrained)
        best_x = (
            np.asarray(ranked_best.x, dtype=float)
            if ranked_best is not None
            else np.full(state.benchmark.dim, 0.5)
        )
        surrogate = make_surrogate(
            kind=str(self.config.options.get("surrogate", "gaussian_process")),
            dim=state.benchmark.dim,
            options={
                "seed": self.config.seed + state.step,
                "noise_level": float(self.config.options.get("noise_level", 1e-6)),
            },
        )
        surrogate.fit(SurrogateDataset(x=observed_x.tolist(), y=observed_y.tolist()))

        descriptor_pool = sample_unit_points(
            n_points=max(16, min(self.config.candidate_pool_size, 256)),
            dim=state.benchmark.dim,
            seed=self.config.seed + 20_000 + state.step,
        )
        descriptor_prediction = surrogate.predict(descriptor_pool)
        if _truthy(self.config.options.get("disable_world_model_descriptor", False)):
            descriptor = LandscapeDescriptor(
                dim=state.benchmark.dim,
                num_observations=len(observed_y),
                best_y=float(np.min(observed_y)),
                y_range=float(np.max(observed_y) - np.min(observed_y)),
                uncertainty=1.0,
                labels={
                    "smoothness": "unknown",
                    "modality": "unknown",
                    "curvature": "unknown",
                    "anisotropy": "unknown",
                },
                calibration={"world_model_entropy": 0.0},
            )
        else:
            descriptor = describe_landscape(
                observed_x=observed_x.tolist(),
                observed_y=observed_y.tolist(),
                surrogate_metadata={
                    **dict(descriptor_prediction.metadata),
                    "mean_std": float(np.mean(descriptor_prediction.std)) if descriptor_prediction.std else 1.0,
                },
            )
        phase = self._control.budget_phase(state.step, self.config.budget)
        hypothesis_tracking = not _truthy(
            self.config.options.get("disable_hypothesis_tracking", False)
        )
        active_hypotheses = self._control.hypothesis_summary() if hypothesis_tracking else []
        if self._portfolio_v5:
            if self._turbo_state is None or self._turbo_state.dim != state.benchmark.dim:
                self._turbo_state = TurboState(dim=state.benchmark.dim)
            if self._cma_state is None or self._cma_state.dim != state.benchmark.dim:
                self._cma_state = OnePlusOneCMAState(
                    dim=state.benchmark.dim, seed=self.config.seed + 90_000
                )
            candidate_options = _build_portfolio_candidate_options(
                surrogate=surrogate,
                observed_x=observed_x,
                observed_y=observed_y,
                dim=state.benchmark.dim,
                n_pool=max(1, self.config.candidate_pool_size),
                n_options_per_strategy=max(
                    1, int(self._control.config.candidate_options_per_strategy)
                ),
                seed=self.config.seed + 25_000 + state.step,
                active_hypotheses=active_hypotheses,
                phase=phase,
                information_gain_weight=float(self._control.config.information_gain_weight),
                world_model_entropy=float(
                    descriptor.calibration.get("world_model_entropy", 1.0)
                ),
                constraint_surrogate=constraint_surrogate,
                has_feasible_observation=bool(np.any(feasible)),
                feasibility_floor=float(
                    self.config.options.get("constraint_pof_floor", 0.05)
                ),
                best_x=best_x,
                feasible_x=observed_x[feasible] if np.any(feasible) else None,
                lengthscales=dict(descriptor_prediction.metadata).get("lengthscales"),
                turbo_state=self._turbo_state,
                cma_state=self._cma_state,
            )
        else:
            candidate_options = _build_strategy_candidate_options(
                surrogate=surrogate,
                observed_x=observed_x,
                observed_y=observed_y,
                dim=state.benchmark.dim,
                n_pool=max(1, self.config.candidate_pool_size),
                n_options_per_strategy=max(
                    1, int(self._control.config.candidate_options_per_strategy)
                ),
                seed=self.config.seed + 25_000 + state.step,
                phase=phase,
                active_hypotheses=active_hypotheses,
                hypothesis_alignment_weight=(
                    float(self._control.config.hypothesis_alignment_weight)
                    if hypothesis_tracking else 0.0
                ),
                information_gain_weight=float(self._control.config.information_gain_weight),
                world_model_entropy=float(
                    descriptor.calibration.get("world_model_entropy", 1.0)
                ),
                constraint_surrogate=constraint_surrogate,
                has_feasible_observation=bool(np.any(feasible)),
                feasibility_floor=float(
                    self.config.options.get("constraint_pof_floor", 0.05)
                ),
                best_x=best_x,
                feasible_x=observed_x[feasible] if np.any(feasible) else None,
            )
        labels = dict(descriptor.labels)
        context_kwargs: dict[str, object] = {
            "phase": phase,
            "trial_number": state.step + 1,
            "completed_trials": state.step,
            "budget": self.config.budget,
            "uncertainty": float(
                descriptor.uncertainty if descriptor.uncertainty is not None else 1.0
            ),
            "smoothness_label": labels.get("smoothness", "unknown"),
            "modality_label": labels.get("modality", "unknown"),
            "flexible_local_follow_up": self._agent.config.mode == "continuous_v4",
        }
        if self._portfolio_v5:
            context_kwargs["has_feasible_observation"] = bool(np.any(feasible))
            context_kwargs["coverage"] = float(descriptor.coverage)
            regime_values = {
                str(name): float(value)
                for name, value in descriptor.regime_posteriors.items()
            }
            weak_value = float(regime_values.get("weakly_identified", 1.0))
            ordered_regimes = sorted(regime_values.values(), reverse=True)
            top2_cutoff = ordered_regimes[min(1, len(ordered_regimes) - 1)] if ordered_regimes else 1.0
            probability_vector = np.asarray(list(regime_values.values()), dtype=float)
            probability_vector /= max(float(np.sum(probability_vector)), 1.0e-12)
            regime_entropy = float(
                -np.sum(
                    probability_vector
                    * np.log(np.clip(probability_vector, 1.0e-12, 1.0))
                ) / math.log(max(2, len(probability_vector)))
            ) if len(probability_vector) else 1.0
            context_kwargs["weakly_identified"] = weak_value
            context_kwargs["weakly_identified_top2"] = bool(weak_value >= top2_cutoff - 1.0e-12)
            context_kwargs["regime_entropy"] = regime_entropy
            context_kwargs["available_strategies"] = sorted(
                {
                    str(option.get("strategy"))
                    for option in candidate_options
                    if option.get("candidate_id") is not None
                }
            )
        strategy_context = self._control.decision_context(**context_kwargs)

        settled_macro = None
        if self._portfolio_v5 and self._macro_action is not None:
            if self._macro_action.strategy not in strategy_context.allowed_strategies:
                settled_macro = self._finish_macro(
                    "strategy_gated", trial_number=state.step + 1
                )
            elif self._control.remaining_budget(state.step, self.config.budget) <= 0:
                settled_macro = self._finish_macro(
                    "budget_exhausted", trial_number=state.step + 1
                )
        macro_continued = self._portfolio_v5 and self._macro_action is not None
        decision_context = strategy_context.to_dict()
        if macro_continued:
            decision_context.update(
                {
                    "forced_strategy": self._macro_action.strategy,
                    "forced_reason": "macro_action_continuation",
                }
            )
            decision = ReasoningDecision(
                strategy=self._macro_action.strategy,
                hypothesis="Continuing the active macro action with an updated surrogate.",
                confidence=0.95,
                rationale="The operator remains legal and its macro duration is not complete.",
                metadata={
                    "source": "macro_continuation",
                    "rule_policy_mode": "portfolio_v5",
                    "rule_policy_version": PORTFOLIO_POLICY_VERSION,
                    "allowed_strategies": list(strategy_context.allowed_strategies),
                    "forced_strategy": self._macro_action.strategy,
                },
            )
            agent_type, llm_error = "macro_continuation", None
        else:
            decision, agent_type, llm_error = self._decide(
                state=state,
                descriptor=descriptor,
                observed_x=observed_x,
                observed_y=observed_y,
                candidate_options=candidate_options,
                phase=phase,
                decision_context=decision_context,
            )
        shared_context_enabled = (
            self._agent.config.mode in {
                "continuous_v2", "continuous_v3", "continuous_v4", "portfolio_v5"
            }
            or _truthy(self.config.options.get("use_llm_agent", False))
        )
        execution_context = strategy_context if shared_context_enabled else None
        if macro_continued:
            executed_strategy = self._macro_action.strategy
            override_reason = None
            allowed_strategies = set(strategy_context.allowed_strategies)
        else:
            executed_strategy, override_reason, allowed_strategies = self._control.choose_strategy(
                proposed_strategy=decision.strategy,
                phase=phase,
                trial_number=state.step + 1,
                uncertainty=float(
                    descriptor.uncertainty if descriptor.uncertainty is not None else 1.0
                ),
                smoothness_label=labels.get("smoothness", "unknown"),
                modality_label=labels.get("modality", "unknown"),
                decision_context=execution_context,
            )

        selected_option, candidate_override = _select_strategy_candidate(
            candidate_options,
            strategy=executed_strategy,
            requested_candidate_id=decision.selected_candidate_id,
        )
        selected_option, gp_verifier_metadata, gp_verifier_override = _verify_candidate_with_gp(
            candidates=candidate_options,
            selected=selected_option,
            strategy=executed_strategy,
            decision_confidence=float(decision.confidence),
            config=self._control.config,
        )
        candidate_override = _append_override(candidate_override, gp_verifier_override)
        candidate = list(selected_option["x_unit"])
        validator = CandidateValidator(dim=state.benchmark.dim)
        is_valid, _issues = validator.validate(candidate)
        if not is_valid:
            candidate = validator.repair(candidate)
            candidate_override = "candidate_repaired" if candidate_override is None else f"{candidate_override};candidate_repaired"

        if self._portfolio_v5 and self._macro_action is None:
            minimum, maximum = MACRO_DURATIONS[executed_strategy]
            remaining = max(
                1, self._control.remaining_budget(state.step, self.config.budget)
            )
            if remaining < minimum:
                minimum = remaining
                maximum = remaining
            else:
                maximum = min(maximum, remaining)
            self._macro_counter += 1
            start_best = (
                _constraint_progress_value(state.observations, constrained=constrained)
                if state.observations else float(np.min(observed_y))
            )
            self._macro_action = MacroActionState(
                macro_id=f"macro_{self._macro_counter:05d}",
                strategy=executed_strategy,
                start_trial=state.step + 1,
                start_best=float(start_best),
                start_scale=objective_scale(observed_y),
                min_steps=minimum,
                max_steps=maximum,
            )

        structured_hypothesis = _structured_hypothesis(
            decision=decision,
            candidate=candidate,
            dim=state.benchmark.dim,
            strategy=executed_strategy,
            phase=phase,
            trial_number=state.step + 1,
            hypothesis_window=int(self._control.config.hypothesis_window),
        )
        if macro_continued and self._macro_action is not None:
            hypothesis_record = next(
                (
                    record for record in self._control.hypotheses
                    if record.hypothesis_id == self._macro_action.hypothesis_id
                ),
                None,
            )
        else:
            hypothesis_record = (
                self._control.create_hypothesis(
                    text=decision.hypothesis,
                    strategy=executed_strategy,
                    trial_number=state.step + 1,
                    baseline_best=(
                        _constraint_progress_value(state.observations, constrained=constrained)
                        if state.observations else float(np.min(observed_y))
                    ),
                    confidence=structured_hypothesis["confidence"],
                    region_center=structured_hypothesis["region_center"],
                    region_radius=structured_hypothesis["region_radius"],
                    sensitive_dims=structured_hypothesis["sensitive_dims"],
                    falsification_rule=structured_hypothesis["falsification_rule"],
                )
                if hypothesis_tracking else None
            )
            if self._macro_action is not None:
                self._macro_action.hypothesis_id = (
                    hypothesis_record.hypothesis_id if hypothesis_record else None
                )

        self._pending_trial = {
            "strategy": executed_strategy,
            "trial_number": state.step + 1,
            "hypothesis_id": hypothesis_record.hypothesis_id if hypothesis_record else None,
            "candidate": list(candidate),
            "predicted_mean": selected_option.get("surrogate_mean"),
            "predicted_std": selected_option.get("surrogate_std"),
            "predicted_feasibility_probability": selected_option.get("predicted_feasibility_probability"),
            "predicted_constraint_log_ratio": selected_option.get("predicted_constraint_log_ratio"),
            "evidence_role": selected_option.get("evidence_role"),
            "target_hypothesis_id": selected_option.get("target_hypothesis_id"),
            "information_gain": selected_option.get("information_gain", 0.0),
            "macro_action_id": (
                self._macro_action.macro_id if self._macro_action is not None else None
            ),
            "macro_step": (
                self._macro_action.completed_steps + 1
                if self._macro_action is not None else None
            ),
        }
        self._last_decision = {
            **decision.to_dict(),
            **{
                key: value
                for key, value in dict(decision.metadata).items()
                if str(key).startswith("llm_")
            },
            "proposed_strategy": decision.strategy,
            "executed_strategy": executed_strategy,
            "override_reason": override_reason,
            "allowed_strategies": sorted(allowed_strategies),
            "strategy_scores": decision.metadata.get("strategy_scores"),
            "strategy_scores_pre_penalty": decision.metadata.get("strategy_scores_pre_penalty"),
            "strategy_score_penalties": decision.metadata.get("strategy_score_penalties"),
            "strategy_penalty_reasons": decision.metadata.get("strategy_penalty_reasons"),
            "local_operator_recent_shares": decision.metadata.get("local_operator_recent_shares"),
            "score_components": decision.metadata.get("score_components"),
            "routing_weights_effective": decision.metadata.get("routing_weights_effective"),
            "rule_policy_version": decision.metadata.get(
                "rule_policy_version",
                PORTFOLIO_POLICY_VERSION if self._portfolio_v5 else None,
            ),
            "masked_strategies": decision.metadata.get("masked_strategies", []),
            "forced_strategy": decision_context.get("forced_strategy"),
            "forced_reason": decision_context.get("forced_reason"),
            "strategy_gate_reasons": list(strategy_context.gate_reasons),
            "macro_action_id": (
                self._macro_action.macro_id if self._macro_action is not None else None
            ),
            "macro_strategy": (
                self._macro_action.strategy if self._macro_action is not None else None
            ),
            "macro_step": (
                self._macro_action.completed_steps + 1
                if self._macro_action is not None else None
            ),
            "macro_min_steps": (
                self._macro_action.min_steps if self._macro_action is not None else None
            ),
            "macro_max_steps": (
                self._macro_action.max_steps if self._macro_action is not None else None
            ),
            "macro_continued": bool(macro_continued),
            "macro_termination_reason": None,
            "macro_reward": None,
            "macro_reward_components": None,
            "previous_macro_settlement": settled_macro,
            "operator_state_summary": {
                "anisotropic_turbo": (
                    self._turbo_state.to_dict() if self._turbo_state is not None else None
                ),
                "cma_local": (
                    self._cma_state.to_dict() if self._cma_state is not None else None
                ),
            },
            "geometry_features": {
                key: getattr(descriptor, key, None)
                for key in (
                    "lengthscale_condition", "local_condition", "rotation_score",
                    "effective_dimension", "valley_score", "geometry_reliability",
                    "lengthscale_reliability",
                )
            },
            "regime_posteriors": dict(descriptor.regime_posteriors),
            "selected_candidate_evidence": decision.metadata.get("selected_candidate_evidence"),
            "landscape_descriptor": descriptor.to_dict(),
            "strategy_decision_context": strategy_context.to_dict(),
            "budget_phase": phase,
            "remaining_budget": self._control.remaining_budget(state.step, self.config.budget),
            "strategy_trust": dict(self._control.trusts),
            "strategy_success_rates": self._control.recent_success_rates(),
            "hypothesis_id": hypothesis_record.hypothesis_id if hypothesis_record else None,
            "hypothesis_status": hypothesis_record.status if hypothesis_record else None,
            "hypothesis_region_center": hypothesis_record.region_center if hypothesis_record else None,
            "hypothesis_region_radius": hypothesis_record.region_radius if hypothesis_record else None,
            "hypothesis_sensitive_dims": hypothesis_record.sensitive_dims if hypothesis_record else None,
            "hypothesis_confidence": hypothesis_record.confidence if hypothesis_record else None,
            "hypothesis_posterior_probability": hypothesis_record.posterior_probability if hypothesis_record else None,
            "hypothesis_relevant_evidence_count": hypothesis_record.relevant_evidence_count if hypothesis_record else 0,
            "falsification_rule": hypothesis_record.falsification_rule if hypothesis_record else None,
            "hypothesis_status_counts": self._control.hypothesis_status_counts(),
            "agent_type": agent_type,
            "llm_error": llm_error,
            "candidate_options": candidate_options,
            "requested_candidate_id": decision.selected_candidate_id,
            "selected_candidate_id": selected_option.get("candidate_id"),
            "evidence_role": selected_option.get("evidence_role"),
            "target_hypothesis_id": selected_option.get("target_hypothesis_id"),
            "joint_score": selected_option.get("joint_score"),
            "expected_improvement": selected_option.get("expected_improvement"),
            "information_gain": selected_option.get("information_gain"),
            "information_gain_weight_effective": selected_option.get("information_gain_weight_effective"),
            "predicted_feasibility_probability": selected_option.get("predicted_feasibility_probability"),
            "predicted_constraint_log_ratio": selected_option.get("predicted_constraint_log_ratio"),
            "candidate_override": candidate_override,
            "gp_verifier": gp_verifier_metadata,
            "wmbo_control": self._control.to_dict(),
        }
        self._last_acquisition = {
            "score": selected_option.get("acquisition_score"),
            "selection_score": selected_option.get("selection_score"),
            "joint_score": selected_option.get("joint_score"),
            "expected_improvement": selected_option.get("expected_improvement"),
            "optimisation_utility": selected_option.get("optimisation_utility"),
            "information_gain": selected_option.get("information_gain"),
            "predicted_feasibility_probability": selected_option.get("predicted_feasibility_probability"),
            "predicted_constraint_log_ratio": selected_option.get("predicted_constraint_log_ratio"),
            "constraint_std": selected_option.get("constraint_std"),
            "confirmation_value": selected_option.get("confirmation_value"),
            "falsification_value": selected_option.get("falsification_value"),
            "selected_candidate_id": selected_option.get("candidate_id"),
            "strategy": executed_strategy,
            "acquisition_strategy": selected_option.get("acquisition_strategy"),
            "evidence_role": selected_option.get("evidence_role"),
            "target_hypothesis_id": selected_option.get("target_hypothesis_id"),
            "surrogate_mean": selected_option.get("surrogate_mean"),
            "surrogate_std": selected_option.get("surrogate_std"),
            "hypothesis_alignment": selected_option.get("hypothesis_alignment"),
            "gp_verifier": gp_verifier_metadata,
            "descriptor": descriptor.to_dict(),
        }
        return candidate

    def _decide(
        self,
        *,
        state: OptimizerState,
        descriptor: LandscapeDescriptor,
        observed_x: np.ndarray,
        observed_y: np.ndarray,
        candidate_options: Sequence[Mapping[str, object]],
        phase: str,
        decision_context: Mapping[str, object],
    ) -> tuple[Any, str, str | None]:
        """Return a rule or LLM decision plus source metadata."""

        rule_state = AgentState(
            observed_x=observed_x.tolist(),
            observed_y=observed_y.tolist(),
            descriptor=descriptor.to_dict(),
            budget_used=state.step,
            budget_total=self.config.budget,
            candidate_options=candidate_options,
            decision_context=decision_context,
        )
        use_llm = _truthy(self.config.options.get("use_llm_agent", False))
        if not use_llm:
            return self._agent.decide(rule_state), "rule", None

        llm_decision_context = {
            **dict(decision_context),
            "budget_phase": phase,
            "remaining_budget": self._control.remaining_budget(state.step, self.config.budget),
            "consecutive_no_improvement": self._control.consecutive_no_improvement,
            "strategy_trust": dict(self._control.trusts),
            "strategy_success_rates": self._control.recent_success_rates(),
            "recent_hypotheses": (
                []
                if _truthy(self.config.options.get("disable_hypothesis_tracking", False))
                else self._control.hypothesis_summary()
            ),
            "feasible_observations": sum(
                bool(item.metadata.get("feasible", False))
                for item in state.observations
            ),
            "best_feasible_gap": min(
                (
                    _observation_objective(item)
                    for item in state.observations
                    if bool(item.metadata.get("feasible", False))
                ),
                default=None,
            ),
            "minimum_constraint_ratio": min(
                (
                    _observation_constraint_ratio(item, failure_ratio=1.0e6)
                    for item in state.observations
                ),
                default=None,
            ),
        }
        try:
            if self._llm_client is None:
                self._llm_client = OpenAIStyleClient.from_mapping(_llm_options(self.config.options))
            decision = decide_with_llm(
                descriptor,
                client=self._llm_client,
                candidates=candidate_options,
                decision_context=llm_decision_context,
                model=_optional_str(self.config.options.get("llm_model") or self.config.options.get("api_model")),
                temperature=float(self.config.options.get("llm_temperature", 0.0)),
                log_io=_truthy(self.config.options.get("llm_log_io", False)),
            )
            return decision, "llm", None
        except LLMAPIError as exc:
            if not _truthy(self.config.options.get("llm_fallback_to_rule", True)):
                raise
            fallback = self._agent.decide(rule_state)
            fallback_metadata = dict(fallback.metadata)
            fallback_metadata["llm_error"] = str(exc)
            fallback_decision = type(fallback)(
                strategy=fallback.strategy,
                hypothesis=fallback.hypothesis,
                confidence=fallback.confidence,
                rationale=fallback.rationale,
                world_model=fallback.world_model,
                selected_candidate_id=fallback.selected_candidate_id,
                hypothesis_region_center=fallback.hypothesis_region_center,
                hypothesis_region_radius=fallback.hypothesis_region_radius,
                hypothesis_sensitive_dims=fallback.hypothesis_sensitive_dims,
                falsification_rule=fallback.falsification_rule,
                metadata=fallback_metadata,
            )
            return fallback_decision, "rule_fallback", str(exc)

    def tell(self, state: OptimizerState, result: EvaluationResult) -> OptimizerState:
        """Update WMBO state after an evaluation.

        Inputs:
            state: Current optimiser state.
            result: New evaluation result.

        Output:
            Updated optimiser state.
        """

        constrained = _constraint_handling_enabled(state, self.config)
        previous_best = _constraint_progress_value(
            state.observations,
            constrained=constrained,
        )
        updated = append_observation(state, result)
        metadata = dict(updated.metadata)

        macro_metadata: dict[str, object] | None = None
        outcome_metadata: dict[str, object] | None = None
        if self._pending_trial is not None and previous_best is not None:
            improved, outcome_y, best_y = _result_improves_constraint_progress(
                state.observations,
                result,
                constrained=constrained,
            )
            outcome = self._control.record_outcome(
                strategy=str(self._pending_trial.get("strategy", "exploit_ei")),
                trial_number=int(self._pending_trial.get("trial_number", updated.step)),
                improved=improved,
                y=outcome_y,
                best_y=best_y,
                candidate=self._pending_trial.get("candidate"),
                predicted_mean=self._pending_trial.get("predicted_mean"),
                predicted_std=self._pending_trial.get("predicted_std"),
                evidence_role=str(self._pending_trial.get("evidence_role") or "optimize"),
                target_hypothesis_id=(
                    str(self._pending_trial["target_hypothesis_id"])
                    if self._pending_trial.get("target_hypothesis_id") is not None
                    else None
                ),
                update_trust=not self._portfolio_v5,
            )
            outcome_metadata = outcome.to_dict()
            if self._portfolio_v5 and self._macro_action is not None:
                strategy = str(self._pending_trial.get("strategy", ""))
                candidate_value = self._pending_trial.get("candidate")
                if strategy == "anisotropic_turbo" and self._turbo_state is not None:
                    self._turbo_state.update(improved)
                if (
                    strategy == "cma_local"
                    and self._cma_state is not None
                    and isinstance(candidate_value, Sequence)
                ):
                    self._cma_state.update(candidate_value, outcome_y, improved)
                self._macro_action.record(
                    best_y=best_y,
                    improved=improved,
                    information_gain=float(
                        self._pending_trial.get("information_gain", 0.0) or 0.0
                    ),
                )
                if improved:
                    extension_ceiling = V51_SUCCESSFUL_MACRO_MAX_STEPS.get(strategy)
                    if extension_ceiling is not None:
                        self._macro_action.extend_after_success(extension_ceiling)
                termination = self._macro_action.should_stop()
                if updated.step >= self.config.budget:
                    termination = "budget_exhausted"
                if termination is not None:
                    macro_metadata = self._finish_macro(
                        termination, trial_number=updated.step
                    )
                else:
                    macro_metadata = {
                        **self._macro_action.to_dict(),
                        "macro_continued": True,
                        "operator_state_summary": {
                            "anisotropic_turbo": self._turbo_state.to_dict(),
                            "cma_local": self._cma_state.to_dict(),
                        },
                    }

        if self._last_decision is not None:
            decision_metadata = dict(self._last_decision)
            if outcome_metadata is not None:
                decision_metadata["last_outcome"] = outcome_metadata
                hypothesis_id = decision_metadata.get("hypothesis_id")
                if hypothesis_id:
                    record = next(
                        (record for record in self._control.hypotheses if record.hypothesis_id == hypothesis_id),
                        None,
                    )
                    if record is not None:
                        decision_metadata["hypothesis_status"] = record.status
                        decision_metadata["hypothesis_posterior_probability"] = record.posterior_probability
                        decision_metadata["hypothesis_relevant_evidence_count"] = record.relevant_evidence_count
                        decision_metadata["hypothesis_supporting_evidence"] = record.supporting_evidence
                        decision_metadata["hypothesis_contradicting_evidence"] = record.contradicting_evidence
                        decision_metadata["hypothesis_last_evidence"] = record.last_evidence
                    else:
                        decision_metadata["hypothesis_status"] = None
            if macro_metadata is not None:
                decision_metadata.update(macro_metadata)
            decision_metadata["strategy_trust"] = dict(self._control.trusts)
            decision_metadata["strategy_success_rates"] = self._control.recent_success_rates()
            decision_metadata["hypothesis_status_counts"] = self._control.hypothesis_status_counts()
            decision_metadata["wmbo_control"] = self._control.to_dict()
            metadata["last_reasoning_decision"] = decision_metadata
        if self._last_acquisition is not None:
            metadata["last_acquisition"] = dict(self._last_acquisition)
        metadata["wmbo_control"] = self._control.to_dict()
        return OptimizerState(
            benchmark=updated.benchmark,
            observations=updated.observations,
            step=updated.step,
            metadata=metadata,
        )


def make_optimizer(method: str, config: OptimizerConfig) -> Optimizer:
    """Construct an optimiser by name.

    Inputs:
        method: Optimisation method name.
        config: Optimiser configuration.

    Output:
        Object implementing the ``Optimizer`` protocol.
    """

    name = str(config.options.get("base_method") or method).strip().lower().replace("-", "_")
    if name in {"random", "random_search"}:
        return RandomSearchOptimizer(config)
    if name in {"sobol", "sobol_search", "quasi_random"}:
        return SobolSearchOptimizer(config)
    if name in {"bo", "bo_ei", "bayesian_optimization", "bayesian_optimisation"}:
        return BayesianOptimizationOptimizer(config, acquisition_strategy="expected_improvement")
    if name in {"bo_pi", "bayesian_optimization_pi"}:
        return BayesianOptimizationOptimizer(config, acquisition_strategy="probability_improvement")
    if name in {"bo_lcb", "bo_ucb", "bayesian_optimization_lcb"}:
        return BayesianOptimizationOptimizer(config, acquisition_strategy="lower_confidence_bound")
    if name in {"es", "simple_es", "evolution", "evolution_strategy"}:
        return EvolutionStrategyOptimizer(config)
    if name in {"tpe", "optuna_tpe"}:
        return TPESearchOptimizer(config)
    if name in {"cma", "cma_es", "cmaes"}:
        return CMAESOptimizer(config)
    if name in {"hebo"}:
        return HEBOOptimizer(config)
    if name == "wmbo_rule":
        return WMBOOptimizer(_config_with_options(config, use_llm_agent=False))
    if name == "wmbo_llm":
        return WMBOOptimizer(
            _config_with_options(
                config,
                use_llm_agent=True,
                llm_fallback_to_rule=False,
            )
        )
    if name in {"wmbo", "world_model", "world_model_bo"}:
        return WMBOOptimizer(config)
    raise ValueError(f"Unknown optimiser method: {method}")


def _config_with_options(config: OptimizerConfig, **updates: object) -> OptimizerConfig:
    return OptimizerConfig(
        method=config.method,
        budget=config.budget,
        initial_samples=config.initial_samples,
        candidate_pool_size=config.candidate_pool_size,
        seed=config.seed,
        options={**dict(config.options), **updates},
    )


def create_initial_state(benchmark: BenchmarkSpec) -> OptimizerState:
    """Create an empty optimiser state for a benchmark.

    Input:
        benchmark: Benchmark specification.

    Output:
        ``OptimizerState`` with no observations and step zero.
    """

    return OptimizerState(benchmark=benchmark, observations=[], step=0)


def append_observation(state: OptimizerState, result: EvaluationResult) -> OptimizerState:
    """Append one evaluation result to an optimiser state.

    Inputs:
        state: Current optimiser state.
        result: Benchmark evaluation result produced for the proposed point.

    Output:
        New immutable-style state containing the added observation.
    """

    _validate_state(state)
    if result.benchmark_name != state.benchmark.name:
        raise ValueError(f"Result benchmark {result.benchmark_name!r} does not match state benchmark {state.benchmark.name!r}.")
    observation = Observation(
        x=list(result.x_unit),
        y=float(result.y),
        metadata={"x_raw": list(result.x_raw), **dict(result.metadata)},
    )
    return OptimizerState(
        benchmark=state.benchmark,
        observations=[*state.observations, observation],
        step=state.step + 1,
        metadata=dict(state.metadata),
    )


def best_observation(
    observations: Sequence[Observation],
    *,
    constrained: bool = False,
) -> Observation | None:
    """Return the best observation for a minimisation run.

    Input:
        observations: Observed input-output pairs.

    Output:
        Feasibility-first best observation for constrained runs, otherwise the
        observation with the smallest scalar objective.
    """

    if not observations:
        return None
    return min(
        observations,
        key=lambda observation: observation_rank_key(observation, constrained=constrained),
    )


def observation_rank_key(
    observation: Observation,
    *,
    constrained: bool,
) -> tuple[float, ...]:
    """Return a deterministic feasibility-first minimisation key."""

    if not constrained:
        return (float(observation.y),)
    feasible = bool(observation.metadata.get("feasible", False))
    objective = _observation_objective(observation)
    violation_ratio = _observation_constraint_ratio(observation, failure_ratio=1.0e6)
    if feasible:
        return (0.0, objective, violation_ratio, float(observation.y))
    return (1.0, violation_ratio, objective, float(observation.y))


def observations_to_arrays(observations: Sequence[Observation], dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert observations to numpy arrays.

    Inputs:
        observations: Observed input-output pairs.
        dim: Expected input dimensionality.

    Output:
        Tuple ``(x, y)`` with shapes ``(n, dim)`` and ``(n,)``.
    """

    if dim <= 0:
        raise ValueError("dim must be positive.")
    if not observations:
        return np.empty((0, dim), dtype=float), np.empty((0,), dtype=float)

    x = np.asarray([observation.x for observation in observations], dtype=float)
    y = np.asarray([observation.y for observation in observations], dtype=float)
    if x.ndim != 2 or x.shape[1] != dim:
        raise ValueError(f"Observation inputs must have shape (n, {dim}); got {x.shape}.")
    if y.ndim != 1 or len(y) != len(x):
        raise ValueError("Observation outputs must be one-dimensional and align with inputs.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Observations must contain only finite values.")
    return x, y


def constrained_observations_to_arrays(
    observations: Sequence[Observation],
    dim: int,
    *,
    failure_ratio: float = 1.0e6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return inputs, physical objective, log constraint ratio, and feasibility."""

    x, _ = observations_to_arrays(observations, dim=dim)
    objective = np.asarray([_observation_objective(item) for item in observations], dtype=float)
    ratio = np.asarray(
        [
            _observation_constraint_ratio(item, failure_ratio=failure_ratio)
            for item in observations
        ],
        dtype=float,
    )
    constraint_log_ratio = np.log10(np.maximum(ratio, 1.0e-12))
    feasible = np.asarray(
        [bool(item.metadata.get("feasible", False)) for item in observations],
        dtype=bool,
    )
    if not np.all(np.isfinite(objective)) or not np.all(np.isfinite(constraint_log_ratio)):
        raise ValueError("Constrained observations must contain finite objective and constraint values.")
    return x, objective, constraint_log_ratio, feasible


def _constraint_handling_enabled(state: OptimizerState, config: OptimizerConfig) -> bool:
    mode = str(config.options.get("constraint_handling", "auto")).strip().lower()
    if mode in {"off", "false", "none", "legacy_penalty"}:
        return False
    return bool(state.benchmark.constrained)


def _constraint_surrogate(
    *,
    config: OptimizerConfig,
    dim: int,
    step: int,
    observed_x: np.ndarray,
    constraint_log_ratio: np.ndarray,
) -> Any:
    surrogate = make_surrogate(
        kind=str(config.options.get("surrogate", "gaussian_process")),
        dim=dim,
        options={
            "seed": config.seed + 500_000 + step,
            "noise_level": float(config.options.get("constraint_noise_level", 1e-6)),
        },
    )
    surrogate.fit(
        SurrogateDataset(
            x=observed_x.tolist(),
            y=constraint_log_ratio.tolist(),
        )
    )
    return surrogate


def _predict_feasibility(
    constraint_surrogate: Any,
    candidates: Sequence[Sequence[float]],
    *,
    feasible_x: np.ndarray | None = None,
) -> tuple[list[float], list[float], list[float]]:
    prediction = constraint_surrogate.predict(candidates)
    mean = np.asarray(prediction.mean, dtype=float)
    std = np.maximum(np.asarray(prediction.std, dtype=float), 1.0e-12)
    z = (0.0 - mean) / std
    erf = np.vectorize(math.erf)
    probability = 0.5 * (1.0 + erf(z / math.sqrt(2.0)))
    if feasible_x is not None and len(feasible_x):
        candidate_array = np.asarray(candidates, dtype=float)
        feasible_array = np.asarray(feasible_x, dtype=float)
        distances = np.linalg.norm(
            candidate_array[:, None, :] - feasible_array[None, :, :],
            axis=2,
        )
        nearest_distance = np.min(distances, axis=1)
        # A small-data GP can be extremely confident when extrapolating far from
        # the observed feasible set.  Retain the learned PoF while applying a
        # conservative locality prior until observations support that region.
        locality = np.exp(-0.5 * np.square(nearest_distance / 0.15))
        probability = probability * (0.05 + 0.95 * locality)
    return (
        np.clip(probability, 0.0, 1.0).astype(float).tolist(),
        mean.astype(float).tolist(),
        std.astype(float).tolist(),
    )


def _observation_objective(observation: Observation) -> float:
    value = observation.metadata.get("normalised_cost_gap")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(observation.y)
    return parsed if math.isfinite(parsed) else float(observation.y)


def _observation_constraint_ratio(
    observation: Observation,
    *,
    failure_ratio: float,
) -> float:
    if not bool(observation.metadata.get("pf_converged", True)):
        return float(failure_ratio)
    value = observation.metadata.get("constraint_ratio")
    if value is None:
        tolerance = observation.metadata.get("feasibility_tolerance", 1.0e-5)
        violation = observation.metadata.get("max_normalized_violation", failure_ratio)
        try:
            value = float(violation) / max(float(tolerance), 1.0e-12)
        except (TypeError, ValueError):
            value = failure_ratio
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = failure_ratio
    if not math.isfinite(parsed):
        return float(failure_ratio)
    return float(min(max(parsed, 0.0), failure_ratio))


def _constraint_progress_value(
    observations: Sequence[Observation],
    *,
    constrained: bool,
) -> float | None:
    if not observations:
        return None
    if not constrained:
        return min(float(item.y) for item in observations)
    feasible = [item for item in observations if bool(item.metadata.get("feasible", False))]
    if feasible:
        return min(_observation_objective(item) for item in feasible)
    return min(
        _observation_constraint_ratio(item, failure_ratio=1.0e6)
        for item in observations
    )


def _result_improves_constraint_progress(
    observations: Sequence[Observation],
    result: EvaluationResult,
    *,
    constrained: bool,
) -> tuple[bool, float, float]:
    previous = _constraint_progress_value(observations, constrained=constrained)
    if previous is None:
        return False, float(result.y), float(result.y)
    if not constrained:
        tolerance = max(1e-8, 0.001 * max(abs(previous), 1.0))
        current = float(result.y)
        return current < previous - tolerance, current, min(previous, current)

    previous_has_feasible = any(
        bool(item.metadata.get("feasible", False)) for item in observations
    )
    result_observation = Observation(
        x=list(result.x_unit),
        y=float(result.y),
        metadata=dict(result.metadata),
    )
    if previous_has_feasible:
        current = _observation_objective(result_observation)
        tolerance = max(1e-8, 0.001 * max(abs(previous), 1.0))
        improved = bool(result.metadata.get("feasible", False)) and current < previous - tolerance
        best = min(previous, current) if bool(result.metadata.get("feasible", False)) else previous
        return improved, current, best

    current = _observation_constraint_ratio(result_observation, failure_ratio=1.0e6)
    tolerance = max(1e-8, 0.001 * max(abs(previous), 1.0))
    return current < previous - tolerance, current, min(previous, current)


def _validate_state(state: OptimizerState) -> None:
    if state.benchmark.dim <= 0:
        raise ValueError("Benchmark dimension must be positive.")
    if state.step < 0:
        raise ValueError("Optimizer step must be non-negative.")
    for observation in state.observations:
        values = list(observation.x)
        if len(values) != state.benchmark.dim:
            raise ValueError(f"Observation dimension mismatch: {len(values)} != {state.benchmark.dim}.")
        if any(float(value) < 0.0 or float(value) > 1.0 for value in values):
            raise ValueError("Observation inputs must be inside [0, 1].")
        if not math.isfinite(float(observation.y)):
            raise ValueError("Observation outputs must be finite.")



def _llm_options(options: Mapping[str, object]) -> dict[str, object]:
    raw = options.get("llm", {}) if isinstance(options, Mapping) else {}
    data = dict(raw) if isinstance(raw, Mapping) else {}
    for key in (
        "api_key",
        "api_key_env",
        "base_url",
        "api_base_url",
        "model",
        "api_model",
        "timeout",
        "max_retries",
        "retry_delay_seconds",
        "request_delay_seconds",
        "organization",
        "project",
        "api_provider",
        "provider",
    ):
        if key in options and key not in data:
            data[key] = options[key]
    return data


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _build_portfolio_candidate_options(
    *,
    surrogate: Any,
    observed_x: np.ndarray,
    observed_y: np.ndarray,
    dim: int,
    n_pool: int,
    n_options_per_strategy: int,
    seed: int,
    active_hypotheses: Sequence[Mapping[str, object]],
    phase: str,
    information_gain_weight: float,
    world_model_entropy: float,
    constraint_surrogate: Any | None,
    has_feasible_observation: bool,
    feasibility_floor: float,
    best_x: np.ndarray,
    feasible_x: np.ndarray | None,
    lengthscales: object,
    turbo_state: TurboState,
    cma_state: OnePlusOneCMAState,
) -> list[dict[str, object]]:
    """Build one pool and make one objective-surrogate prediction per v5 operator."""

    count = max(1, int(n_pool))
    option_count = max(1, int(n_options_per_strategy))
    hypotheses = [
        item for item in active_hypotheses
        if str(item.get("status", "active")) == "active"
    ]
    phase_key = str(phase).strip().lower()
    roles = (
        ("optimize",)
        if phase_key == "late" or not hypotheses
        else ("optimize", "confirm", "falsify")
    )
    phase_multiplier = V51_CANDIDATE_INFORMATION_MULTIPLIERS.get(phase_key, 0.35)
    effective_information_weight = float(
        np.clip(float(information_gain_weight) * phase_multiplier, 0.0, 0.85)
    )
    shape = portfolio_shape(observed_x, observed_y, lengthscales)
    best_y = float(np.min(observed_y)) if len(observed_y) else 0.0
    cma_state.ensure(best_x, best_y, shape)
    options: list[dict[str, object]] = []

    for strategy_index, strategy in enumerate(PORTFOLIO_STRATEGIES):
        if strategy == "cma_local" and not has_feasible_observation:
            continue
        operator_seed = int(seed) + 10_000 * strategy_index
        sobol = qmc.Sobol(d=dim, scramble=True, seed=operator_seed)
        sobol_count = count
        sobol_pool = sobol.random_base2(int(math.ceil(math.log2(max(1, sobol_count)))))[:sobol_count]
        if strategy in {"global_sobol", "gp_ucb", "gp_ei"}:
            pool = np.asarray(sobol_pool, dtype=float)
        elif strategy == "anisotropic_turbo":
            local_count = max(1, int(math.ceil(0.80 * count)))
            global_count = max(0, count - local_count)
            rng = np.random.default_rng(operator_seed + 1)
            local = rng.multivariate_normal(
                np.asarray(best_x, dtype=float),
                (float(turbo_state.radius) ** 2) * shape,
                size=local_count,
                check_valid="ignore",
            )
            pool = np.vstack([np.clip(local, 0.0, 1.0), sobol_pool[:global_count]])
        else:
            sample_count = count * (8 if constraint_surrogate is not None else 1)
            raw_pool = cma_state.sample(sample_count, operator_seed)
            if constraint_surrogate is not None:
                raw_pool = np.vstack([np.asarray(best_x, dtype=float), raw_pool])
                raw_pof, _raw_constraint_mean, _raw_constraint_std = _predict_feasibility(
                    constraint_surrogate, raw_pool, feasible_x=feasible_x
                )
                eligible = np.flatnonzero(
                    np.asarray(raw_pof, dtype=float) >= float(feasibility_floor)
                )
                if not len(eligible):
                    continue
                pool = raw_pool[eligible[:count]]
            else:
                pool = raw_pool[:count]

        prediction = surrogate.predict(pool)
        feasibility_probability = None
        constraint_mean = None
        constraint_std = None
        if constraint_surrogate is not None:
            feasibility_probability, constraint_mean, constraint_std = _predict_feasibility(
                constraint_surrogate, pool, feasible_x=feasible_x
            )
        acquisition_strategy = {
            "global_sobol": "global_diverse",
            "gp_ucb": "explore_ucb",
            "gp_ei": "exploit_ei",
            "anisotropic_turbo": "exploit_ei",
            "cma_local": "exploit_ei",
        }[strategy]
        acquisition = AcquisitionInput(
            candidates=pool.tolist(),
            observed_x=observed_x.tolist(),
            observed_y=observed_y.tolist(),
            surrogate_mean=prediction.mean,
            surrogate_std=prediction.std,
            strategy=acquisition_strategy,
            feasibility_probability=feasibility_probability,
            constraint_std=constraint_std,
            has_feasible_observation=has_feasible_observation,
            feasibility_floor=feasibility_floor,
        )
        strategy_scores = np.asarray(score_candidates(acquisition), dtype=float)
        raw_ei = np.asarray(
            expected_improvement(
                prediction.mean, prediction.std, best_y=best_y, xi=0.001
            ),
            dtype=float,
        )
        constrained_ei = np.asarray(
            score_candidates(
                AcquisitionInput(
                    candidates=pool.tolist(),
                    observed_x=observed_x.tolist(),
                    observed_y=observed_y.tolist(),
                    surrogate_mean=prediction.mean,
                    surrogate_std=prediction.std,
                    strategy="exploit_ei",
                    feasibility_probability=feasibility_probability,
                    constraint_std=constraint_std,
                    has_feasible_observation=has_feasible_observation,
                    feasibility_floor=feasibility_floor,
                )
            ),
            dtype=float,
        )
        information, confirmation, falsification, confirm_targets, falsify_targets = (
            _candidate_information_values(
                candidates=pool.tolist(),
                surrogate_std=prediction.std,
                observed_x=observed_x,
                hypotheses=hypotheses,
                world_model_entropy=world_model_entropy,
            )
        )
        optimisation = (
            0.65 * _scale_candidates_01(strategy_scores)
            + 0.35 * _scale_candidates_01(constrained_ei)
        )
        used_indices: set[int] = set()
        for option_index in range(option_count):
            role = roles[option_index % len(roles)]
            if role == "confirm":
                role_values = confirmation
                targets = confirm_targets
            elif role == "falsify":
                role_values = falsification
                targets = falsify_targets
            else:
                role_values = information
                targets = [None] * len(pool)
            joint = (
                (1.0 - effective_information_weight) * optimisation
                + effective_information_weight * _scale_candidates_01(role_values)
            )
            order = np.argsort(-np.asarray(joint, dtype=float), kind="stable")
            selected_index = next(
                (int(index) for index in order if int(index) not in used_indices),
                int(order[0]),
            )
            used_indices.add(selected_index)
            x = [float(value) for value in pool[selected_index]]
            pof = (
                float(feasibility_probability[selected_index])
                if feasibility_probability is not None
                else 1.0
            )
            target = targets[selected_index] if selected_index < len(targets) else None
            options.append(
                {
                    "candidate_id": f"{strategy}_{option_index + 1}",
                    "strategy": strategy,
                    "x_unit": x,
                    "acquisition_strategy": acquisition_strategy,
                    "acquisition_score": float(strategy_scores[selected_index]),
                    "expected_improvement": float(raw_ei[selected_index]),
                    "constrained_expected_improvement": float(constrained_ei[selected_index]),
                    "optimisation_utility": float(optimisation[selected_index]),
                    "information_gain": float(information[selected_index]),
                    "information_gain_weight_effective": effective_information_weight,
                    "confirmation_value": float(confirmation[selected_index]),
                    "falsification_value": float(falsification[selected_index]),
                    "joint_score": float(joint[selected_index]),
                    "selection_score": float(joint[selected_index]),
                    "evidence_role": role,
                    "target_hypothesis_id": target,
                    "hypothesis_alignment": _hypothesis_alignment(x, hypotheses),
                    "surrogate_mean": float(prediction.mean[selected_index]),
                    "surrogate_std": float(prediction.std[selected_index]),
                    "probability_feasible": pof,
                    "predicted_feasibility_probability": pof,
                    "predicted_constraint_log_ratio": (
                        float(constraint_mean[selected_index])
                        if constraint_mean is not None else None
                    ),
                    "constraint_std": (
                        float(constraint_std[selected_index])
                        if constraint_std is not None else None
                    ),
                    "distance_to_best": float(
                        np.linalg.norm(np.asarray(x, dtype=float) - np.asarray(best_x, dtype=float))
                    ),
                    "distance_to_nearest_observation": _distance_to_nearest_observation(
                        x, observed_x
                    ),
                }
            )
    return options
def _build_strategy_candidate_options(
    *,
    surrogate: Any,
    observed_x: np.ndarray,
    observed_y: np.ndarray,
    dim: int,
    n_pool: int,
    n_options_per_strategy: int,
    seed: int,
    phase: str = "middle",
    active_hypotheses: Sequence[Mapping[str, object]] | None = None,
    hypothesis_alignment_weight: float = 0.0,
    information_gain_weight: float = 0.35,
    world_model_entropy: float = 1.0,
    constraint_surrogate: Any | None = None,
    has_feasible_observation: bool = True,
    feasibility_floor: float = 0.05,
    best_x: np.ndarray | None = None,
    feasible_x: np.ndarray | None = None,
) -> list[dict[str, object]]:
    """Build scored candidate options grouped by WMBO strategy."""

    best = (
        np.asarray(best_x, dtype=float)
        if best_x is not None
        else observed_x[int(np.argmin(observed_y))]
        if len(observed_y)
        else np.full(dim, 0.5)
    )
    hypotheses = [
        hypothesis
        for hypothesis in (active_hypotheses or [])
        if str(hypothesis.get("status", "active")) == "active"
    ]
    phase_multiplier = {"early": 1.0, "middle": 0.80, "late": 0.35}.get(str(phase), 0.80)
    information_weight = float(np.clip(float(information_gain_weight) * phase_multiplier, 0.0, 0.85))
    roles = ("optimize", "confirm", "falsify") if hypotheses else ("optimize",)
    options: list[dict[str, object]] = []
    for strategy_index, strategy in enumerate(STRATEGIES):
        acquisition_strategy = _strategy_to_acquisition(strategy)
        for option_index in range(max(1, n_options_per_strategy)):
            option_seed = int(seed) + strategy_index * 10_000 + option_index
            evidence_role = roles[option_index % len(roles)]
            pool = _make_wmbo_candidate_pool(
                strategy=strategy,
                observed_x=observed_x,
                observed_y=observed_y,
                dim=dim,
                n_points=n_pool,
                seed=option_seed,
                best_x=best,
                constrained=constraint_surrogate is not None,
            )
            if feasible_x is not None and len(feasible_x):
                pool = _augment_pool_with_feasible_neighbourhood(
                    pool,
                    feasible_x=feasible_x,
                    center=best,
                    seed=option_seed + 500_000,
                )
            pool = _augment_pool_with_hypothesis_tests(
                pool,
                hypotheses=hypotheses,
                dim=dim,
                n_points=n_pool,
                seed=option_seed + 1_000_000,
            )
            prediction = surrogate.predict(pool)
            feasibility_probability = None
            constraint_mean = None
            constraint_std = None
            if constraint_surrogate is not None:
                feasibility_probability, constraint_mean, constraint_std = _predict_feasibility(
                    constraint_surrogate,
                    pool,
                    feasible_x=feasible_x,
                )
            acquisition_input = AcquisitionInput(
                candidates=pool,
                observed_x=observed_x.tolist(),
                observed_y=observed_y.tolist(),
                surrogate_mean=prediction.mean,
                surrogate_std=prediction.std,
                strategy=acquisition_strategy,
                feasibility_probability=feasibility_probability,
                constraint_std=constraint_std,
                has_feasible_observation=has_feasible_observation,
                feasibility_floor=feasibility_floor,
            )
            strategy_scores = np.asarray(score_candidates(acquisition_input), dtype=float)
            best_y = float(np.min(observed_y)) if len(observed_y) else 0.0
            raw_ei_scores = np.asarray(
                expected_improvement(
                    prediction.mean,
                    prediction.std,
                    best_y=best_y,
                    xi=0.001,
                ),
                dtype=float,
            )
            constrained_ei_scores = np.asarray(
                score_candidates(
                    AcquisitionInput(
                        candidates=pool,
                        observed_x=observed_x.tolist(),
                        observed_y=observed_y.tolist(),
                        surrogate_mean=prediction.mean,
                        surrogate_std=prediction.std,
                        strategy="exploit_ei",
                        feasibility_probability=feasibility_probability,
                        constraint_std=constraint_std,
                        has_feasible_observation=has_feasible_observation,
                        feasibility_floor=feasibility_floor,
                    )
                ),
                dtype=float,
            )
            optimisation_utility = (
                0.65 * _scale_candidates_01(strategy_scores)
                + 0.35 * _scale_candidates_01(constrained_ei_scores)
            )
            information, confirmation, falsification, confirm_targets, falsify_targets = (
                _candidate_information_values(
                    candidates=pool,
                    surrogate_std=prediction.std,
                    observed_x=observed_x,
                    hypotheses=hypotheses,
                    world_model_entropy=world_model_entropy,
                )
            )
            if evidence_role == "confirm":
                role_information = confirmation
                targets = confirm_targets
            elif evidence_role == "falsify":
                role_information = falsification
                targets = falsify_targets
            else:
                role_information = information
                targets = [None] * len(pool)
            if evidence_role == "optimize":
                joint_scores = (
                    (1.0 - information_weight) * optimisation_utility
                    + information_weight * _scale_candidates_01(information)
                )
            else:
                joint_scores = (
                    (1.0 - information_weight) * optimisation_utility
                    + information_weight * _scale_candidates_01(role_information)
                )
            selected_index = int(np.argmax(joint_scores))
            x = [float(value) for value in pool[selected_index]]
            score = float(strategy_scores[selected_index])
            alignment = _hypothesis_alignment(x, active_hypotheses or [])
            selection_score = _adjust_score_for_hypothesis(
                score=float(joint_scores[selected_index]),
                alignment=alignment,
                weight=float(hypothesis_alignment_weight),
            )
            target_hypothesis_id = targets[selected_index] if selected_index < len(targets) else None
            options.append(
                {
                    "candidate_id": f"{strategy}_{option_index + 1}",
                    "strategy": strategy,
                    "x_unit": x,
                    "acquisition_strategy": acquisition_strategy,
                    "acquisition_score": score,
                    "expected_improvement": float(raw_ei_scores[selected_index]),
                    "constrained_expected_improvement": float(constrained_ei_scores[selected_index]),
                    "optimisation_utility": float(optimisation_utility[selected_index]),
                    "information_gain": float(information[selected_index]),
                    "confirmation_value": float(confirmation[selected_index]),
                    "falsification_value": float(falsification[selected_index]),
                    "joint_score": float(joint_scores[selected_index]),
                    "selection_score": selection_score,
                    "evidence_role": evidence_role,
                    "target_hypothesis_id": target_hypothesis_id,
                    "hypothesis_alignment": alignment,
                    "surrogate_mean": float(prediction.mean[selected_index]),
                    "surrogate_std": float(prediction.std[selected_index]),
                    "predicted_feasibility_probability": (
                        float(feasibility_probability[selected_index])
                        if feasibility_probability is not None
                        else None
                    ),
                    "predicted_constraint_log_ratio": (
                        float(constraint_mean[selected_index])
                        if constraint_mean is not None
                        else None
                    ),
                    "constraint_std": (
                        float(constraint_std[selected_index])
                        if constraint_std is not None
                        else None
                    ),
                    "distance_to_best": float(np.linalg.norm(np.asarray(x, dtype=float) - best)),
                    "distance_to_nearest_observation": _distance_to_nearest_observation(x, observed_x),
                }
            )
    return options


def _augment_pool_with_hypothesis_tests(
    pool: Sequence[Sequence[float]],
    *,
    hypotheses: Sequence[Mapping[str, object]],
    dim: int,
    n_points: int,
    seed: int,
) -> list[list[float]]:
    """Inject confirmation points and boundary counterexamples into a pool."""

    base = [[float(value) for value in point] for point in pool]
    if not hypotheses or n_points <= 0:
        return base[: max(0, int(n_points))]
    rng = np.random.default_rng(int(seed))
    tests: list[list[float]] = []
    for hypothesis in hypotheses:
        center = _valid_region_center(hypothesis.get("region_center"), dim=dim)
        radius = _valid_region_radius(hypothesis.get("region_radius"))
        if center is None or radius is None:
            continue
        dims = _valid_sensitive_dims(hypothesis.get("sensitive_dims"), dim=dim)
        confirm = np.asarray(center, dtype=float)
        confirm[dims] += rng.normal(0.0, max(0.01, 0.30 * radius), size=len(dims))
        tests.append(np.clip(confirm, 0.0, 1.0).astype(float).tolist())

        direction = rng.normal(0.0, 1.0, size=len(dims))
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            direction = np.ones(len(dims), dtype=float) / math.sqrt(max(len(dims), 1))
        else:
            direction /= norm
        falsify = np.asarray(center, dtype=float)
        falsify[dims] += direction * radius * 1.15
        tests.append(np.clip(falsify, 0.0, 1.0).astype(float).tolist())

    combined = [*tests, *base]
    unique: list[list[float]] = []
    seen: set[tuple[float, ...]] = set()
    for point in combined:
        key = tuple(round(float(value), 12) for value in point)
        if key in seen:
            continue
        seen.add(key)
        unique.append(point)
        if len(unique) >= int(n_points):
            break
    return unique


def _candidate_information_values(
    *,
    candidates: Sequence[Sequence[float]],
    surrogate_std: Sequence[float],
    observed_x: np.ndarray,
    hypotheses: Sequence[Mapping[str, object]],
    world_model_entropy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str | None], list[str | None]]:
    """Approximate expected entropy reduction for world-model hypotheses.

    Confirmation value is largest inside a hypothesis region; falsification
    value is largest near its boundary. Both are weighted by the current binary
    hypothesis entropy and predictive uncertainty. A global component continues
    to reduce uncertain world-model properties when no explicit hypothesis is
    active.
    """

    points = np.asarray(candidates, dtype=float)
    std = np.asarray(surrogate_std, dtype=float)
    n = len(points)
    if n == 0:
        empty = np.empty((0,), dtype=float)
        return empty, empty, empty, [], []
    uncertainty = _scale_candidates_01(std)
    novelty = np.asarray(
        [_distance_to_nearest_observation(point, observed_x) for point in points],
        dtype=float,
    )
    novelty = _scale_candidates_01(novelty)
    model_entropy = float(np.clip(world_model_entropy, 0.0, 1.0))
    global_information = model_entropy * uncertainty * (0.25 + 0.75 * novelty)

    confirmation = np.zeros(n, dtype=float)
    falsification = np.zeros(n, dtype=float)
    confirm_targets: list[str | None] = [None] * n
    falsify_targets: list[str | None] = [None] * n
    for hypothesis in hypotheses:
        posterior = _candidate_float(
            hypothesis,
            "posterior_probability",
            default=_candidate_float(hypothesis, "confidence", default=0.5),
        )
        entropy = _binary_entropy(posterior)
        if entropy <= 1e-12:
            continue
        ratios = _hypothesis_distance_ratios(points, hypothesis)
        confirm_kernel = np.exp(-0.5 * (ratios / 0.75) ** 2)
        falsify_kernel = np.exp(-0.5 * ((ratios - 1.0) / 0.35) ** 2)
        uncertainty_factor = 0.20 + 0.80 * uncertainty
        confirm_value = entropy * uncertainty_factor * confirm_kernel
        falsify_value = entropy * uncertainty_factor * falsify_kernel
        hypothesis_id = str(hypothesis.get("hypothesis_id")) if hypothesis.get("hypothesis_id") else None
        for index in range(n):
            if confirm_value[index] > confirmation[index]:
                confirmation[index] = float(confirm_value[index])
                confirm_targets[index] = hypothesis_id
            if falsify_value[index] > falsification[index]:
                falsification[index] = float(falsify_value[index])
                falsify_targets[index] = hypothesis_id
    information = np.maximum(global_information, np.maximum(confirmation, falsification))
    return information, confirmation, falsification, confirm_targets, falsify_targets


def _hypothesis_distance_ratios(
    candidates: np.ndarray,
    hypothesis: Mapping[str, object],
) -> np.ndarray:
    dim = int(candidates.shape[1])
    center = _valid_region_center(hypothesis.get("region_center"), dim=dim)
    radius = _valid_region_radius(hypothesis.get("region_radius"))
    if center is None or radius is None:
        return np.full(len(candidates), np.inf, dtype=float)
    dims = _valid_sensitive_dims(hypothesis.get("sensitive_dims"), dim=dim)
    delta = candidates[:, dims] - np.asarray(center, dtype=float)[dims]
    return np.linalg.norm(delta, axis=1) / max(radius, 1e-12)


def _binary_entropy(probability: float) -> float:
    probability = float(np.clip(probability, 1e-9, 1.0 - 1e-9))
    entropy = -probability * math.log(probability) - (1.0 - probability) * math.log(1.0 - probability)
    return float(entropy / math.log(2.0))


def _scale_candidates_01(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array
    lower = float(np.min(array))
    spread = float(np.max(array) - lower)
    if spread <= 1e-12:
        return np.ones_like(array)
    return (array - lower) / spread


def _select_strategy_candidate(
    candidates: Sequence[Mapping[str, object]],
    *,
    strategy: str,
    requested_candidate_id: str | None,
) -> tuple[Mapping[str, object], str | None]:
    matching = [candidate for candidate in candidates if str(candidate.get("strategy")) == strategy]
    if not matching:
        raise ValueError(f"No candidate options available for strategy: {strategy}")

    if requested_candidate_id:
        requested = next(
            (candidate for candidate in matching if str(candidate.get("candidate_id")) == requested_candidate_id),
            None,
        )
        if requested is not None:
            return requested, None
        known = any(str(candidate.get("candidate_id")) == requested_candidate_id for candidate in candidates)
        reason = "candidate_ignored_after_strategy_override" if known else "unknown_selected_candidate_id"
    else:
        reason = "candidate_id_missing"

    selected = max(matching, key=_candidate_selection_score)
    return selected, reason


def _candidate_selection_score(candidate: Mapping[str, object]) -> float:
    score = candidate.get("selection_score", candidate.get("acquisition_score", float("-inf")))
    try:
        value = float(score)
    except (TypeError, ValueError):
        return float("-inf")
    return value if math.isfinite(value) else float("-inf")


def _hypothesis_alignment(candidate: Sequence[float], hypotheses: Sequence[Mapping[str, object]]) -> float:
    x = np.asarray(candidate, dtype=float)
    best_alignment = 0.0
    for hypothesis in hypotheses:
        if str(hypothesis.get("status", "active")) != "active":
            continue
        center_value = hypothesis.get("region_center")
        radius_value = hypothesis.get("region_radius")
        if center_value is None or radius_value is None:
            continue
        try:
            center = np.asarray(center_value, dtype=float)
            radius = float(radius_value)
            confidence = float(hypothesis.get("confidence", 1.0))
        except (TypeError, ValueError):
            continue
        if center.shape != x.shape or not math.isfinite(radius) or radius <= 0.0:
            continue
        dims = _valid_sensitive_dims(hypothesis.get("sensitive_dims"), dim=len(x))
        diff = x[dims] - center[dims] if dims else x - center
        distance = float(np.linalg.norm(diff))
        alignment = max(0.0, 1.0 - distance / radius) * max(0.0, min(confidence, 1.0))
        best_alignment = max(best_alignment, alignment)
    return float(min(best_alignment, 1.0))


def _adjust_score_for_hypothesis(*, score: float, alignment: float, weight: float) -> float:
    if not math.isfinite(float(score)):
        return float("-inf")
    bounded_alignment = max(0.0, min(float(alignment), 1.0))
    bounded_weight = max(0.0, float(weight))
    if bounded_alignment <= 0.0 or bounded_weight <= 0.0:
        return float(score)
    if score >= 0.0:
        return float(score * (1.0 + bounded_weight * bounded_alignment))
    return float(score * (1.0 - min(0.95, bounded_weight * bounded_alignment)))


def _verify_candidate_with_gp(
    *,
    candidates: Sequence[Mapping[str, object]],
    selected: Mapping[str, object],
    strategy: str,
    decision_confidence: float,
    config: WMBOControlConfig,
) -> tuple[Mapping[str, object], dict[str, object], str | None]:
    """Accept or refine a selected option using same-strategy GP diagnostics."""

    selected_id = str(selected.get("candidate_id"))
    matching = [candidate for candidate in candidates if str(candidate.get("strategy")) == strategy]
    enabled = _truthy(config.gp_verifier_enabled)
    metadata: dict[str, object] = {
        "enabled": enabled,
        "action": "accepted",
        "reason": None,
        "original_candidate_id": selected_id,
        "verified_candidate_id": selected_id,
        "decision_confidence": float(decision_confidence),
        "original_acquisition_score": _candidate_acquisition_score(selected),
        "original_surrogate_std": _candidate_float(selected, "surrogate_std"),
        "original_feasibility_probability": _candidate_float(
            selected,
            "predicted_feasibility_probability",
            default=1.0,
        ),
    }
    evidence_role = str(selected.get("evidence_role", "optimize"))
    targeted_information_test = bool(
        evidence_role in {"confirm", "falsify"} and selected.get("target_hypothesis_id")
    )
    selected_probability = _candidate_float(
        selected,
        "predicted_feasibility_probability",
        default=1.0,
    )
    min_probability = float(
        np.clip(config.gp_verifier_min_feasibility_probability, 0.0, 1.0)
    )
    if targeted_information_test and selected_probability >= min_probability:
        metadata["action"] = "accepted_information_test"
        metadata["evidence_role"] = evidence_role
        metadata["target_hypothesis_id"] = selected.get("target_hypothesis_id")
        return selected, metadata, None
    if not enabled or len(matching) <= 1:
        return selected, metadata, None

    best_by_acquisition = max(matching, key=_candidate_acquisition_score)
    best_score = _candidate_acquisition_score(best_by_acquisition)
    selected_score = _candidate_acquisition_score(selected)
    min_ratio = max(0.0, min(float(config.gp_verifier_min_score_ratio), 1.0))
    confidence = max(0.0, min(float(decision_confidence), 1.0))

    refined = selected
    reason: str | None = None
    best_by_feasibility = max(
        matching,
        key=lambda candidate: (
            _candidate_float(
                candidate,
                "predicted_feasibility_probability",
                default=1.0,
            ),
            _candidate_acquisition_score(candidate),
        ),
    )
    best_probability = _candidate_float(
        best_by_feasibility,
        "predicted_feasibility_probability",
        default=1.0,
    )
    if (
        selected_probability < min_probability
        and best_probability > selected_probability + 0.05
    ):
        refined = best_by_feasibility
        reason = "gp_refined_low_feasibility_probability"
    elif confidence < 0.85 and _score_is_much_worse(selected_score, best_score, min_ratio):
        refined = best_by_acquisition
        reason = "gp_refined_low_acquisition_score"
    elif _candidate_float(selected, "distance_to_nearest_observation") < float(config.gp_verifier_duplicate_distance):
        less_duplicate = max(
            matching,
            key=lambda candidate: (
                _candidate_float(candidate, "distance_to_nearest_observation"),
                _candidate_acquisition_score(candidate),
            ),
        )
        if less_duplicate is not selected:
            refined = less_duplicate
            reason = "gp_refined_duplicate_candidate"
    elif strategy in {"explore_ucb", "global_diverse"} and confidence < 0.80:
        std_values = [_candidate_float(candidate, "surrogate_std") for candidate in matching]
        median_std = float(np.median(std_values)) if std_values else 0.0
        selected_std = _candidate_float(selected, "surrogate_std")
        if median_std > 0.0 and selected_std < 0.75 * median_std:
            refined = max(
                matching,
                key=lambda candidate: (
                    _candidate_float(candidate, "surrogate_std"),
                    _candidate_float(candidate, "distance_to_nearest_observation"),
                    _candidate_acquisition_score(candidate),
                ),
            )
            reason = "gp_refined_low_exploration_uncertainty"

    metadata.update(
        {
            "best_candidate_id": str(best_by_acquisition.get("candidate_id")),
            "best_acquisition_score": best_score,
            "verified_candidate_id": str(refined.get("candidate_id")),
            "verified_acquisition_score": _candidate_acquisition_score(refined),
            "verified_surrogate_std": _candidate_float(refined, "surrogate_std"),
            "best_feasibility_probability": best_probability,
            "verified_feasibility_probability": _candidate_float(
                refined,
                "predicted_feasibility_probability",
                default=1.0,
            ),
        }
    )
    if reason is None or refined is selected:
        return selected, metadata, None

    metadata["action"] = "refined"
    metadata["reason"] = reason
    return refined, metadata, reason


def _score_is_much_worse(selected_score: float, best_score: float, min_ratio: float) -> bool:
    if not math.isfinite(selected_score) or not math.isfinite(best_score):
        return False
    if best_score > 0.0:
        return selected_score < best_score * min_ratio
    if best_score < 0.0:
        tolerated_gap = abs(best_score) * (1.0 - min_ratio)
        return selected_score < best_score - tolerated_gap
    return selected_score < -1e-12


def _candidate_acquisition_score(candidate: Mapping[str, object]) -> float:
    return _candidate_float(candidate, "acquisition_score", default=float("-inf"))


def _candidate_float(candidate: Mapping[str, object], key: str, default: float = 0.0) -> float:
    try:
        value = float(candidate.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _append_override(current: str | None, extra: str | None) -> str | None:
    if not extra:
        return current
    return extra if not current else f"{current};{extra}"


def _structured_hypothesis(
    *,
    decision: Any,
    candidate: Sequence[float],
    dim: int,
    strategy: str,
    phase: str,
    trial_number: int,
    hypothesis_window: int,
) -> dict[str, object]:
    candidate_center = [float(min(max(value, 0.0), 1.0)) for value in candidate]
    center = _valid_region_center(getattr(decision, "hypothesis_region_center", None), dim=dim) or candidate_center
    radius = _valid_region_radius(getattr(decision, "hypothesis_region_radius", None))
    if radius is None:
        radius = _default_hypothesis_radius(strategy=strategy, phase=phase)
    dims = _valid_sensitive_dims(getattr(decision, "hypothesis_sensitive_dims", None), dim=dim)
    falsification_rule = getattr(decision, "falsification_rule", None)
    if not falsification_rule:
        expires_trial = int(trial_number) + max(1, int(hypothesis_window)) - 1
        falsification_rule = f"Refine or reject if no best-value improvement by trial {expires_trial}."
    return {
        "region_center": center,
        "region_radius": radius,
        "sensitive_dims": dims,
        "confidence": float(max(0.0, min(float(getattr(decision, "confidence", 0.0)), 1.0))),
        "falsification_rule": str(falsification_rule).strip(),
    }


def _valid_region_center(value: object, *, dim: int) -> list[float] | None:
    if value is None or isinstance(value, (str, bytes, bytearray)):
        return None
    try:
        center = [float(item) for item in value]  # type: ignore[iteration-over-optional]
    except (TypeError, ValueError):
        return None
    if len(center) != dim or not all(math.isfinite(item) for item in center):
        return None
    return [float(min(max(item, 0.0), 1.0)) for item in center]


def _valid_region_radius(value: object) -> float | None:
    if value is None:
        return None
    try:
        radius = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(radius) or radius <= 0.0:
        return None
    return float(min(max(radius, 0.02), 0.75))


def _valid_sensitive_dims(value: object, *, dim: int) -> list[int]:
    if value is None or isinstance(value, (str, bytes, bytearray)):
        return list(range(dim))
    result: list[int] = []
    try:
        iterator = iter(value)  # type: ignore[arg-type]
    except TypeError:
        return list(range(dim))
    for item in iterator:
        try:
            parsed = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= parsed < dim and parsed not in result:
            result.append(parsed)
    return result or list(range(dim))


def _default_hypothesis_radius(*, strategy: str, phase: str) -> float:
    key = strategy.strip().lower().replace("-", "_")
    if key == "trust_region":
        return 0.12 if phase != "early" else 0.18
    if key == "exploit_ei":
        return 0.16 if phase != "early" else 0.22
    if key == "explore_ucb":
        return 0.28
    return 0.35


def _distance_to_nearest_observation(candidate: Sequence[float], observed_x: np.ndarray) -> float:
    if len(observed_x) == 0:
        return 1.0
    x = np.asarray(candidate, dtype=float)
    distances = np.linalg.norm(observed_x - x[None, :], axis=1)
    return float(np.min(distances))


def _wmbo_control_config_from_options(options: Mapping[str, object]) -> WMBOControlConfig:
    raw_control = options.get("wmbo_control", options.get("control", {})) if isinstance(options, Mapping) else {}
    if not isinstance(raw_control, Mapping):
        raw_control = {}
    fields = WMBOControlConfig.__dataclass_fields__
    values: dict[str, Any] = {}
    for key in fields:
        if key in raw_control:
            values[key] = raw_control[key]
    return WMBOControlConfig(**values)

def _strategy_to_acquisition(strategy: str) -> str:
    key = strategy.strip().lower().replace("-", "_")
    if key == "explore_ucb":
        return "explore_ucb"
    if key == "global_diverse":
        return "global_diverse"
    if key == "trust_region":
        return "exploit_ei"
    if key == "exploit_ei":
        return "exploit_ei"
    return "expected_improvement"


def _make_wmbo_candidate_pool(
    strategy: str,
    observed_x: np.ndarray,
    observed_y: np.ndarray,
    dim: int,
    n_points: int,
    seed: int,
    best_x: np.ndarray | None = None,
    constrained: bool = False,
) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    key = strategy.strip().lower().replace("-", "_")
    if key == "trust_region" and len(observed_x):
        best = (
            np.asarray(best_x, dtype=float)
            if best_x is not None
            else observed_x[int(np.argmin(observed_y))]
        )
        radius = max(0.04, 0.25 * (0.97 ** len(observed_y)))
        if constrained:
            radius = min(radius, 0.10)
        local_count = max(1, int(0.8 * n_points))
        global_count = max(0, n_points - local_count)
        local = best + rng.normal(0.0, radius, size=(local_count, dim))
        local = np.clip(local, 0.0, 1.0)
        if global_count:
            global_points = rng.uniform(0.0, 1.0, size=(global_count, dim))
            pool = np.vstack([local, global_points])
        else:
            pool = local
        return pool.astype(float).tolist()

    return sample_unit_points(n_points=n_points, dim=dim, seed=seed)


def _augment_pool_with_feasible_neighbourhood(
    pool: Sequence[Sequence[float]],
    *,
    feasible_x: np.ndarray,
    center: np.ndarray,
    seed: int,
) -> list[list[float]]:
    """Reserve half of a candidate pool for strict-feasible neighbourhood search."""

    candidate_array = np.asarray(pool, dtype=float)
    if candidate_array.ndim != 2 or not len(candidate_array):
        return candidate_array.astype(float).tolist()
    feasible_array = np.asarray(feasible_x, dtype=float)
    if feasible_array.ndim != 2 or not len(feasible_array):
        return candidate_array.astype(float).tolist()

    rng = np.random.default_rng(seed)
    local_count = max(1, len(candidate_array) // 2)
    tight_count = max(1, (2 * local_count) // 3)
    broad_count = local_count - tight_count
    local_parts = [
        np.asarray(center, dtype=float)[None, :]
        + rng.normal(0.0, 0.015, size=(tight_count, candidate_array.shape[1]))
    ]
    if broad_count:
        anchors = feasible_array[rng.integers(0, len(feasible_array), size=broad_count)]
        local_parts.append(
            anchors + rng.normal(0.0, 0.05, size=(broad_count, candidate_array.shape[1]))
        )
    local = np.clip(np.vstack(local_parts), 0.0, 1.0)
    retained = candidate_array[: len(candidate_array) - local_count]
    return np.vstack([local, retained]).astype(float).tolist()


__all__ = [
    "Observation",
    "OptimizerState",
    "Optimizer",
    "RandomSearchOptimizer",
    "SobolSearchOptimizer",
    "BayesianOptimizationOptimizer",
    "EvolutionStrategyOptimizer",
    "WMBOOptimizer",
    "make_optimizer",
    "create_initial_state",
    "append_observation",
    "best_observation",
    "observations_to_arrays",
]
