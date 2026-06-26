"""Baseline optimiser implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Mapping, Protocol, Sequence

import numpy as np
from scipy.stats import qmc

from .acquisition import AcquisitionInput, select_next_candidate
from .agents import AgentState, CandidateValidator, WorldModelAgent
from .benchmarks import BenchmarkSpec, EvaluationResult, sample_unit_points
from .control import OptimizerConfig
from .descriptors import describe_landscape
from .surrogate import SurrogateDataset, make_surrogate

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
        self._initial = SobolSearchOptimizer(config)

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

        observed_x, observed_y = observations_to_arrays(state.observations, dim=state.benchmark.dim)
        candidate_pool = sample_unit_points(
            n_points=max(1, self.config.candidate_pool_size),
            dim=state.benchmark.dim,
            seed=self.config.seed + 10_000 + state.step,
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
        acquisition = select_next_candidate(
            AcquisitionInput(
                candidates=candidate_pool,
                observed_x=observed_x.tolist(),
                observed_y=observed_y.tolist(),
                surrogate_mean=prediction.mean,
                surrogate_std=prediction.std,
                strategy=self.acquisition_strategy,
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
        self._initial = SobolSearchOptimizer(config)
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
        elite_indices = np.argsort(y_obs)[:elite_count]
        elite = x_obs[elite_indices]
        center = np.mean(elite, axis=0)
        scale = np.maximum(np.std(elite, axis=0), float(self.config.options.get("mutation_scale", 0.08)))
        candidate = center + self._rng.normal(0.0, scale, size=state.benchmark.dim)
        return np.clip(candidate, 0.0, 1.0).astype(float).tolist()

    def tell(self, state: OptimizerState, result: EvaluationResult) -> OptimizerState:
        """Update evolutionary state after an evaluation."""

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
        self._initial = SobolSearchOptimizer(config)
        self._agent = WorldModelAgent()
        self._last_decision: Mapping[str, object] | None = None
        self._last_acquisition: Mapping[str, object] | None = None

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
            self._last_decision = {
                "strategy": "initial_design",
                "rationale": "Collecting initial Sobol design points before fitting a world model.",
            }
            self._last_acquisition = {"strategy": "sobol"}
            return candidate

        observed_x, observed_y = observations_to_arrays(state.observations, dim=state.benchmark.dim)
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
        descriptor = describe_landscape(
            observed_x=observed_x.tolist(),
            observed_y=observed_y.tolist(),
            surrogate_metadata={
                **dict(descriptor_prediction.metadata),
                "mean_std": float(np.mean(descriptor_prediction.std)) if descriptor_prediction.std else 1.0,
            },
        )
        decision = self._agent.decide(
            AgentState(
                observed_x=observed_x.tolist(),
                observed_y=observed_y.tolist(),
                descriptor=descriptor.to_dict(),
                budget_used=state.step,
                budget_total=self.config.budget,
            )
        )

        candidate_pool = _make_wmbo_candidate_pool(
            strategy=decision.strategy,
            observed_x=observed_x,
            observed_y=observed_y,
            dim=state.benchmark.dim,
            n_points=max(1, self.config.candidate_pool_size),
            seed=self.config.seed + 30_000 + state.step,
        )
        prediction = surrogate.predict(candidate_pool)
        acquisition_strategy = _strategy_to_acquisition(decision.strategy)
        acquisition = select_next_candidate(
            AcquisitionInput(
                candidates=candidate_pool,
                observed_x=observed_x.tolist(),
                observed_y=observed_y.tolist(),
                surrogate_mean=prediction.mean,
                surrogate_std=prediction.std,
                strategy=acquisition_strategy,
            )
        )
        validator = CandidateValidator(dim=state.benchmark.dim)
        candidate = list(acquisition.selected_x)
        is_valid, _issues = validator.validate(candidate)
        if not is_valid:
            candidate = validator.repair(candidate)

        self._last_decision = decision.to_dict()
        self._last_acquisition = {
            **dict(acquisition.metadata),
            "score": acquisition.score,
            "selected_index": acquisition.selected_index,
            "descriptor": descriptor.to_dict(),
        }
        return candidate

    def tell(self, state: OptimizerState, result: EvaluationResult) -> OptimizerState:
        """Update WMBO state after an evaluation.

        Inputs:
            state: Current optimiser state.
            result: New evaluation result.

        Output:
            Updated optimiser state.
        """

        updated = append_observation(state, result)
        metadata = dict(updated.metadata)
        if self._last_decision is not None:
            metadata["last_reasoning_decision"] = dict(self._last_decision)
        if self._last_acquisition is not None:
            metadata["last_acquisition"] = dict(self._last_acquisition)
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

    name = method.strip().lower().replace("-", "_")
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
    if name in {"wmbo", "world_model", "world_model_bo"}:
        return WMBOOptimizer(config)
    raise ValueError(f"Unknown optimiser method: {method}")


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


def best_observation(observations: Sequence[Observation]) -> Observation | None:
    """Return the best observation for a minimisation run.

    Input:
        observations: Observed input-output pairs.

    Output:
        Observation with the smallest objective value, or ``None`` when empty.
    """

    if not observations:
        return None
    return min(observations, key=lambda observation: float(observation.y))


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
) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    key = strategy.strip().lower().replace("-", "_")
    if key == "trust_region" and len(observed_x):
        best = observed_x[int(np.argmin(observed_y))]
        radius = max(0.04, 0.25 * (0.97 ** len(observed_y)))
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
