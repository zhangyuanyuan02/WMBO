"""Optimiser interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Mapping, Protocol, Sequence

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
    """Initial baseline optimiser interface."""

    def __init__(self, config: OptimizerConfig) -> None:
        """Create a random-search optimiser.

        Input:
            config: Optimiser configuration.

        Output:
            Optimiser instance.
        """

        self.config = config

    def ask(self, state: OptimizerState) -> Vector:
        """Propose a random candidate.

        Input:
            state: Current optimiser state.

        Output:
            Candidate point in normalised coordinates.
        """

        rng = random.Random(self.config.seed + state.step)
        return [rng.random() for _ in range(state.benchmark.dim)]

    def tell(self, state: OptimizerState, result: EvaluationResult) -> OptimizerState:
        """Update random-search state after an evaluation.

        Inputs:
            state: Current optimiser state.
            result: New evaluation result.

        Output:
            Updated optimiser state.
        """

        observations = list(state.observations)
        observations.append(Observation(x=list(result.x_unit), y=float(result.y), metadata=result.metadata))
        return OptimizerState(
            benchmark=state.benchmark,
            observations=observations,
            step=state.step + 1,
            metadata=state.metadata,
        )


class WMBOOptimizer:
    """World-model black-box optimiser interface."""

    def __init__(self, config: OptimizerConfig) -> None:
        """Create a WMBO optimiser.

        Input:
            config: Optimiser configuration.

        Output:
            Optimiser instance.
        """

        self.config = config
        self.agent = WorldModelAgent()

    def ask(self, state: OptimizerState) -> Vector:
        """Propose a candidate using world-model reasoning.

        Input:
            state: Current optimiser state.

        Output:
            Candidate point in normalised coordinates.
        """

        if len(state.observations) < self.config.initial_samples:
            return RandomSearchOptimizer(self.config).ask(state)

        observed_x = [obs.x for obs in state.observations]
        observed_y = [obs.y for obs in state.observations]
        candidates = sample_unit_points(
            max(1, self.config.candidate_pool_size),
            state.benchmark.dim,
            seed=self.config.seed + 10_000 + state.step,
        )
        surrogate = make_surrogate(
            "gaussian_process",
            dim=state.benchmark.dim,
            options={"seed": self.config.seed, **dict(self.config.options)},
        ).fit(SurrogateDataset(x=observed_x, y=observed_y))
        prediction = surrogate.predict(candidates)
        descriptor = describe_landscape(
            observed_x,
            observed_y,
            surrogate_metadata={**dict(prediction.metadata), "std": prediction.std},
        )
        decision = self.agent.decide(
            AgentState(
                observed_x=observed_x,
                observed_y=observed_y,
                descriptor={
                    "dim": descriptor.dim,
                    "best_y": descriptor.best_y,
                    "y_range": descriptor.y_range,
                    "smoothness": descriptor.smoothness,
                    "modality": descriptor.modality,
                    "uncertainty": descriptor.uncertainty,
                    "labels": dict(descriptor.labels),
                },
                budget_used=len(state.observations),
                budget_total=self.config.budget,
            )
        )
        selected = select_next_candidate(
            AcquisitionInput(
                candidates=candidates,
                observed_x=observed_x,
                observed_y=observed_y,
                surrogate_mean=prediction.mean,
                surrogate_std=prediction.std,
                strategy=decision.strategy,
            )
        )
        return CandidateValidator(state.benchmark.dim).repair(selected.selected_x)

    def tell(self, state: OptimizerState, result: EvaluationResult) -> OptimizerState:
        """Update WMBO state after an evaluation.

        Inputs:
            state: Current optimiser state.
            result: New evaluation result.

        Output:
            Updated optimiser state.
        """

        return RandomSearchOptimizer(self.config).tell(state, result)


def make_optimizer(method: str, config: OptimizerConfig) -> Optimizer:
    """Construct an optimiser by name.

    Inputs:
        method: Optimisation method name.
        config: Optimiser configuration.

    Output:
        Object implementing the ``Optimizer`` protocol.
    """

    key = method.strip().lower().replace("-", "_")
    if key in {"random", "random_search", "baseline"}:
        return RandomSearchOptimizer(config)
    if key in {"wmbo", "world_model", "world_model_bo"}:
        return WMBOOptimizer(config)
    raise ValueError(f"Unknown optimiser method: {method}")
