"""Optimiser interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

from .benchmarks import BenchmarkSpec, EvaluationResult
from .control import OptimizerConfig

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

        raise NotImplementedError("Random-search candidate generation is not implemented yet.")

    def tell(self, state: OptimizerState, result: EvaluationResult) -> OptimizerState:
        """Update random-search state after an evaluation.

        Inputs:
            state: Current optimiser state.
            result: New evaluation result.

        Output:
            Updated optimiser state.
        """

        raise NotImplementedError("Random-search state update is not implemented yet.")


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

    def ask(self, state: OptimizerState) -> Vector:
        """Propose a candidate using world-model reasoning.

        Input:
            state: Current optimiser state.

        Output:
            Candidate point in normalised coordinates.
        """

        raise NotImplementedError("WMBO candidate generation is not implemented yet.")

    def tell(self, state: OptimizerState, result: EvaluationResult) -> OptimizerState:
        """Update WMBO state after an evaluation.

        Inputs:
            state: Current optimiser state.
            result: New evaluation result.

        Output:
            Updated optimiser state.
        """

        raise NotImplementedError("WMBO state update is not implemented yet.")


def make_optimizer(method: str, config: OptimizerConfig) -> Optimizer:
    """Construct an optimiser by name.

    Inputs:
        method: Optimisation method name.
        config: Optimiser configuration.

    Output:
        Object implementing the ``Optimizer`` protocol.
    """

    raise NotImplementedError("Optimiser factory is not implemented yet.")
