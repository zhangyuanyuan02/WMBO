"""Benchmark function interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

Vector = Sequence[float]
Bounds = Sequence[tuple[float, float]]
ObjectiveFunction = Callable[[Vector], float]


@dataclass(frozen=True)
class BenchmarkSpec:
    """Description of a black-box optimisation benchmark.

    Inputs:
        name: Benchmark identifier.
        dim: Input dimensionality.
        bounds: Natural-domain bounds for each dimension.
        optimum_value: Known global optimum value, if available.
        tags: Optional metadata such as smoothness or modality labels.

    Output:
        Used to construct benchmark evaluation requests.
    """

    name: str
    dim: int
    bounds: Bounds
    optimum_value: float | None = None
    tags: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationRequest:
    """Input required for one objective evaluation.

    Inputs:
        benchmark_name: Benchmark to evaluate.
        x_unit: Point in the normalised ``[0, 1]^d`` search space.
        seed: Optional seed for noisy benchmarks.

    Output:
        Passed to ``evaluate`` to produce an ``EvaluationResult``.
    """

    benchmark_name: str
    x_unit: Vector
    seed: int | None = None


@dataclass(frozen=True)
class EvaluationResult:
    """Output produced by one objective evaluation.

    Inputs:
        benchmark_name: Benchmark that was evaluated.
        x_unit: Input point in normalised coordinates.
        x_raw: Input point mapped to benchmark coordinates.
        y: Objective value.
        metadata: Optional evaluation details.

    Output:
        Stored as an observation for the optimiser.
    """

    benchmark_name: str
    x_unit: Vector
    x_raw: Vector
    y: float
    metadata: Mapping[str, object] = field(default_factory=dict)


def list_benchmarks() -> list[str]:
    """List benchmark names available in the project.

    Input:
        None.

    Output:
        Benchmark names as strings.
    """

    raise NotImplementedError("Benchmark registry is not implemented yet.")


def get_benchmark(name: str) -> BenchmarkSpec:
    """Look up the specification for a benchmark.

    Input:
        name: Benchmark identifier.

    Output:
        ``BenchmarkSpec`` for the requested benchmark.
    """

    raise NotImplementedError("Benchmark lookup is not implemented yet.")


def denormalise(x_unit: Vector, bounds: Bounds) -> list[float]:
    """Map a point from ``[0, 1]^d`` to benchmark coordinates.

    Inputs:
        x_unit: Normalised candidate point.
        bounds: Natural-domain bounds for each dimension.

    Output:
        Candidate point in benchmark coordinates.
    """

    raise NotImplementedError("Input denormalisation is not implemented yet.")


def normalise(x_raw: Vector, bounds: Bounds) -> list[float]:
    """Map a point from benchmark coordinates to ``[0, 1]^d``.

    Inputs:
        x_raw: Candidate point in benchmark coordinates.
        bounds: Natural-domain bounds for each dimension.

    Output:
        Candidate point in normalised coordinates.
    """

    raise NotImplementedError("Input normalisation is not implemented yet.")


def evaluate(request: EvaluationRequest) -> EvaluationResult:
    """Evaluate a benchmark at one normalised point.

    Input:
        request: Benchmark name, candidate point, and optional random seed.

    Output:
        ``EvaluationResult`` containing the objective value and mapped input.
    """

    raise NotImplementedError("Benchmark evaluation is not implemented yet.")
