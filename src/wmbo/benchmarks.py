"""Benchmark functions and search-space utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
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

    return [
        "branin",
        "hartmann3",
        "hartmann6",
        "ackley5",
        "ackley10",
        "rastrigin5",
        "rosenbrock3",
        "rosenbrock5",
        "levy5",
        "griewank5",
        "styblinski5",
    ]


def get_benchmark(name: str) -> BenchmarkSpec:
    """Look up the specification for a benchmark.

    Input:
        name: Benchmark identifier.

    Output:
        ``BenchmarkSpec`` for the requested benchmark.
    """

    key, dim_arg = _parse_benchmark_name(name)

    if key == "branin":
        return BenchmarkSpec(
            name="branin",
            dim=2,
            bounds=[(-5.0, 10.0), (0.0, 15.0)],
            optimum_value=0.397887,
            tags={
                "smoothness": "smooth",
                "modality": "multimodal",
                "curvature": "moderate",
                "anisotropy": "moderate",
            },
        )

    if key == "hartmann3":
        return BenchmarkSpec(
            name="hartmann3",
            dim=3,
            bounds=[(0.0, 1.0)] * 3,
            optimum_value=-3.86278,
            tags={
                "smoothness": "smooth",
                "modality": "multimodal",
                "curvature": "moderate",
                "anisotropy": "moderate",
            },
        )

    if key == "hartmann6":
        return BenchmarkSpec(
            name="hartmann6",
            dim=6,
            bounds=[(0.0, 1.0)] * 6,
            optimum_value=-3.32237,
            tags={
                "smoothness": "smooth",
                "modality": "multimodal",
                "curvature": "moderate",
                "anisotropy": "moderate",
            },
        )

    if key == "ackley":
        dim = dim_arg or 5
        return BenchmarkSpec(
            name=f"ackley{dim}",
            dim=dim,
            bounds=[(-32.768, 32.768)] * dim,
            optimum_value=0.0,
            tags={
                "smoothness": "rugged",
                "modality": "highly_multimodal",
                "curvature": "moderate",
                "anisotropy": "low",
            },
        )

    if key == "rastrigin":
        dim = dim_arg or 5
        return BenchmarkSpec(
            name=f"rastrigin{dim}",
            dim=dim,
            bounds=[(-5.12, 5.12)] * dim,
            optimum_value=0.0,
            tags={
                "smoothness": "rugged",
                "modality": "highly_multimodal",
                "curvature": "moderate",
                "anisotropy": "low",
            },
        )

    if key == "rosenbrock":
        dim = dim_arg or 3
        return BenchmarkSpec(
            name=f"rosenbrock{dim}",
            dim=dim,
            bounds=[(-2.0, 2.0)] * dim,
            optimum_value=0.0,
            tags={
                "smoothness": "smooth",
                "modality": "mostly_unimodal",
                "curvature": "high",
                "anisotropy": "high",
            },
        )

    if key == "levy":
        dim = dim_arg or 5
        return BenchmarkSpec(
            name=f"levy{dim}",
            dim=dim,
            bounds=[(-10.0, 10.0)] * dim,
            optimum_value=0.0,
            tags={
                "smoothness": "rugged",
                "modality": "multimodal",
                "curvature": "moderate",
                "anisotropy": "low",
            },
        )

    if key == "griewank":
        dim = dim_arg or 5
        return BenchmarkSpec(
            name=f"griewank{dim}",
            dim=dim,
            bounds=[(-600.0, 600.0)] * dim,
            optimum_value=0.0,
            tags={
                "smoothness": "smooth",
                "modality": "highly_multimodal",
                "curvature": "low",
                "anisotropy": "low",
            },
        )

    if key == "styblinski":
        dim = dim_arg or 5
        return BenchmarkSpec(
            name=f"styblinski{dim}",
            dim=dim,
            bounds=[(-5.0, 5.0)] * dim,
            optimum_value=-39.16599 * dim,
            tags={
                "smoothness": "smooth",
                "modality": "multimodal",
                "curvature": "moderate",
                "anisotropy": "low",
            },
        )

    raise ValueError(f"Unknown benchmark: {name}")


def denormalise(x_unit: Vector, bounds: Bounds) -> list[float]:
    """Map a point from ``[0, 1]^d`` to benchmark coordinates.

    Inputs:
        x_unit: Normalised candidate point.
        bounds: Natural-domain bounds for each dimension.

    Output:
        Candidate point in benchmark coordinates.
    """

    unit = _as_float_list(x_unit, name="x_unit")
    validated_bounds = _validate_bounds(bounds)
    if len(unit) != len(validated_bounds):
        raise ValueError(f"Expected {len(validated_bounds)} values, got {len(unit)}.")
    for value in unit:
        if value < 0.0 or value > 1.0:
            raise ValueError("Normalised inputs must be inside [0, 1].")
    return [lo + value * (hi - lo) for value, (lo, hi) in zip(unit, validated_bounds)]


def normalise(x_raw: Vector, bounds: Bounds) -> list[float]:
    """Map a point from benchmark coordinates to ``[0, 1]^d``.

    Inputs:
        x_raw: Candidate point in benchmark coordinates.
        bounds: Natural-domain bounds for each dimension.

    Output:
        Candidate point in normalised coordinates.
    """

    raw = _as_float_list(x_raw, name="x_raw")
    validated_bounds = _validate_bounds(bounds)
    if len(raw) != len(validated_bounds):
        raise ValueError(f"Expected {len(validated_bounds)} values, got {len(raw)}.")
    return [(value - lo) / (hi - lo) for value, (lo, hi) in zip(raw, validated_bounds)]


def evaluate(request: EvaluationRequest) -> EvaluationResult:
    """Evaluate a benchmark at one normalised point.

    Input:
        request: Benchmark name, candidate point, and optional random seed.

    Output:
        ``EvaluationResult`` containing the objective value and mapped input.
    """

    spec = get_benchmark(request.benchmark_name)
    x_unit = _as_float_list(request.x_unit, name="x_unit")
    x_raw = denormalise(x_unit, spec.bounds)
    objective = _get_objective(spec.name)
    y = objective(x_raw)
    return EvaluationResult(
        benchmark_name=spec.name,
        x_unit=x_unit,
        x_raw=x_raw,
        y=float(y),
        metadata={
            "dim": spec.dim,
            "optimum_value": spec.optimum_value,
            "seed": request.seed,
            "tags": dict(spec.tags),
        },
    )


def sample_unit_points(n_points: int, dim: int, seed: int | None = None) -> list[list[float]]:
    """Draw random points from the normalised search space.

    Inputs:
        n_points: Number of points to sample.
        dim: Dimensionality of each point.
        seed: Optional random seed.

    Output:
        List of points in ``[0, 1]^d``.
    """

    if n_points < 0:
        raise ValueError("n_points must be non-negative.")
    if dim <= 0:
        raise ValueError("dim must be positive.")
    rng = random.Random(seed)
    return [[rng.random() for _ in range(dim)] for _ in range(n_points)]


def sample_raw_points(n_points: int, bounds: Bounds, seed: int | None = None) -> list[list[float]]:
    """Draw random points and map them to the raw benchmark domain.

    Inputs:
        n_points: Number of points to sample.
        bounds: Natural-domain bounds for each dimension.
        seed: Optional random seed.

    Output:
        List of points in benchmark coordinates.
    """

    validated_bounds = _validate_bounds(bounds)
    return [denormalise(point, validated_bounds) for point in sample_unit_points(n_points, len(validated_bounds), seed)]


def _as_float_list(values: Vector, name: str) -> list[float]:
    try:
        return [float(value) for value in values]
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of numeric values.") from exc


def _validate_bounds(bounds: Bounds) -> list[tuple[float, float]]:
    validated: list[tuple[float, float]] = []
    for index, pair in enumerate(bounds):
        if len(pair) != 2:
            raise ValueError(f"Bound at index {index} must contain two values.")
        lo, hi = float(pair[0]), float(pair[1])
        if not lo < hi:
            raise ValueError(f"Invalid bound at index {index}: lower value must be less than upper value.")
        validated.append((lo, hi))
    if not validated:
        raise ValueError("At least one bound is required.")
    return validated


def _parse_benchmark_name(name: str) -> tuple[str, int | None]:
    key = name.strip().lower()
    for prefix in ("ackley", "rastrigin", "rosenbrock", "levy", "griewank", "styblinski"):
        if key.startswith(prefix):
            suffix = key[len(prefix) :]
            if suffix == "":
                return prefix, None
            if not suffix.isdigit():
                raise ValueError(f"Invalid benchmark dimension in name: {name}")
            dim = int(suffix)
            if dim <= 0:
                raise ValueError("Benchmark dimension must be positive.")
            return prefix, dim
    return key, None


def _get_objective(name: str) -> ObjectiveFunction:
    key, _ = _parse_benchmark_name(name)
    objectives: dict[str, ObjectiveFunction] = {
        "branin": _branin,
        "hartmann3": lambda x: _hartmann(x, _HARTMANN3_ALPHA, _HARTMANN3_A, _HARTMANN3_P),
        "hartmann6": lambda x: _hartmann(x, _HARTMANN6_ALPHA, _HARTMANN6_A, _HARTMANN6_P),
        "ackley": _ackley,
        "rastrigin": _rastrigin,
        "rosenbrock": _rosenbrock,
        "levy": _levy,
        "griewank": _griewank,
        "styblinski": _styblinski_tang,
    }
    return objectives[key]


def _branin(x: Vector) -> float:
    x1, x2 = x
    a = 1.0
    b = 5.1 / (4.0 * math.pi**2)
    c = 5.0 / math.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * math.pi)
    return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1.0 - t) * math.cos(x1) + s


def _ackley(x: Vector) -> float:
    values = _as_float_list(x, name="x")
    dim = len(values)
    square_mean = sum(value**2 for value in values) / dim
    cosine_mean = sum(math.cos(2.0 * math.pi * value) for value in values) / dim
    return -20.0 * math.exp(-0.2 * math.sqrt(square_mean)) - math.exp(cosine_mean) + 20.0 + math.e


def _rastrigin(x: Vector) -> float:
    values = _as_float_list(x, name="x")
    dim = len(values)
    return 10.0 * dim + sum(value**2 - 10.0 * math.cos(2.0 * math.pi * value) for value in values)


def _rosenbrock(x: Vector) -> float:
    values = _as_float_list(x, name="x")
    return sum(100.0 * (right - left**2) ** 2 + (1.0 - left) ** 2 for left, right in zip(values[:-1], values[1:]))


def _levy(x: Vector) -> float:
    values = _as_float_list(x, name="x")
    w = [1.0 + (value - 1.0) / 4.0 for value in values]
    term1 = math.sin(math.pi * w[0]) ** 2
    term3 = (w[-1] - 1.0) ** 2 * (1.0 + math.sin(2.0 * math.pi * w[-1]) ** 2)
    term2 = sum(
        (wi - 1.0) ** 2 * (1.0 + 10.0 * math.sin(math.pi * w_next) ** 2)
        for wi, w_next in zip(w[:-1], w[1:])
    )
    return term1 + term2 + term3


def _griewank(x: Vector) -> float:
    values = _as_float_list(x, name="x")
    sum_term = sum(value**2 for value in values) / 4000.0
    product_term = 1.0
    for index, value in enumerate(values, start=1):
        product_term *= math.cos(value / math.sqrt(index))
    return 1.0 + sum_term - product_term


def _styblinski_tang(x: Vector) -> float:
    values = _as_float_list(x, name="x")
    return 0.5 * sum(value**4 - 16.0 * value**2 + 5.0 * value for value in values)


_HARTMANN3_ALPHA = [1.0, 1.2, 3.0, 3.2]
_HARTMANN3_A = [
    [3.0, 10.0, 30.0],
    [0.1, 10.0, 35.0],
    [3.0, 10.0, 30.0],
    [0.1, 10.0, 35.0],
]
_HARTMANN3_P = [
    [0.3689, 0.1170, 0.2673],
    [0.4699, 0.4387, 0.7470],
    [0.1091, 0.8732, 0.5547],
    [0.0381, 0.5743, 0.8828],
]

_HARTMANN6_ALPHA = [1.0, 1.2, 3.0, 3.2]
_HARTMANN6_A = [
    [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
    [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
    [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
    [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
]
_HARTMANN6_P = [
    [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
    [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
    [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
    [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
]


def _hartmann(x: Vector, alpha: Sequence[float], a: Sequence[Vector], p: Sequence[Vector]) -> float:
    values = _as_float_list(x, name="x")
    total = 0.0
    for alpha_i, a_row, p_row in zip(alpha, a, p):
        inner = sum(a_ij * (x_j - p_ij) ** 2 for x_j, a_ij, p_ij in zip(values, a_row, p_row))
        total += alpha_i * math.exp(-inner)
    return -total


__all__ = [
    "BenchmarkSpec",
    "EvaluationRequest",
    "EvaluationResult",
    "list_benchmarks",
    "get_benchmark",
    "denormalise",
    "normalise",
    "evaluate",
    "sample_unit_points",
    "sample_raw_points",
]
