"""Benchmark runner interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .control import RunConfig
from .metrics import RunSummary


@dataclass(frozen=True)
class BenchmarkRunRequest:
    """Input for one benchmark-method-seed run.

    Inputs:
        benchmark_name: Benchmark identifier.
        method: Optimisation method name.
        seed: Random seed.
        budget: Evaluation budget.
        output_dir: Directory for future run artifacts.
        metadata: Optional run metadata.

    Output:
        Passed to ``run_single_benchmark``.
    """

    benchmark_name: str
    method: str
    seed: int
    budget: int
    output_dir: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkRunResult:
    """Output for one completed optimisation run.

    Inputs:
        request: Original run request.
        summary: Run-level metric summary.
        observations: Recorded evaluation dictionaries.
        metadata: Optional run metadata.

    Output:
        Returned to the suite runner and future analysis code.
    """

    request: BenchmarkRunRequest
    summary: RunSummary
    observations: Sequence[Mapping[str, object]]
    metadata: Mapping[str, object] = field(default_factory=dict)


def run_single_benchmark(request: BenchmarkRunRequest) -> BenchmarkRunResult:
    """Run one optimiser on one benchmark.

    Input:
        request: Benchmark name, method, seed, budget, and optional output directory.

    Output:
        ``BenchmarkRunResult`` with observations and summary metrics.
    """

    raise NotImplementedError("Single benchmark execution is not implemented yet.")


def run_benchmark_suite(config: RunConfig) -> list[BenchmarkRunResult]:
    """Run a collection of benchmarks, methods, and seeds.

    Input:
        config: Suite-level run configuration.

    Output:
        List of ``BenchmarkRunResult`` objects.
    """

    raise NotImplementedError("Benchmark suite execution is not implemented yet.")


def save_run_result(result: BenchmarkRunResult, output_dir: str) -> None:
    """Persist a benchmark run result.

    Inputs:
        result: Run result to save.
        output_dir: Destination directory.

    Output:
        None. Future implementation will write files to disk.
    """

    raise NotImplementedError("Run result persistence is not implemented yet.")
