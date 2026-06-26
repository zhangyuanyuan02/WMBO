"""Benchmark runner interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .benchmarks import EvaluationRequest, evaluate, get_benchmark
from .control import RunConfig, build_default_optimizer_config
from .metrics import RunSummary, summarise_run
from .optimizers import OptimizerState, make_optimizer
from .utils import write_json


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

    benchmark = get_benchmark(request.benchmark_name)
    config = build_default_optimizer_config(request.method, request.budget, request.seed)
    optimizer = make_optimizer(request.method, config)
    state = OptimizerState(benchmark=benchmark, observations=[], step=0)

    for index in range(request.budget):
        candidate = optimizer.ask(state)
        result = evaluate(EvaluationRequest(benchmark_name=benchmark.name, x_unit=candidate, seed=request.seed + index))
        state = optimizer.tell(state, result)

    values = [obs.y for obs in state.observations]
    summary = summarise_run(benchmark.name, request.method, request.seed, values, benchmark.optimum_value)
    observations = [
        {
            "step": index,
            "x_unit": list(obs.x),
            "y": obs.y,
            "metadata": dict(obs.metadata),
        }
        for index, obs in enumerate(state.observations)
    ]
    result = BenchmarkRunResult(
        request=request,
        summary=summary,
        observations=observations,
        metadata={"benchmark": benchmark.name, "dim": benchmark.dim},
    )
    if request.output_dir:
        save_run_result(result, request.output_dir)
    return result


def run_benchmark_suite(config: RunConfig) -> list[BenchmarkRunResult]:
    """Run a collection of benchmarks, methods, and seeds.

    Input:
        config: Suite-level run configuration.

    Output:
        List of ``BenchmarkRunResult`` objects.
    """

    results: list[BenchmarkRunResult] = []
    for benchmark_name in config.benchmarks:
        for method in config.methods:
            for seed in config.seeds:
                results.append(
                    run_single_benchmark(
                        BenchmarkRunRequest(
                            benchmark_name=benchmark_name,
                            method=method,
                            seed=seed,
                            budget=config.optimizer.budget,
                            output_dir=config.output_dir,
                        )
                    )
                )
    return results


def save_run_result(result: BenchmarkRunResult, output_dir: str) -> None:
    """Persist a benchmark run result.

    Inputs:
        result: Run result to save.
        output_dir: Destination directory.

    Output:
        None. Future implementation will write files to disk.
    """

    path = f"{output_dir}/{result.request.benchmark_name}_{result.request.method}_seed{result.request.seed}.json"
    write_json(
        path,
        {
            "request": {
                "benchmark_name": result.request.benchmark_name,
                "method": result.request.method,
                "seed": result.request.seed,
                "budget": result.request.budget,
                "metadata": dict(result.request.metadata),
            },
            "summary": {
                "benchmark_name": result.summary.benchmark_name,
                "method": result.summary.method,
                "seed": result.summary.seed,
                "final_best": result.summary.final_best,
                "final_regret": result.summary.final_regret,
                "num_evaluations": result.summary.num_evaluations,
                "metadata": dict(result.summary.metadata),
            },
            "observations": list(result.observations),
            "metadata": dict(result.metadata),
        },
    )
