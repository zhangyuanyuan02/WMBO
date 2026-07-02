"""Benchmark runner utilities."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .benchmarks import EvaluationRequest, evaluate, get_benchmark
from .control import OptimizerConfig, RunConfig, build_default_optimizer_config
from .metrics import RunSummary, cumulative_best, simple_regret, summarise_run
from .optimizers import create_initial_state, make_optimizer
from .utils import ensure_dir, write_json


@dataclass(frozen=True)
class BenchmarkRunRequest:
    """Input for one benchmark-method-seed run.

    Inputs:
        benchmark_name: Benchmark identifier.
        method: Optimisation method name.
        seed: Random seed.
        budget: Evaluation budget.
        output_dir: Directory for run artifacts.
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
        Returned to the suite runner and analysis code.
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

    if request.budget <= 0:
        raise ValueError("budget must be positive.")

    benchmark = get_benchmark(request.benchmark_name)
    optimizer_config = _make_optimizer_config(request)
    optimizer = make_optimizer(request.method, optimizer_config)
    state = create_initial_state(benchmark)
    log_enabled = _logging_enabled(request.metadata)
    run_label = _run_label(request)

    if log_enabled:
        _log(
            f"{run_label} start benchmark={benchmark.name} dim={benchmark.dim} "
            f"method={request.method} seed={request.seed} budget={request.budget}"
        )

    observations: list[dict[str, object]] = []
    objective_values: list[float] = []

    for step in range(request.budget):
        if log_enabled:
            _log(f"{run_label} step {step + 1}/{request.budget} ask")
        x_unit = optimizer.ask(state)
        result = evaluate(
            EvaluationRequest(
                benchmark_name=benchmark.name,
                x_unit=x_unit,
                seed=request.seed,
            )
        )
        state = optimizer.tell(state, result)
        objective_values.append(float(result.y))
        best_curve = cumulative_best(objective_values, minimise=True)
        regret_curve = simple_regret(best_curve, benchmark.optimum_value)
        optimiser_metadata = _extract_observation_metadata(state.metadata)
        observation = {
            "step": step,
            "benchmark": benchmark.name,
            "method": request.method,
            "seed": request.seed,
            "x_unit": [float(value) for value in result.x_unit],
            "x_raw": [float(value) for value in result.x_raw],
            "y": float(result.y),
            "best_y": best_curve[-1],
            "simple_regret": regret_curve[-1],
            **optimiser_metadata,
        }
        observations.append(
            observation
        )
        if log_enabled:
            _log(_format_step_log(run_label, step + 1, request.budget, observation))

    summary = summarise_run(
        benchmark_name=benchmark.name,
        method=request.method,
        seed=request.seed,
        values=objective_values,
        optimum_value=benchmark.optimum_value,
    )
    result = BenchmarkRunResult(
        request=request,
        summary=summary,
        observations=observations,
        metadata={
            "benchmark": benchmark.name,
            "dim": benchmark.dim,
            "bounds": [list(bound) for bound in benchmark.bounds],
            "optimum_value": benchmark.optimum_value,
            "optimizer_state": dict(state.metadata),
        },
    )
    if request.output_dir:
        save_run_result(result, request.output_dir)
    if log_enabled:
        _log(
            f"{run_label} done final_best={_format_number(summary.final_best)} "
            f"final_regret={_format_number(summary.final_regret)} output={request.output_dir or '<memory>'}"
        )
    return result


def run_benchmark_suite(config: RunConfig) -> list[BenchmarkRunResult]:
    """Run a collection of benchmarks, methods, and seeds.

    Input:
        config: Suite-level run configuration.

    Output:
        List of ``BenchmarkRunResult`` objects.
    """

    benchmarks = list(config.benchmarks) or ["branin"]
    methods = list(config.methods) or ["random"]
    seeds = list(config.seeds) or [0]

    results: list[BenchmarkRunResult] = []
    total_runs = len(benchmarks) * len(methods) * len(seeds)
    run_index = 0
    suite_logging = _logging_enabled(config.optimizer.options)
    if suite_logging:
        _log(
            f"[suite] start benchmarks={len(benchmarks)} methods={len(methods)} "
            f"seeds={len(seeds)} total_runs={total_runs} budget={config.optimizer.budget} "
            f"output={config.output_dir}"
        )
    for benchmark_name in benchmarks:
        for method in methods:
            for seed in seeds:
                run_index += 1
                request = BenchmarkRunRequest(
                    benchmark_name=benchmark_name,
                    method=method,
                    seed=int(seed),
                    budget=int(config.optimizer.budget),
                    output_dir=config.output_dir,
                    metadata={
                        "initial_samples": config.optimizer.initial_samples,
                        "candidate_pool_size": config.optimizer.candidate_pool_size,
                        "options": dict(config.optimizer.options),
                        "run_index": run_index,
                        "total_runs": total_runs,
                    },
                )
                try:
                    results.append(run_single_benchmark(request))
                except Exception as exc:
                    if suite_logging:
                        _log(
                            f"[run {run_index}/{total_runs}] failed "
                            f"benchmark={benchmark_name} method={method} seed={seed}: {exc}"
                        )
                    raise

    if config.output_dir:
        save_suite_summary(results, config.output_dir)
    if suite_logging:
        _log(f"[suite] done completed_runs={len(results)} output={config.output_dir}")
    return results


def save_run_result(result: BenchmarkRunResult, output_dir: str) -> None:
    """Persist a benchmark run result.

    Inputs:
        result: Run result to save.
        output_dir: Destination directory.

    Output:
        None. Writes JSON and CSV artifacts to disk.
    """

    run_dir = _run_directory(output_dir, result.request.benchmark_name, result.request.method, result.request.seed)
    ensure_dir(run_dir)
    write_json(run_dir / "summary.json", _summary_to_dict(result.summary))
    write_json(run_dir / "run.json", _to_jsonable(result))
    _write_observations_csv(run_dir / "observations.csv", result.observations)


def save_suite_summary(results: Sequence[BenchmarkRunResult], output_dir: str) -> None:
    """Persist a compact summary for a benchmark suite.

    Inputs:
        results: Completed run results.
        output_dir: Destination directory.

    Output:
        None. Writes ``summary.csv`` and ``summary.json``.
    """

    directory = ensure_dir(output_dir)
    rows = [_summary_to_dict(result.summary) for result in results]
    write_json(directory / "summary.json", rows)
    _write_summary_csv(directory / "summary.csv", rows)


def _make_optimizer_config(request: BenchmarkRunRequest) -> OptimizerConfig:
    config = build_default_optimizer_config(
        method=request.method,
        budget=request.budget,
        seed=request.seed,
    )
    metadata = dict(request.metadata)
    initial_samples = int(metadata.get("initial_samples", config.initial_samples) or config.initial_samples)
    candidate_pool_size = int(metadata.get("candidate_pool_size", config.candidate_pool_size) or config.candidate_pool_size)
    raw_options = metadata.get("options", {})
    options = dict(raw_options) if isinstance(raw_options, Mapping) else {}
    return OptimizerConfig(
        method=request.method,
        budget=config.budget,
        initial_samples=max(1, min(initial_samples, config.budget)),
        candidate_pool_size=max(1, candidate_pool_size),
        seed=request.seed,
        options=options,
    )


def _run_directory(output_dir: str, benchmark_name: str, method: str, seed: int) -> Path:
    return Path(output_dir) / benchmark_name / method / f"seed_{seed}"


def _summary_to_dict(summary: RunSummary) -> dict[str, object]:
    data = asdict(summary)
    metadata = data.pop("metadata", {})
    if isinstance(metadata, Mapping):
        data.update(metadata)
    return _to_jsonable(data)


def _write_observations_csv(path: Path, observations: Sequence[Mapping[str, object]]) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "step",
        "benchmark",
        "method",
        "seed",
        "x_unit",
        "x_raw",
        "y",
        "best_y",
        "simple_regret",
        "strategy",
        "proposed_strategy",
        "executed_strategy",
        "override_reason",
        "budget_phase",
        "remaining_budget",
        "consecutive_no_improvement",
        "hypothesis_id",
        "hypothesis_status",
        "hypothesis_status_counts",
        "strategy_trust",
        "strategy_success_rates",
        "agent_type",
        "llm_error",
        "selected_candidate_id",
        "candidate_override",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for observation in observations:
            row = dict(observation)
            row["x_unit"] = _format_vector(row.get("x_unit"))
            row["x_raw"] = _format_vector(row.get("x_raw"))
            for key in ("strategy_trust", "strategy_success_rates", "hypothesis_status_counts"):
                row[key] = _format_mapping(row.get(key))
            writer.writerow({name: row.get(name) for name in fieldnames})


def _write_summary_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    ensure_dir(path.parent)
    fieldnames = ["benchmark_name", "method", "seed", "final_best", "final_regret", "num_evaluations"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})



def _extract_observation_metadata(state_metadata: Mapping[str, object]) -> dict[str, object]:
    decision = state_metadata.get("last_reasoning_decision") if isinstance(state_metadata, Mapping) else None
    if not isinstance(decision, Mapping):
        return {}
    control = decision.get("wmbo_control")
    control_map = control if isinstance(control, Mapping) else {}
    return {
        "strategy": decision.get("executed_strategy", decision.get("strategy")),
        "proposed_strategy": decision.get("proposed_strategy"),
        "executed_strategy": decision.get("executed_strategy"),
        "override_reason": decision.get("override_reason"),
        "budget_phase": decision.get("budget_phase"),
        "remaining_budget": decision.get("remaining_budget"),
        "consecutive_no_improvement": control_map.get("consecutive_no_improvement"),
        "hypothesis_id": decision.get("hypothesis_id"),
        "hypothesis_status": decision.get("hypothesis_status"),
        "hypothesis_status_counts": decision.get("hypothesis_status_counts"),
        "strategy_trust": decision.get("strategy_trust"),
        "strategy_success_rates": decision.get("strategy_success_rates"),
        "agent_type": decision.get("agent_type"),
        "llm_error": decision.get("llm_error"),
        "selected_candidate_id": decision.get("selected_candidate_id"),
        "candidate_override": decision.get("candidate_override"),
    }

def _format_vector(value: object) -> str:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return "[" + ", ".join(f"{float(item):.8g}" for item in value) + "]"
    return str(value)



def _format_mapping(value: object) -> str:
    if isinstance(value, Mapping):
        parts = [f"{key}={value[key]}" for key in sorted(value)]
        return ";".join(parts)
    return "" if value is None else str(value)

def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _logging_enabled(metadata: Mapping[str, object]) -> bool:
    raw_options = metadata.get("options", metadata) if isinstance(metadata, Mapping) else {}
    options = raw_options if isinstance(raw_options, Mapping) else {}
    logging_options = options.get("logging", {}) if isinstance(options, Mapping) else {}
    if isinstance(logging_options, Mapping):
        return _truthy(logging_options.get("verbose", True))
    return True


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _run_label(request: BenchmarkRunRequest) -> str:
    run_index = request.metadata.get("run_index") if isinstance(request.metadata, Mapping) else None
    total_runs = request.metadata.get("total_runs") if isinstance(request.metadata, Mapping) else None
    if run_index is not None and total_runs is not None:
        return f"[run {run_index}/{total_runs}]"
    return "[run]"


def _format_step_log(
    run_label: str,
    step_number: int,
    budget: int,
    observation: Mapping[str, object],
) -> str:
    parts = [
        f"{run_label} step {step_number}/{budget}",
        f"y={_format_number(observation.get('y'))}",
        f"best={_format_number(observation.get('best_y'))}",
    ]
    agent = observation.get("agent_type")
    strategy = observation.get("executed_strategy") or observation.get("strategy")
    phase = observation.get("budget_phase")
    if agent:
        parts.append(f"agent={agent}")
    if strategy:
        parts.append(f"strategy={strategy}")
    if phase:
        parts.append(f"phase={phase}")
    llm_error = observation.get("llm_error")
    if llm_error:
        parts.append(f"llm_error={str(llm_error)[:160]}")
    return " ".join(parts)


def _log(message: str) -> None:
    print(message, flush=True)


def _format_number(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


__all__ = [
    "BenchmarkRunRequest",
    "BenchmarkRunResult",
    "run_single_benchmark",
    "run_benchmark_suite",
    "save_run_result",
    "save_suite_summary",
]
