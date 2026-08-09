"""Benchmark runner utilities."""

from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import math
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from scipy.stats import qmc

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
    shared_initial = _shared_initial_points(
        benchmark_name=benchmark.name,
        dim=benchmark.dim,
        count=optimizer_config.initial_samples,
        seed=request.seed,
        mode=str(optimizer_config.options.get("shared_initial_design", "none")),
    )
    run_started = time.perf_counter()

    for step in range(request.budget):
        if log_enabled:
            _log(f"{run_label} step {step + 1}/{request.budget} ask")
        ask_started = time.perf_counter()
        if step < len(shared_initial):
            x_unit = list(shared_initial[step])
        elif step == 0 and benchmark.recommended_start_unit is not None:
            x_unit = [float(value) for value in benchmark.recommended_start_unit]
        else:
            x_unit = optimizer.ask(state)
        optimizer_ask_time = time.perf_counter() - ask_started
        evaluation_started = time.perf_counter()
        result = evaluate(
            EvaluationRequest(
                benchmark_name=benchmark.name,
                x_unit=x_unit,
                seed=request.seed,
                options=_evaluation_options(request.metadata),
            )
        )
        objective_evaluation_time = time.perf_counter() - evaluation_started
        tell_started = time.perf_counter()
        state = optimizer.tell(state, result)
        optimizer_tell_time = time.perf_counter() - tell_started
        objective_values.append(float(result.y))
        best_curve = cumulative_best(objective_values, minimise=True)
        regret_curve = simple_regret(best_curve, benchmark.optimum_value)
        optimiser_metadata = _extract_observation_metadata(state.metadata)
        evaluation_metadata = dict(result.metadata)
        feasible_gaps = [
            gap
            for gap in (
                _finite_float(item.get("normalised_cost_gap"))
                for item in observations
                if bool(item.get("feasible", False))
            )
            if gap is not None
        ]
        current_gap = _finite_float(evaluation_metadata.get("normalised_cost_gap"))
        if bool(evaluation_metadata.get("feasible", False)) and current_gap is not None:
            feasible_gaps.append(current_gap)
        primary_best = (
            min(feasible_gaps)
            if benchmark.constrained and feasible_gaps
            else None
            if benchmark.constrained
            else best_curve[-1]
        )
        observation = {
            "step": step,
            "benchmark": benchmark.name,
            "method": request.method,
            "seed": request.seed,
            "x_unit": [float(value) for value in result.x_unit],
            "x_raw": [float(value) for value in result.x_raw],
            "y": float(result.y),
            "best_y": best_curve[-1],
            "primary_best": primary_best,
            "simple_regret": regret_curve[-1],
            "optimizer_ask_time": optimizer_ask_time,
            "optimizer_tell_time": optimizer_tell_time,
            "objective_evaluation_time": objective_evaluation_time,
            "is_shared_initial": step < len(shared_initial),
            **evaluation_metadata,
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
        initial_samples=optimizer_config.initial_samples,
        metadata={
            **(_summarise_opf_observations(observations) if benchmark.family == "opf" else {}),
            **_summarise_llm_observations(observations),
            "total_run_time": time.perf_counter() - run_started,
            "total_optimizer_time": sum(
                float(item["optimizer_ask_time"]) + float(item["optimizer_tell_time"])
                for item in observations
            ),
            "total_objective_evaluation_time": sum(
                float(item["objective_evaluation_time"]) for item in observations
            ),
        },
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
            "family": benchmark.family,
            "constrained": benchmark.constrained,
            "benchmark_metadata": dict(benchmark.metadata),
            "optimizer_state": dict(state.metadata),
        },
    )
    if request.output_dir:
        save_run_result(result, request.output_dir)
    if log_enabled:
        primary_score = summary.metadata.get("primary_score")
        _log(
            f"{run_label} done primary_score={_format_number(primary_score)} "
            f"final_best={_format_number(summary.final_best)} "
            f"final_regret={_format_number(summary.final_regret)} output={request.output_dir or '<memory>'}"
        )
    return result


def run_benchmark_suite(
    config: RunConfig,
    *,
    resume: bool = False,
    continue_on_error: bool = False,
    workers: int | None = None,
) -> list[BenchmarkRunResult]:
    """Run a collection of benchmarks, methods, and seeds.

    Input:
        config: Suite-level run configuration.

    Output:
        List of ``BenchmarkRunResult`` objects.
    """

    benchmarks = list(config.benchmarks) or ["branin"]
    methods = list(config.methods) or ["random"]
    seeds = list(config.seeds) or [0]

    results_by_index: dict[int, BenchmarkRunResult] = {}
    failures: list[dict[str, object]] = []
    fingerprint = _config_fingerprint(config)
    if config.output_dir:
        _prepare_manifest(config, fingerprint=fingerprint, resume=resume)
    total_runs = len(benchmarks) * len(methods) * len(seeds)
    worker_count = _resolve_worker_count(config, workers)
    if worker_count > 1 and any(
        str(method).lower().replace("-", "_") == "wmbo_llm" for method in methods
    ):
        raise ValueError(
            "Parallel execution is disabled for wmbo_llm to avoid uncontrolled concurrent API calls. "
            "Run the LLM suite with workers=1."
        )
    run_index = 0
    suite_logging = _logging_enabled(config.optimizer.options)
    if suite_logging:
        _log(
            f"[suite] start benchmarks={len(benchmarks)} methods={len(methods)} "
            f"seeds={len(seeds)} total_runs={total_runs} budget={config.optimizer.budget} "
            f"workers={worker_count} output={config.output_dir}"
        )
    pending: list[tuple[int, BenchmarkRunRequest]] = []
    for benchmark_name in benchmarks:
        for method in methods:
            for seed in seeds:
                run_index += 1
                request = BenchmarkRunRequest(
                    benchmark_name=benchmark_name,
                    method=method,
                    seed=int(seed),
                    budget=_budget_for_benchmark(config, benchmark_name),
                    output_dir=config.output_dir,
                    metadata={
                        "initial_samples": config.optimizer.initial_samples,
                        "candidate_pool_size": config.optimizer.candidate_pool_size,
                        "options": dict(config.optimizer.options),
                        "evaluation": dict(config.evaluation),
                        "run_index": run_index,
                        "total_runs": total_runs,
                    },
                )
                try:
                    if resume:
                        saved = _load_completed_run(request, fingerprint=fingerprint)
                        if saved is not None:
                            results_by_index[run_index] = saved
                            continue
                except Exception as exc:
                    _record_failure(
                        failures=failures,
                        request=request,
                        error=exc,
                        suite_logging=suite_logging,
                    )
                    if not continue_on_error:
                        raise
                else:
                    pending.append((run_index, request))

    if worker_count == 1:
        for run_index, request in pending:
            try:
                result = _execute_run_request(request, fingerprint)
                results_by_index[run_index] = result
            except Exception as exc:
                _record_failure(
                    failures=failures,
                    request=request,
                    error=exc,
                    suite_logging=suite_logging,
                )
                if not continue_on_error:
                    raise
    elif pending:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_request = {
                executor.submit(_execute_run_request, _quiet_parallel_request(request), fingerprint): (run_index, request)
                for run_index, request in pending
            }
            for future in as_completed(future_to_request):
                run_index, request = future_to_request[future]
                try:
                    results_by_index[run_index] = future.result()
                    if suite_logging:
                        _log(
                            f"[run {run_index}/{total_runs}] done "
                            f"benchmark={request.benchmark_name} method={request.method} seed={request.seed}"
                        )
                except Exception as exc:
                    _record_failure(
                        failures=failures,
                        request=request,
                        error=exc,
                        suite_logging=suite_logging,
                    )
                    if not continue_on_error:
                        for pending_future in future_to_request:
                            pending_future.cancel()
                        raise

    results = [results_by_index[index] for index in sorted(results_by_index)]

    if config.output_dir:
        save_suite_summary(results, config.output_dir)
        write_json(Path(config.output_dir) / "failed_runs.json", failures)
        _finish_manifest(config.output_dir, completed=len(results), failed=len(failures))
    if suite_logging:
        _log(f"[suite] done completed_runs={len(results)} output={config.output_dir}")
    return results


def _execute_run_request(request: BenchmarkRunRequest, fingerprint: str) -> BenchmarkRunResult:
    """Run and atomically mark one request; suitable for a worker process."""

    result = run_single_benchmark(request)
    _write_run_fingerprint(request, fingerprint)
    return result


def _quiet_parallel_request(request: BenchmarkRunRequest) -> BenchmarkRunRequest:
    """Suppress per-step worker output while retaining parent progress messages."""

    metadata = dict(request.metadata)
    options = metadata.get("options", {})
    copied_options = dict(options) if isinstance(options, Mapping) else {}
    logging_options = copied_options.get("logging", {})
    copied_logging = dict(logging_options) if isinstance(logging_options, Mapping) else {}
    copied_logging["verbose"] = False
    copied_options["logging"] = copied_logging
    metadata["options"] = copied_options
    return BenchmarkRunRequest(
        benchmark_name=request.benchmark_name,
        method=request.method,
        seed=request.seed,
        budget=request.budget,
        output_dir=request.output_dir,
        metadata=metadata,
    )


def _resolve_worker_count(config: RunConfig, workers: int | None) -> int:
    if workers is not None:
        if int(workers) < 1:
            raise ValueError("workers must be at least 1.")
        return int(workers)
    execution = config.optimizer.options.get("execution", {})
    configured = execution.get("workers", 1) if isinstance(execution, Mapping) else 1
    if int(configured) < 1:
        raise ValueError("execution.workers must be at least 1.")
    return int(configured)


def _record_failure(
    *,
    failures: list[dict[str, object]],
    request: BenchmarkRunRequest,
    error: Exception,
    suite_logging: bool,
) -> None:
    if suite_logging:
        _log(
            f"{_run_label(request)} failed benchmark={request.benchmark_name} "
            f"method={request.method} seed={request.seed}: {error}"
        )
    failure: dict[str, object] = {
        "benchmark_name": request.benchmark_name,
        "method": request.method,
        "seed": request.seed,
        "error_type": type(error).__name__,
        "error": str(error),
    }
    telemetry = getattr(error, "telemetry", None)
    if isinstance(telemetry, Mapping):
        failure["llm_telemetry"] = _to_jsonable(telemetry)
    failures.append(failure)


def describe_benchmark_suite(config: RunConfig) -> dict[str, object]:
    """Return an execution estimate without evaluating any objective."""

    run_count = len(config.benchmarks) * len(config.methods) * len(config.seeds)
    evaluations = 0
    llm_calls = 0
    for benchmark_name in config.benchmarks:
        budget = _budget_for_benchmark(config, benchmark_name)
        initial = _initial_samples_for_benchmark(config, benchmark_name, budget)
        evaluations += budget * len(config.methods) * len(config.seeds)
        if "wmbo_llm" in {str(method).lower().replace("-", "_") for method in config.methods}:
            llm_calls += max(0, budget - initial) * len(config.seeds)
    return {
        "benchmarks": len(config.benchmarks),
        "methods": len(config.methods),
        "seeds": len(config.seeds),
        "runs": run_count,
        "evaluations": evaluations,
        "estimated_llm_calls": llm_calls,
    }


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
    aggregate_rows = _aggregate_summary_rows(rows)
    write_json(directory / "aggregate_summary.json", aggregate_rows)
    _write_aggregate_summary_csv(directory / "aggregate_summary.csv", aggregate_rows)


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
    per_dim = options.get("initial_samples_per_dimension")
    if per_dim is not None:
        benchmark = get_benchmark(request.benchmark_name)
        initial_samples = max(1, int(per_dim) * benchmark.dim)
    method_options = options.get("method_options", {})
    if isinstance(method_options, Mapping):
        override = method_options.get(request.method, {})
        if isinstance(override, Mapping):
            options.update(dict(override))
    return OptimizerConfig(
        method=request.method,
        budget=config.budget,
        initial_samples=max(1, min(initial_samples, config.budget)),
        candidate_pool_size=max(1, candidate_pool_size),
        seed=request.seed,
        options=options,
    )


def _initial_samples_for_benchmark(config: RunConfig, benchmark_name: str, budget: int) -> int:
    per_dim = config.optimizer.options.get("initial_samples_per_dimension")
    if per_dim is None:
        return max(1, min(config.optimizer.initial_samples, budget))
    return max(1, min(int(per_dim) * get_benchmark(benchmark_name).dim, budget))


def _budget_for_benchmark(config: RunConfig, benchmark_name: str) -> int:
    per_dim = config.optimizer.options.get("budget_per_dimension")
    if per_dim is None:
        return int(config.optimizer.budget)
    return max(1, int(per_dim) * get_benchmark(benchmark_name).dim)


def _shared_initial_points(
    *,
    benchmark_name: str,
    dim: int,
    count: int,
    seed: int,
    mode: str,
) -> list[list[float]]:
    if mode.strip().lower() not in {"sobol", "shared_sobol"}:
        return []
    digest = hashlib.sha256(f"{benchmark_name}|{int(seed)}".encode("utf-8")).digest()
    sobol_seed = int.from_bytes(digest[:4], "big", signed=False)
    power = int(math.ceil(math.log2(max(1, int(count)))))
    points = qmc.Sobol(d=int(dim), scramble=True, seed=sobol_seed).random_base2(power)
    return points[: int(count)].astype(float).tolist()


def _run_directory(output_dir: str, benchmark_name: str, method: str, seed: int) -> Path:
    return Path(output_dir) / benchmark_name / method / f"seed_{seed}"


def _config_fingerprint(config: RunConfig) -> str:
    sanitized = _sanitize_config_value(asdict(config))
    payload = json.dumps(sanitized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sanitize_config_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): ("<redacted>" if "key" in str(key).lower() else _sanitize_config_value(item))
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_sanitize_config_value(item) for item in value]
    return value


def _prepare_manifest(config: RunConfig, *, fingerprint: str, resume: bool) -> None:
    output = ensure_dir(config.output_dir)
    path = output / "manifest.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            existing = json.load(file)
        existing_fingerprint = existing.get("config_fingerprint") if isinstance(existing, Mapping) else None
        if existing_fingerprint != fingerprint:
            raise ValueError(
                "Output directory contains results from a different configuration; "
                "choose a new output directory or restore the matching config."
            )
        if not resume:
            raise ValueError("Output directory already has a manifest; use --resume to continue.")
        return
    write_json(
        path,
        {
            "config_fingerprint": fingerprint,
            "status": "running",
            "created_unix": time.time(),
            "config": _sanitize_config_value(asdict(config)),
            "dependencies": _dependency_versions(),
        },
    )


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("numpy", "scipy", "scikit-learn", "ioh", "optuna", "cma", "HEBO"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _write_run_fingerprint(request: BenchmarkRunRequest, fingerprint: str) -> None:
    if request.output_dir is None:
        return
    write_json(
        _run_directory(request.output_dir, request.benchmark_name, request.method, request.seed)
        / "complete.json",
        {"config_fingerprint": fingerprint, "status": "complete"},
    )


def _load_completed_run(
    request: BenchmarkRunRequest,
    *,
    fingerprint: str,
) -> BenchmarkRunResult | None:
    if request.output_dir is None:
        return None
    directory = _run_directory(request.output_dir, request.benchmark_name, request.method, request.seed)
    marker = directory / "complete.json"
    run_path = directory / "run.json"
    if not marker.exists() or not run_path.exists():
        return None
    with marker.open("r", encoding="utf-8") as file:
        marker_data = json.load(file)
    if marker_data.get("config_fingerprint") != fingerprint:
        raise ValueError(f"Run fingerprint mismatch: {directory}")
    with run_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    summary_data = dict(data["summary"])
    summary = RunSummary(
        benchmark_name=str(summary_data["benchmark_name"]),
        method=str(summary_data["method"]),
        seed=int(summary_data["seed"]),
        final_best=_finite_float(summary_data.get("final_best")),
        final_regret=_finite_float(summary_data.get("final_regret")),
        num_evaluations=int(summary_data["num_evaluations"]),
        metadata=dict(summary_data.get("metadata", {})),
    )
    return BenchmarkRunResult(
        request=request,
        summary=summary,
        observations=list(data.get("observations", [])),
        metadata=dict(data.get("metadata", {})),
    )


def _finish_manifest(output_dir: str, *, completed: int, failed: int) -> None:
    path = Path(output_dir) / "manifest.json"
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    data.update(
        {
            "status": "complete" if failed == 0 else "complete_with_failures",
            "completed_runs": completed,
            "failed_runs": failed,
            "finished_unix": time.time(),
        }
    )
    write_json(path, data)


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
        "primary_best",
        "simple_regret",
        "is_shared_initial",
        "optimizer_ask_time",
        "optimizer_tell_time",
        "objective_evaluation_time",
        "family",
        "constrained",
        "generation_cost",
        "control_names",
        "control_values",
        "pg_mw",
        "vg_pu",
        "reference_cost",
        "normalised_cost_gap",
        "feasible",
        "pf_converged",
        "total_violation",
        "max_normalized_violation",
        "feasibility_tolerance",
        "constraint_ratio",
        "constraint_excess",
        "legacy_penalised_score",
        "max_voltage_violation",
        "max_thermal_violation",
        "max_angle_violation",
        "generator_violation",
        "power_balance_residual",
        "evaluation_time",
        "solver_time",
        "termination_status",
        "solver_error",
        "scenario_id",
        "score_kind",
        "reference_kind",
        "penalty_weight",
        "failure_penalty",
        "strategy",
        "proposed_strategy",
        "executed_strategy",
        "override_reason",
        "budget_phase",
        "remaining_budget",
        "macro_action_id",
        "macro_strategy",
        "macro_step",
        "macro_min_steps",
        "macro_max_steps",
        "macro_continued",
        "macro_termination_reason",
        "macro_reward",
        "macro_reward_components",
        "previous_macro_settlement",
        "operator_state_summary",
        "geometry_features",
        "regime_posteriors",
        "lengthscale_condition",
        "local_condition",
        "rotation_score",
        "effective_dimension",
        "valley_score",
        "regime_separable_smooth",
        "regime_rotated_ill_conditioned",
        "regime_curved_valley",
        "regime_rugged_multimodal",
        "regime_weakly_identified",
        "consecutive_no_improvement",
        "hypothesis_id",
        "hypothesis_status",
        "hypothesis_region_center",
        "hypothesis_region_radius",
        "hypothesis_sensitive_dims",
        "hypothesis_confidence",
        "hypothesis_posterior_probability",
        "hypothesis_relevant_evidence_count",
        "hypothesis_supporting_evidence",
        "hypothesis_contradicting_evidence",
        "falsification_rule",
        "hypothesis_status_counts",
        "strategy_trust",
        "strategy_success_rates",
        "agent_type",
        "llm_error",
        "llm_model",
        "llm_calls",
        "llm_prompt_tokens",
        "llm_completion_tokens",
        "llm_reasoning_tokens",
        "llm_total_tokens",
        "llm_latency_seconds",
        "llm_attempts",
        "requested_candidate_id",
        "selected_candidate_id",
        "evidence_role",
        "target_hypothesis_id",
        "joint_score",
        "expected_improvement",
        "information_gain",
        "predicted_feasibility_probability",
        "predicted_constraint_log_ratio",
        "constraint_std",
        "candidate_override",
        "gp_verifier_action",
        "gp_verifier_reason",
        "verified_candidate_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for observation in observations:
            row = dict(observation)
            row["x_unit"] = _format_vector(row.get("x_unit"))
            row["control_names"] = _format_vector(row.get("control_names"), numeric=False)
            row["control_values"] = _format_vector(row.get("control_values"))
            row["pg_mw"] = _format_vector(row.get("pg_mw"))
            row["vg_pu"] = _format_vector(row.get("vg_pu"))
            row["x_raw"] = _format_vector(row.get("x_raw"))
            row["hypothesis_region_center"] = _format_vector(row.get("hypothesis_region_center"))
            row["hypothesis_sensitive_dims"] = _format_vector(row.get("hypothesis_sensitive_dims"))
            for key in ("strategy_trust", "strategy_success_rates", "hypothesis_status_counts"):
                row[key] = _format_mapping(row.get(key))
            writer.writerow({name: row.get(name) for name in fieldnames})


def _write_summary_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "benchmark_name", "method", "seed", "final_best", "final_regret", "num_evaluations",
        "log10_final_regret", "relative_regret_auc", "initial_design_regret",
        "target_success_rate", "target_auc", "total_run_time", "total_optimizer_time",
        "total_objective_evaluation_time", "llm_calls", "llm_prompt_tokens",
        "llm_completion_tokens", "llm_reasoning_tokens", "llm_total_tokens",
        "llm_latency_seconds",
        "primary_score", "feasible_improvement_found", "evals_to_first_feasible_improvement",
        "best_feasible_cost", "best_feasible_gap", "feasibility_rate", "evals_to_first_feasible",
        "feasible_evaluations", "min_total_violation", "min_max_normalized_violation",
        "near_feasible_rate_10x", "pf_failure_rate", "total_evaluation_time",
        "anytime_feasible_gap_auc",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _write_aggregate_summary_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "benchmark_name",
        "method",
        "runs",
        "primary_score_median",
        "primary_score_q1",
        "primary_score_q3",
        "feasible_improvement_rate",
        "median_feasibility_rate",
        "median_near_feasible_rate_10x",
        "median_min_max_normalized_violation",
        "final_regret_median",
        "final_regret_q1",
        "final_regret_q3",
        "log10_final_regret_median",
        "relative_regret_auc_median",
        "target_auc_median",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})



def _evaluation_options(metadata: Mapping[str, object]) -> Mapping[str, object]:
    raw = metadata.get("evaluation", {}) if isinstance(metadata, Mapping) else {}
    return raw if isinstance(raw, Mapping) else {}


def _summarise_opf_observations(observations: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not observations:
        return {}
    feasible = [observation for observation in observations if bool(observation.get("feasible", False))]
    costs = [
        value for value in (_finite_float(observation.get("generation_cost")) for observation in feasible)
        if value is not None
    ]
    gaps = [
        value for value in (_finite_float(observation.get("normalised_cost_gap")) for observation in feasible)
        if value is not None
    ]
    violations = [
        value for value in (_finite_float(observation.get("total_violation")) for observation in observations)
        if value is not None
    ]
    max_violations = [
        value
        for value in (
            _finite_float(observation.get("max_normalized_violation"))
            for observation in observations
        )
        if value is not None
    ]
    times = [
        value for value in (_finite_float(observation.get("evaluation_time")) for observation in observations)
        if value is not None
    ]
    first_feasible = next(
        (index for index, observation in enumerate(observations, start=1) if bool(observation.get("feasible", False))),
        None,
    )
    initial_feasible_gap = next(
        (
            gap
            for observation in observations
            if bool(observation.get("feasible", False))
            for gap in [_finite_float(observation.get("normalised_cost_gap"))]
            if gap is not None
        ),
        None,
    )
    improvement_tolerance = 1.0e-9
    first_feasible_improvement = next(
        (
            index
            for index, observation in enumerate(observations, start=1)
            if bool(observation.get("feasible", False))
            and initial_feasible_gap is not None
            and _finite_float(observation.get("normalised_cost_gap")) is not None
            and float(observation["normalised_cost_gap"])
                < initial_feasible_gap - improvement_tolerance
        ),
        None,
    )
    feasibility_tolerance = next(
        (
            value
            for value in (
                _finite_float(observation.get("feasibility_tolerance"))
                for observation in observations
            )
            if value is not None and value > 0.0
        ),
        1.0e-5,
    )
    near_feasible = sum(
        value <= 10.0 * feasibility_tolerance for value in max_violations
    )
    best_gap = None
    gap_curve: list[float] = []
    for observation in observations:
        gap = _finite_float(observation.get("normalised_cost_gap")) if observation.get("feasible") else None
        if gap is not None:
            best_gap = gap if best_gap is None else min(best_gap, gap)
        if best_gap is not None:
            gap_curve.append(max(0.0, best_gap))
    return {
        "primary_score": min(gaps) if gaps else None,
        "feasible_improvement_found": first_feasible_improvement is not None,
        "evals_to_first_feasible_improvement": first_feasible_improvement,
        "best_feasible_cost": min(costs) if costs else None,
        "best_feasible_gap": min(gaps) if gaps else None,
        "feasibility_rate": len(feasible) / len(observations),
        "evals_to_first_feasible": first_feasible,
        "feasible_evaluations": len(feasible),
        "min_total_violation": min(violations) if violations else None,
        "min_max_normalized_violation": min(max_violations) if max_violations else None,
        "near_feasible_rate_10x": near_feasible / len(observations),
        "pf_failure_rate": sum(not bool(item.get("pf_converged", False)) for item in observations) / len(observations),
        "total_evaluation_time": sum(times),
        "anytime_feasible_gap_auc": sum(gap_curve) / len(gap_curve) if gap_curve else None,
        "regret_label": "legacy_penalised_score_regret",
    }


def _summarise_llm_observations(observations: Sequence[Mapping[str, object]]) -> dict[str, object]:
    llm_rows = [row for row in observations if _finite_float(row.get("llm_calls")) is not None]
    if not llm_rows:
        return {
            "llm_calls": 0,
            "llm_prompt_tokens": 0,
            "llm_completion_tokens": 0,
            "llm_reasoning_tokens": 0,
            "llm_total_tokens": 0,
            "llm_latency_seconds": 0.0,
        }
    return {
        key: sum(_finite_float(row.get(key)) or 0.0 for row in llm_rows)
        for key in (
            "llm_calls",
            "llm_prompt_tokens",
            "llm_completion_tokens",
            "llm_reasoning_tokens",
            "llm_total_tokens",
            "llm_latency_seconds",
        )
    }


def _aggregate_summary_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        key = (str(row.get("benchmark_name", "")), str(row.get("method", "")))
        grouped.setdefault(key, []).append(row)

    result: list[dict[str, object]] = []
    for (benchmark_name, method), group in sorted(grouped.items()):
        primary = [
            value
            for value in (_finite_float(row.get("primary_score")) for row in group)
            if value is not None
        ]
        feasibility = [
            value
            for value in (_finite_float(row.get("feasibility_rate")) for row in group)
            if value is not None
        ]
        near_feasible = [
            value
            for value in (_finite_float(row.get("near_feasible_rate_10x")) for row in group)
            if value is not None
        ]
        min_violations = [
            value
            for value in (
                _finite_float(row.get("min_max_normalized_violation"))
                for row in group
            )
            if value is not None
        ]
        final_regrets = [
            value for value in (_finite_float(row.get("final_regret")) for row in group)
            if value is not None
        ]
        log_regrets = [
            value for value in (_finite_float(row.get("log10_final_regret")) for row in group)
            if value is not None
        ]
        relative_aucs = [
            value for value in (_finite_float(row.get("relative_regret_auc")) for row in group)
            if value is not None
        ]
        target_aucs = [
            value for value in (_finite_float(row.get("target_auc")) for row in group)
            if value is not None
        ]
        result.append(
            {
                "benchmark_name": benchmark_name,
                "method": method,
                "runs": len(group),
                "primary_score_median": _percentile(primary, 0.50),
                "primary_score_q1": _percentile(primary, 0.25),
                "primary_score_q3": _percentile(primary, 0.75),
                "feasible_improvement_rate": sum(
                    bool(row.get("feasible_improvement_found", False)) for row in group
                )
                / len(group),
                "median_feasibility_rate": _percentile(feasibility, 0.50),
                "median_near_feasible_rate_10x": _percentile(near_feasible, 0.50),
                "median_min_max_normalized_violation": _percentile(min_violations, 0.50),
                "final_regret_median": _percentile(final_regrets, 0.50),
                "final_regret_q1": _percentile(final_regrets, 0.25),
                "final_regret_q3": _percentile(final_regrets, 0.75),
                "log10_final_regret_median": _percentile(log_regrets, 0.50),
                "relative_regret_auc_median": _percentile(relative_aucs, 0.50),
                "target_auc_median": _percentile(target_aucs, 0.50),
            }
        )
    return result


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = float(probability) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _extract_observation_metadata(state_metadata: Mapping[str, object]) -> dict[str, object]:
    decision = state_metadata.get("last_reasoning_decision") if isinstance(state_metadata, Mapping) else None
    if not isinstance(decision, Mapping):
        return {}
    control = decision.get("wmbo_control")
    control_map = control if isinstance(control, Mapping) else {}
    verifier = decision.get("gp_verifier")
    verifier_map = verifier if isinstance(verifier, Mapping) else {}
    geometry = decision.get("geometry_features")
    geometry_map = geometry if isinstance(geometry, Mapping) else {}
    regimes = decision.get("regime_posteriors")
    regime_map = regimes if isinstance(regimes, Mapping) else {}
    return {
        "strategy": decision.get("executed_strategy", decision.get("strategy")),
        "proposed_strategy": decision.get("proposed_strategy"),
        "executed_strategy": decision.get("executed_strategy"),
        "override_reason": decision.get("override_reason"),
        "allowed_strategies": decision.get("allowed_strategies"),
        "masked_strategies": decision.get("masked_strategies"),
        "strategy_scores": decision.get("strategy_scores"),
        "score_components": decision.get("score_components"),
        "forced_strategy": decision.get("forced_strategy"),
        "forced_reason": decision.get("forced_reason"),
        "strategy_gate_reasons": decision.get("strategy_gate_reasons"),
        "selected_candidate_evidence": decision.get("selected_candidate_evidence"),
        "landscape_descriptor": decision.get("landscape_descriptor"),
        "strategy_decision_context": decision.get("strategy_decision_context"),
        "candidate_options": decision.get("candidate_options"),
        "budget_phase": decision.get("budget_phase"),
        "remaining_budget": decision.get("remaining_budget"),
        "macro_action_id": decision.get("macro_action_id"),
        "macro_strategy": decision.get("macro_strategy"),
        "macro_step": decision.get("macro_step"),
        "macro_min_steps": decision.get("macro_min_steps"),
        "macro_max_steps": decision.get("macro_max_steps"),
        "macro_continued": decision.get("macro_continued"),
        "previous_macro_settlement": decision.get("previous_macro_settlement"),
        "macro_termination_reason": decision.get("macro_termination_reason"),
        "macro_reward": decision.get("macro_reward"),
        "macro_reward_components": decision.get("macro_reward_components"),
        "operator_state_summary": decision.get("operator_state_summary"),
        "geometry_features": decision.get("geometry_features"),
        "lengthscale_condition": geometry_map.get("lengthscale_condition"),
        "local_condition": geometry_map.get("local_condition"),
        "rotation_score": geometry_map.get("rotation_score"),
        "effective_dimension": geometry_map.get("effective_dimension"),
        "valley_score": geometry_map.get("valley_score"),
        "regime_separable_smooth": regime_map.get("separable_smooth"),
        "regime_rotated_ill_conditioned": regime_map.get("rotated_ill_conditioned"),
        "regime_curved_valley": regime_map.get("curved_valley"),
        "regime_rugged_multimodal": regime_map.get("rugged_multimodal"),
        "regime_weakly_identified": regime_map.get("weakly_identified"),
        "regime_posteriors": decision.get("regime_posteriors"),
        "consecutive_no_improvement": control_map.get("consecutive_no_improvement"),
        "hypothesis_id": decision.get("hypothesis_id"),
        "hypothesis_status": decision.get("hypothesis_status"),
        "hypothesis_region_center": decision.get("hypothesis_region_center"),
        "hypothesis_region_radius": decision.get("hypothesis_region_radius"),
        "hypothesis_sensitive_dims": decision.get("hypothesis_sensitive_dims"),
        "hypothesis_confidence": decision.get("hypothesis_confidence"),
        "hypothesis_posterior_probability": decision.get("hypothesis_posterior_probability"),
        "hypothesis_relevant_evidence_count": decision.get("hypothesis_relevant_evidence_count"),
        "hypothesis_supporting_evidence": decision.get("hypothesis_supporting_evidence"),
        "hypothesis_contradicting_evidence": decision.get("hypothesis_contradicting_evidence"),
        "falsification_rule": decision.get("falsification_rule"),
        "hypothesis_status_counts": decision.get("hypothesis_status_counts"),
        "strategy_trust": decision.get("strategy_trust"),
        "strategy_success_rates": decision.get("strategy_success_rates"),
        "agent_type": decision.get("agent_type"),
        "llm_error": decision.get("llm_error"),
        "llm_model": decision.get("llm_model"),
        "llm_calls": decision.get("llm_calls"),
        "llm_prompt_tokens": decision.get("llm_prompt_tokens"),
        "llm_completion_tokens": decision.get("llm_completion_tokens"),
        "llm_reasoning_tokens": decision.get("llm_reasoning_tokens"),
        "llm_total_tokens": decision.get("llm_total_tokens"),
        "llm_latency_seconds": decision.get("llm_latency_seconds"),
        "llm_attempts": decision.get("llm_attempts"),
        "requested_candidate_id": decision.get("requested_candidate_id"),
        "selected_candidate_id": decision.get("selected_candidate_id"),
        "evidence_role": decision.get("evidence_role"),
        "target_hypothesis_id": decision.get("target_hypothesis_id"),
        "joint_score": decision.get("joint_score"),
        "expected_improvement": decision.get("expected_improvement"),
        "information_gain": decision.get("information_gain"),
        "predicted_feasibility_probability": decision.get("predicted_feasibility_probability"),
        "predicted_constraint_log_ratio": decision.get("predicted_constraint_log_ratio"),
        "constraint_std": decision.get("constraint_std"),
        "candidate_override": decision.get("candidate_override"),
        "gp_verifier_action": verifier_map.get("action"),
        "gp_verifier_reason": verifier_map.get("reason"),
        "verified_candidate_id": verifier_map.get("verified_candidate_id"),
    }

def _format_vector(value: object, *, numeric: bool = True) -> str:
    if value is None:
        return ""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if numeric:
            return "[" + ", ".join(f"{float(item):.8g}" for item in value) + "]"
        return "[" + ", ".join(str(item) for item in value) + "]"
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
    if "primary_best" in observation:
        parts.append(f"primary={_format_number(observation.get('primary_best'))}")
    agent = observation.get("agent_type")
    strategy = observation.get("executed_strategy") or observation.get("strategy")
    phase = observation.get("budget_phase")
    if agent:
        parts.append(f"agent={agent}")
    if strategy:
        parts.append(f"strategy={strategy}")
    if phase:
        parts.append(f"phase={phase}")
    if observation.get("pf_converged") is not None:
        parts.append(f"pf={observation.get('pf_converged')}")
        parts.append(f"feasible={observation.get('feasible')}")
        parts.append(f"violation={_format_number(observation.get('total_violation'))}")
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
    "describe_benchmark_suite",
    "save_run_result",
    "save_suite_summary",
]
