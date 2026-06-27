"""Configuration loading utilities for reproducible benchmark runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .control import OptimizerConfig, RunConfig


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment configuration.

    Input:
        path: Path to a YAML configuration file.

    Output:
        Parsed configuration dictionary.
    """

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError("Configuration file must contain a YAML mapping at the top level.")
    return data


def run_config_from_mapping(data: Mapping[str, Any]) -> RunConfig:
    """Convert a configuration mapping into a ``RunConfig``.

    Input:
        data: Parsed YAML mapping with optional ``experiment``, ``optimizer``,
            and ``benchmark_suite`` sections.

    Output:
        ``RunConfig`` for the benchmark runner.
    """

    experiment = _mapping(data.get("experiment", {}), section="experiment")
    optimizer_section = _mapping(data.get("optimizer", {}), section="optimizer")
    suite = _mapping(data.get("benchmark_suite", {}), section="benchmark_suite")

    benchmarks = _as_str_list(suite.get("benchmarks", data.get("benchmarks", ["branin"])))
    methods = _as_str_list(experiment.get("methods", data.get("methods", ["random"])))
    seeds = _as_int_list(experiment.get("seeds", data.get("seeds", [0])))
    budget = _positive_int(experiment.get("budget", data.get("budget", 20)), name="budget")

    initial_samples = _positive_int(
        experiment.get("initial_samples", experiment.get("n_initial", optimizer_section.get("initial_samples", 5))),
        name="initial_samples",
    )
    candidate_pool_size = _positive_int(
        experiment.get(
            "candidate_pool_size",
            experiment.get("n_candidates", optimizer_section.get("candidate_pool_size", 256)),
        ),
        name="candidate_pool_size",
    )
    output_dir = str(experiment.get("output_dir", data.get("output_dir", "results")))
    options = dict(_mapping(optimizer_section.get("options", {}), section="optimizer.options"))
    control_options = optimizer_section.get("wmbo_control", data.get("wmbo_control", {}))
    if control_options is not None:
        control_mapping = _mapping(control_options, section="optimizer.wmbo_control")
        if control_mapping:
            options["wmbo_control"] = dict(control_mapping)

    optimizer = OptimizerConfig(
        method=methods[0],
        budget=budget,
        initial_samples=max(1, min(initial_samples, budget)),
        candidate_pool_size=max(1, candidate_pool_size),
        seed=seeds[0],
        options=options,
    )
    return RunConfig(
        benchmarks=benchmarks,
        methods=methods,
        seeds=seeds,
        output_dir=output_dir,
        optimizer=optimizer,
    )


def load_run_config(path: str | Path) -> RunConfig:
    """Load a YAML file directly into a ``RunConfig``.

    Input:
        path: Path to YAML config.

    Output:
        ``RunConfig`` instance.
    """

    return run_config_from_mapping(load_config(path))


def merge_run_config(
    base: RunConfig,
    *,
    benchmarks: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    budget: int | None = None,
    initial_samples: int | None = None,
    candidate_pool_size: int | None = None,
    output_dir: str | None = None,
) -> RunConfig:
    """Apply explicit command-line overrides to a ``RunConfig``.

    Inputs:
        base: Existing run configuration.
        benchmarks: Optional benchmark override.
        methods: Optional method override.
        seeds: Optional seed override.
        budget: Optional budget override.
        initial_samples: Optional initial design size override.
        candidate_pool_size: Optional candidate pool size override.
        output_dir: Optional result directory override.

    Output:
        Updated ``RunConfig``.
    """

    new_benchmarks = list(benchmarks) if benchmarks is not None else list(base.benchmarks)
    new_methods = list(methods) if methods is not None else list(base.methods)
    new_seeds = [int(seed) for seed in seeds] if seeds is not None else [int(seed) for seed in base.seeds]
    new_budget = _positive_int(budget if budget is not None else base.optimizer.budget, name="budget")
    new_initial = _positive_int(
        initial_samples if initial_samples is not None else base.optimizer.initial_samples,
        name="initial_samples",
    )
    new_candidate_pool = _positive_int(
        candidate_pool_size if candidate_pool_size is not None else base.optimizer.candidate_pool_size,
        name="candidate_pool_size",
    )
    optimizer = OptimizerConfig(
        method=new_methods[0],
        budget=new_budget,
        initial_samples=max(1, min(new_initial, new_budget)),
        candidate_pool_size=max(1, new_candidate_pool),
        seed=new_seeds[0],
        options=base.optimizer.options,
    )
    return RunConfig(
        benchmarks=new_benchmarks,
        methods=new_methods,
        seeds=new_seeds,
        output_dir=output_dir if output_dir is not None else base.output_dir,
        optimizer=optimizer,
    )


def default_run_config() -> RunConfig:
    """Create the default CLI run configuration.

    Input:
        None.

    Output:
        Default ``RunConfig`` matching the no-config command-line behaviour.
    """

    return run_config_from_mapping({})


def _mapping(value: Any, *, section: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{section} must be a mapping.")
    return value


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = [str(item).strip() for item in value if str(item).strip()]
    else:
        raise ValueError("expected a string or list of strings")
    if not items:
        raise ValueError("at least one item is required")
    return items


def _as_int_list(value: Any) -> list[int]:
    if isinstance(value, str):
        parts = [item.strip() for item in value.split(",") if item.strip()]
        items = [int(item) for item in parts]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = [int(item) for item in value]
    else:
        raise ValueError("expected a string or list of integers")
    if not items:
        raise ValueError("at least one seed is required")
    return items


def _positive_int(value: Any, *, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive.")
    return parsed


__all__ = [
    "default_run_config",
    "load_config",
    "load_run_config",
    "merge_run_config",
    "run_config_from_mapping",
]
