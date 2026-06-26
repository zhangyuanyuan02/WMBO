"""Control-flow data structures for benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass(frozen=True)
class OptimizerConfig:
    """Inputs controlling a single optimiser instance.

    Inputs:
        method: Optimisation method name.
        budget: Maximum number of objective evaluations.
        initial_samples: Number of initial design points.
        candidate_pool_size: Number of candidates considered per step.
        seed: Random seed.
        options: Method-specific options.

    Output:
        Passed to optimiser construction.
    """

    method: str
    budget: int
    initial_samples: int
    candidate_pool_size: int
    seed: int
    options: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RunConfig:
    """Inputs controlling a benchmark suite run.

    Inputs:
        benchmarks: Benchmark names to run.
        methods: Optimiser names to compare.
        seeds: Random seeds for repeated runs.
        output_dir: Directory for future results.
        optimizer: Shared optimiser configuration.

    Output:
        Passed to ``runner.run_benchmark_suite``.
    """

    benchmarks: Sequence[str]
    methods: Sequence[str]
    seeds: Sequence[int]
    output_dir: str
    optimizer: OptimizerConfig


@dataclass(frozen=True)
class RunState:
    """Mutable-style state snapshot for one optimisation run.

    Inputs:
        step: Current optimisation step.
        best_y: Best objective value found so far.
        evaluations_used: Number of evaluations consumed.
        evaluations_remaining: Number of evaluations still available.

    Output:
        Used by stopping and logging utilities.
    """

    step: int
    best_y: float | None
    evaluations_used: int
    evaluations_remaining: int


def build_default_optimizer_config(method: str, budget: int, seed: int) -> OptimizerConfig:
    """Create a minimal optimiser configuration.

    Inputs:
        method: Optimiser name.
        budget: Evaluation budget.
        seed: Random seed.

    Output:
        ``OptimizerConfig`` with project defaults.
    """

    if budget <= 0:
        raise ValueError("budget must be positive.")
    return OptimizerConfig(
        method=method,
        budget=int(budget),
        initial_samples=min(max(2, budget // 5), budget),
        candidate_pool_size=256,
        seed=int(seed),
    )


def should_stop(state: RunState) -> bool:
    """Decide whether an optimisation run should stop.

    Input:
        state: Current run state.

    Output:
        ``True`` when no more evaluations should be performed.
    """

    return state.evaluations_remaining <= 0


def update_run_state(state: RunState, new_y: float) -> RunState:
    """Update run state after one new objective value.

    Inputs:
        state: Previous run state.
        new_y: Newly observed objective value.

    Output:
        Updated ``RunState``.
    """

    best_y = float(new_y) if state.best_y is None else min(float(state.best_y), float(new_y))
    used = state.evaluations_used + 1
    return RunState(
        step=state.step + 1,
        best_y=best_y,
        evaluations_used=used,
        evaluations_remaining=max(0, state.evaluations_remaining - 1),
    )
