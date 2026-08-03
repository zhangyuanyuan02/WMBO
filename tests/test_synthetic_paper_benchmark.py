from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from wmbo.agents import ReasoningDecision
from wmbo.analysis import bootstrap_median_ci, holm_adjust, vargha_delaney_a12
from wmbo.benchmarks import (
    EvaluationRequest,
    evaluate,
    expand_bbob_benchmarks,
    get_benchmark,
)
from wmbo.config import load_run_config, run_config_from_mapping
from wmbo.descriptors import LandscapeDescriptor
from wmbo.llm_api import decide_with_llm
from wmbo.metrics import synthetic_run_metrics
from wmbo.optimizers import create_initial_state, make_optimizer
from wmbo.runner import (
    BenchmarkRunRequest,
    _shared_initial_points,
    describe_benchmark_suite,
    run_benchmark_suite,
    run_single_benchmark,
)


def test_bbob_matrix_expands_to_240_unique_names() -> None:
    names = expand_bbob_benchmarks(range(1, 25), [5, 10], range(1, 6))
    assert len(names) == 240
    assert len(set(names)) == 240
    assert names[0] == "bbob_f01_d05_i01"
    assert names[-1] == "bbob_f24_d10_i05"


def test_bbob_optimum_roundtrip() -> None:
    pytest.importorskip("ioh")
    import ioh

    spec = get_benchmark("bbob_f01_d05_i01")
    problem = ioh.get_problem(1, instance=1, dimension=5, problem_class=ioh.ProblemClass.BBOB)
    optimum_raw = [float(value) for value in problem.optimum.x]
    x_unit = [
        (value - lo) / (hi - lo)
        for value, (lo, hi) in zip(optimum_raw, spec.bounds)
    ]
    result = evaluate(EvaluationRequest(spec.name, x_unit, seed=0))
    assert result.y == pytest.approx(spec.optimum_value, abs=1.0e-8)
    assert result.metadata["bbob_function"] == 1
    assert result.metadata["bbob_instance"] == 1


def test_paper_config_dry_run_counts() -> None:
    pytest.importorskip("ioh")
    config = load_run_config("configs/synthetic_bbob_paper.yaml")
    estimate = describe_benchmark_suite(config)
    assert estimate == {
        "benchmarks": 240,
        "methods": 10,
        "seeds": 1,
        "runs": 2400,
        "evaluations": 360000,
        "estimated_llm_calls": 0,
    }


def test_shared_sobol_initial_design_is_reproducible_and_method_independent() -> None:
    first = _shared_initial_points(
        benchmark_name="branin",
        dim=2,
        count=4,
        seed=7,
        mode="sobol",
    )
    second = _shared_initial_points(
        benchmark_name="branin",
        dim=2,
        count=4,
        seed=7,
        mode="sobol",
    )
    assert first == second
    assert len({tuple(row) for row in first}) == 4
    assert all(0.0 <= value <= 1.0 for row in first for value in row)

    observed = []
    for method in ("random", "sobol", "bo_ei", "wmbo_rule"):
        run = run_single_benchmark(
            BenchmarkRunRequest(
                benchmark_name="branin",
                method=method,
                seed=7,
                budget=4,
                metadata={
                    "initial_samples": 4,
                    "options": {"shared_initial_design": "sobol", "logging": {"verbose": False}},
                },
            )
        )
        observed.append([row["x_unit"] for row in run.observations])
    assert all(points == observed[0] for points in observed[1:])


@pytest.mark.parametrize(
    ("method", "module"),
    [("tpe", "optuna"), ("cma_es", "cma"), ("hebo", "hebo")],
)
def test_external_baselines_accept_warm_start_and_stay_in_bounds(
    method: str,
    module: str,
    tmp_path: Path,
) -> None:
    pytest.importorskip(module)
    config = run_config_from_mapping(
        {
            "experiment": {
                "methods": [method],
                "budget": 7,
                "initial_samples": 4,
                "shared_initial_design": "sobol",
                "seeds": [3],
                "output_dir": str(tmp_path / method),
            },
            "benchmark_suite": {"benchmarks": ["branin"]},
            "logging": {"verbose": False},
        }
    )
    run = run_benchmark_suite(config)
    assert len(run) == 1
    assert len(run[0].observations) == 7
    assert all(
        0.0 <= float(value) <= 1.0
        for row in run[0].observations
        for value in row["x_unit"]
    )


def test_synthetic_metrics_have_exact_auc_and_targets() -> None:
    metrics = synthetic_run_metrics(
        [10.0, 5.0, 1.0, 0.01, 0.0001],
        optimum_value=0.0,
        initial_samples=2,
    )
    assert metrics["initial_design_regret"] == 5.0
    assert metrics["relative_regret_auc"] == pytest.approx((1.0 + 0.2 + 0.002 + 0.00002) / 4)
    assert metrics["evals_to_target_1em2"] == 4
    assert metrics["evals_to_target_1em4"] == 5
    assert metrics["success_target_1em8"] is False


def test_statistical_helpers_are_deterministic() -> None:
    assert bootstrap_median_ci([1, 2, 3], samples=500, seed=4) == bootstrap_median_ci(
        [1, 2, 3], samples=500, seed=4
    )
    assert holm_adjust([0.01, 0.04, 0.5]) == pytest.approx([0.03, 0.08, 0.5])
    assert vargha_delaney_a12([1, 2, 3], [2, 2, 4]) == pytest.approx((2 + 0.5) / 3)


class _FakeCompletions:
    def create(self, **_kwargs):
        return {
            "model": "fake-reasoner",
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "world_model": {
                                    "smoothness": "smooth",
                                    "modality": "mostly_unimodal",
                                    "curvature": "low",
                                    "anisotropy": "low",
                                },
                                "strategy": "exploit_ei",
                                "hypothesis": "local basin",
                                "confidence": 0.8,
                                "falsification_rule": "no improvement",
                                "rationale": "test",
                                "selected_candidate_id": "c1",
                            }
                        )
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "completion_tokens_details": {"reasoning_tokens": 5},
            },
            "_wmbo_transport": {"latency_seconds": 0.25, "attempts": 2},
        }


def test_llm_decision_carries_transport_and_token_telemetry() -> None:
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=_FakeCompletions()),
        config=SimpleNamespace(default_model="fallback"),
    )
    decision = decide_with_llm(
        LandscapeDescriptor(
            dim=2,
            num_observations=3,
            best_y=1.0,
            y_range=2.0,
            labels={},
        ),
        client=client,
    )
    assert isinstance(decision, ReasoningDecision)
    assert decision.metadata["llm_model"] == "fake-reasoner"
    assert decision.metadata["llm_prompt_tokens"] == 100
    assert decision.metadata["llm_reasoning_tokens"] == 5
    assert decision.metadata["llm_latency_seconds"] == pytest.approx(0.25)
    assert decision.metadata["llm_attempts"] == 2


def test_resume_skips_complete_matching_run(tmp_path: Path) -> None:
    output = tmp_path / "resume"
    config = run_config_from_mapping(
        {
            "experiment": {
                "methods": ["random"],
                "seeds": [0],
                "budget": 3,
                "output_dir": str(output),
            },
            "benchmark_suite": {"benchmarks": ["branin"]},
            "logging": {"verbose": False},
        }
    )
    first = run_benchmark_suite(config)
    second = run_benchmark_suite(config, resume=True)
    assert len(first) == len(second) == 1
    assert first[0].summary.final_best == second[0].summary.final_best
    assert (output / "branin" / "random" / "seed_0" / "complete.json").exists()


def test_parallel_workers_complete_runs_and_preserve_shared_design(tmp_path: Path) -> None:
    output = tmp_path / "parallel"
    config = run_config_from_mapping(
        {
            "experiment": {
                "methods": ["random", "sobol"],
                "seeds": [0],
                "budget": 4,
                "initial_samples": 4,
                "shared_initial_design": "sobol",
                "output_dir": str(output),
            },
            "benchmark_suite": {"benchmarks": ["branin"]},
            "execution": {"workers": 2},
            "logging": {"verbose": False},
        }
    )
    results = run_benchmark_suite(config)
    assert [result.request.method for result in results] == ["random", "sobol"]
    assert [row["x_unit"] for row in results[0].observations] == [
        row["x_unit"] for row in results[1].observations
    ]
    assert all((output / "branin" / method / "seed_0" / "complete.json").exists() for method in ("random", "sobol"))
