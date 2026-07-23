from __future__ import annotations

import math
from typing import Mapping

import pytest

from wmbo.benchmarks import EvaluationRequest, evaluate, get_benchmark, list_benchmarks
from wmbo.config import run_config_from_mapping
from wmbo.runner import BenchmarkRunRequest, run_single_benchmark


class StaticBackend:
    def __init__(
        self,
        *,
        cost: float | None = 110.0,
        reference_cost: float = 100.0,
        violation: float = 0.002,
        converged: bool = True,
        feasible: bool = False,
    ) -> None:
        self.cost = cost
        self.reference_cost = reference_cost
        self.violation = violation
        self.converged = converged
        self.feasible = feasible
        self.requests: list[dict[str, object]] = []

    def request(self, payload: Mapping[str, object], *, timeout: float | None = None) -> Mapping[str, object]:
        self.requests.append(dict(payload))
        return {
            "benchmark": payload["benchmark"],
            "scenario_id": "fake_case14",
            "pf_converged": self.converged,
            "feasible": self.feasible,
            "generation_cost": self.cost,
            "reference_cost": self.reference_cost,
            "total_violation": self.violation,
            "max_normalized_violation": math.sqrt(self.violation),
            "max_voltage_violation": 0.0,
            "max_thermal_violation": 0.0,
            "max_angle_violation": 0.0,
            "generator_violation": self.violation,
            "power_balance_residual": 0.0,
            "evaluation_time": 0.01,
            "solver_time": 0.005,
            "termination_status": "LOCALLY_SOLVED" if self.converged else "FAILED",
        }


def test_opf_benchmarks_are_registered_without_starting_julia() -> None:
    assert "opf_pglib_case14_typ_pg" in list_benchmarks()
    assert "opf_pglib_case14_api_pg" in list_benchmarks()

    typical = get_benchmark("opf_pglib_case14_typ_pg")
    api = get_benchmark("opf_pglib_case14_api_pg")

    assert typical.family == "opf"
    assert typical.constrained is True
    assert typical.dim == 1
    assert list(typical.bounds) == [(0.0, 59.0)]
    assert list(typical.recommended_start_unit or []) == [0.5]
    assert api.dim == 1
    assert list(api.bounds) == [(0.0, 230.0)]


def test_opf_scalarisation_and_metadata_use_backend_result() -> None:
    backend = StaticBackend()
    result = evaluate(
        EvaluationRequest(
            benchmark_name="opf_pglib_case14_typ_pg",
            x_unit=[0.5],
            seed=7,
            options={"opf": {"_backend": backend, "penalty_weight": 100.0}},
        )
    )

    assert result.x_raw == [29.5]
    assert result.y == pytest.approx(0.3)
    assert result.metadata["normalised_cost_gap"] == pytest.approx(0.1)
    assert result.metadata["generation_cost"] == 110.0
    assert backend.requests[0]["pg_mw"] == [29.5]


def test_nonconverged_power_flow_returns_finite_failure_penalty() -> None:
    backend = StaticBackend(cost=None, converged=False, violation=1.0)
    result = evaluate(
        EvaluationRequest(
            benchmark_name="opf_pglib_case14_typ_pg",
            x_unit=[0.5],
            options={"opf": {"_backend": backend, "failure_penalty": 12345.0}},
        )
    )

    assert result.y == 12345.0
    assert math.isfinite(result.y)
    assert result.metadata["pf_converged"] is False


def test_runner_uses_shared_start_and_persists_opf_metrics() -> None:
    first_points: list[list[float]] = []
    for method in ("random", "sobol"):
        backend = StaticBackend(cost=100.0, violation=0.0, feasible=True)
        run = run_single_benchmark(
            BenchmarkRunRequest(
                benchmark_name="opf_pglib_case14_typ_pg",
                method=method,
                seed=0,
                budget=3,
                metadata={
                    "evaluation": {"opf": {"_backend": backend}},
                    "options": {"logging": {"verbose": False}},
                },
            )
        )
        first_points.append(list(run.observations[0]["x_unit"]))
        assert run.observations[0]["generation_cost"] == 100.0
        assert run.summary.metadata["feasibility_rate"] == 1.0
        assert run.summary.metadata["evals_to_first_feasible"] == 1
        assert run.summary.metadata["pf_failure_rate"] == 0.0

    assert first_points == [[0.5], [0.5]]


def test_evaluation_config_is_kept_separate_from_optimizer_options() -> None:
    config = run_config_from_mapping(
        {
            "evaluation": {
                "opf": {
                    "penalty_weight": 77.0,
                    "backend": {"timeout_seconds": 12},
                }
            }
        }
    )

    assert config.evaluation["opf"]["penalty_weight"] == 77.0
    assert "opf" not in config.optimizer.options


def test_existing_analytic_benchmark_is_unchanged() -> None:
    result = evaluate(EvaluationRequest("branin", [0.5, 0.5], seed=0))

    assert result.benchmark_name == "branin"
    assert result.metadata["family"] == "synthetic"
    assert math.isfinite(result.y)
