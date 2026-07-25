from __future__ import annotations

import math
from typing import Mapping

import pytest

from wmbo.benchmarks import EvaluationRequest, evaluate, get_benchmark, list_benchmarks
from wmbo.config import run_config_from_mapping
from wmbo.runner import BenchmarkRunRequest, run_single_benchmark


TYP = "opf_pglib_case14_typ_pgvg"
API = "opf_pglib_case14_api_pgvg"
CONTROL_NAMES = ["Pg2", "Vg1", "Vg2", "Vg3", "Vg6", "Vg8"]


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
        controls = [float(value) for value in payload["control_values"]]  # type: ignore[index]
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
            "reference_kind": "full_ac_opf_pg_vg",
            "control_names": CONTROL_NAMES,
            "control_values": controls,
            "pg_mw": controls[:1],
            "vg_pu": controls[1:],
        }


def test_pgvg_benchmarks_replace_pg_only_names_without_starting_julia() -> None:
    names = list_benchmarks()
    assert TYP in names
    assert API in names
    assert "opf_pglib_case14_typ_pg" not in names
    assert "opf_pglib_case14_api_pg" not in names
    with pytest.raises(ValueError, match="Unknown benchmark"):
        get_benchmark("opf_pglib_case14_typ_pg")

    typical = get_benchmark(TYP)
    api = get_benchmark(API)

    assert typical.family == "opf"
    assert typical.constrained is True
    assert typical.dim == 6
    assert list(typical.bounds) == [(0.0, 59.0)] + [(0.94, 1.06)] * 5
    assert [variable["name"] for variable in typical.metadata["variables"]] == CONTROL_NAMES
    assert len(typical.recommended_start_unit or []) == 6
    assert all(0.0 <= value <= 1.0 for value in typical.recommended_start_unit or [])
    assert api.dim == 6
    assert list(api.bounds) == [(0.0, 230.0)] + [(0.94, 1.06)] * 5


def test_opf_scalarisation_and_metadata_use_pgvg_backend_result() -> None:
    backend = StaticBackend()
    result = evaluate(
        EvaluationRequest(
            benchmark_name=TYP,
            x_unit=[0.5] * 6,
            seed=7,
            options={"opf": {"_backend": backend, "penalty_weight": 100.0}},
        )
    )

    assert result.x_raw == pytest.approx([29.5, 1.0, 1.0, 1.0, 1.0, 1.0])
    expected_ratio = math.sqrt(0.002) / 1.0e-5
    assert result.y == pytest.approx(0.1 + 100.0 * (expected_ratio - 1.0) ** 2)
    assert result.metadata["normalised_cost_gap"] == pytest.approx(0.1)
    assert result.metadata["constraint_ratio"] == pytest.approx(expected_ratio)
    assert result.metadata["legacy_penalised_score"] == pytest.approx(0.3)
    assert result.metadata["score_kind"] == "normalised_cost_gap_plus_tolerance_scaled_excess_penalty"
    assert result.metadata["generation_cost"] == 110.0
    assert result.metadata["control_names"] == CONTROL_NAMES
    assert backend.requests[0]["control_values"] == pytest.approx(result.x_raw)
    assert "pg_mw" not in backend.requests[0]


def test_nonconverged_power_flow_returns_finite_failure_penalty() -> None:
    backend = StaticBackend(cost=None, converged=False, violation=1.0)
    result = evaluate(
        EvaluationRequest(
            benchmark_name=TYP,
            x_unit=[0.5] * 6,
            options={"opf": {"_backend": backend, "failure_penalty": 12345.0}},
        )
    )

    assert result.y == 12345.0
    assert math.isfinite(result.y)
    assert result.metadata["pf_converged"] is False
    assert result.metadata["control_values"] == pytest.approx(result.x_raw)


def test_violation_exactly_at_tolerance_has_no_excess_penalty() -> None:
    tolerance = 1.0e-5
    backend = StaticBackend(
        cost=110.0,
        violation=tolerance * tolerance,
        feasible=True,
    )
    result = evaluate(
        EvaluationRequest(
            benchmark_name=TYP,
            x_unit=[0.5] * 6,
            options={
                "opf": {
                    "_backend": backend,
                    "penalty_weight": 100.0,
                    "feasibility_tolerance": tolerance,
                }
            },
        )
    )

    assert result.metadata["constraint_ratio"] == pytest.approx(1.0)
    assert result.metadata["constraint_excess"] == pytest.approx(0.0)
    assert result.y == pytest.approx(0.1)


def test_runner_uses_shared_pgvg_start_and_persists_opf_metrics() -> None:
    benchmark = get_benchmark(TYP)
    first_points: list[list[float]] = []
    for method in ("random", "sobol"):
        backend = StaticBackend(cost=100.0, violation=0.0, feasible=True)
        run = run_single_benchmark(
            BenchmarkRunRequest(
                benchmark_name=TYP,
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
        assert run.observations[0]["control_names"] == CONTROL_NAMES
        assert run.observations[0]["pg_mw"] == pytest.approx([59.0])
        assert len(run.observations[0]["vg_pu"]) == 5
        assert run.summary.metadata["feasibility_rate"] == 1.0
        assert run.summary.metadata["evals_to_first_feasible"] == 1
        assert run.summary.metadata["pf_failure_rate"] == 0.0

    assert first_points[0] == pytest.approx(benchmark.recommended_start_unit or [])
    assert first_points[1] == pytest.approx(benchmark.recommended_start_unit or [])


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
