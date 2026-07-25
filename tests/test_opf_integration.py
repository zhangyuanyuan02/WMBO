from __future__ import annotations

import os
import shutil

import pytest

from wmbo.benchmarks import EvaluationRequest, evaluate, get_benchmark
from wmbo.opf_backend import close_opf_backends


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("WMBO_RUN_OPF_INTEGRATION") != "1" or shutil.which("julia") is None,
        reason="set WMBO_RUN_OPF_INTEGRATION=1 after running julia/setup.jl",
    ),
]


@pytest.mark.parametrize(
    ("benchmark_name", "expected_reference"),
    [
        ("opf_pglib_case14_typ_pgvg", 2178.080410819643),
        ("opf_pglib_case14_api_pgvg", 5999.363080387212),
    ],
)
def test_suboptimal_start_is_feasible_deterministic_and_separated_from_reference(
    benchmark_name: str,
    expected_reference: float,
) -> None:
    benchmark = get_benchmark(benchmark_name)
    request = EvaluationRequest(
        benchmark_name,
        benchmark.recommended_start_unit or [],
        seed=0,
        options={"opf": {"backend": {"startup_timeout_seconds": 180}}},
    )
    first = evaluate(request)
    second = evaluate(request)

    assert benchmark.dim == 6
    assert first.metadata["control_names"] == ["Pg2", "Vg1", "Vg2", "Vg3", "Vg6", "Vg8"]
    assert first.metadata["pf_converged"] is True
    assert first.metadata["feasible"] is True
    assert first.metadata["reference_cost"] == pytest.approx(expected_reference, rel=1.0e-8)
    assert first.metadata["reference_kind"] == "full_ac_opf_pg_vg"
    assert first.metadata["normalised_cost_gap"] >= 0.01
    assert first.y == pytest.approx(second.y, abs=1.0e-10)
    assert first.metadata["generation_cost"] == pytest.approx(second.metadata["generation_cost"], abs=1.0e-8)
    assert first.metadata["total_violation"] < 1.0e-10


def test_vg_control_is_applied_and_evaluation_order_does_not_leak_state() -> None:
    benchmark = get_benchmark("opf_pglib_case14_typ_pgvg")
    start = list(benchmark.recommended_start_unit or [])
    changed = list(start)
    changed[2] = max(0.0, changed[2] - 0.05)
    options = {"opf": {"backend": {"startup_timeout_seconds": 180}}}

    first = evaluate(EvaluationRequest(benchmark.name, start, options=options))
    perturbed = evaluate(EvaluationRequest(benchmark.name, changed, options=options))
    repeated = evaluate(EvaluationRequest(benchmark.name, start, options=options))

    assert perturbed.metadata["vg_pu"][1] == pytest.approx(perturbed.x_raw[2])
    assert perturbed.metadata["vg_pu"][1] != pytest.approx(first.metadata["vg_pu"][1])
    assert perturbed.metadata["generation_cost"] != pytest.approx(first.metadata["generation_cost"], abs=1.0e-8)
    assert repeated.y == pytest.approx(first.y, abs=1.0e-10)
    assert repeated.metadata["generation_cost"] == pytest.approx(first.metadata["generation_cost"], abs=1.0e-8)


def teardown_module() -> None:
    close_opf_backends()