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
        ("opf_pglib_case14_typ_pg", 2178.0804108368666),
        ("opf_pglib_case14_api_pg", 5999.363080435895),
    ],
)
def test_reference_candidate_is_feasible_and_deterministic(
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

    assert first.metadata["pf_converged"] is True
    assert first.metadata["feasible"] is True
    assert first.metadata["reference_cost"] == pytest.approx(expected_reference, rel=1.0e-8)
    assert first.metadata["reference_kind"] == "restricted_pg_with_frozen_ac_opf_voltage_setpoints"
    assert first.y == pytest.approx(second.y, abs=1.0e-10)
    assert first.metadata["generation_cost"] == pytest.approx(second.metadata["generation_cost"], abs=1.0e-8)
    assert first.metadata["total_violation"] < 1.0e-10


def teardown_module() -> None:
    close_opf_backends()
