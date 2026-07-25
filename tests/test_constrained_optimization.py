from __future__ import annotations

import pytest

from wmbo.acquisition import AcquisitionInput, score_candidates
from wmbo.benchmarks import EvaluationResult
from wmbo.optimizers import (
    Observation,
    _result_improves_constraint_progress,
    best_observation,
)
from wmbo.runner import _summarise_opf_observations


def _observation(
    *,
    y: float,
    gap: float,
    ratio: float,
    feasible: bool,
) -> Observation:
    return Observation(
        x=[0.5],
        y=y,
        metadata={
            "normalised_cost_gap": gap,
            "constraint_ratio": ratio,
            "max_normalized_violation": ratio * 1.0e-5,
            "feasibility_tolerance": 1.0e-5,
            "feasible": feasible,
            "pf_converged": True,
        },
    )


def test_feasibility_first_ranking_rejects_lower_cost_infeasible_point() -> None:
    feasible = _observation(y=0.40, gap=0.40, ratio=0.5, feasible=True)
    infeasible = _observation(y=0.01, gap=0.01, ratio=1.01, feasible=False)

    assert best_observation([infeasible, feasible], constrained=True) is feasible
    assert best_observation([infeasible, feasible], constrained=False) is infeasible


def test_infeasible_ranking_uses_violation_before_cost_gap() -> None:
    lower_violation = _observation(y=100.0, gap=0.9, ratio=2.0, feasible=False)
    lower_cost = _observation(y=1.0, gap=0.01, ratio=3.0, feasible=False)

    assert best_observation([lower_cost, lower_violation], constrained=True) is lower_violation


def test_constrained_acquisition_multiplies_objective_utility_by_pof() -> None:
    scores = score_candidates(
        AcquisitionInput(
            candidates=[[0.1], [0.9]],
            observed_x=[[0.5]],
            observed_y=[0.5],
            surrogate_mean=[0.2, 0.2],
            surrogate_std=[0.1, 0.1],
            strategy="expected_improvement",
            feasibility_probability=[0.9, 0.1],
            constraint_std=[0.2, 0.2],
            has_feasible_observation=True,
        )
    )

    assert scores[0] > scores[1]


def test_constraint_restoration_prioritises_probability_before_first_feasible() -> None:
    scores = score_candidates(
        AcquisitionInput(
            candidates=[[0.1], [0.9]],
            observed_x=[[0.5]],
            observed_y=[0.5],
            surrogate_mean=[0.1, 0.9],
            surrogate_std=[0.1, 0.1],
            strategy="expected_improvement",
            feasibility_probability=[0.2, 0.8],
            constraint_std=[0.1, 0.1],
            has_feasible_observation=False,
        )
    )

    assert scores[1] > scores[0]


def test_wmbo_progress_does_not_reward_infeasible_low_objective() -> None:
    history = [_observation(y=0.40, gap=0.40, ratio=0.5, feasible=True)]
    result = EvaluationResult(
        benchmark_name="opf",
        x_unit=[0.2],
        x_raw=[0.2],
        y=0.01,
        metadata={
            "normalised_cost_gap": 0.01,
            "constraint_ratio": 1.01,
            "feasible": False,
            "pf_converged": True,
        },
    )

    improved, _current, best = _result_improves_constraint_progress(
        history,
        result,
        constrained=True,
    )

    assert improved is False
    assert best == pytest.approx(0.40)


def test_opf_summary_uses_strict_feasible_primary_score_and_near_feasible_rate() -> None:
    observations = [
        {
            "normalised_cost_gap": 0.40,
            "generation_cost": 140.0,
            "feasible": True,
            "pf_converged": True,
            "total_violation": 0.0,
            "max_normalized_violation": 0.5e-5,
            "feasibility_tolerance": 1.0e-5,
            "evaluation_time": 0.01,
        },
        {
            "normalised_cost_gap": 0.01,
            "generation_cost": 101.0,
            "feasible": False,
            "pf_converged": True,
            "total_violation": 1.21e-10,
            "max_normalized_violation": 1.1e-5,
            "feasibility_tolerance": 1.0e-5,
            "evaluation_time": 0.01,
        },
        {
            "normalised_cost_gap": 0.30,
            "generation_cost": 130.0,
            "feasible": True,
            "pf_converged": True,
            "total_violation": 0.0,
            "max_normalized_violation": 0.1e-5,
            "feasibility_tolerance": 1.0e-5,
            "evaluation_time": 0.01,
        },
    ]

    summary = _summarise_opf_observations(observations)

    assert summary["primary_score"] == pytest.approx(0.30)
    assert summary["feasible_improvement_found"] is True
    assert summary["evals_to_first_feasible_improvement"] == 3
    assert summary["feasible_evaluations"] == 2
    assert summary["near_feasible_rate_10x"] == pytest.approx(1.0)
