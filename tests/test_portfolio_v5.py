from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wmbo.agents import AgentState, WorldModelAgent
from wmbo.control import PortfolioWMBOState, WMBOControlConfig
from wmbo.llm_api import LLMAPIError, parse_reasoning_decision
from wmbo.optimizers import _build_portfolio_candidate_options
from wmbo.portfolio import (
    MACRO_DURATIONS,
    MacroActionState,
    OnePlusOneCMAState,
    PORTFOLIO_STRATEGIES,
    TurboState,
    geometry_features,
    objective_scale,
    portfolio_shape,
    regime_posteriors,
)
from wmbo.runner import BenchmarkRunRequest, run_single_benchmark


def _posteriors(**updates: float) -> dict[str, float]:
    values = {
        "separable_smooth": 0.2,
        "rotated_ill_conditioned": 0.2,
        "curved_valley": 0.2,
        "rugged_multimodal": 0.2,
        "weakly_identified": 0.2,
    }
    values.update(updates)
    total = sum(values.values())
    return {key: value / total for key, value in values.items()}


def _agent_state(
    *,
    allowed: list[str] | None = None,
    posteriors: dict[str, float] | None = None,
) -> AgentState:
    options = []
    for index, strategy in enumerate(PORTFOLIO_STRATEGIES):
        options.append(
            {
                "candidate_id": f"{strategy}_1",
                "strategy": strategy,
                "selection_score": 1.0 + index,
                "expected_improvement": 0.1 + index,
                "information_gain": 0.2 + index,
                "distance_to_nearest_observation": 0.3 + index,
                "surrogate_std": 0.4 + index,
                "probability_feasible": 1.0,
                "confirmation_value": 0.5,
            }
        )
    return AgentState(
        observed_x=[[0.1, 0.2], [0.8, 0.7], [0.4, 0.5]],
        observed_y=[3.0, 2.0, 1.0],
        descriptor={
            "dim": 2,
            "num_observations": 3,
            "uncertainty": 0.4,
            "coverage": 0.6,
            "stagnation": 0.2,
            "regime_posteriors": posteriors or _posteriors(),
            "calibration": {
                "world_model_entropy": 0.4,
                "posterior_confidence": 0.7,
            },
            "labels": {},
        },
        budget_used=12,
        budget_total=30,
        candidate_options=options,
        decision_context={
            "budget_phase": "middle",
            "allowed_strategies": allowed or list(PORTFOLIO_STRATEGIES),
            "strategy_trust": {name: 0.5 for name in PORTFOLIO_STRATEGIES},
        },
    )


def test_regime_posteriors_are_normalised_and_geometry_sensitive() -> None:
    separable = regime_posteriors(
        num_observations=100,
        dim=2,
        smoothness=0.0,
        modality=0.0,
        curvature=0.1,
        uncertainty=0.1,
        geometry={
            "rotation_score": 0.0,
            "local_condition": 0.0,
            "lengthscale_condition": 0.0,
            "effective_dimension": 1.0,
        },
    )
    rotated = regime_posteriors(
        num_observations=100,
        dim=2,
        smoothness=0.1,
        modality=0.0,
        curvature=0.6,
        uncertainty=0.2,
        geometry={
            "rotation_score": 1.0,
            "local_condition": 1.0,
            "lengthscale_condition": 1.0,
            "effective_dimension": 0.2,
        },
    )
    assert sum(separable.values()) == pytest.approx(1.0)
    assert sum(rotated.values()) == pytest.approx(1.0)
    assert separable["separable_smooth"] > separable["rotated_ill_conditioned"]
    assert rotated["rotated_ill_conditioned"] > separable["rotated_ill_conditioned"]
    assert rotated["curved_valley"] > separable["curved_valley"]


def test_geometry_safely_handles_missing_lengthscales_and_rank_deficiency() -> None:
    x = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
    features = geometry_features(x, [2.0, 1.0], None)
    assert all(np.isfinite(list(features.values())))
    assert all(0.0 <= value <= 1.0 for value in features.values())
    shape = portfolio_shape(x, [2.0, 1.0], None)
    assert shape.shape == (3, 3)
    assert np.all(np.linalg.eigvalsh(shape) > 0.0)
    assert np.linalg.cond(shape) <= 1.0e4 * 1.01


def test_turbo_and_cma_updates_are_deterministic_and_bounded() -> None:
    turbo = TurboState(dim=2)
    for _ in range(3):
        turbo.update(True)
    assert turbo.radius == pytest.approx(0.30)
    for _ in range(4):
        turbo.update(False)
    assert turbo.radius == pytest.approx(0.15)

    cma_a = OnePlusOneCMAState(dim=2, seed=7)
    cma_b = OnePlusOneCMAState(dim=2, seed=7)
    for cma in (cma_a, cma_b):
        cma.ensure([0.5, 0.5], 1.0, np.eye(2))
    assert np.allclose(cma_a.sample(8, 3), cma_b.sample(8, 3))
    cma_a.update([0.55, 0.45], 0.8, True)
    assert cma_a.sigma == pytest.approx(0.24)
    cma_a.update([0.6, 0.4], 0.9, False)
    assert cma_a.sigma == pytest.approx(0.24 * 0.82)
    assert np.linalg.cond(cma_a.covariance) <= 1.0e4 * 1.01


def test_macro_reward_is_positive_affine_scale_invariant_and_once_per_macro() -> None:
    base = MacroActionState("m1", "gp_ei", 1, 10.0, objective_scale([10, 12, 15]), 2, 3)
    shifted = MacroActionState(
        "m2", "gp_ei", 1, 25.0, objective_scale([25, 29, 35]), 2, 3
    )
    for best, info in [(9.0, 0.2), (8.0, 0.4)]:
        base.record(best, True, info)
    for best, info in [(23.0, 0.2), (21.0, 0.4)]:
        shifted.record(best, True, info)
    assert base.reward()[0] == pytest.approx(shifted.reward()[0])
    assert base.should_stop() is None
    base.record(8.0, False, 0.1)
    assert base.should_stop() == "max_steps"


def test_portfolio_controller_masks_cma_until_feasible() -> None:
    control = PortfolioWMBOState(WMBOControlConfig())
    context = control.decision_context(
        phase="early",
        trial_number=1,
        completed_trials=0,
        budget=20,
        uncertainty=1.0,
        has_feasible_observation=False,
    )
    assert "cma_local" not in context.allowed_strategies
    assert "cma_local_requires_feasible_observation" in context.gate_reasons


def test_portfolio_rule_only_selects_legal_strategy_and_candidate() -> None:
    state = _agent_state(
        allowed=["gp_ei", "anisotropic_turbo"],
        posteriors=_posteriors(curved_valley=0.8),
    )
    first = WorldModelAgent({"mode": "portfolio_v5"}).decide(state)
    second = WorldModelAgent({"mode": "portfolio_v5"}).decide(state)
    assert first == second
    assert first.strategy in {"gp_ei", "anisotropic_turbo"}
    assert first.selected_candidate_id.startswith(first.strategy)
    assert first.metadata["strategy_scores"]["global_sobol"] is None


def test_llm_parser_uses_dynamic_v5_strategy_list() -> None:
    payload = json.dumps(
        {
            "world_model": {},
            "strategy": "cma_local",
            "hypothesis": "rotated basin",
            "confidence": 0.7,
            "rationale": "geometry",
        }
    )
    decision = parse_reasoning_decision(payload, allowed_strategies=PORTFOLIO_STRATEGIES)
    assert decision.strategy == "cma_local"
    with pytest.raises(LLMAPIError):
        parse_reasoning_decision(payload)


class _Prediction:
    def __init__(self, n: int) -> None:
        self.mean = np.linspace(1.0, 0.0, n).tolist()
        self.std = np.linspace(0.1, 1.0, n).tolist()
        self.metadata = {}


class _CountingSurrogate:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, candidates: object) -> _Prediction:
        self.calls += 1
        return _Prediction(len(candidates))


class _RejectingConstraint:
    def predict(self, candidates: object) -> _Prediction:
        prediction = _Prediction(len(candidates))
        prediction.mean = np.full(len(candidates), 10.0).tolist()
        prediction.std = np.full(len(candidates), 0.01).tolist()
        return prediction
def test_each_operator_uses_one_candidate_pool_prediction() -> None:
    surrogate = _CountingSurrogate()
    x = np.asarray([[0.1, 0.2], [0.8, 0.7], [0.4, 0.5], [0.2, 0.9]])
    y = np.asarray([4.0, 2.0, 1.0, 3.0])
    options = _build_portfolio_candidate_options(
        surrogate=surrogate,
        observed_x=x,
        observed_y=y,
        dim=2,
        n_pool=16,
        n_options_per_strategy=3,
        seed=11,
        active_hypotheses=[],
        information_gain_weight=0.35,
        world_model_entropy=0.5,
        constraint_surrogate=None,
        has_feasible_observation=True,
        feasibility_floor=0.05,
        best_x=x[2],
        feasible_x=x,
        lengthscales=[0.2, 0.8],
        turbo_state=TurboState(dim=2),
        cma_state=OnePlusOneCMAState(dim=2, seed=12),
    )
    assert surrogate.calls == len(PORTFOLIO_STRATEGIES)
    assert len(options) == len(PORTFOLIO_STRATEGIES) * 3
    assert len({option["candidate_id"] for option in options}) == len(options)


def test_cma_masks_itself_when_eightfold_pool_fails_pof_threshold() -> None:
    surrogate = _CountingSurrogate()
    x = np.asarray([[0.1, 0.2], [0.8, 0.7], [0.4, 0.5], [0.2, 0.9]])
    y = np.asarray([4.0, 2.0, 1.0, 3.0])
    options = _build_portfolio_candidate_options(
        surrogate=surrogate,
        observed_x=x,
        observed_y=y,
        dim=2,
        n_pool=8,
        n_options_per_strategy=1,
        seed=13,
        active_hypotheses=[],
        information_gain_weight=0.35,
        world_model_entropy=0.5,
        constraint_surrogate=_RejectingConstraint(),
        has_feasible_observation=True,
        feasibility_floor=0.5,
        best_x=x[2],
        feasible_x=x[2:3],
        lengthscales=[1.0, 1.0],
        turbo_state=TurboState(dim=2),
        cma_state=OnePlusOneCMAState(dim=2, seed=14),
    )
    assert "cma_local" not in {option["strategy"] for option in options}
    control = PortfolioWMBOState(WMBOControlConfig())
    context = control.decision_context(
        phase="middle",
        trial_number=8,
        completed_trials=7,
        budget=20,
        uncertainty=0.5,
        has_feasible_observation=True,
        available_strategies=sorted({option["strategy"] for option in options}),
    )
    assert "cma_local" not in context.allowed_strategies
    assert "cma_local_candidate_unavailable" in context.gate_reasons

def test_v5_smoke_has_macro_reuse_zero_override_and_valid_candidate_ids() -> None:

    run = run_single_benchmark(
        BenchmarkRunRequest(
            benchmark_name="branin",
            method="wmbo_rule",
            seed=3,
            budget=8,
            metadata={
                "initial_samples": 2,
                "candidate_pool_size": 32,
                "options": {
                    "rule_policy": {"mode": "portfolio_v5"},
                    "wmbo_control": {"candidate_options_per_strategy": 2},
                    "logging": {"verbose": False},
                },
            },
        )
    )
    rows = [
        row for row in run.observations
        if row.get("executed_strategy") not in {None, "initial_design"}
    ]
    assert rows
    assert all(row["executed_strategy"] in PORTFOLIO_STRATEGIES for row in rows)
    assert all(row.get("override_reason") is None for row in rows)
    assert all(row.get("selected_candidate_id") is not None for row in rows)
    assert all(row.get("macro_action_id") is not None for row in rows)
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["macro_action_id"]), []).append(row)
    for group in grouped.values():
        hypothesis_ids = {row.get("hypothesis_id") for row in group}
        assert len(hypothesis_ids) == 1


def test_paper_configuration_remains_on_v4() -> None:
    text = Path("configs/synthetic_bbob_paper.yaml").read_text(encoding="utf-8")
    assert "mode: continuous_v4" in text
    assert "mode: portfolio_v5" not in text

