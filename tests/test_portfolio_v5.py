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
    PORTFOLIO_POLICY_VERSION,
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
    phase: str = "middle",
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
            "budget_phase": phase,
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



def test_v51_geometry_is_reliability_shrunk_with_small_samples() -> None:
    small_x = np.asarray([[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]])
    small = geometry_features(
        small_x,
        [2.0, 1.0],
        [0.01, 1.0, 10.0],
        curvature=0.8,
        modality=0.2,
    )
    rng = np.random.default_rng(12)
    large_x = rng.random((30, 3))
    large = geometry_features(
        large_x,
        np.sum((large_x - 0.3) ** 2, axis=1),
        [0.01, 1.0, 10.0],
        curvature=0.8,
        modality=0.2,
    )
    assert small["geometry_reliability"] == pytest.approx(0.0)
    assert small["rotation_score"] == pytest.approx(0.0)
    assert small["local_condition"] == pytest.approx(0.0)
    assert small["effective_dimension"] == pytest.approx(1.0)
    assert large["geometry_reliability"] > small["geometry_reliability"]
    assert large["lengthscale_reliability"] > small["lengthscale_reliability"]
    assert large["lengthscale_condition"] > small["lengthscale_condition"]


def test_v51_rc2_lengthscale_reliability_is_half_at_two_d_samples() -> None:
    # The BBOB policy starts routing after a 2d initial design. RC2 deliberately
    # trusts ARD condition at 0.5 here instead of Formal's 1/3.
    x = np.linspace(0.05, 0.95, 12).reshape(6, 2)[:4]
    features = geometry_features(
        x,
        np.sum((x - 0.3) ** 2, axis=1),
        [0.01, 1.0],
        curvature=0.5,
        modality=0.2,
    )
    assert len(x) == 2 * x.shape[1]
    assert features["lengthscale_reliability"] == pytest.approx(0.5)


def test_v51_rc2_successful_local_macro_extends_only_after_progress() -> None:
    cma = MacroActionState("m-cma", "cma_local", 1, 10.0, 1.0, 2, 4)
    cma.record(10.0, False, 0.0)
    assert cma.max_steps == 4
    assert cma.successful_extensions == 0
    cma.record(9.5, True, 0.0)
    cma.extend_after_success(6)
    assert cma.max_steps == 6
    assert cma.successful_extensions == 1
    cma.extend_after_success(6)
    assert cma.successful_extensions == 1

    turbo = MacroActionState("m-turbo", "anisotropic_turbo", 1, 10.0, 1.0, 2, 4)
    turbo.record(9.8, True, 0.0)
    turbo.extend_after_success(5)
    assert turbo.max_steps == 5
    assert turbo.successful_extensions == 1


def test_v51_regime_separates_epistemic_uncertainty_from_ruggedness() -> None:
    common = dict(
        num_observations=24,
        dim=4,
        smoothness=0.8,
        modality=0.8,
        curvature=0.4,
        coverage=0.8,
        world_model_entropy=0.5,
        geometry={
            "rotation_score": 0.1,
            "local_condition": 0.1,
            "lengthscale_condition": 0.1,
            "effective_dimension": 0.9,
        },
    )
    low_uncertainty = regime_posteriors(uncertainty=0.1, **common)
    high_uncertainty = regime_posteriors(uncertainty=0.9, **common)
    assert high_uncertainty["weakly_identified"] > low_uncertainty["weakly_identified"]
    assert high_uncertainty["rugged_multimodal"] <= low_uncertainty["rugged_multimodal"]


def test_v51_macro_durations_are_short_and_adaptive() -> None:
    assert PORTFOLIO_POLICY_VERSION == "5.1-rc3"
    assert MACRO_DURATIONS["global_sobol"] == (1, 1)
    assert MACRO_DURATIONS["gp_ucb"] == (1, 1)
    assert MACRO_DURATIONS["gp_ei"] == (1, 1)
    assert MACRO_DURATIONS["anisotropic_turbo"] == (2, 4)
    assert MACRO_DURATIONS["cma_local"] == (2, 4)


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


def test_v51_macro_reward_is_progress_centred_and_phase_decayed() -> None:
    failed = MacroActionState("m_fail", "global_sobol", 1, 10.0, 5.0, 1, 1)
    failed.record(10.0, False, 1.0)
    early_reward, early_components = failed.reward(phase="early")
    middle_reward, _ = failed.reward(phase="middle")
    late_reward, late_components = failed.reward(phase="late")
    assert early_reward == pytest.approx(0.15)
    assert middle_reward == pytest.approx(0.05)
    assert late_reward == pytest.approx(0.0)
    assert early_components["objective_success"] == 0.0
    assert late_components["information_weight"] == 0.0

    improved = MacroActionState("m_win", "cma_local", 1, 10.0, 5.0, 1, 1)
    improved.record(9.5, True, 0.0)
    reward, components = improved.reward(phase="late")
    assert reward >= 0.55
    assert components["objective_success"] == 1.0


def test_v51_trust_uses_objective_success_and_exploration_failure_cooldown() -> None:
    control = PortfolioWMBOState(WMBOControlConfig(failure_cooldown_trials=2))
    assert control.recent_success_rates()["global_sobol"] == pytest.approx(0.5)

    cooldown = None
    for trial in (3, 6, 9):
        cooldown = control.record_delayed_reward(
            "global_sobol", reward=0.15, trial_number=trial, improved=False
        )
    assert cooldown == 12
    assert control.cooldown_until["global_sobol"] == 12
    assert control.recent_success_rates()["global_sobol"] < 0.5
    assert control.trusts["global_sobol"] < 0.5

    before = control.trusts["cma_local"]
    control.record_delayed_reward("cma_local", reward=0.60, trial_number=10, improved=True)
    assert control.trusts["cma_local"] > before
    assert control.recent_success_rates()["cma_local"] > 0.5


def test_v51_rc3_penalises_stale_dominant_local_operator_without_masking() -> None:
    control = PortfolioWMBOState(WMBOControlConfig(trust_window=5))
    control.executed_strategies.extend(
        ["cma_local"] * 9 + ["gp_ei", "anisotropic_turbo", "gp_ei"]
    )
    for _ in range(4):
        control.outcomes["cma_local"].append(False)

    penalties, reasons, shares = control.local_routing_penalties()
    assert shares["cma_local"] == pytest.approx(0.75)
    assert penalties["cma_local"] == pytest.approx(0.65)
    assert "stale_local_dominance" in reasons["cma_local"]
    assert penalties["gp_ei"] == pytest.approx(1.0)

    context = control.decision_context(
        phase="middle", trial_number=20, completed_trials=19, budget=40, uncertainty=0.2
    )
    assert "cma_local" in context.allowed_strategies
    assert context.strategy_routing_penalties["cma_local"] == pytest.approx(0.65)


def test_v51_rc3_dominance_penalty_clears_after_recent_macro_success() -> None:
    control = PortfolioWMBOState(WMBOControlConfig(trust_window=5))
    control.executed_strategies.extend(["cma_local"] * 10 + ["gp_ei"] * 2)
    control.outcomes["cma_local"].extend([False, False, False, True])
    penalties, reasons, shares = control.local_routing_penalties()
    assert shares["cma_local"] > 0.70
    assert penalties["cma_local"] == pytest.approx(1.0)
    assert "cma_local" not in reasons


def test_v51_rc3_penalises_evidence_gated_underperforming_local_operator() -> None:
    control = PortfolioWMBOState(WMBOControlConfig(trust_window=5))
    # Keep recent evaluation shares balanced so this test isolates performance evidence.
    control.executed_strategies.extend(
        ["gp_ei", "cma_local", "anisotropic_turbo"] * 4
    )
    control.outcomes["gp_ei"].extend([False, False, False, False, False])
    control.outcomes["cma_local"].extend([True, True, True, True, False])
    control.outcomes["anisotropic_turbo"].extend([False, True, False, True, False])

    penalties, reasons, _ = control.local_routing_penalties()
    assert control.recent_success_rates()["gp_ei"] == pytest.approx(1.0 / 7.0)
    assert control.recent_success_rates()["cma_local"] == pytest.approx(5.0 / 7.0)
    assert penalties["gp_ei"] == pytest.approx(0.65)
    assert "evidence_gated_underperformance" in reasons["gp_ei"]
    assert penalties["cma_local"] == pytest.approx(1.0)


def test_v51_rc3_penalty_does_not_stack_when_both_reasons_apply() -> None:
    control = PortfolioWMBOState(WMBOControlConfig(trust_window=5))
    control.executed_strategies.extend(["gp_ei"] * 9 + ["cma_local"] * 3)
    control.outcomes["gp_ei"].extend([False] * 5)
    control.outcomes["cma_local"].extend([True] * 5)
    penalties, reasons, _ = control.local_routing_penalties()
    assert penalties["gp_ei"] == pytest.approx(0.65)
    assert set(reasons["gp_ei"]) == {
        "stale_local_dominance",
        "evidence_gated_underperformance",
    }


def test_v51_new_best_follow_up_is_portfolio_local_only() -> None:
    control = PortfolioWMBOState(WMBOControlConfig())
    control.record_outcome(
        strategy="gp_ucb",
        trial_number=4,
        improved=True,
        y=0.8,
        best_y=0.8,
        update_trust=False,
    )
    context = control.decision_context(
        phase="early",
        trial_number=5,
        completed_trials=4,
        budget=20,
        uncertainty=1.0,
        coverage=0.1,
        weakly_identified=1.0,
        modality_label="highly_multimodal",
        has_feasible_observation=True,
    )
    assert set(context.allowed_strategies) == {"gp_ei", "anisotropic_turbo", "cma_local"}
    assert "new_best_local_follow_up_local_only" in context.gate_reasons
    assert context.forced_strategy is None


def test_v51_sparse_exploration_gates_do_not_reexplore_immediately() -> None:
    control = PortfolioWMBOState(WMBOControlConfig())
    first = control.decision_context(
        phase="early",
        trial_number=5,
        completed_trials=4,
        budget=30,
        uncertainty=1.0,
        coverage=0.9,
        weakly_identified=0.22,
        weakly_identified_top2=True,
        regime_entropy=0.95,
        modality_label="highly_multimodal",
    )
    assert "global_sobol" not in first.allowed_strategies
    assert "gp_ucb" not in first.allowed_strategies
    assert "early_global_sparse_gate" in first.gate_reasons
    assert "early_ucb_sparse_gate" in first.gate_reasons

    control.executed_strategies.extend(["gp_ei"] * 6)
    control.consecutive_no_improvement = 2
    later = control.decision_context(
        phase="early",
        trial_number=11,
        completed_trials=10,
        budget=30,
        uncertainty=0.9,
        coverage=0.95,
        weakly_identified=0.21,
        weakly_identified_top2=True,
        regime_entropy=0.9,
    )
    assert "global_sobol" in later.allowed_strategies
    assert "gp_ucb" in later.allowed_strategies

    control.consecutive_no_improvement = 1
    middle = control.decision_context(
        phase="middle",
        trial_number=15,
        completed_trials=14,
        budget=30,
        uncertainty=1.0,
        coverage=1.0,
        weakly_identified=0.2,
        weakly_identified_top2=True,
        regime_entropy=0.9,
    )
    assert "global_sobol" not in middle.allowed_strategies
    assert "gp_ucb" not in middle.allowed_strategies


def test_v51_global_emergency_probe_is_sparse_and_capped() -> None:
    control = PortfolioWMBOState(WMBOControlConfig())
    control.executed_strategies.extend(["gp_ei"] * 6)
    control.consecutive_no_improvement = 3
    context = control.decision_context(
        phase="early",
        trial_number=12,
        completed_trials=11,
        budget=40,
        uncertainty=0.9,
        weakly_identified=0.2,
        weakly_identified_top2=True,
        regime_entropy=0.9,
    )
    assert context.forced_strategy == "global_sobol"
    assert context.forced_reason == "weak_identification_emergency_probe"

    control.executed_strategies.extend(["global_sobol", "gp_ei", "gp_ei", "gp_ei", "gp_ei", "gp_ei", "gp_ei", "global_sobol"])
    control.executed_strategies.extend(["gp_ei"] * 6)
    capped = control.decision_context(
        phase="early",
        trial_number=26,
        completed_trials=25,
        budget=80,
        uncertainty=0.95,
        weakly_identified=0.2,
        weakly_identified_top2=True,
        regime_entropy=0.95,
    )
    assert "global_sobol" not in capped.allowed_strategies
    assert "early_global_action_cap" in capped.gate_reasons

def test_v51_candidate_information_weight_decays_by_phase() -> None:
    surrogate = _CountingSurrogate()
    x = np.asarray([[0.1, 0.2], [0.8, 0.7], [0.4, 0.5], [0.2, 0.9]])
    y = np.asarray([4.0, 2.0, 1.0, 3.0])
    weights = {}
    for phase in ("early", "middle", "late"):
        options = _build_portfolio_candidate_options(
            surrogate=surrogate,
            observed_x=x,
            observed_y=y,
            dim=2,
            n_pool=8,
            n_options_per_strategy=1,
            seed=21,
            active_hypotheses=[],
            phase=phase,
            information_gain_weight=0.35,
            world_model_entropy=0.5,
            constraint_surrogate=None,
            has_feasible_observation=True,
            feasibility_floor=0.05,
            best_x=x[2],
            feasible_x=x,
            lengthscales=[0.2, 0.8],
            turbo_state=TurboState(dim=2),
            cma_state=OnePlusOneCMAState(dim=2, seed=22),
        )
        weights[phase] = options[0]["information_gain_weight_effective"]
    assert weights["early"] == pytest.approx(0.35 * 0.75)
    assert weights["middle"] == pytest.approx(0.35 * 0.35)
    assert weights["late"] == pytest.approx(0.35 * 0.10)
    assert weights["early"] > weights["middle"] > weights["late"]



def test_v51_late_candidates_are_optimisation_only() -> None:
    surrogate = _CountingSurrogate()
    x = np.asarray([[0.1, 0.2], [0.8, 0.7], [0.4, 0.5], [0.2, 0.9]])
    y = np.asarray([4.0, 2.0, 1.0, 3.0])
    options = _build_portfolio_candidate_options(
        surrogate=surrogate,
        observed_x=x,
        observed_y=y,
        dim=2,
        n_pool=8,
        n_options_per_strategy=3,
        seed=41,
        active_hypotheses=[{"hypothesis_id": "h1", "status": "active", "region_center": [0.4, 0.5], "region_radius": 0.2}],
        phase="late",
        information_gain_weight=0.35,
        world_model_entropy=0.8,
        constraint_surrogate=None,
        has_feasible_observation=True,
        feasibility_floor=0.05,
        best_x=x[2],
        feasible_x=x,
        lengthscales=[0.2, 0.8],
        turbo_state=TurboState(dim=2),
        cma_state=OnePlusOneCMAState(dim=2, seed=42),
    )
    assert options
    assert {option["evidence_role"] for option in options} == {"optimize"}


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



def test_v51_routing_weights_change_by_phase() -> None:
    expected = {
        "early": {"landscape": 0.25, "geometry": 0.20, "candidate": 0.35, "history": 0.20},
        "middle": {"landscape": 0.20, "geometry": 0.30, "candidate": 0.30, "history": 0.20},
        "late": {"landscape": 0.15, "geometry": 0.30, "candidate": 0.30, "history": 0.25},
    }
    agent = WorldModelAgent({"mode": "portfolio_v5"})
    for phase, target in expected.items():
        decision = agent.decide(_agent_state(phase=phase))
        weights = decision.metadata["routing_weights_effective"]
        for name, value in target.items():
            assert weights[name] == pytest.approx(value)
        assert decision.metadata["rule_policy_version"] == "5.1-rc3"


def test_v51_rc3_agent_applies_soft_routing_penalty_after_base_score() -> None:
    base = _agent_state(phase="middle")
    context = dict(base.decision_context)
    context["strategy_routing_penalties"] = {"cma_local": 0.65}
    context["strategy_penalty_reasons"] = {"cma_local": ["stale_local_dominance"]}
    context["local_operator_recent_shares"] = {
        "gp_ei": 0.1, "anisotropic_turbo": 0.15, "cma_local": 0.75
    }
    state = AgentState(
        observed_x=base.observed_x,
        observed_y=base.observed_y,
        descriptor=base.descriptor,
        budget_used=base.budget_used,
        budget_total=base.budget_total,
        candidate_options=base.candidate_options,
        decision_context=context,
    )
    decision = WorldModelAgent({"mode": "portfolio_v5"}).decide(state)
    pre = decision.metadata["strategy_scores_pre_penalty"]["cma_local"]
    post = decision.metadata["strategy_scores"]["cma_local"]
    assert post == pytest.approx(0.65 * pre)
    assert decision.metadata["strategy_score_penalties"]["cma_local"] == pytest.approx(0.65)
    assert decision.metadata["strategy_penalty_reasons"]["cma_local"] == [
        "stale_local_dominance"
    ]


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
        phase="middle",
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
        phase="middle",
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

