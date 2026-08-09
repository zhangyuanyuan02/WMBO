from __future__ import annotations

import pytest

from wmbo.agents import AgentState, RulePolicyConfig, WorldModelAgent
from wmbo.control import STRATEGIES, WMBOControlConfig, WMBOState
from wmbo.runner import BenchmarkRunRequest, run_single_benchmark


def _descriptor(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "dim": 2,
        "num_observations": 6,
        "uncertainty": 0.5,
        "coverage": 0.5,
        "stagnation": 0.5,
        "smoothness": 0.5,
        "modality": 0.5,
        "curvature": 0.5,
        "anisotropy": 0.5,
        "labels": {
            "smoothness": "mixed",
            "modality": "multimodal",
            "curvature": "moderate",
            "anisotropy": "moderate",
        },
        "property_posteriors": {},
        "calibration": {
            "world_model_entropy": 0.5,
            "posterior_confidence": 0.6,
        },
    }
    value.update(updates)
    return value


def _candidates() -> list[dict[str, object]]:
    return [
        {
            "candidate_id": f"{strategy}_1",
            "strategy": strategy,
            "x_unit": [0.25, 0.75],
            "selection_score": 0.5,
            "expected_improvement": 0.5,
            "constrained_expected_improvement": 0.5,
            "information_gain": 0.5,
            "confirmation_value": 0.5,
            "distance_to_nearest_observation": 0.5,
            "surrogate_std": 0.5,
        }
        for strategy in STRATEGIES
    ]


def _state(
    *,
    descriptor: dict[str, object],
    observed_y: list[float],
    used: int,
    total: int = 20,
    candidates: list[dict[str, object]] | None = None,
    context: dict[str, object] | None = None,
) -> AgentState:
    return AgentState(
        observed_x=[[0.1, 0.2] for _ in observed_y],
        observed_y=observed_y,
        descriptor=descriptor,
        budget_used=used,
        budget_total=total,
        candidate_options=candidates or _candidates(),
        decision_context=context or {"allowed_strategies": list(STRATEGIES)},
    )


def test_continuous_policy_respects_forced_and_masked_strategies() -> None:
    decision = WorldModelAgent({"mode": "continuous_v2"}).decide(
        _state(
            descriptor=_descriptor(),
            observed_y=[10.0, 8.0, 7.0],
            used=8,
            context={
                "allowed_strategies": ["exploit_ei", "trust_region"],
                "forced_strategy": "trust_region",
                "forced_reason": "new_best_local_follow_up",
            },
        )
    )
    assert decision.strategy == "trust_region"
    assert decision.selected_candidate_id == "trust_region_1"
    assert decision.metadata["masked_strategies"] == ["global_diverse", "explore_ucb"]
    assert decision.metadata["strategy_scores"]["global_diverse"] is None


def test_continuous_policy_changes_from_exploration_to_exploitation() -> None:
    agent = WorldModelAgent({"mode": "continuous_v2"})
    early = agent.decide(
        _state(
            descriptor=_descriptor(
                uncertainty=0.9,
                coverage=0.1,
                stagnation=1.0,
                curvature=0.8,
                calibration={"world_model_entropy": 1.0, "posterior_confidence": 0.5},
            ),
            observed_y=[10.0, 10.0, 10.0],
            used=2,
        )
    )
    late = agent.decide(
        _state(
            descriptor=_descriptor(
                uncertainty=0.1,
                coverage=0.9,
                stagnation=0.2,
                modality=0.1,
                curvature=0.1,
                calibration={"world_model_entropy": 0.1, "posterior_confidence": 0.9},
            ),
            observed_y=[10.0, 8.0, 5.0, 4.0],
            used=17,
            context={"allowed_strategies": ["exploit_ei", "trust_region"]},
        )
    )
    assert early.strategy in {"global_diverse", "explore_ucb"}
    assert late.strategy == "exploit_ei"


def test_recent_improvement_is_invariant_to_positive_affine_scaling() -> None:
    agent = WorldModelAgent({"mode": "continuous_v2"})
    base = agent.decide(
        _state(descriptor=_descriptor(), observed_y=[10.0, 8.0, 5.0], used=10)
    )
    scaled = agent.decide(
        _state(descriptor=_descriptor(), observed_y=[110.0, 90.0, 60.0], used=10)
    )
    assert scaled.strategy == base.strategy
    assert scaled.metadata["normalised_recent_improvement"] == pytest.approx(
        base.metadata["normalised_recent_improvement"]
    )


def test_candidate_selection_is_legal_and_deterministic() -> None:
    candidates = _candidates()
    candidates.extend(
        [
            {
                **candidates[-1],
                "candidate_id": "trust_region_2",
                "selection_score": 0.9,
            },
            {
                **candidates[-1],
                "candidate_id": "trust_region_3",
                "selection_score": 0.9,
            },
        ]
    )
    decision = WorldModelAgent({"mode": "continuous_v2"}).decide(
        _state(
            descriptor=_descriptor(),
            observed_y=[3.0, 2.0],
            used=10,
            candidates=candidates,
            context={
                "allowed_strategies": ["trust_region"],
                "forced_strategy": "trust_region",
            },
        )
    )
    assert decision.selected_candidate_id == "trust_region_3"


def test_missing_posterior_and_candidate_fields_degrade_safely() -> None:
    decision = WorldModelAgent({"mode": "continuous_v2"}).decide(
        _state(
            descriptor={"labels": {}, "calibration": {}},
            observed_y=[1.0, 1.0],
            used=4,
            candidates=[
                {
                    "candidate_id": f"{strategy}_missing",
                    "strategy": strategy,
                    "x_unit": [0.5, 0.5],
                }
                for strategy in STRATEGIES
            ],
        )
    )
    assert decision.strategy in STRATEGIES
    assert decision.selected_candidate_id is not None
    assert 0.0 <= decision.confidence <= 1.0


def test_controller_context_prevents_rule_override() -> None:
    control = WMBOState(WMBOControlConfig())
    context = control.decision_context(
        phase="late",
        trial_number=18,
        completed_trials=17,
        budget=20,
        uncertainty=0.1,
        smoothness_label="smooth",
        modality_label="mostly_unimodal",
    )
    decision = WorldModelAgent({"mode": "continuous_v2"}).decide(
        _state(
            descriptor=_descriptor(uncertainty=0.1, coverage=0.9),
            observed_y=[5.0, 4.0, 3.0],
            used=17,
            context=context.to_dict(),
        )
    )
    executed, override, allowed = control.choose_strategy(
        proposed_strategy=decision.strategy,
        phase="late",
        trial_number=18,
        uncertainty=0.1,
        decision_context=context,
    )
    assert decision.strategy in allowed
    assert executed == decision.strategy
    assert override is None


def test_rule_policy_config_validates_weights_and_mode() -> None:
    with pytest.raises(ValueError):
        RulePolicyConfig.from_mapping({"mode": "unknown"})
    with pytest.raises(ValueError):
        RulePolicyConfig.from_mapping(
            {"landscape_weight": 0.0, "candidate_weight": 0.0, "history_weight": 0.0}
        )
    assert RulePolicyConfig.from_mapping({"mode": "continuous_v4"}).mode == "continuous_v4"



def test_real_run_persists_replayable_rule_diagnostics() -> None:
    run = run_single_benchmark(
        BenchmarkRunRequest(
            benchmark_name="branin",
            method="wmbo_rule",
            seed=4,
            budget=3,
            metadata={
                "initial_samples": 2,
                "candidate_pool_size": 16,
                "options": {
                    "rule_policy": {"mode": "continuous_v4"},
                    "logging": {"verbose": False},
                },
            },
        )
    )
    decision = run.observations[-1]
    assert decision["proposed_strategy"] in decision["allowed_strategies"]
    assert decision["override_reason"] is None
    assert decision["requested_candidate_id"] is not None
    assert decision["strategy_scores"]
    assert decision["score_components"]
    assert decision["landscape_descriptor"]
    assert decision["strategy_decision_context"]
    assert len(decision["candidate_options"]) == 12


def test_continuous_v3_adds_final_regret_phase_priors() -> None:
    state = _state(
        descriptor=_descriptor(),
        observed_y=[10.0, 8.0, 7.0],
        used=10,
        context={
            "budget_phase": "middle",
            "allowed_strategies": list(STRATEGIES),
        },
    )
    v2 = WorldModelAgent({"mode": "continuous_v2"}).decide(state)
    v3 = WorldModelAgent({"mode": "continuous_v3"}).decide(state)
    assert v2.metadata["score_components"]["explore_ucb"]["phase_prior"] == 0.0
    assert v3.metadata["score_components"]["explore_ucb"]["phase_prior"] == -0.10
    assert v3.metadata["score_components"]["trust_region"]["phase_prior"] == 0.06
    assert v3.metadata["rule_policy_mode"] == "continuous_v3"


def test_continuous_v4_strengthens_local_phase_priors() -> None:
    state = _state(
        descriptor=_descriptor(),
        observed_y=[10.0, 8.0, 7.0],
        used=10,
        context={
            "budget_phase": "middle",
            "allowed_strategies": list(STRATEGIES),
        },
    )
    v3 = WorldModelAgent({"mode": "continuous_v3"}).decide(state)
    v4 = WorldModelAgent({"mode": "continuous_v4"}).decide(state)
    assert v4.metadata["score_components"]["explore_ucb"]["phase_prior"] == -0.11
    assert v4.metadata["score_components"]["trust_region"]["phase_prior"] == 0.10
    assert (
        v4.metadata["strategy_scores"]["trust_region"]
        > v3.metadata["strategy_scores"]["trust_region"]
    )
    assert v4.metadata["rule_policy_mode"] == "continuous_v4"


def test_continuous_v4_local_follow_up_allows_ei_trust_region_competition() -> None:
    control = WMBOState(WMBOControlConfig())
    control.follow_up_local = True
    context = control.decision_context(
        phase="middle",
        trial_number=11,
        completed_trials=10,
        budget=20,
        uncertainty=0.2,
        smoothness_label="smooth",
        modality_label="mostly_unimodal",
        flexible_local_follow_up=True,
    )
    assert set(context.allowed_strategies) == {"exploit_ei", "trust_region"}
    assert context.forced_strategy is None
    assert "new_best_local_follow_up_local_only" in context.gate_reasons

    decision = WorldModelAgent({"mode": "continuous_v4"}).decide(
        _state(
            descriptor=_descriptor(uncertainty=0.2, coverage=0.8),
            observed_y=[5.0, 4.0, 3.0],
            used=10,
            context=context.to_dict(),
        )
    )
    executed, override, allowed = control.choose_strategy(
        proposed_strategy=decision.strategy,
        phase="middle",
        trial_number=11,
        uncertainty=0.2,
        smoothness_label="smooth",
        modality_label="mostly_unimodal",
        decision_context=context,
    )
    assert decision.strategy in allowed
    assert executed == decision.strategy
    assert override is None


