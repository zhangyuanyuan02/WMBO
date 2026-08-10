"""Portfolio-v5 geometry, operator state, macro actions, and delayed rewards."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import numpy as np

PORTFOLIO_POLICY_VERSION = "5.1-rc3"

PORTFOLIO_STRATEGIES = (
    "global_sobol",
    "gp_ucb",
    "gp_ei",
    "anisotropic_turbo",
    "cma_local",
)
PORTFOLIO_EXPLORATION_STRATEGIES = {"global_sobol", "gp_ucb"}
PORTFOLIO_LOCAL_STRATEGIES = {"gp_ei", "anisotropic_turbo", "cma_local"}
REGIMES = (
    "separable_smooth",
    "rotated_ill_conditioned",
    "curved_valley",
    "rugged_multimodal",
    "weakly_identified",
)
MACRO_DURATIONS = {
    # Stateless GP acquisition rules should return to the controller after every
    # evaluation. Local stateful methods start with a short macro; RC2 only grants
    # extra persistence after the operator has actually improved the incumbent.
    "global_sobol": (1, 1),
    "gp_ucb": (1, 1),
    "gp_ei": (1, 1),
    "anisotropic_turbo": (2, 4),
    "cma_local": (2, 4),
}
V51_SUCCESSFUL_MACRO_MAX_STEPS = {
    "anisotropic_turbo": 5,
    "cma_local": 6,
}

# RC3 adds only soft, evidence-gated local anti-lock-in penalties.  These are
# deliberately conservative: neither mechanism masks an operator, and multiple
# penalty reasons do not compound below the same 0.65 multiplier.
V51_RC3_LOCAL_DOMINANCE_WINDOW = 12
V51_RC3_LOCAL_DOMINANCE_SHARE = 0.70
V51_RC3_RECENT_FAILURE_MACROS = 4
V51_RC3_UNDERPERFORM_MIN_MACROS = 5
V51_RC3_UNDERPERFORM_RATIO = 0.25
V51_RC3_ROUTING_PENALTY = 0.65

# V5.1 formal policy constants.
V51_CANDIDATE_INFORMATION_MULTIPLIERS = {
    "early": 0.75,
    "middle": 0.35,
    "late": 0.10,
}
V51_REWARD_INFORMATION_WEIGHTS = {
    "early": 0.15,
    "middle": 0.05,
    "late": 0.00,
}
V51_TRUST_ALPHA = 0.18
V51_REWARD_EMA_ALPHA = 0.35
V51_REWARD_REFERENCE = 0.60
V51_EXPLORATION_FAILURE_WINDOW = 3
V51_EXPLORATION_GATES = {
    # The Stage-1 absolute weak-ID/coverage thresholds were outside the empirical
    # posterior scale and killed Sobol entirely.  V5.1 uses relative weak-ID or
    # entropy evidence, plus explicit spacing and stagnation requirements.
    "early_global_interval": 6,
    "early_global_no_improvement": 2,
    "early_global_max_actions": 2,
    "early_global_regime_entropy": 0.78,
    "early_global_uncertainty": 0.55,
    "early_ucb_uncertainty": 0.60,
    "early_ucb_no_improvement": 1,
    "early_ucb_interval": 5,
    "early_ucb_regime_entropy": 0.72,
    "middle_ucb_uncertainty": 0.55,
    "middle_ucb_no_improvement": 2,
    "middle_ucb_interval": 5,
    "late_ucb_uncertainty": 0.75,
    "late_ucb_no_improvement": 4,
    "late_ucb_interval": 8,
}

# Multipliers are applied to user/configured portfolio weights, so setting a
# component to zero still performs a valid ablation.
V51_ROUTING_WEIGHT_MULTIPLIERS = {
    # RC2 backs away from Formal's candidate-heavy routing. With the default
    # configured weights (0.25/0.30/0.30/0.15), these multipliers normalise to:
    # early  = 0.25 / 0.20 / 0.35 / 0.20
    # middle = 0.20 / 0.30 / 0.30 / 0.20
    # late   = 0.15 / 0.30 / 0.30 / 0.25
    "early": {
        "landscape": 1.00,
        "geometry": 2.0 / 3.0,
        "candidate": 7.0 / 6.0,
        "history": 4.0 / 3.0,
    },
    "middle": {
        "landscape": 0.80,
        "geometry": 1.00,
        "candidate": 1.00,
        "history": 4.0 / 3.0,
    },
    "late": {
        "landscape": 0.60,
        "geometry": 1.00,
        "candidate": 1.00,
        "history": 5.0 / 3.0,
    },
}


def _bounded(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    if not math.isfinite(parsed):
        parsed = float(default)
    return float(np.clip(parsed, 0.0, 1.0))


def _normalise_shape(matrix: np.ndarray, condition_limit: float = 1.0e4) -> np.ndarray:
    """Return a symmetric positive-definite shape with bounded condition."""

    value = np.asarray(matrix, dtype=float)
    value = 0.5 * (value + value.T)
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    upper = max(float(np.max(eigenvalues)), 1.0e-12)
    lower = max(upper / max(float(condition_limit), 1.0), 1.0e-12)
    eigenvalues = np.clip(eigenvalues, lower, upper)
    result = (eigenvectors * eigenvalues[None, :]) @ eigenvectors.T
    scale = float(np.trace(result) / max(len(result), 1))
    return result / max(scale, 1.0e-12)


def elite_geometry(
    observed_x: Sequence[Sequence[float]],
    observed_y: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Return elite centre/shape, local condition, rotation, and effective dimension."""

    x = np.asarray(observed_x, dtype=float)
    y = np.asarray(observed_y, dtype=float)
    if x.ndim != 2 or not len(x):
        dim = int(x.shape[1]) if x.ndim == 2 else 1
        return np.full(dim, 0.5), np.eye(dim), 0.0, 0.0, 1.0
    dim = x.shape[1]
    elite_count = min(len(y), max(dim + 2, int(math.ceil(0.25 * len(y)))))
    indices = np.argsort(y, kind="stable")[:elite_count]
    elite = x[indices]
    ranks = np.arange(len(elite), dtype=float)
    weights = np.exp(-ranks / max(1.0, len(elite) / 3.0))
    weights /= max(float(np.sum(weights)), 1.0e-12)
    centre = np.sum(elite * weights[:, None], axis=0)
    centred = elite - centre[None, :]
    covariance = (centred * weights[:, None]).T @ centred + 1.0e-6 * np.eye(dim)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 1.0e-12)
    condition = float(np.max(eigenvalues) / np.min(eigenvalues))
    local_condition = float(np.clip(np.log10(condition) / 6.0, 0.0, 1.0))
    standard = np.sqrt(np.maximum(np.diag(covariance), 1.0e-12))
    correlation = covariance / np.maximum(standard[:, None] * standard[None, :], 1.0e-12)
    off_diagonal = correlation - np.diag(np.diag(correlation))
    rotation = float(
        np.clip(
            np.linalg.norm(off_diagonal, ord="fro")
            / max(np.linalg.norm(correlation, ord="fro"), 1.0e-12),
            0.0,
            1.0,
        )
    )
    participation = float(np.sum(eigenvalues) ** 2 / max(np.sum(eigenvalues**2), 1.0e-12))
    effective_dimension = float(np.clip(participation / max(dim, 1), 0.0, 1.0))
    return centre, _normalise_shape(covariance), local_condition, rotation, effective_dimension


def geometry_features(
    observed_x: Sequence[Sequence[float]],
    observed_y: Sequence[float],
    lengthscales: object = None,
    *,
    curvature: object = 0.5,
    modality: object = 0.5,
) -> dict[str, float]:
    """Estimate reliability-shrunk, scale-free geometry features.

    Elite covariance is extremely noisy near the 2d initial design used by the
    BBOB experiments.  V5.1 therefore shrinks covariance-derived rotation and
    condition evidence towards the neutral geometry until enough observations
    have accumulated.  GP ARD condition receives a separate, smoother
    reliability curve.
    """

    x = np.asarray(observed_x, dtype=float)
    dim = int(x.shape[1]) if x.ndim == 2 and x.shape[1] else 1
    n = int(len(x)) if x.ndim == 2 else 0
    _, _, raw_local_condition, raw_rotation, raw_effective = elite_geometry(
        observed_x, observed_y
    )
    geometry_reliability = float(
        np.clip((n - (dim + 2)) / max(6.0 * dim, 1.0), 0.0, 1.0)
    )
    lengthscale_reliability = float(
        np.clip(n / max(n + 2.0 * dim, 1.0), 0.0, 1.0)
    )
    local_condition = geometry_reliability * raw_local_condition
    rotation = geometry_reliability * raw_rotation
    effective = (
        geometry_reliability * raw_effective
        + (1.0 - geometry_reliability) * 1.0
    )

    try:
        scales = np.asarray(lengthscales, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        scales = np.asarray([], dtype=float)
    if len(scales) == dim and np.all(np.isfinite(scales)):
        positive = np.maximum(scales, 1.0e-12)
        raw_length_condition = float(
            np.clip(np.log10(np.max(positive) / np.min(positive)) / 4.0, 0.0, 1.0)
        )
    else:
        raw_length_condition = 0.0
    length_condition = lengthscale_reliability * raw_length_condition
    condition = max(length_condition, local_condition)
    valley = float(
        np.clip(
            np.mean(
                [
                    condition,
                    1.0 - effective,
                    _bounded(curvature, 0.5),
                    1.0 - _bounded(modality, 0.5),
                ]
            ),
            0.0,
            1.0,
        )
    )
    return {
        "lengthscale_condition": float(length_condition),
        "local_condition": float(local_condition),
        "rotation_score": float(rotation),
        "effective_dimension": float(effective),
        "valley_score": valley,
        "geometry_reliability": geometry_reliability,
        "lengthscale_reliability": lengthscale_reliability,
    }

def regime_posteriors(
    *,
    num_observations: int,
    dim: int,
    smoothness: object,
    modality: object,
    curvature: object,
    uncertainty: object,
    geometry: Mapping[str, object],
    coverage: object = 0.0,
    world_model_entropy: object = 1.0,
    temperature: float | None = None,
) -> dict[str, float]:
    """Return reliability-aware posteriors over the five V5.1 regimes.

    V5 coupled ruggedness to epistemic uncertainty, which could turn "we do not
    know yet" into evidence that the objective itself is rugged.  V5.1 separates
    these concepts: rugged/multimodal is driven only by landscape evidence,
    while weak identification absorbs low sample evidence, poor coverage,
    surrogate uncertainty, and world-model entropy.
    """

    smooth = _bounded(smoothness, 0.5)
    multi = _bounded(modality, 0.5)
    curve = _bounded(curvature, 0.5)
    uncertain = _bounded(uncertainty, 1.0)
    coverage_value = _bounded(coverage, 0.0)
    entropy = _bounded(world_model_entropy, 1.0)
    rotation = _bounded(geometry.get("rotation_score"), 0.0)
    condition = max(
        _bounded(geometry.get("lengthscale_condition"), 0.0),
        _bounded(geometry.get("local_condition"), 0.0),
    )
    effective = _bounded(geometry.get("effective_dimension"), 1.0)
    evidence = float(num_observations / (num_observations + 4 * max(1, int(dim))))
    logits = np.asarray(
        [
            np.mean([1.0 - smooth, 1.0 - multi, 1.0 - rotation, 1.0 - condition]),
            np.mean([condition, rotation, 1.0 - multi]),
            np.mean([condition, 1.0 - effective, curve, 1.0 - multi]),
            np.mean([smooth, multi]),
            np.mean([1.0 - evidence, 1.0 - coverage_value, uncertain, entropy]),
        ],
        dtype=float,
    )
    # Low-evidence states remain deliberately flatter.  As evidence accumulates
    # the posterior sharpens without requiring the descriptor layer to know the
    # optimisation budget phase.
    effective_temperature = (
        float(temperature)
        if temperature is not None
        else 0.18 + 0.17 * (1.0 - evidence)
    )
    shifted = (logits - float(np.max(logits))) / max(effective_temperature, 1.0e-6)
    values = np.exp(np.clip(shifted, -60.0, 0.0))
    values /= max(float(np.sum(values)), 1.0e-12)
    values = evidence * values + (1.0 - evidence) / len(REGIMES)
    values /= max(float(np.sum(values)), 1.0e-12)
    return {name: float(value) for name, value in zip(REGIMES, values)}

def portfolio_shape(
    observed_x: Sequence[Sequence[float]],
    observed_y: Sequence[float],
    lengthscales: object,
) -> np.ndarray:
    """Blend elite covariance and GP ARD only in proportion to their reliability."""

    x = np.asarray(observed_x, dtype=float)
    dim = int(x.shape[1]) if x.ndim == 2 and x.shape[1] else 1
    n = int(len(x)) if x.ndim == 2 else 0
    _, local_shape, _, _, _ = elite_geometry(observed_x, observed_y)
    geometry_reliability = float(
        np.clip((n - (dim + 2)) / max(6.0 * dim, 1.0), 0.0, 1.0)
    )
    lengthscale_reliability = float(
        np.clip(n / max(n + 2.0 * dim, 1.0), 0.0, 1.0)
    )
    try:
        scales = np.asarray(lengthscales, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        scales = np.ones(dim, dtype=float)
    if len(scales) != dim or not np.all(np.isfinite(scales)):
        scales = np.ones(dim, dtype=float)
        lengthscale_reliability = 0.0
    scales = np.maximum(scales, 1.0e-6)
    scales /= max(float(np.exp(np.mean(np.log(scales)))), 1.0e-12)
    ard_shape = _normalise_shape(np.diag(scales**2))

    local_weight = 0.60 * geometry_reliability
    ard_weight = 0.40 * lengthscale_reliability
    neutral_weight = max(0.0, 1.0 - local_weight - ard_weight)
    blended = (
        local_weight * local_shape
        + ard_weight * ard_shape
        + neutral_weight * np.eye(dim)
    )
    return _normalise_shape(blended)


@dataclass
class TurboState:
    dim: int
    radius: float = 0.20
    consecutive_successes: int = 0
    consecutive_failures: int = 0

    def update(self, improved: bool) -> None:
        if improved:
            self.consecutive_successes += 1
            self.consecutive_failures = 0
            if self.consecutive_successes >= 3:
                self.radius = min(0.50, self.radius * 1.50)
                self.consecutive_successes = 0
        else:
            self.consecutive_failures += 1
            self.consecutive_successes = 0
            if self.consecutive_failures >= max(4, int(self.dim)):
                self.radius = max(0.02, self.radius * 0.50)
                self.consecutive_failures = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "radius": float(self.radius),
            "consecutive_successes": int(self.consecutive_successes),
            "consecutive_failures": int(self.consecutive_failures),
        }


@dataclass
class OnePlusOneCMAState:
    dim: int
    seed: int
    mean: np.ndarray | None = None
    sigma: float = 0.20
    covariance: np.ndarray | None = None
    path: np.ndarray | None = None
    incumbent: float | None = None

    def ensure(
        self,
        best_x: Sequence[float],
        best_y: float,
        initial_shape: np.ndarray | None = None,
    ) -> None:
        best = np.asarray(best_x, dtype=float)
        if self.mean is None:
            self.mean = best.copy()
            self.covariance = _normalise_shape(
                np.eye(self.dim) if initial_shape is None else initial_shape
            )
            self.path = np.zeros(self.dim, dtype=float)
            self.incumbent = float(best_y)
            return
        covariance = self.covariance if self.covariance is not None else np.eye(self.dim)
        try:
            inverse = np.linalg.pinv(covariance)
            distance = float(np.sqrt((best - self.mean) @ inverse @ (best - self.mean)))
        except Exception:
            distance = float("inf")
        if distance > 3.0 * max(self.sigma, 1.0e-12):
            self.mean = best.copy()
            self.sigma = 0.20
            self.path = np.zeros(self.dim, dtype=float)
        elif self.incumbent is None or float(best_y) < self.incumbent:
            self.mean = 0.5 * self.mean + 0.5 * best
        self.incumbent = min(float(best_y), self.incumbent if self.incumbent is not None else float(best_y))

    def sample(self, n_points: int, step_seed: int) -> np.ndarray:
        if self.mean is None:
            self.mean = np.full(self.dim, 0.5)
        if self.covariance is None:
            self.covariance = np.eye(self.dim)
        rng = np.random.default_rng(int(self.seed) + int(step_seed))
        values = rng.multivariate_normal(
            self.mean,
            (self.sigma**2) * _normalise_shape(self.covariance),
            size=max(1, int(n_points)),
            check_valid="ignore",
        )
        return np.clip(values, 0.0, 1.0)

    def update(self, candidate: Sequence[float], value: float, improved: bool) -> None:
        x = np.asarray(candidate, dtype=float)
        if self.mean is None:
            self.mean = x.copy()
        if self.covariance is None:
            self.covariance = np.eye(self.dim)
        if self.path is None:
            self.path = np.zeros(self.dim)
        delta = (x - self.mean) / max(self.sigma, 1.0e-12)
        if improved:
            self.path = 0.80 * self.path + math.sqrt(1.0 - 0.80**2) * delta
            self.covariance = _normalise_shape(
                0.90 * self.covariance + 0.10 * np.outer(self.path, self.path)
            )
            self.mean = x.copy()
            self.sigma = min(0.30, self.sigma * 1.20)
            self.incumbent = float(value)
        else:
            self.sigma = max(0.01, self.sigma * 0.82)

    def to_dict(self) -> dict[str, object]:
        covariance = self.covariance if self.covariance is not None else np.eye(self.dim)
        return {
            "mean": self.mean.astype(float).tolist() if self.mean is not None else None,
            "sigma": float(self.sigma),
            "covariance_condition": float(np.linalg.cond(covariance)),
            "incumbent": self.incumbent,
        }


@dataclass
class MacroActionState:
    macro_id: str
    strategy: str
    start_trial: int
    start_best: float
    start_scale: float
    min_steps: int
    max_steps: int
    completed_steps: int = 0
    best_trajectory: list[float] = field(default_factory=list)
    information_values: list[float] = field(default_factory=list)
    consecutive_failures: int = 0
    termination_reason: str | None = None
    hypothesis_id: str | None = None
    successful_extensions: int = 0

    def extend_after_success(self, ceiling: int) -> None:
        """Extend a local macro only after measured objective progress."""

        target = max(int(self.max_steps), int(ceiling))
        if target > self.max_steps:
            self.max_steps = target
            self.successful_extensions += 1

    def record(self, best_y: float, improved: bool, information_gain: float) -> None:
        self.completed_steps += 1
        self.best_trajectory.append(float(best_y))
        self.information_values.append(_bounded(information_gain, 0.0))
        self.consecutive_failures = 0 if improved else self.consecutive_failures + 1

    def should_stop(self) -> str | None:
        if self.completed_steps >= self.max_steps:
            return "max_steps"
        if (
            self.completed_steps >= self.min_steps
            and self.consecutive_failures >= 2
        ):
            return "early_no_improvement"
        return None

    def reward(self, phase: str = "middle") -> tuple[float, dict[str, float]]:
        """Return progress-centred V5.1 macro credit.

        Information is deliberately a small, phase-decaying bonus.  A macro that
        does not improve the incumbent therefore cannot accumulate high long-term
        trust merely by visiting uncertain points.
        """

        steps = max(1, int(self.completed_steps))
        end_best = min(self.best_trajectory, default=self.start_best)
        objective_success = bool(end_best < self.start_best - 1.0e-15)
        terminal_gain = max(0.0, self.start_best - end_best) / max(
            self.start_scale * steps, 1.0e-12
        )
        anytime_gain = float(
            np.mean(
                [
                    max(0.0, self.start_best - best) / max(self.start_scale, 1.0e-12)
                    for best in self.best_trajectory
                ]
            )
        ) if self.best_trajectory else 0.0
        information = float(np.mean(self.information_values)) if self.information_values else 0.0
        phase_key = str(phase).strip().lower()
        information_weight = V51_REWARD_INFORMATION_WEIGHTS.get(phase_key, 0.05)
        components = {
            "objective_success": 1.0 if objective_success else 0.0,
            "terminal": float(1.0 - math.exp(-terminal_gain)),
            "anytime": float(1.0 - math.exp(-anytime_gain)),
            "information": float(np.clip(information, 0.0, 1.0)),
            "information_weight": float(information_weight),
        }
        reward = float(
            np.clip(
                0.55 * components["objective_success"]
                + 0.25 * components["terminal"]
                + 0.15 * components["anytime"]
                + information_weight * components["information"],
                0.0,
                1.0,
            )
        )
        return reward, components

    def to_dict(self, phase: str = "middle") -> dict[str, object]:
        reward, components = self.reward(phase=phase)
        return {
            "macro_action_id": self.macro_id,
            "macro_strategy": self.strategy,
            "macro_step": self.completed_steps,
            "macro_min_steps": self.min_steps,
            "macro_max_steps": self.max_steps,
            "macro_successful_extensions": self.successful_extensions,
            "macro_termination_reason": self.termination_reason,
            "macro_reward": reward if self.termination_reason is not None else None,
            "macro_reward_components": components if self.termination_reason is not None else None,
        }


def objective_scale(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return 1.0
    return float(
        max(
            np.percentile(array, 75.0) - np.percentile(array, 25.0),
            np.max(array) - np.min(array),
            abs(float(np.median(array))) * 1.0e-6,
            1.0e-12,
        )
    )
