"""Portfolio-v5 geometry, operator state, macro actions, and delayed rewards."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import numpy as np

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
    "global_sobol": (1, 1),
    "gp_ucb": (2, 2),
    "gp_ei": (2, 3),
    "anisotropic_turbo": (3, 4),
    "cma_local": (4, 6),
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
    """Estimate scale-free geometry features for portfolio selection."""

    x = np.asarray(observed_x, dtype=float)
    dim = int(x.shape[1]) if x.ndim == 2 and x.shape[1] else 1
    _, _, local_condition, rotation, effective = elite_geometry(observed_x, observed_y)
    try:
        scales = np.asarray(lengthscales, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        scales = np.asarray([], dtype=float)
    if len(scales) == dim and np.all(np.isfinite(scales)):
        positive = np.maximum(scales, 1.0e-12)
        length_condition = float(
            np.clip(np.log10(np.max(positive) / np.min(positive)) / 4.0, 0.0, 1.0)
        )
    else:
        length_condition = 0.0
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
        "lengthscale_condition": length_condition,
        "local_condition": local_condition,
        "rotation_score": rotation,
        "effective_dimension": effective,
        "valley_score": valley,
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
    temperature: float = 0.20,
) -> dict[str, float]:
    """Return deterministic evidence-shrunk posteriors over v5 regimes."""

    smooth = _bounded(smoothness, 0.5)
    multi = _bounded(modality, 0.5)
    curve = _bounded(curvature, 0.5)
    uncertain = _bounded(uncertainty, 1.0)
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
            np.mean([smooth, multi, uncertain]),
            1.0 - evidence,
        ],
        dtype=float,
    )
    shifted = (logits - float(np.max(logits))) / max(float(temperature), 1.0e-6)
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
    """Blend rotated elite covariance with the GP ARD diagonal shape."""

    x = np.asarray(observed_x, dtype=float)
    dim = int(x.shape[1]) if x.ndim == 2 and x.shape[1] else 1
    _, local_shape, _, _, _ = elite_geometry(observed_x, observed_y)
    try:
        scales = np.asarray(lengthscales, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        scales = np.ones(dim, dtype=float)
    if len(scales) != dim or not np.all(np.isfinite(scales)):
        scales = np.ones(dim, dtype=float)
    scales = np.maximum(scales, 1.0e-6)
    scales /= max(float(np.exp(np.mean(np.log(scales)))), 1.0e-12)
    ard_shape = _normalise_shape(np.diag(scales**2))
    return _normalise_shape(0.60 * local_shape + 0.40 * ard_shape)


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

    def reward(self) -> tuple[float, dict[str, float]]:
        steps = max(1, int(self.completed_steps))
        end_best = min(self.best_trajectory, default=self.start_best)
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
        components = {
            "terminal": float(1.0 - math.exp(-terminal_gain)),
            "anytime": float(1.0 - math.exp(-anytime_gain)),
            "information": float(np.clip(information, 0.0, 1.0)),
        }
        reward = float(
            np.clip(
                0.70 * components["terminal"]
                + 0.20 * components["anytime"]
                + 0.10 * components["information"],
                0.0,
                1.0,
            )
        )
        return reward, components

    def to_dict(self) -> dict[str, object]:
        reward, components = self.reward()
        return {
            "macro_action_id": self.macro_id,
            "macro_strategy": self.strategy,
            "macro_step": self.completed_steps,
            "macro_min_steps": self.min_steps,
            "macro_max_steps": self.max_steps,
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
