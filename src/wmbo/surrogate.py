"""Surrogate-model utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

Vector = Sequence[float]
Matrix = Sequence[Vector]


@dataclass(frozen=True)
class SurrogateDataset:
    """Training data for a surrogate model.

    Inputs:
        x: Evaluated points in normalised coordinates.
        y: Objective values for the evaluated points.
        metadata: Optional dataset metadata.

    Output:
        Passed to ``SurrogateModel.fit``.
    """

    x: Matrix
    y: Sequence[float]
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SurrogatePrediction:
    """Predicted objective values and uncertainty.

    Inputs:
        mean: Predicted objective mean for each candidate.
        std: Predicted uncertainty for each candidate.
        metadata: Optional model diagnostics.

    Output:
        Used by acquisition functions and descriptors.
    """

    mean: Sequence[float]
    std: Sequence[float]
    metadata: Mapping[str, object] = field(default_factory=dict)


class SurrogateModel(Protocol):
    """Protocol implemented by surrogate models."""

    def fit(self, data: SurrogateDataset) -> "SurrogateModel":
        """Fit the surrogate model.

        Input:
            data: Training inputs and objective values.

        Output:
            Fitted surrogate model.
        """

        ...

    def predict(self, candidates: Matrix) -> SurrogatePrediction:
        """Predict objective statistics for candidate points.

        Input:
            candidates: Points in normalised coordinates.

        Output:
            ``SurrogatePrediction`` with mean and uncertainty values.
        """

        ...


class GaussianProcessSurrogate:
    """Small Gaussian-process surrogate for normalised black-box inputs.

    This class intentionally keeps the implementation compact. It is good enough
    for early experiments and can later be replaced with a richer model wrapper.
    """

    def __init__(self, dim: int, noise_level: float = 1e-6, seed: int | None = None) -> None:
        """Create a Gaussian-process surrogate.

        Inputs:
            dim: Input dimensionality.
            noise_level: Observation-noise level.
            seed: Random seed.

        Output:
            Surrogate instance.
        """

        if dim <= 0:
            raise ValueError("dim must be positive.")
        if noise_level < 0.0:
            raise ValueError("noise_level must be non-negative.")

        self.dim = int(dim)
        self.noise_level = float(noise_level)
        self.seed = seed
        self._model: GaussianProcessRegressor | None = None
        self._is_fit = False
        self._x_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None

    @property
    def is_fit(self) -> bool:
        """Whether the wrapped Gaussian process has been fitted."""

        return self._is_fit

    def fit(self, data: SurrogateDataset) -> "GaussianProcessSurrogate":
        """Fit the surrogate on observed data.

        Input:
            data: Training data.

        Output:
            Fitted surrogate instance.
        """

        x = _as_2d_array(data.x, dim=self.dim, name="data.x")
        y = _as_1d_array(data.y, name="data.y")
        if len(x) != len(y):
            raise ValueError(f"data.x and data.y length mismatch: {len(x)} != {len(y)}.")

        self._x_train = x
        self._y_train = y

        if len(x) < 2:
            self._model = None
            self._is_fit = False
            return self

        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(
                length_scale=np.ones(self.dim) * 0.3,
                length_scale_bounds=(1e-2, 1e2),
                nu=2.5,
            )
            + WhiteKernel(
                noise_level=max(self.noise_level, 1e-12),
                noise_level_bounds=(1e-10, 1e-1),
            )
        )
        self._model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=self.seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self._model.fit(x, y)
        self._is_fit = True
        return self

    def predict(self, candidates: Matrix) -> SurrogatePrediction:
        """Predict mean and uncertainty for candidates.

        Input:
            candidates: Candidate points to evaluate with the surrogate.

        Output:
            ``SurrogatePrediction``.
        """

        x = _as_2d_array(candidates, dim=self.dim, name="candidates")
        if self._is_fit and self._model is not None:
            mean, std = self._model.predict(x, return_std=True)
            std = np.maximum(np.asarray(std, dtype=float), 1e-12)
            return SurrogatePrediction(
                mean=np.asarray(mean, dtype=float).tolist(),
                std=std.tolist(),
                metadata={
                    "model": "gaussian_process",
                    "is_fit": True,
                    "kernel": str(self._model.kernel_),
                    "lengthscales": _safe_lengthscales(self._model),
                },
            )

        fallback_mean = 0.0
        fallback_std = 1.0
        if self._y_train is not None and len(self._y_train):
            fallback_mean = float(np.mean(self._y_train))
            fallback_std = float(max(np.std(self._y_train), 1.0))

        return SurrogatePrediction(
            mean=[fallback_mean] * len(x),
            std=[fallback_std] * len(x),
            metadata={"model": "gaussian_process", "is_fit": False},
        )

    def lengthscales(self) -> list[float] | None:
        """Return fitted Matern lengthscales when available."""

        if not self._is_fit or self._model is None:
            return None
        return _safe_lengthscales(self._model)


def make_surrogate(kind: str, dim: int, options: Mapping[str, object] | None = None) -> SurrogateModel:
    """Construct a surrogate model by name.

    Inputs:
        kind: Surrogate type name, for example ``gaussian_process``.
        dim: Input dimensionality.
        options: Optional model-specific settings.

    Output:
        Object implementing ``SurrogateModel``.
    """

    opts = dict(options or {})
    normalised_kind = kind.strip().lower().replace("-", "_")
    if normalised_kind in {"gp", "gaussian_process", "gaussianprocess"}:
        return GaussianProcessSurrogate(
            dim=dim,
            noise_level=float(opts.get("noise_level", 1e-6)),
            seed=opts.get("seed") if isinstance(opts.get("seed"), int) else None,
        )
    raise ValueError(f"Unknown surrogate kind: {kind}")


def _as_1d_array(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _as_2d_array(values: Matrix, dim: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional.")
    if array.shape[1] != dim:
        raise ValueError(f"{name} must have shape (n, {dim}); got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _safe_lengthscales(model: GaussianProcessRegressor) -> list[float] | None:
    try:
        length_scale = model.kernel_.k1.k2.length_scale
    except Exception:
        return None
    return np.asarray(length_scale, dtype=float).reshape(-1).tolist()
