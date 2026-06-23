"""Surrogate-model interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

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
    """Gaussian-process surrogate interface."""

    def __init__(self, dim: int, noise_level: float = 1e-6, seed: int | None = None) -> None:
        """Create a Gaussian-process surrogate placeholder.

        Inputs:
            dim: Input dimensionality.
            noise_level: Observation-noise level.
            seed: Random seed.

        Output:
            Surrogate instance.
        """

        self.dim = dim
        self.noise_level = noise_level
        self.seed = seed

    def fit(self, data: SurrogateDataset) -> "GaussianProcessSurrogate":
        """Fit the surrogate on observed data.

        Input:
            data: Training data.

        Output:
            Fitted surrogate instance.
        """

        raise NotImplementedError("Gaussian-process fitting is not implemented yet.")

    def predict(self, candidates: Matrix) -> SurrogatePrediction:
        """Predict mean and uncertainty for candidates.

        Input:
            candidates: Candidate points to evaluate with the surrogate.

        Output:
            ``SurrogatePrediction``.
        """

        raise NotImplementedError("Gaussian-process prediction is not implemented yet.")


def make_surrogate(kind: str, dim: int, options: Mapping[str, object] | None = None) -> SurrogateModel:
    """Construct a surrogate model by name.

    Inputs:
        kind: Surrogate type name, for example ``gaussian_process``.
        dim: Input dimensionality.
        options: Optional model-specific settings.

    Output:
        Object implementing ``SurrogateModel``.
    """

    raise NotImplementedError("Surrogate factory is not implemented yet.")
