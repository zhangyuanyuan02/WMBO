from wmbo.descriptors import describe_landscape, estimate_progress


def test_describe_landscape_exposes_world_model_features() -> None:
    x = [
        [0.0, 0.0],
        [0.0, 1.0],
        [0.25, 0.0],
        [0.25, 1.0],
        [0.5, 0.0],
        [0.5, 1.0],
        [0.75, 0.0],
        [0.75, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
    y = [(point[0] - 0.55) ** 2 + 0.02 * point[1] for point in x]

    descriptor = describe_landscape(x, y, surrogate_metadata={"mean_std": 0.1})
    data = descriptor.to_dict()

    for key in (
        "curvature",
        "anisotropy",
        "coverage",
        "boundary_bias",
        "stagnation",
        "improvement_rate",
        "dimension_sensitivity",
        "sensitive_dims",
    ):
        assert key in data

    labels = data["labels"]
    assert labels["curvature"] in {"low", "moderate", "high"}
    assert labels["anisotropy"] in {"low", "moderate", "high"}
    assert labels["coverage"] in {"low", "moderate", "high"}
    assert labels["progress"] in {"active", "slow", "stalled"}


def test_anisotropy_identifies_dominant_dimension() -> None:
    x = [
        [a, b]
        for a in (0.0, 0.25, 0.5, 0.75, 1.0)
        for b in (0.0, 0.5, 1.0)
    ]
    y = [point[0] ** 2 for point in x]

    descriptor = describe_landscape(x, y)

    assert descriptor.anisotropy is not None
    assert descriptor.anisotropy > 0.20
    assert 0 in descriptor.sensitive_dims
    assert descriptor.dimension_sensitivity[0] > descriptor.dimension_sensitivity[1]


def test_progress_marks_stalled_runs() -> None:
    stagnation, improvement_rate = estimate_progress([10.0, 6.0, 6.5, 6.4, 6.3, 6.2, 6.1])

    assert stagnation >= 0.75
    assert improvement_rate > 0.0


def test_empty_landscape_descriptor_is_serialisable() -> None:
    descriptor = describe_landscape([], [])
    data = descriptor.to_dict()

    assert data["num_observations"] == 0
    assert data["coverage"] == 0.0
    assert data["stagnation"] == 1.0
    assert data["dimension_sensitivity"] == []
    assert data["sensitive_dims"] == []
