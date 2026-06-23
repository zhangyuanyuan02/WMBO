"""Small utility function interfaces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist.

    Input:
        path: Directory path.

    Output:
        Normalised ``Path`` object for the directory.
    """

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object from disk.

    Input:
        path: JSON file path.

    Output:
        Parsed JSON dictionary.
    """

    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object at the top level.")
    return data


def write_json(path: str | Path, data: Any) -> None:
    """Write JSON data to disk.

    Inputs:
        path: Destination JSON file path.
        data: JSON-serialisable value.

    Output:
        None.
    """

    destination = Path(path)
    ensure_dir(destination.parent)
    with destination.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def parse_csv(value: str | None) -> list[str] | None:
    """Parse a comma-separated command-line argument.

    Input:
        value: Comma-separated string or ``None``.

    Output:
        List of stripped strings, or ``None`` when no value is provided.
    """

    if value is None or not value.strip():
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv(value: str | None) -> list[int] | None:
    """Parse a comma-separated integer argument.

    Input:
        value: Comma-separated integer string or ``None``.

    Output:
        List of integers, or ``None`` when no value is provided.
    """

    parsed = parse_csv(value)
    if parsed is None:
        return None
    return [int(item) for item in parsed]


def flatten(values: Sequence[Sequence[float]]) -> list[float]:
    """Flatten a nested sequence of floats.

    Input:
        values: Nested numeric sequence.

    Output:
        One-dimensional list of floats.
    """

    return [float(item) for row in values for item in row]
