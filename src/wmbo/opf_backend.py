"""Persistent Julia/PowerModels backend for OPF benchmark evaluations."""

from __future__ import annotations

import atexit
from collections import deque
from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import threading
from typing import Any, Mapping, Protocol, Sequence


class OPFBackendError(RuntimeError):
    """Raised when the Julia OPF backend cannot serve a request."""


class OPFBackend(Protocol):
    """Minimal backend contract used by benchmark evaluation and tests."""

    def request(self, payload: Mapping[str, object], *, timeout: float | None = None) -> Mapping[str, object]:
        """Send one protocol request and return its result mapping."""


@dataclass(frozen=True)
class OPFBackendConfig:
    """Runtime options for the persistent Julia subprocess."""

    julia_executable: str = "julia"
    timeout_seconds: float = 30.0
    startup_timeout_seconds: float = 180.0
    warmup: bool = True
    project_dir: str | None = None
    server_script: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, object] | None = None) -> "OPFBackendConfig":
        """Build a validated configuration from YAML-style values."""

        options = dict(value or {})
        project_root = Path(__file__).resolve().parents[2]
        project_dir = Path(str(options.get("project_dir") or project_root / "julia")).resolve()
        server_script = Path(str(options.get("server_script") or project_dir / "opf_server.jl")).resolve()
        timeout = float(options.get("timeout_seconds", options.get("timeout", 30.0)))
        startup_timeout = float(options.get("startup_timeout_seconds", 180.0))
        if timeout <= 0.0 or startup_timeout <= 0.0:
            raise ValueError("OPF backend timeouts must be positive.")
        return cls(
            julia_executable=str(options.get("julia_executable", "julia")),
            timeout_seconds=timeout,
            startup_timeout_seconds=startup_timeout,
            warmup=_truthy(options.get("warmup", True)),
            project_dir=str(project_dir),
            server_script=str(server_script),
        )


class PersistentJuliaOPFBackend:
    """Line-delimited JSON client backed by one long-lived Julia process."""

    def __init__(self, config: OPFBackendConfig) -> None:
        self.config = config
        self._process: subprocess.Popen[str] | None = None
        self._responses: queue.Queue[str | None] = queue.Queue()
        self._stderr_tail: deque[str] = deque(maxlen=80)
        self._lock = threading.RLock()
        self._request_id = 0

    def request(self, payload: Mapping[str, object], *, timeout: float | None = None) -> Mapping[str, object]:
        """Send one request, restarting once after an unexpected process exit."""

        with self._lock:
            last_error: Exception | None = None
            for attempt in range(2):
                try:
                    self._ensure_started()
                    return self._exchange(payload, timeout=timeout or self.config.timeout_seconds)
                except (BrokenPipeError, EOFError, OSError) as exc:
                    last_error = exc
                    self._terminate()
                    if attempt == 0:
                        continue
            raise OPFBackendError(f"Julia OPF backend stopped unexpectedly: {last_error}") from last_error

    def close(self) -> None:
        """Request a graceful shutdown and release subprocess resources."""

        with self._lock:
            if self._process is None:
                return
            try:
                if self._process.poll() is None:
                    self._exchange({"action": "shutdown"}, timeout=5.0)
            except Exception:
                pass
            self._terminate()

    def _ensure_started(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        self._terminate()
        julia = _resolve_executable(self.config.julia_executable)
        project_dir = Path(str(self.config.project_dir))
        server_script = Path(str(self.config.server_script))
        if not project_dir.is_dir():
            raise OPFBackendError(f"Julia project directory does not exist: {project_dir}")
        if not server_script.is_file():
            raise OPFBackendError(f"Julia OPF server script does not exist: {server_script}")

        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self._responses = queue.Queue()
        self._stderr_tail.clear()
        try:
            self._process = subprocess.Popen(
                [
                    julia,
                    "--startup-file=no",
                    f"--project={project_dir}",
                    str(server_script),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
        except OSError as exc:
            raise OPFBackendError(f"Failed to start Julia OPF backend with {julia!r}: {exc}") from exc
        threading.Thread(target=self._read_stdout, name="wmbo-opf-stdout", daemon=True).start()
        threading.Thread(target=self._read_stderr, name="wmbo-opf-stderr", daemon=True).start()
        self._exchange(
            {"action": "describe", "benchmark": "opf_pglib_case14_typ_pgvg"},
            timeout=self.config.startup_timeout_seconds,
        )
        if self.config.warmup:
            self._exchange({"action": "warmup"}, timeout=self.config.startup_timeout_seconds)

    def _exchange(self, payload: Mapping[str, object], *, timeout: float) -> Mapping[str, object]:
        process = self._process
        if process is None or process.poll() is not None or process.stdin is None:
            raise EOFError("Julia OPF backend is not running.")
        self._request_id += 1
        request_id = self._request_id
        request = {**dict(payload), "request_id": request_id}
        process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        process.stdin.flush()
        try:
            line = self._responses.get(timeout=timeout)
        except queue.Empty as exc:
            stderr = self._stderr_summary()
            self._terminate()
            raise OPFBackendError(
                f"Julia OPF backend timed out after {timeout:.1f}s."
                + (f" stderr: {stderr}" if stderr else "")
            ) from exc
        if line is None:
            stderr = self._stderr_summary()
            raise EOFError("Julia OPF backend closed its output." + (f" stderr: {stderr}" if stderr else ""))
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise OPFBackendError(f"Julia OPF backend returned invalid JSON: {line[:500]}") from exc
        if response.get("request_id") != request_id:
            raise OPFBackendError(
                f"Julia OPF response id mismatch: expected {request_id}, got {response.get('request_id')}"
            )
        if not response.get("ok"):
            raise OPFBackendError(str(response.get("error", "unknown Julia OPF backend error")))
        result = response.get("result")
        if not isinstance(result, Mapping):
            raise OPFBackendError("Julia OPF backend response did not contain a result mapping.")
        return dict(result)

    def _read_stdout(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            for line in process.stdout:
                self._responses.put(line.rstrip("\r\n"))
        finally:
            self._responses.put(None)

    def _read_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        for line in process.stderr:
            self._stderr_tail.append(line.rstrip())

    def _stderr_summary(self) -> str:
        return " | ".join(line for line in self._stderr_tail if line)[-2000:]

    def _terminate(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass


_BACKENDS: dict[OPFBackendConfig, PersistentJuliaOPFBackend] = {}
_BACKENDS_LOCK = threading.Lock()


def evaluate_opf(
    benchmark_name: str,
    control_values: Sequence[float],
    options: Mapping[str, object] | None = None,
) -> tuple[float, dict[str, object]]:
    """Evaluate one fixed-``Pg+Vg`` AC power-flow candidate and scalarise its diagnostics."""

    evaluation_options = dict(options or {})
    backend_options = _mapping(evaluation_options.get("backend", evaluation_options))
    backend = evaluation_options.get("_backend")
    if backend is None:
        backend = get_opf_backend(OPFBackendConfig.from_mapping(backend_options))
    if not hasattr(backend, "request"):
        raise TypeError("evaluation.opf._backend must implement request(payload, timeout=...).")

    penalty_weight = float(evaluation_options.get("penalty_weight", 100.0))
    failure_penalty = float(evaluation_options.get("failure_penalty", 1.0e6))
    feasibility_tolerance = float(evaluation_options.get("feasibility_tolerance", 1.0e-5))
    constraint_failure_ratio = float(evaluation_options.get("constraint_failure_ratio", 1.0e6))
    if (
        penalty_weight < 0.0
        or failure_penalty <= 0.0
        or feasibility_tolerance <= 0.0
        or constraint_failure_ratio <= 1.0
    ):
        raise ValueError("Invalid OPF scalarisation options.")

    result = dict(
        backend.request(
            {
                "action": "evaluate",
                "benchmark": benchmark_name,
                "control_values": [float(value) for value in control_values],
                "feasibility_tolerance": feasibility_tolerance,
            }
        )
    )
    converged = bool(result.get("pf_converged", False))
    generation_cost = _optional_float(result.get("generation_cost"))
    reference_cost = _optional_float(result.get("reference_cost"))
    total_violation = max(0.0, float(result.get("total_violation", 1.0)))
    max_normalized_violation = max(0.0, float(result.get("max_normalized_violation", 1.0)))
    if not converged or generation_cost is None or reference_cost is None:
        score = failure_penalty
        legacy_penalised_score = failure_penalty
        normalised_cost_gap = None
        constraint_ratio = constraint_failure_ratio
        constraint_excess = constraint_failure_ratio - 1.0
    else:
        normalised_cost_gap = max(0.0, (generation_cost - reference_cost) / max(abs(reference_cost), 1.0))
        constraint_ratio = max_normalized_violation / feasibility_tolerance
        constraint_excess = max(0.0, constraint_ratio - 1.0)
        legacy_penalised_score = normalised_cost_gap + penalty_weight * total_violation
        score = normalised_cost_gap + penalty_weight * constraint_excess * constraint_excess
    result.update(
        {
            "normalised_cost_gap": normalised_cost_gap,
            "penalty_weight": penalty_weight,
            "failure_penalty": failure_penalty,
            "feasibility_tolerance": feasibility_tolerance,
            "constraint_ratio": constraint_ratio,
            "constraint_excess": constraint_excess,
            "legacy_penalised_score": legacy_penalised_score,
            "score_kind": "normalised_cost_gap_plus_tolerance_scaled_excess_penalty",
        }
    )
    return float(score), result


def get_opf_backend(config: OPFBackendConfig) -> PersistentJuliaOPFBackend:
    """Return a process singleton for an exact backend configuration."""

    with _BACKENDS_LOCK:
        backend = _BACKENDS.get(config)
        if backend is None:
            backend = PersistentJuliaOPFBackend(config)
            _BACKENDS[config] = backend
        return backend


def close_opf_backends() -> None:
    """Close all cached Julia subprocesses."""

    with _BACKENDS_LOCK:
        backends = list(_BACKENDS.values())
        _BACKENDS.clear()
    for backend in backends:
        backend.close()


def _resolve_executable(value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.parent != Path(".") or candidate.is_absolute():
        if candidate.is_file():
            return str(candidate.resolve())
        raise OPFBackendError(f"Julia executable does not exist: {candidate}")
    resolved = shutil.which(value)
    if resolved:
        return resolved
    raise OPFBackendError(
        f"Julia executable {value!r} was not found. Install Julia 1.10 with juliaup "
        "or set evaluation.opf.backend.julia_executable."
    )


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


atexit.register(close_opf_backends)


__all__ = [
    "OPFBackend",
    "OPFBackendConfig",
    "OPFBackendError",
    "PersistentJuliaOPFBackend",
    "evaluate_opf",
    "get_opf_backend",
    "close_opf_backends",
]
