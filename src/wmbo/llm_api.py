"""OpenAI-compatible LLM client and WMBO decision parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Protocol, Sequence

import requests

from .agents import ReasoningDecision
from .descriptors import LandscapeDescriptor

Message = Mapping[str, str]
API_PROVIDER_ENV = "WMBO_API_PROVIDER"

WORLD_MODEL_LABELS: dict[str, set[str]] = {
    "smoothness": {"smooth", "mixed", "rugged", "unknown"},
    "modality": {"mostly_unimodal", "multimodal", "highly_multimodal", "unknown"},
    "curvature": {"low", "moderate", "high", "unknown"},
    "anisotropy": {"low", "moderate", "high", "unknown"},
}


class LLMAPIError(RuntimeError):
    """Raised when a remote LLM call or response parse fails."""


def _candidate_secret_paths() -> list[Path]:
    project_root = Path(__file__).resolve().parents[2]
    return [
        Path.cwd() / "configs" / "llm_secrets.yaml",
        project_root / "configs" / "llm_secrets.yaml",
    ]


def resolve_provider_config(data: Mapping[str, Any], provider: str | None = None) -> dict[str, Any]:
    """Resolve one API provider section from a local secrets mapping."""

    providers = data.get("providers")
    selected = provider or os.getenv(API_PROVIDER_ENV) or data.get("active_provider")
    if isinstance(providers, Mapping):
        if not selected:
            raise LLMAPIError(f"Select an API provider with active_provider or {API_PROVIDER_ENV}.")
        section = providers.get(str(selected))
        if not isinstance(section, Mapping):
            available = ", ".join(sorted(str(name) for name in providers))
            raise LLMAPIError(f"Unknown API provider {selected!r}. Available providers: {available}.")
    elif selected and isinstance(data.get(str(selected)), Mapping):
        section = data[str(selected)]
    elif selected and isinstance(data.get(str(selected).lower()), Mapping):
        section = data[str(selected).lower()]
    elif selected:
        lower_to_key = {str(key).lower(): key for key in data}
        matched_key = lower_to_key.get(str(selected).lower())
        if matched_key is not None and isinstance(data.get(matched_key), Mapping):
            section = data[matched_key]
        else:
            section = data.get("openai") or data.get("llm") or data.get("llm_agent") or data
    else:
        section = data.get("openai") or data.get("llm") or data.get("llm_agent") or data

    if not isinstance(section, Mapping):
        raise LLMAPIError("Local LLM config section must be a YAML object.")

    config = dict(section)
    if "url" in config and "base_url" not in config:
        config["base_url"] = config["url"]
    if "default_model" in config and "model" not in config:
        config["model"] = config["default_model"]
    return config


def load_local_llm_config(provider: str | None = None) -> dict[str, Any]:
    """Load optional local LLM credentials from configs/llm_secrets.yaml."""

    path = next((candidate for candidate in _candidate_secret_paths() if candidate.exists()), None)
    if path is None:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise LLMAPIError("Reading configs/llm_secrets.yaml requires PyYAML.") from exc

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, Mapping):
        raise LLMAPIError(f"Local LLM config must be a YAML object: {path}")
    return resolve_provider_config(data, provider=provider)


@dataclass(frozen=True)
class LLMRequest:
    """Provider-agnostic generation request kept for local backend compatibility."""

    prompt: str
    system_message: str | None = None
    temperature: float = 0.0
    max_tokens: int = 512
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMResponse:
    """Provider-agnostic generation response kept for local backend compatibility."""

    text: str
    model_name: str
    metadata: Mapping[str, object] = field(default_factory=dict)


class LLMClient(Protocol):
    """Protocol implemented by remote or local text-generation clients."""

    def generate(self, request: LLMRequest) -> LLMResponse:
        ...


@dataclass(frozen=True)
class LLMClientConfig:
    """Configuration for an OpenAI-compatible chat-completions endpoint."""

    api_key: str | None = None
    base_url: str = "https://api.openai.com/v1"
    default_model: str = "gpt-4o-mini"
    timeout: float = 120.0
    max_retries: int = 3
    retry_delay_seconds: float = 2.0
    request_delay_seconds: float = 0.0
    organization: str | None = None
    project: str | None = None

    @classmethod
    def from_env(cls) -> "LLMClientConfig":
        """Build a client config from environment variables."""

        local = load_local_llm_config()
        return cls(
            api_key=os.getenv("OPENAI_API_KEY") or local.get("api_key"),
            base_url=os.getenv("OPENAI_BASE_URL") or str(local.get("base_url", "https://api.openai.com/v1")),
            default_model=os.getenv("OPENAI_MODEL") or str(local.get("model", "gpt-4o-mini")),
            timeout=float(os.getenv("OPENAI_TIMEOUT") or local.get("timeout", 120)),
            max_retries=int(os.getenv("OPENAI_MAX_RETRIES") or local.get("max_retries", 3)),
            retry_delay_seconds=float(os.getenv("OPENAI_RETRY_DELAY_SECONDS") or local.get("retry_delay_seconds", 2)),
            request_delay_seconds=float(os.getenv("OPENAI_REQUEST_DELAY_SECONDS") or local.get("request_delay_seconds", 0)),
            organization=os.getenv("OPENAI_ORG_ID") or local.get("organization"),
            project=os.getenv("OPENAI_PROJECT_ID") or local.get("project"),
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "LLMClientConfig":
        """Build a client config from direct optimiser options."""

        provider = data.get("api_provider") or data.get("provider")
        local = load_local_llm_config(provider=str(provider)) if provider else load_local_llm_config()
        merged = {**local, **dict(data)}

        key_env = merged.get("api_key_env") or merged.get("key_env")
        api_key = merged.get("api_key")
        if not api_key and key_env:
            key_env_text = str(key_env)
            api_key = os.getenv(key_env_text)
            if not api_key and key_env_text.startswith("sk-"):
                api_key = key_env_text
        return cls(
            api_key=str(api_key) if api_key else os.getenv("OPENAI_API_KEY"),
            base_url=str(merged.get("base_url") or merged.get("api_base_url") or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")),
            default_model=str(merged.get("model") or merged.get("api_model") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
            timeout=float(merged.get("timeout", os.getenv("OPENAI_TIMEOUT", "120"))),
            max_retries=int(merged.get("max_retries", os.getenv("OPENAI_MAX_RETRIES", "3"))),
            retry_delay_seconds=float(merged.get("retry_delay_seconds", os.getenv("OPENAI_RETRY_DELAY_SECONDS", "2"))),
            request_delay_seconds=float(merged.get("request_delay_seconds", os.getenv("OPENAI_REQUEST_DELAY_SECONDS", "0"))),
            organization=str(merged.get("organization") or os.getenv("OPENAI_ORG_ID") or "") or None,
            project=str(merged.get("project") or os.getenv("OPENAI_PROJECT_ID") or "") or None,
        )


class _ChatCompletionsResource:
    def __init__(self, client: "OpenAIStyleClient") -> None:
        self._client = client

    def create(
        self,
        *,
        messages: Sequence[Message],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: Mapping[str, Any] | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model or self._client.config.default_model,
            "messages": [dict(message) for message in messages],
            "temperature": float(temperature),
        }
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        if response_format is not None:
            payload["response_format"] = dict(response_format)
        payload.update(extra)
        return self._client.post_json("/chat/completions", payload)


class _ChatResource:
    def __init__(self, client: "OpenAIStyleClient") -> None:
        self.completions = _ChatCompletionsResource(client)


class OpenAIStyleClient:
    """Small HTTP client for OpenAI-compatible chat-completions APIs."""

    def __init__(
        self,
        config: LLMClientConfig | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        retry_delay_seconds: float | None = None,
        request_delay_seconds: float | None = None,
        organization: str | None = None,
        project: str | None = None,
    ) -> None:
        cfg = config or LLMClientConfig.from_env()
        self.config = LLMClientConfig(
            api_key=api_key if api_key is not None else cfg.api_key,
            base_url=base_url if base_url is not None else cfg.base_url,
            default_model=default_model if default_model is not None else cfg.default_model,
            timeout=timeout if timeout is not None else cfg.timeout,
            max_retries=max_retries if max_retries is not None else cfg.max_retries,
            retry_delay_seconds=retry_delay_seconds if retry_delay_seconds is not None else cfg.retry_delay_seconds,
            request_delay_seconds=request_delay_seconds if request_delay_seconds is not None else cfg.request_delay_seconds,
            organization=organization if organization is not None else cfg.organization,
            project=project if project is not None else cfg.project,
        )
        self.chat = _ChatResource(self)

    @classmethod
    def from_env(cls) -> "OpenAIStyleClient":
        return cls(LLMClientConfig.from_env())

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "OpenAIStyleClient":
        return cls(LLMClientConfig.from_mapping(data))

    def post_json(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        """POST JSON to the configured API endpoint with retry handling."""

        if not self.config.api_key:
            raise LLMAPIError("Missing API key. Set OPENAI_API_KEY or pass api_key/API key env in options.")

        url = self.config.base_url.rstrip("/") + "/" + path.lstrip("/")
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.config.organization:
            headers["OpenAI-Organization"] = self.config.organization
        if self.config.project:
            headers["OpenAI-Project"] = self.config.project

        body = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
        attempts = max(1, int(self.config.max_retries) + 1)
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                response = requests.post(url, data=body, headers=headers, timeout=self.config.timeout)
                if response.ok:
                    if self.config.request_delay_seconds > 0:
                        time.sleep(float(self.config.request_delay_seconds))
                    try:
                        return response.json()
                    except requests.exceptions.JSONDecodeError as exc:
                        preview = response.text[:500].replace("\n", "\\n")
                        raise LLMAPIError(f"LLM API returned non-JSON response: {preview!r}") from exc

                retryable = response.status_code == 429 or response.status_code >= 500
                if not retryable or attempt == attempts:
                    raise LLMAPIError(
                        f"LLM API request failed with HTTP {response.status_code} at {url}: {response.text[:1000]}"
                    )
                retry_after = response.headers.get("Retry-After")
                delay = _parse_retry_after(retry_after, fallback=self.config.retry_delay_seconds * attempt)
                last_error = LLMAPIError(f"HTTP {response.status_code}: {response.text[:500]}")
                time.sleep(delay)
            except requests.exceptions.RequestException as exc:
                last_error = exc
                if attempt == attempts:
                    raise LLMAPIError(f"LLM API request failed after {attempts} attempt(s): {exc}") from exc
                time.sleep(float(self.config.retry_delay_seconds) * attempt)

        raise LLMAPIError(f"LLM API request failed: {last_error}")


def _parse_retry_after(value: str | None, fallback: float) -> float:
    if value is None:
        return max(0.0, float(fallback))
    try:
        return max(0.0, float(value))
    except ValueError:
        return max(0.0, float(fallback))


def extract_message_text(response: Mapping[str, Any]) -> str:
    """Extract assistant content from a chat-completions-style response."""

    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMAPIError("Response does not contain choices.")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise LLMAPIError("Response choice must be an object.")
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise LLMAPIError("Response choice does not contain message.")
    content = message.get("content", "")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, Mapping):
                parts.append(str(part.get("text") or part.get("content") or ""))
        text = "".join(parts)
    else:
        text = str(content)
    if not text.strip():
        raise LLMAPIError("LLM returned empty assistant content.")
    return text


def inspect_reasoning_response(response: Mapping[str, Any], assistant_text: str = "") -> tuple[bool, int | None]:
    """Return whether hidden/reasoning fields appear in a provider response."""

    has_reasoning = "<think>" in assistant_text.lower()
    reasoning_tokens: int | None = None
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, Mapping):
            message = choice.get("message")
            if isinstance(message, Mapping):
                for key in ("reasoning_content", "reasoning", "thinking"):
                    if message.get(key):
                        has_reasoning = True
    usage = response.get("usage")
    if isinstance(usage, Mapping):
        for details_key in ("completion_tokens_details", "output_tokens_details"):
            details = usage.get(details_key)
            if isinstance(details, Mapping):
                value = details.get("reasoning_tokens")
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    reasoning_tokens = int(value)
                    has_reasoning = has_reasoning or reasoning_tokens > 0
                    break
    return has_reasoning, reasoning_tokens


def build_world_model_messages(
    desc: LandscapeDescriptor,
    candidates: Sequence[Mapping[str, Any]] | None = None,
    decision_context: Mapping[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Build messages asking an LLM for a structured WMBO decision."""

    system = (
        "You are a world-model black-box optimisation agent. "
        "The objective is minimisation under a small evaluation budget. "
        "Choose exactly one strategy from: global_diverse, explore_ucb, exploit_ei, trust_region. "
        "Return strict JSON only with keys: world_model, strategy, hypothesis, confidence, rationale, selected_candidate_id. "
        "world_model must contain categorical fields: smoothness, modality, curvature, anisotropy. "
        "Allowed smoothness: smooth, mixed, rugged, unknown. "
        "Allowed modality: mostly_unimodal, multimodal, highly_multimodal, unknown. "
        "Allowed curvature/anisotropy: low, moderate, high, unknown. "
        "confidence must be numeric in [0, 1]. "
        "When candidate_options are provided, selected_candidate_id must be one of their candidate_id values. "
        "Never invent coordinates; select an existing candidate by ID."
    )
    payload = {
        "landscape_descriptor": desc.to_dict(),
        "objective": "minimise the unknown black-box function with as few evaluations as possible",
        "candidate_options": list(candidates or []),
        "decision_context": dict(decision_context or {}),
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def parse_reasoning_decision(text: str) -> ReasoningDecision:
    """Parse assistant JSON into a ``ReasoningDecision``."""

    data = _extract_json_object(text)
    strategy = str(data.get("strategy", "")).strip().lower().replace("-", "_")
    allowed = {"global_diverse", "explore_ucb", "exploit_ei", "trust_region"}
    if strategy not in allowed:
        raise LLMAPIError(f"Unsupported strategy from LLM: {strategy!r}.")
    confidence = _parse_confidence(data.get("confidence", 0.0))
    return ReasoningDecision(
        world_model=normalise_world_model(data.get("world_model")),
        strategy=strategy,
        hypothesis=str(data.get("hypothesis", "")).strip(),
        confidence=float(min(max(confidence, 0.0), 1.0)),
        rationale=str(data.get("rationale", "")).strip(),
        selected_candidate_id=(str(data["selected_candidate_id"]) if data.get("selected_candidate_id") is not None else None),
        metadata={"source": "llm"},
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = text.strip()
    if not raw:
        raise LLMAPIError("LLM returned empty assistant content.")
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    if not raw.startswith("{"):
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start : end + 1]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        preview = raw[:240].replace("\n", "\\n")
        raise LLMAPIError(f"LLM assistant content is not valid JSON: {preview!r}") from exc
    if not isinstance(data, dict):
        raise LLMAPIError("LLM response JSON must be an object.")
    return data


def _parse_confidence(value: Any) -> float:
    if isinstance(value, bool):
        raise LLMAPIError("LLM confidence must be numeric, not boolean.")
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        labels = {"low": 0.25, "medium": 0.5, "moderate": 0.5, "high": 0.75, "very_high": 0.9}
        key = value.strip().lower().replace(" ", "_").replace("-", "_")
        if key in labels:
            parsed = labels[key]
        else:
            try:
                parsed = float(key)
            except ValueError as exc:
                raise LLMAPIError(f"LLM confidence must be numeric, got {value!r}.") from exc
    else:
        raise LLMAPIError(f"LLM confidence must be numeric, got {type(value).__name__}.")
    if not math.isfinite(parsed):
        raise LLMAPIError("LLM confidence must be finite.")
    return parsed


def normalise_world_model(value: Any) -> dict[str, str]:
    """Validate or repair the LLM world-model labels."""

    if not isinstance(value, Mapping):
        raise LLMAPIError("LLM response world_model must be an object.")
    result: dict[str, str] = {}
    for key, allowed in WORLD_MODEL_LABELS.items():
        if key not in value:
            result[key] = "unknown"
            continue
        raw = value[key]
        if isinstance(raw, bool):
            raise LLMAPIError(f"world_model.{key} must be a categorical label.")
        if isinstance(raw, (int, float)):
            result[key] = _numeric_world_model_label(key, float(raw))
            continue
        label = str(raw).strip().lower().replace(" ", "_").replace("-", "_")
        if label not in allowed:
            choices = ", ".join(sorted(allowed))
            raise LLMAPIError(f"Unsupported world_model.{key} value {raw!r}; expected one of {choices}.")
        result[key] = label
    return result


def _numeric_world_model_label(key: str, score: float) -> str:
    if not math.isfinite(score):
        return "unknown"
    score = min(max(score, 0.0), 1.0)
    if key == "smoothness":
        if score >= 0.55:
            return "rugged"
        return "mixed" if score >= 0.25 else "smooth"
    if key == "modality":
        if score >= 0.65:
            return "highly_multimodal"
        return "multimodal" if score >= 0.25 else "mostly_unimodal"
    if key in {"curvature", "anisotropy"}:
        if score >= 0.65:
            return "high"
        return "moderate" if score >= 0.25 else "low"
    return "unknown"


def decide_with_llm(
    desc: LandscapeDescriptor,
    client: OpenAIStyleClient | None = None,
    *,
    candidates: Sequence[Mapping[str, Any]] | None = None,
    decision_context: Mapping[str, Any] | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    log_io: bool = False,
) -> ReasoningDecision:
    """Call an OpenAI-compatible model and parse a WMBO decision."""

    client = client or OpenAIStyleClient.from_env()
    messages = build_world_model_messages(desc, candidates=candidates, decision_context=decision_context)
    if log_io:
        print("[LLM input]")
        print(json.dumps(messages, ensure_ascii=False, indent=2))
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=float(temperature),
        response_format={"type": "json_object"},
    )
    text = extract_message_text(response)
    has_reasoning, reasoning_tokens = inspect_reasoning_response(response, text)
    if log_io:
        token_text = str(reasoning_tokens) if reasoning_tokens is not None else "unknown"
        print(f"[LLM thinking] present={has_reasoning}, reasoning_tokens={token_text}")
        print("[LLM output]")
        print(text)
    return parse_reasoning_decision(text)


def available_api_providers() -> list[str]:
    """Return provider names configured in configs/llm_secrets.yaml."""

    path = next((candidate for candidate in _candidate_secret_paths() if candidate.exists()), None)
    if path is None:
        return []
    try:
        import yaml
    except ImportError as exc:
        raise LLMAPIError("Reading configs/llm_secrets.yaml requires PyYAML.") from exc
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, Mapping):
        return []
    providers = data.get("providers")
    if isinstance(providers, Mapping):
        return sorted(str(name) for name in providers)
    return []


__all__ = [
    "API_PROVIDER_ENV",
    "LLMAPIError",
    "LLMClient",
    "LLMClientConfig",
    "LLMRequest",
    "LLMResponse",
    "OpenAIStyleClient",
    "available_api_providers",
    "build_world_model_messages",
    "decide_with_llm",
    "extract_message_text",
    "inspect_reasoning_response",
    "load_local_llm_config",
    "normalise_world_model",
    "parse_reasoning_decision",
    "resolve_provider_config",
]
