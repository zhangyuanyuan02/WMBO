"""Interfaces for optional local Hugging Face LLM support."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .llm_api import LLMClient, LLMRequest, LLMResponse


@dataclass(frozen=True)
class LocalHFConfig:
    """Inputs needed to load a local Hugging Face model.

    Inputs:
        model_path: Local model path or model identifier.
        device: Runtime device, for example ``cpu`` or ``cuda``.
        torch_dtype: Preferred tensor dtype.
        max_new_tokens: Maximum generated tokens per call.
        options: Extra backend options.

    Output:
        Passed to ``make_local_hf_client``.
    """

    model_path: str
    device: str = "auto"
    torch_dtype: str = "auto"
    max_new_tokens: int = 512
    options: Mapping[str, object] = field(default_factory=dict)


class LocalHFClient:
    """Placeholder client for local Hugging Face generation."""

    def __init__(self, config: LocalHFConfig) -> None:
        """Create a local model client.

        Input:
            config: Local model loading and generation settings.

        Output:
            A client instance ready for future generation calls.
        """

        self.config = config

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate text with a local model.

        Input:
            request: Prompt and decoding options.

        Output:
            ``LLMResponse`` containing generated text.
        """

        raise NotImplementedError("Local Hugging Face generation is not implemented yet.")


def make_local_hf_client(config: LocalHFConfig) -> LLMClient:
    """Create a local Hugging Face LLM client.

    Input:
        config: Local model loading settings.

    Output:
        Object implementing ``LLMClient``.
    """

    raise NotImplementedError("Local Hugging Face client creation is not implemented yet.")
