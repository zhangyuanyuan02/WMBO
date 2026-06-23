"""Interfaces for optional remote LLM reasoning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol


@dataclass(frozen=True)
class LLMRequest:
    """Input sent to an external language model.

    Inputs:
        prompt: Full model prompt.
        system_message: Optional system instruction.
        temperature: Sampling temperature.
        max_tokens: Maximum generated tokens.
        metadata: Optional request metadata.

    Output:
        Passed to an ``LLMClient`` implementation.
    """

    prompt: str
    system_message: str | None = None
    temperature: float = 0.0
    max_tokens: int = 512
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMResponse:
    """Output returned by an external language model.

    Inputs:
        text: Raw generated text.
        model_name: Name of the model that produced the response.
        metadata: Optional provider-specific metadata.

    Output:
        Parsed into a structured optimisation decision.
    """

    text: str
    model_name: str
    metadata: Mapping[str, object] = field(default_factory=dict)


class LLMClient(Protocol):
    """Protocol implemented by remote or local LLM clients."""

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate text from a language model request.

        Input:
            request: Prompt, decoding options, and metadata.

        Output:
            ``LLMResponse`` containing generated text.
        """

        ...


def build_reasoning_prompt(context: Mapping[str, object]) -> str:
    """Build an optimisation-reasoning prompt.

    Input:
        context: Observations, descriptors, budget, and candidate information.

    Output:
        Prompt string to send to an LLM.
    """

    raise NotImplementedError("LLM prompt construction is not implemented yet.")


def parse_reasoning_response(response: LLMResponse) -> dict[str, object]:
    """Parse an LLM response into structured optimisation fields.

    Input:
        response: Raw LLM response.

    Output:
        Dictionary containing strategy, hypothesis, confidence, and rationale.
    """

    raise NotImplementedError("LLM response parsing is not implemented yet.")


def make_llm_client(provider: str, config: Mapping[str, object]) -> LLMClient:
    """Create an LLM client from provider settings.

    Inputs:
        provider: Provider name.
        config: Provider configuration such as model name and credentials.

    Output:
        Object implementing ``LLMClient``.
    """

    raise NotImplementedError("Remote LLM clients are not implemented yet.")
