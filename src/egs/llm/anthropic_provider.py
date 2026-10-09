"""Anthropic backend for guidance extraction.

Uses `messages.create` with `output_config` set to the caller's JSON schema,
not `messages.parse`. `messages.parse` runs `GuidanceExtraction`'s pydantic
validators (the `no_guidance_given`/`low<=high` checks) as part of parsing the
response, and raises straight out of the call if they fail -- which throws
away the raw text before this provider ever sees it. Since `complete_json`'s
whole job is to hand back raw text and let `extract.pipeline` decide what to
do with validation failures, that's the wrong tool here even though it reads
nicer for the happy path.
"""

import anthropic

from egs.config import get_settings
from egs.llm.base import Completion, LLMProvider, ProviderError

MAX_TOKENS = 16000


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(
        self,
        *,
        model: str | None = None,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        settings = get_settings()
        self.model = model or settings.anthropic_model
        self.client = client or anthropic.Anthropic(api_key=settings.anthropic_api_key)
        # Raw text of the last response, so a repair call can replay it as
        # conversation history instead of needing the caller to pass it in.
        self._last_raw: str | None = None

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        repair_hint: str | None = None,
    ) -> Completion:
        messages: list[dict] = [{"role": "user", "content": user}]
        if repair_hint is not None:
            if self._last_raw is not None:
                messages.append({"role": "assistant", "content": self._last_raw})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"That response failed schema validation:\n{repair_hint}\n\n"
                        "Return corrected JSON that satisfies the schema. No prose, "
                        "just the JSON."
                    ),
                }
            )

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=system,
                messages=messages,
                thinking={"type": "adaptive"},
                output_config={
                    "format": {"type": "json_schema", "schema": schema}
                },
            )
        except anthropic.NotFoundError as e:
            raise ProviderError(f"model not found: {self.model}") from e
        except anthropic.RateLimitError as e:
            raise ProviderError(f"rate limited by anthropic: {e}") from e
        except anthropic.APIStatusError as e:
            raise ProviderError(f"anthropic API error {e.status_code}: {e}") from e
        except anthropic.APIConnectionError as e:
            raise ProviderError(f"could not reach anthropic: {e}") from e

        # A declined request comes back as a normal 200 with stop_reason
        # "refusal" and an empty-ish content list, so this has to be checked
        # before reading content or it surfaces as a confusing "no text block".
        if resp.stop_reason == "refusal":
            category = getattr(resp.stop_details, "category", None)
            raise ProviderError(f"anthropic declined the request (category={category})")
        if resp.stop_reason == "max_tokens":
            raise ProviderError(
                f"response hit the {MAX_TOKENS} token cap and is truncated JSON"
            )

        text = next((b.text for b in resp.content if b.type == "text"), None)
        if text is None:
            raise ProviderError("anthropic response had no text block")

        self._last_raw = text
        return Completion(
            text=text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )
