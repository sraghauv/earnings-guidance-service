"""Anthropic backend for guidance extraction.

Uses `create` with `output_config` set to the caller's JSON schema, not
`messages.parse`. `messages.parse` runs `GuidanceExtraction`'s pydantic
validators (the `no_guidance_given`/`low<=high` checks) as part of parsing the
response, and raises straight out of the call if they fail -- which throws
away the raw text before this provider ever sees it. Since `complete_json`'s
whole job is to hand back raw text and let `extract.pipeline` decide what to
do with validation failures, that's the wrong tool here even though it reads
nicer for the happy path.

Requests go through the beta namespace so they can carry `fallbacks`, which
re-runs a request the safety classifiers decline on another model server-side
instead of handing back the refusal. `"default"` routes by refusal category
rather than pinning a substitute model, so there's no list to maintain when a
pinned model is retired. Not available on the Batches API, which is why
`extract.batch_runner` doesn't set it.
"""

import anthropic

from egs.config import get_settings
from egs.llm.base import Completion, LLMProvider, ProviderError

MAX_TOKENS = 16000

# Gates the `fallbacks: "default"` scalar form specifically. The array form
# (`fallbacks=[{"model": ...}]`) is gated by -2026-06-01 instead, and pairing
# either header with the other form is a 400 -- so this string and the value
# below have to move together.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def _rejects_fallbacks(error: anthropic.BadRequestError) -> bool:
    """Whether a 400 is about the fallback beta rather than our request.

    Deliberately narrow: a malformed schema or a bad message list also returns
    400, and retrying those unchanged would just burn a second request and
    bury the real error.
    """
    text = str(error).lower()
    return "fallback" in text or FALLBACK_BETA in text


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
        # Latches off if the API rejects the fallback beta, so one rejection
        # costs one wasted request rather than one per call for the whole run.
        self._fallbacks_enabled = True

    def _create(self, system: str, messages: list[dict], schema: dict):
        """One request, with the fallback beta unless it has been rejected.

        The beta namespace is required -- `fallbacks` isn't a parameter on
        plain `client.messages.create`.
        """
        extra = (
            {"betas": [FALLBACK_BETA], "fallbacks": "default"}
            if self._fallbacks_enabled
            else {}
        )
        return self.client.beta.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"format": {"type": "json_schema", "schema": schema}},
            **extra,
        )

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
            resp = self._create(system, messages, schema)
        except anthropic.BadRequestError as e:
            if not self._fallbacks_enabled or not _rejects_fallbacks(e):
                raise ProviderError(f"anthropic rejected the request: {e}") from e
            # The beta isn't available to this key. Drop it and carry on
            # without the safety net rather than failing the extraction.
            self._fallbacks_enabled = False
            try:
                resp = self._create(system, messages, schema)
            except anthropic.APIStatusError as retry_error:
                raise ProviderError(
                    f"anthropic API error {retry_error.status_code}: {retry_error}"
                ) from retry_error
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
