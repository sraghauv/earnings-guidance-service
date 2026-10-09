"""Local Ollama backend for guidance extraction.

Ollama supports a JSON-schema `format` field on `/api/chat`, which is the
closest thing it has to Anthropic's `output_config`. There's no equivalent of
`messages.parse` -- we just POST and read `message.content` back as text.
"""

import httpx

from egs.config import get_settings
from egs.llm.base import Completion, LLMProvider, ProviderError

TIMEOUT = 300.0

# Ollama defaults num_ctx to 2048, which silently truncates both the prompt and
# the generation -- a full transcript runs about 12k tokens, so the default
# produces cut-off JSON that looks like a model failure rather than a config
# one. gemma3 handles far more than this; 32k leaves room for the transcript,
# the schema and the answer.
NUM_CTX = 32768
NUM_PREDICT = 4096


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, *, host: str | None = None, model: str | None = None) -> None:
        settings = get_settings()
        self.host = (host or settings.ollama_host).rstrip("/")
        self.model = model or settings.ollama_model
        self._last_raw: str | None = None

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        repair_hint: str | None = None,
    ) -> Completion:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
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
            resp = httpx.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "format": schema,
                    "stream": False,
                    "options": {
                        "num_ctx": NUM_CTX,
                        "num_predict": NUM_PREDICT,
                        "temperature": 0,
                    },
                },
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                f"ollama returned {e.response.status_code} for model "
                f"{self.model}: {e.response.text}"
            ) from e
        except httpx.HTTPError as e:
            raise ProviderError(f"could not reach ollama at {self.host}: {e}") from e

        body = resp.json()
        try:
            text = body["message"]["content"]
        except KeyError as e:
            raise ProviderError(
                f"ollama response missing message.content: {body}"
            ) from e

        # Hitting the generation cap leaves invalid JSON mid-string, so say so
        # here rather than letting it surface as a confusing parse error.
        if body.get("done_reason") == "length":
            raise ProviderError(
                f"ollama hit the {NUM_PREDICT} token generation cap; "
                "output is truncated JSON"
            )

        self._last_raw = text
        return Completion(text=text, input_tokens=None, output_tokens=None)
