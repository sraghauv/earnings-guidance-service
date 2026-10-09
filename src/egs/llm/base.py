"""Provider contract for guidance extraction.

A provider's only job is: given a prompt and the target JSON schema, return raw
JSON text. Validation, repair, caching and persistence all live above this
layer, so a new backend only has to implement `complete_json`.
"""

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass

SCHEMA_VERSION = "1"


class ProviderError(RuntimeError):
    """Backend failed in a way retrying the same request will not fix."""


class SchemaViolation(ValueError):
    """Model returned JSON that does not satisfy the schema.

    Carries the validation message so the repair loop can hand it back to the
    model verbatim.
    """

    def __init__(self, message: str, raw: str) -> None:
        super().__init__(message)
        self.raw = raw


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMProvider(ABC):
    """Returns JSON text conforming to `schema`."""

    name: str
    model: str

    @abstractmethod
    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        repair_hint: str | None = None,
    ) -> Completion:
        """Produce JSON text for one transcript.

        `repair_hint` is the validation error from a previous attempt. When set,
        the provider should include it so the model can correct itself.
        """

    def cache_key(self, *, transcript_text: str, prompt_version: str) -> str:
        """Content address for one extraction.

        Anything that can change the output is in the key, so a prompt edit or
        a model swap correctly misses the cache instead of serving a stale row.
        """
        h = hashlib.sha256()
        for part in (
            self.name,
            self.model,
            prompt_version,
            SCHEMA_VERSION,
            transcript_text,
        ):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()[:32]
