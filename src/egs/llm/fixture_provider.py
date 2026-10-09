"""Canned-response provider for tests. No network, no API key.

Looks a call up in `mapping` first, keyed by the exact `user` prompt text --
in a test you already have that string (you built the transcript), so this
needs no hashing to use. A mapping value can be a list: each call against
that key consumes the next entry and holds on the last one once the list
runs out, which is how a single fixture set can walk a repair sequence (bad
response, then a corrected one). Anything not in `mapping` falls back to
`<directory>/<cache_key>.json`, so a directory of previously-cached real
responses can double as a fixture set.
"""

from pathlib import Path

from egs.llm.base import Completion, LLMProvider, ProviderError


class FixtureProvider(LLMProvider):
    name = "fixture"

    def __init__(
        self,
        directory: Path,
        *,
        mapping: dict[str, str | list[str]] | None = None,
        model: str = "fixture",
    ) -> None:
        self.directory = Path(directory)
        self.mapping = mapping or {}
        self.model = model
        self.calls = 0
        self._attempts: dict[str, int] = {}

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        repair_hint: str | None = None,
    ) -> Completion:
        self.calls += 1

        if user in self.mapping:
            return Completion(text=self._read(user, self.mapping[user]))

        key = self.cache_key(transcript_text=user, prompt_version="fixture")
        path = self.directory / f"{key}.json"
        if not path.exists():
            raise ProviderError(f"no fixture for key {key!r} in {self.directory}")
        return Completion(text=path.read_text())

    def _read(self, key: str, entry: str | list[str]) -> str:
        filenames = entry if isinstance(entry, list) else [entry]
        attempt = self._attempts.get(key, 0)
        self._attempts[key] = attempt + 1
        filename = filenames[min(attempt, len(filenames) - 1)]
        path = self.directory / filename
        if not path.exists():
            raise ProviderError(
                f"fixture {filename!r} for key {key!r} not found in {self.directory}"
            )
        return path.read_text()
