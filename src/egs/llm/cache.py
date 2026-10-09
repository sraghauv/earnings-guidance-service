"""Disk cache for validated extractions, one JSON file per cache key.

Keys come from `LLMProvider.cache_key()`, not from here -- this module just
reads and writes files under `settings.cache_dir`.
"""

import json
import os
import tempfile
from pathlib import Path


def get(cache_dir: Path, key: str) -> dict | None:
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def put(cache_dir: Path, key: str, payload: dict) -> None:
    """Write `payload` for `key`, atomically.

    Write to a temp file in the same directory and `os.replace` it into
    place, so a process killed mid-write never leaves a half-written entry
    for a later run to read.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=cache_dir, prefix=f".{key}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
        os.replace(tmp_name, cache_dir / f"{key}.json")
    except Exception:
        os.unlink(tmp_name)
        raise
