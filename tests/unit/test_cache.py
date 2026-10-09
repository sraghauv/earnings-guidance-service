"""Disk cache tests: roundtrip, missing keys, and that a write never leaves
a half-written file behind."""

from egs.llm import cache


def test_roundtrip(tmp_path):
    cache.put(tmp_path, "key-1", {"ticker": "NVDA", "items": []})
    assert cache.get(tmp_path, "key-1") == {"ticker": "NVDA", "items": []}


def test_missing_key_returns_none(tmp_path):
    assert cache.get(tmp_path, "does-not-exist") is None


def test_get_on_nonexistent_dir_returns_none(tmp_path):
    missing_dir = tmp_path / "never-created"
    assert cache.get(missing_dir, "key-1") is None


def test_put_creates_cache_dir(tmp_path):
    fresh_dir = tmp_path / "fresh"
    cache.put(fresh_dir, "key-1", {"a": 1})
    assert fresh_dir.exists()
    assert cache.get(fresh_dir, "key-1") == {"a": 1}


def test_overwrite_replaces_value(tmp_path):
    cache.put(tmp_path, "key-1", {"a": 1})
    cache.put(tmp_path, "key-1", {"a": 2})
    assert cache.get(tmp_path, "key-1") == {"a": 2}


def test_put_leaves_no_leftover_tmp_file(tmp_path):
    cache.put(tmp_path, "key-1", {"a": 1})
    entries = list(tmp_path.iterdir())
    assert entries == [tmp_path / "key-1.json"]
