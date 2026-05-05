"""Unit tests for the gif_reply package.

Run from repo root:  python -m pytest tests/

Tests that need heavy dependencies (torch, transformers, the PEPE checkpoint)
are skipped automatically when the dependencies aren't available, so the suite
is meaningful even on a fresh CPU-only checkout.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gif_reply.index import Index, IndexEntry  # noqa: E402
from gif_reply.safety import SafetyFilter  # noqa: E402


# ---------- Index ----------

def _make_index(n: int = 8, dim: int = 16) -> Index:
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((n, dim)).astype(np.float32)
    entries = [
        IndexEntry(gif_id=f"g{i}", giphy_id=f"giphy{i}", permalink="", alt_text=f"alt {i}", rating="g")
        for i in range(n)
    ]
    return Index(emb, entries)


def test_index_search_returns_self_for_own_vector():
    idx = _make_index()
    q = idx.embeddings[3].copy()
    results = idx.search(q, k=3)
    assert results[0][0].gif_id == "g3"
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)


def test_index_search_excludes():
    idx = _make_index()
    q = idx.embeddings[3].copy()
    results = idx.search(q, k=3, exclude={"g3"})
    assert all(e.gif_id != "g3" for e, _ in results)


def test_index_round_trip(tmp_path):
    idx = _make_index()
    idx.save(str(tmp_path), backend="siglip")
    loaded = Index.load(str(tmp_path), backend="siglip")
    assert len(loaded) == len(idx)
    np.testing.assert_allclose(loaded.embeddings, idx.embeddings, atol=1e-6)
    assert [e.gif_id for e in loaded.entries] == [e.gif_id for e in idx.entries]


# ---------- Safety ----------

def test_safety_blocks_nsfw_rating():
    f = SafetyFilter()
    assert f.gif_is_safe(giphy_id="x", alt_text="hello", rating="r") is False
    assert f.gif_is_safe(giphy_id="x", alt_text="hello", rating="g") is True


def test_safety_blocks_banned_word(tmp_path):
    words = tmp_path / "banned.txt"
    words.write_text("badword\n")
    f = SafetyFilter(banned_words_path=str(words))
    assert f.query_is_safe("this is a badword here") is False
    assert f.query_is_safe("totally fine") is True
    assert f.gif_is_safe(giphy_id="x", alt_text="contains badword", rating="g") is False


def test_safety_word_boundary(tmp_path):
    words = tmp_path / "banned.txt"
    words.write_text("ass\n")
    f = SafetyFilter(banned_words_path=str(words))
    # "class" must not match the banned word "ass"
    assert f.query_is_safe("good class today") is True
    assert f.query_is_safe("you ass") is False


def test_safety_blocks_banned_id(tmp_path):
    ids = tmp_path / "banned_ids.txt"
    ids.write_text("badgif\n")
    f = SafetyFilter(banned_ids_path=str(ids))
    assert f.gif_is_safe(giphy_id="badgif", alt_text="fine", rating="g") is False
    assert f.gif_is_safe(giphy_id="goodgif", alt_text="fine", rating="g") is True


# ---------- Rate limiter (replicates main.py logic against a fresh dict) ----------

def _check_rate(store, key, limit, now):
    cutoff = now - timedelta(hours=1)
    history = [t for t in store.get(key, []) if datetime.fromisoformat(t) > cutoff]
    if len(history) >= limit:
        store[key] = history
        return False
    history.append(now.isoformat())
    store[key] = history
    return True


def test_rate_limiter_caps_at_limit():
    store: dict = {}
    now = datetime(2026, 5, 2, 12, 0, 0)
    for _ in range(5):
        assert _check_rate(store, "u", 5, now) is True
    assert _check_rate(store, "u", 5, now) is False


def test_rate_limiter_window_expires():
    store: dict = {}
    t0 = datetime(2026, 5, 2, 12, 0, 0)
    for i in range(5):
        _check_rate(store, "u", 5, t0 + timedelta(minutes=i))
    assert _check_rate(store, "u", 5, t0 + timedelta(minutes=5)) is False
    # Two hours later, all old entries expire.
    assert _check_rate(store, "u", 5, t0 + timedelta(hours=2)) is True


# ---------- Encoder smoke tests (skipped if deps missing) ----------

def _has_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not (_has_module("torch") and _has_module("transformers")),
                    reason="torch/transformers not installed")
def test_siglip_encoder_shape():
    from gif_reply.encoders import SiglipEncoder
    enc = SiglipEncoder()
    v = enc.encode_text("a happy cat")
    assert v.ndim == 1
    assert v.dtype == np.float32
    assert abs(np.linalg.norm(v) - 1.0) < 1e-3


def test_giphy_extend_index_dedups_and_appends(tmp_path):
    """Atomic-extend should append new vectors and skip duplicate giphy_ids."""
    from gif_reply.giphy_refresh import _atomic_extend_index, _load_existing_giphy_ids

    v1 = np.eye(3, dtype=np.float32)
    e1 = [{"gif_id": f"giphy:{i}", "giphy_id": str(i), "permalink": "", "alt_text": "", "rating": "g"} for i in range(3)]
    _atomic_extend_index(str(tmp_path), "siglip", v1, e1)
    assert _load_existing_giphy_ids(str(tmp_path)) == {"0", "1", "2"}

    # Append two new ids with the same dim — should grow.
    v2 = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    e2 = [{"gif_id": "giphy:3", "giphy_id": "3", "permalink": "", "alt_text": "", "rating": "g"},
          {"gif_id": "giphy:4", "giphy_id": "4", "permalink": "", "alt_text": "", "rating": "g"}]
    _atomic_extend_index(str(tmp_path), "siglip", v2, e2)
    assert _load_existing_giphy_ids(str(tmp_path)) == {"0", "1", "2", "3", "4"}

    loaded = np.load(tmp_path / "siglip_embeddings.npy")
    assert loaded.shape == (5, 3)


def test_giphy_extend_index_rejects_dim_mismatch(tmp_path):
    from gif_reply.giphy_refresh import _atomic_extend_index
    _atomic_extend_index(str(tmp_path), "siglip", np.eye(2, dtype=np.float32),
                         [{"gif_id": "g0", "giphy_id": "0", "permalink": "", "alt_text": "", "rating": "g"},
                          {"gif_id": "g1", "giphy_id": "1", "permalink": "", "alt_text": "", "rating": "g"}])
    with pytest.raises(ValueError, match="embedding dim mismatch"):
        _atomic_extend_index(str(tmp_path), "siglip", np.eye(3, dtype=np.float32),
                             [{"gif_id": "g2", "giphy_id": "2", "permalink": "", "alt_text": "", "rating": "g"},
                              {"gif_id": "g3", "giphy_id": "3", "permalink": "", "alt_text": "", "rating": "g"},
                              {"gif_id": "g4", "giphy_id": "4", "permalink": "", "alt_text": "", "rating": "g"}])


def test_pepe_encoder_missing_checkpoint_raises():
    if not _has_module("torch"):
        pytest.skip("torch not installed")
    from gif_reply.encoders import PepeEncoder
    with pytest.raises(FileNotFoundError):
        PepeEncoder("/nope/not-a-real-checkpoint.pth")


@pytest.mark.skipif(
    not (_has_module("torch")
         and os.path.exists("/shared/2/projects/gif-reply/data/release/PEPE-model-checkpoint.pth")),
    reason="PEPE checkpoint or torch not available",
)
def test_pepe_encoder_loads():
    from gif_reply.encoders import PepeEncoder
    try:
        enc = PepeEncoder("/shared/2/projects/gif-reply/data/release/PEPE-model-checkpoint.pth")
    except ImportError:
        pytest.skip("PEPE model class not importable in this environment (expected until vendored)")
    v = enc.encode_text("a happy cat")
    assert v.ndim == 1
    assert v.dtype == np.float32
