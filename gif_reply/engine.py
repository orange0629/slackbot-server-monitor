"""End-to-end retrieval engine: text -> GIF candidate."""
from __future__ import annotations

import logging
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .index import Index, IndexEntry
from .safety import SafetyFilter

logger = logging.getLogger(__name__)


@dataclass
class GifSuggestion:
    gif_id: str
    giphy_id: str
    permalink: str
    alt_text: str
    score: float


class GifReplyEngine:
    """Loads encoder + index lazily, then maps text to a GIF.

    The encoder load is the expensive part (~hundreds of MB of weights), so
    construction is cheap and the encoder is materialized on the first call.
    """

    def __init__(
        self,
        index_dir: str | list[str],
        backend: str = "siglip",
        encoder_kwargs: dict | None = None,
        safety: SafetyFilter | None = None,
        aux_reload_interval: float = 30.0,
    ):
        # Accept either a single dir (legacy) or a list — first entry is the
        # primary (frozen) index; subsequent entries are auxiliary indexes
        # (e.g. the discoverer's growing aux). Search merges them in RAM and
        # the bot live-reloads aux on disk changes without restart.
        self.index_dirs: list[str] = [index_dir] if isinstance(index_dir, str) else list(index_dir)
        self.backend = backend
        self._encoder_kwargs = encoder_kwargs or {}
        self._encoder = None
        self._primary_index: Index | None = None  # never reloads
        self._merged_index: Index | None = None   # primary + current aux snapshot
        self._aux_mtimes: dict[str, float | None] = {}
        self._last_aux_check: float = 0.0
        self._aux_reload_interval = aux_reload_interval
        self.safety = safety or SafetyFilter()
        self._encoder_lock = threading.Lock()
        self._index_lock = threading.Lock()

    @property
    def index_dir(self) -> str:
        """Back-compat accessor; primary dir."""
        return self.index_dirs[0]

    def _aux_emb_paths(self) -> list[str]:
        return [os.path.join(d, f"{self.backend}_embeddings.npy") for d in self.index_dirs[1:]]

    def _current_aux_mtimes(self) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for p in self._aux_emb_paths():
            out[p] = os.path.getmtime(p) if os.path.exists(p) else None
        return out

    def _build_merged_index(self) -> Index:
        """Reuse cached primary; load each aux dir from disk; concat in RAM."""
        if self._primary_index is None:
            self._primary_index = Index.load(self.index_dirs[0], self.backend)
        parts: list[Index] = [self._primary_index]
        for d in self.index_dirs[1:]:
            try:
                parts.append(Index.load(d, self.backend))
            except FileNotFoundError as e:
                logger.warning("aux index %s missing; skipping (%s)", d, e)
        merged = Index.concat(parts) if len(parts) > 1 else parts[0]
        logger.info("merged index ready: primary=%d + aux_dirs=%d → total=%d",
                    len(self._primary_index), len(self.index_dirs) - 1, len(merged))
        return merged

    def _get_encoder(self):
        if self._encoder is None:
            # transformers 5.x lazy-imports via _LazyModule, which isn't thread-safe; serialize.
            with self._encoder_lock:
                if self._encoder is None:
                    from .encoders import load_encoder  # heavy import deferred
                    self._encoder = load_encoder(self.backend, **self._encoder_kwargs)
        return self._encoder

    def _get_index(self) -> Index:
        """Return the current merged index, live-reloading aux on mtime change.

        Polled at most once per `aux_reload_interval` seconds to keep the
        per-query overhead negligible. Reference swap is atomic, so an
        in-flight `search` always sees a consistent (old or new) Index.
        """
        if self._merged_index is None:
            with self._index_lock:
                if self._merged_index is None:
                    self._merged_index = self._build_merged_index()
                    self._aux_mtimes = self._current_aux_mtimes()
                    self._last_aux_check = time.time()
            return self._merged_index

        if len(self.index_dirs) > 1 and time.time() - self._last_aux_check >= self._aux_reload_interval:
            with self._index_lock:
                if time.time() - self._last_aux_check >= self._aux_reload_interval:
                    current = self._current_aux_mtimes()
                    if current != self._aux_mtimes:
                        logger.info("aux index mtime changed; reloading (was=%s now=%s)",
                                    self._aux_mtimes, current)
                        self._merged_index = self._build_merged_index()
                        self._aux_mtimes = current
                    self._last_aux_check = time.time()
        return self._merged_index

    def index_size(self) -> int:
        try:
            return len(self._get_index())
        except FileNotFoundError:
            return 0

    def suggest(
        self,
        text: str,
        exclude_ids: Iterable[str] = (),
        k: int = 25,
        sample_top_k: int = 10,
        temperature: float = 1.0,
        rng: random.Random | None = None,
    ) -> GifSuggestion | None:
        """Retrieve, safety-filter, and sample one gif.

        Pulls top-`k` candidates, drops ones the safety filter rejects, then
        softmax-samples from the first `sample_top_k` survivors using
        `temperature` (lower = greedier; <=0 means argmax).
        """
        if not self.safety.query_is_safe(text):
            logger.info("query failed safety filter; suppressing reply")
            return None
        encoder = self._get_encoder()
        index = self._get_index()
        q = encoder.encode_text(text)
        candidates = index.search(q, k=max(k, sample_top_k), exclude=exclude_ids)

        safe: list[tuple[IndexEntry, float]] = []
        for entry, score in candidates:
            if self.safety.gif_is_safe(giphy_id=entry.giphy_id, alt_text=entry.alt_text, rating=entry.rating):
                safe.append((entry, score))
            if len(safe) >= sample_top_k:
                break
        if not safe:
            return None

        if sample_top_k <= 1 or temperature <= 0 or len(safe) == 1:
            entry, score = safe[0]
        else:
            scores = np.array([s for _, s in safe], dtype=np.float64)
            logits = scores / float(temperature)
            logits -= logits.max()  # numeric stability
            probs = np.exp(logits)
            probs /= probs.sum()
            r = rng if rng is not None else random
            idx = r.choices(range(len(safe)), weights=probs.tolist(), k=1)[0]
            entry, score = safe[idx]

        return GifSuggestion(
            gif_id=entry.gif_id,
            giphy_id=entry.giphy_id,
            permalink=entry.permalink,
            alt_text=entry.alt_text or text,
            score=score,
        )
