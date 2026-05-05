"""End-to-end retrieval engine: text -> GIF candidate."""
from __future__ import annotations

import logging
import os
import random
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
        index_dir: str,
        backend: str = "siglip",
        encoder_kwargs: dict | None = None,
        safety: SafetyFilter | None = None,
    ):
        self.index_dir = index_dir
        self.backend = backend
        self._encoder_kwargs = encoder_kwargs or {}
        self._encoder = None
        self._index: Index | None = None
        self.safety = safety or SafetyFilter()

    def _get_encoder(self):
        if self._encoder is None:
            from .encoders import load_encoder  # heavy import deferred
            self._encoder = load_encoder(self.backend, **self._encoder_kwargs)
        return self._encoder

    def _get_index(self) -> Index:
        if self._index is None:
            self._index = Index.load(self.index_dir, self.backend)
        return self._index

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
