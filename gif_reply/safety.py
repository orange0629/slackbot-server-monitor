"""Content-safety filters for GIF candidates."""
from __future__ import annotations

import os
import re
from typing import Iterable


_ALLOWED_RATINGS = {"g", "pg"}


def load_word_list(path: str | None) -> set[str]:
    if not path or not os.path.exists(path):
        return set()
    out: set[str] = set()
    with open(path) as f:
        for line in f:
            w = line.strip().lower()
            if w and not w.startswith("#"):
                out.add(w)
    return out


def load_id_list(path: str | None) -> set[str]:
    if not path or not os.path.exists(path):
        return set()
    out: set[str] = set()
    with open(path) as f:
        for line in f:
            x = line.strip()
            if x and not x.startswith("#"):
                out.add(x)
    return out


class SafetyFilter:
    def __init__(self, banned_words_path: str | None = None, banned_ids_path: str | None = None):
        self.banned_words = load_word_list(banned_words_path)
        self.banned_ids = load_id_list(banned_ids_path)
        # Match banned words as whole words to avoid e.g. "ass" hitting "class".
        if self.banned_words:
            pattern = r"\b(" + "|".join(re.escape(w) for w in self.banned_words) + r")\b"
            self._word_re = re.compile(pattern, re.IGNORECASE)
        else:
            self._word_re = None

    def query_is_safe(self, text: str) -> bool:
        if self._word_re and self._word_re.search(text or ""):
            return False
        return True

    def gif_is_safe(self, *, giphy_id: str, alt_text: str, rating: str) -> bool:
        if rating and rating.lower() not in _ALLOWED_RATINGS:
            return False
        if giphy_id and giphy_id in self.banned_ids:
            return False
        if self._word_re and self._word_re.search(alt_text or ""):
            return False
        return True

    def filter(self, candidates: Iterable) -> list:
        return [c for c in candidates if self.gif_is_safe(
            giphy_id=getattr(c, "giphy_id", ""),
            alt_text=getattr(c, "alt_text", ""),
            rating=getattr(c, "rating", "g"),
        )]
