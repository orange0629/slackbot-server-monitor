"""On-disk candidate index: a numpy embedding matrix + a metadata table."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class IndexEntry:
    gif_id: str
    giphy_id: str
    permalink: str
    alt_text: str
    rating: str  # giphy content rating: g, pg, pg-13, r


class Index:
    """In-memory cosine-similarity index over L2-normalized float32 vectors."""

    def __init__(self, embeddings: np.ndarray, entries: list[IndexEntry]):
        if embeddings.shape[0] != len(entries):
            raise ValueError("embeddings/entries length mismatch")
        self.embeddings = embeddings.astype(np.float32, copy=False)
        # Re-normalize defensively; cheap and avoids drift.
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True).clip(min=1e-8)
        self.embeddings = self.embeddings / norms
        self.entries = entries
        self._id_to_row = {e.gif_id: i for i, e in enumerate(entries)}

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])

    def __len__(self) -> int:
        return len(self.entries)

    @classmethod
    def load(cls, index_dir: str, backend: str) -> "Index":
        emb_path = os.path.join(index_dir, f"{backend}_embeddings.npy")
        meta_path = os.path.join(index_dir, "index_metadata.jsonl")
        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"missing {emb_path} — run build_index first")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"missing {meta_path} — run build_index first")
        embeddings = np.load(emb_path)
        entries: list[IndexEntry] = []
        with open(meta_path) as f:
            for line in f:
                row = json.loads(line)
                entries.append(IndexEntry(
                    gif_id=row["gif_id"],
                    giphy_id=row.get("giphy_id", ""),
                    permalink=row.get("permalink", ""),
                    alt_text=row.get("alt_text", ""),
                    rating=row.get("rating", "g"),
                ))
        return cls(embeddings, entries)

    @classmethod
    def concat(cls, parts: list["Index"]) -> "Index":
        """Concat multiple in-memory indexes, dedup by gif_id, first-wins.

        Useful when one part (typically the large primary) is already loaded
        and we only want to reload smaller aux parts from disk.
        """
        if not parts:
            raise ValueError("concat needs at least one Index")
        if len(parts) == 1:
            return parts[0]
        mats: list[np.ndarray] = []
        merged_entries: list[IndexEntry] = []
        seen_gids: set[str] = set()
        for p in parts:
            if mats and p.embeddings.shape[1] != mats[0].shape[1]:
                raise ValueError(
                    f"index dim mismatch: {p.embeddings.shape[1]} vs {mats[0].shape[1]}"
                )
            keep_rows: list[int] = []
            for row, e in enumerate(p.entries):
                if e.gif_id in seen_gids:
                    continue
                seen_gids.add(e.gif_id)
                keep_rows.append(row)
                merged_entries.append(e)
            if keep_rows:
                mats.append(p.embeddings[keep_rows])
        return cls(np.concatenate(mats, axis=0), merged_entries)

    @classmethod
    def load_many(cls, index_dirs: list[str], backend: str) -> "Index":
        """Load multiple index dirs and concat them into a single search matrix.

        Use case: a frozen primary index (e.g. the 147k PEPE-v2 union) + small
        auxiliary indexes that grow incrementally (e.g. the slow Giphy
        discoverer's output). Each aux index is its own self-contained
        embeddings.npy + metadata.jsonl; the primary is never touched.

        Dedup is by `gif_id`, primary-wins (first occurrence kept). Missing
        aux dirs are skipped with a warning; missing primary raises.
        """
        if not index_dirs:
            raise ValueError("load_many requires at least one index dir")
        parts: list[Index] = []
        for i, d in enumerate(index_dirs):
            try:
                part = cls.load(d, backend)
            except FileNotFoundError as e:
                if i == 0:
                    raise
                logger.warning("aux index %s missing; skipping (%s)", d, e)
                continue
            logger.info("loaded index %s: %d entries", d, len(part.entries))
            parts.append(part)
        return cls.concat(parts)

    def save(self, index_dir: str, backend: str) -> None:
        os.makedirs(index_dir, exist_ok=True)
        np.save(os.path.join(index_dir, f"{backend}_embeddings.npy"), self.embeddings)
        with open(os.path.join(index_dir, "index_metadata.jsonl"), "w") as f:
            for e in self.entries:
                f.write(json.dumps({
                    "gif_id": e.gif_id,
                    "giphy_id": e.giphy_id,
                    "permalink": e.permalink,
                    "alt_text": e.alt_text,
                    "rating": e.rating,
                }) + "\n")

    def search(self, query: np.ndarray, k: int = 10, exclude: Iterable[str] = ()) -> list[tuple[IndexEntry, float]]:
        q = query.astype(np.float32, copy=False)
        q = q / max(float(np.linalg.norm(q)), 1e-8)
        scores = self.embeddings @ q  # cosine since both are normalized
        if exclude:
            for gif_id in exclude:
                row = self._id_to_row.get(gif_id)
                if row is not None:
                    scores[row] = -np.inf
        # top-k
        if k >= len(scores):
            order = np.argsort(-scores)
        else:
            top = np.argpartition(-scores, k)[:k]
            order = top[np.argsort(-scores[top])]
        return [(self.entries[i], float(scores[i])) for i in order if scores[i] > -np.inf]
