"""Bi-encoder embeddings for papers and member profiles.

Lazy-loads sentence-transformers on first call. Caches per-member embeddings
to data/profile_embeds.npz keyed by mtime of profiles_cache.json.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import PAPER_CURATOR_BIENCODER
from . import paths

logger = logging.getLogger(__name__)

EMB_PATH = paths.PROFILE_EMBEDS
PROFILES_PATH = paths.PROFILES_CACHE

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        # Route HF downloads + model snapshots into the shared cache so vLLM
        # on burger and bge-small here both reuse a single download.
        paths.apply_hf_env()
        from sentence_transformers import SentenceTransformer
        logger.info("loading bi-encoder %s (cache=%s)",
                    PAPER_CURATOR_BIENCODER, paths.HF_CACHE_DIR)
        _MODEL = SentenceTransformer(PAPER_CURATOR_BIENCODER, device="cpu")
    return _MODEL


def _paper_text(p: Dict) -> str:
    return f"{p.get('title','')}. {p.get('abstract','')}".strip()


def embed_papers(papers: List[Dict]) -> np.ndarray:
    if not papers:
        return np.zeros((0, 384), dtype=np.float32)
    texts = [_paper_text(p) for p in papers]
    vecs = _model().encode(texts, batch_size=16, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=False)
    return vecs.astype(np.float32)


def embed_members(members: List[Dict]) -> Tuple[List[str], np.ndarray]:
    """Per member: one row per themed interest line (separate query vectors),
    plus one mean-pooled row over publication titles + role/affiliation.

    Returns (row_member_names, vectors[N_rows, d]) where row_member_names[i]
    tells you which member row i belongs to. Each row is L2-normalized — paper
    score = max cosine across all rows (caller folds back to members for tags).
    """
    if not members:
        return [], np.zeros((0, 384), dtype=np.float32)
    row_names: List[str] = []
    rows: List[np.ndarray] = []
    model = _model()
    for m in members:
        themes = _coerce_member_themes(m)
        # 1) one row per themed interest line — these are the explicit query
        #    vectors. Encoded individually (NOT mean-pooled) so a paper that
        #    matches one theme keeps its full cosine score.
        for theme in themes:
            v = model.encode([theme], batch_size=1, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False)
            rows.append(v[0].astype(np.float32))
            row_names.append(m["name"])
        # 2) one mean-pooled row over (role + affiliation) + publication titles.
        #    This keeps a member matchable on adjacent work even before themes
        #    are filled in, and gives the bi-encoder a fallback for members
        #    without any interest entries.
        chunks: List[str] = []
        blurb = " ".join(filter(None, [m.get("role", ""), m.get("affiliation", "")]))
        if blurb:
            chunks.append(blurb)
        for pub in m.get("publications", []) or []:
            t = pub.get("title", "")
            if t:
                chunks.append(t)
        if not chunks and not themes:
            chunks = [m.get("name", "")]
        if chunks:
            v = model.encode(chunks, batch_size=16, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False)
            vec = v.mean(axis=0)
            n = np.linalg.norm(vec)
            if n > 0:
                vec = vec / n
            rows.append(vec.astype(np.float32))
            row_names.append(m["name"])
    return row_names, np.stack(rows) if rows else np.zeros((0, 384), dtype=np.float32)


def _coerce_member_themes(m: Dict) -> List[str]:
    raw = m.get("interests")
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    # legacy string: split on newlines, treat each non-empty line as a theme.
    lines = [ln.strip(" -\t") for ln in str(raw).splitlines()]
    themes = [ln for ln in lines if ln]
    return themes or [str(raw).strip()]


def load_or_build_member_embeds(members: List[Dict]) -> Tuple[List[str], np.ndarray]:
    """Cache profile embeddings against the mtime of profiles_cache.json."""
    if not os.path.exists(PROFILES_PATH) or not members:
        return embed_members(members)
    mtime = os.path.getmtime(PROFILES_PATH)
    if os.path.exists(EMB_PATH):
        try:
            with np.load(EMB_PATH, allow_pickle=False) as z:
                # mtime alone is authoritative: profiles_cache.json is rewritten
                # whenever member set OR interests change, so a matching mtime
                # implies the same row layout.
                if float(z["mtime"]) == mtime:
                    return list(z["names"]), z["vectors"].astype(np.float32)
        except Exception as e:
            logger.warning("profile embed cache unreadable (%s); rebuilding", e)
    names, vecs = embed_members(members)
    try:
        np.savez(EMB_PATH, names=np.array(names),
                 vectors=vecs, mtime=np.array(mtime))
    except Exception as e:
        logger.warning("could not save profile embed cache: %s", e)
    return names, vecs


def rank_papers_against_members(paper_vecs: np.ndarray,
                                member_vecs: np.ndarray,
                                k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (top_idx, sim_matrix). sim_matrix is (N_papers, N_members);
    paper score = max sim across members. top_idx = paper indices sorted by
    descending score, capped at k."""
    if paper_vecs.shape[0] == 0 or member_vecs.shape[0] == 0:
        return np.array([], dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
    sim = paper_vecs @ member_vecs.T   # both L2-normalized → cosine
    paper_score = sim.max(axis=1)
    order = np.argsort(-paper_score)[:k]
    return order, sim


def top_k_per_member(sim: np.ndarray, k: int) -> np.ndarray:
    """Given an (N_papers, N_members) sim matrix, return the union of each
    member's top-k papers by sim, as a 1-D array of paper indices (deduped,
    order undefined). Guarantees every member gets representation regardless
    of how the global score distribution looks."""
    if sim.size == 0 or k <= 0:
        return np.array([], dtype=np.int64)
    n_papers = sim.shape[0]
    k = min(k, n_papers)
    # For each member (column), the top-k paper indices by sim.
    # argpartition is O(n) per column; we don't need them sorted here.
    picks = set()
    for j in range(sim.shape[1]):
        col = sim[:, j]
        idx = np.argpartition(-col, k - 1)[:k]
        picks.update(int(i) for i in idx)
    return np.fromiter(picks, dtype=np.int64, count=len(picks))
