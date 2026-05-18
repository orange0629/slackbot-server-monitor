"""Ollama LLM judge — fallback when the remote vLLM dispatcher is unavailable.

Per-(paper, theme) judging. For each interest theme on each member, ask "would
someone whose research focus is THEME care about this paper?" and collect a
score + short why. Aggregate per paper: pick each member's best theme; tag the
top-2 members whose best score >= PAPER_CURATOR_TAG_SCORE_THRESHOLD.

Returns judgments aligned 1:1 to the input papers, each shaped:
    {"relevant": bool, "score": int, "tags": [name,...], "one_line_why": str,
     "per_member": {name: {"theme": str, "score": int, "why": str}}}
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from config import (
    PAPER_CURATOR_OLLAMA_FALLBACK,
    PAPER_CURATOR_OLLAMA_HOST,
    PAPER_CURATOR_OLLAMA_MODEL,
    PAPER_CURATOR_TAG_SCORE_THRESHOLD,
    PAPER_CURATOR_USE_REMOTE,
)

logger = logging.getLogger(__name__)


SYSTEM = (
    "You judge whether a research paper would be useful today to someone "
    "whose research focus is the one stated. Be VERY strict — when in doubt, "
    "score lower. Most papers should score 0-5; reserve 8+ for unambiguous "
    "matches the focus-holder would clearly want to read.\n"
    "Score guide (0-10):\n"
    " 9-10 = the paper IS the focus area: directly studies it as its central "
    "contribution, and the focus-holder would reliably read it.\n"
    " 8    = clear and substantive match; the focus-holder would almost "
    "certainly read it. Use only when the methodological or empirical core "
    "of the paper directly engages the focus, not just the topic area.\n"
    " 5-7  = related but not a clear match (adjacent subfield, partial "
    "overlap, or applies the focus to an unrelated problem).\n"
    " 0-4  = tangential, keyword-only match (e.g. 'mentions LLMs' without "
    "engaging the focus), or a survey/tutorial/review/comprehensive guide/"
    "position paper.\n"
    "If you are not confident the focus-holder would actively want to read "
    "this paper, score <= 7.\n"
    "Quote a specific phrase or claim from the paper in your `why`.\n"
    "/no_think"
)


def _interest_themes(raw) -> List[str]:
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    parts = [ln.strip(" -\t") for ln in str(raw).splitlines()]
    return [p for p in parts if p]


def _user_prompt(paper: Dict, theme: str) -> str:
    """Paper-first / theme-last so any prefix-caching machinery upstream can
    reuse the SYSTEM + PAPER block across themes."""
    return (
        f"PAPER\n"
        f"title: {paper.get('title','')}\n"
        f"abstract: {paper.get('abstract','')[:1500]}\n"
        f"venue: {paper.get('source','')}\n"
        f"\n---\n\n"
        f"RESEARCH FOCUS: {theme}\n\n"
        "Respond with strict JSON only:\n"
        '{"score": 0-10, "why": "max 10 words; quote a specific phrase from '
        'the paper"}'
    )


def _chat_no_think(client, **kwargs):
    try:
        return client.chat(think=False, **kwargs)
    except TypeError:
        return client.chat(**kwargs)


def _judge_one(client, model: str, paper: Dict, theme: str) -> Dict:
    resp = _chat_no_think(
        client,
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": _user_prompt(paper, theme)},
        ],
        format="json",
        options={"num_ctx": 4096, "temperature": 0.2},
    )
    raw = resp["message"]["content"]
    data = json.loads(raw)
    return {
        "score": int(data.get("score", 0)),
        "why": str(data.get("why", ""))[:200],
    }


def _aggregate(papers: List[Dict],
               raw_rows: List[tuple],
               threshold: int) -> List[Dict]:
    """raw_rows: list of (paper_idx, member_name, theme, score, why)."""
    by_paper: Dict[int, Dict[str, Dict]] = {}
    for p_idx, name, theme, score, why in raw_rows:
        slot = by_paper.setdefault(p_idx, {})
        prev = slot.get(name)
        if prev is None or score > prev["score"]:
            slot[name] = {"theme": theme, "score": score, "why": why}

    out: List[Dict] = []
    for p_idx, _ in enumerate(papers):
        per_member = by_paper.get(p_idx, {})
        candidates = sorted(
            ((name, info) for name, info in per_member.items()
             if info["score"] >= threshold),
            key=lambda x: -x[1]["score"],
        )
        top = candidates[:2]
        if top:
            tags = [name for name, _ in top]
            best_why = top[0][1]["why"]
            best_score = top[0][1]["score"]
        else:
            tags = []
            best_why = ""
            best_score = max((info["score"] for info in per_member.values()),
                             default=0)
        out.append({
            "relevant": bool(top),
            "score": best_score,
            "tags": tags,
            "one_line_why": best_why,
            "per_member": per_member,
        })
    return out


def judge_papers(papers: List[Dict], members: List[Dict]) -> List[Optional[Dict]]:
    """Returns a list aligned to `papers`. None = LLM failed for that paper.

    Tries remote vLLM (if PAPER_CURATOR_USE_REMOTE) first, falls back to local
    Ollama. Final fallback is None for every paper, which the caller treats
    as 'LLM offline -> bi-encoder ranking only'."""
    if not papers:
        return []

    if PAPER_CURATOR_USE_REMOTE:
        try:
            from .remote_dispatch import judge_remotely
            remote = judge_remotely(papers, members)
            if remote is not None:
                return [r for r in remote]
            logger.info("remote judge unavailable; falling back to Ollama")
        except Exception as e:
            logger.warning("remote judge errored (%s); falling back to Ollama", e)

    try:
        import ollama  # lazy import
    except ImportError:
        logger.error("ollama python client not installed; skipping LLM step")
        return [None] * len(papers)

    client = ollama.Client(host=PAPER_CURATOR_OLLAMA_HOST)

    # Pre-compute themes per member; skip members without any interests.
    member_themes = [(m["name"], _interest_themes(m.get("interests")))
                     for m in members]
    member_themes = [(n, ts) for n, ts in member_themes if ts]
    if not member_themes:
        return [None] * len(papers)

    # Build (paper_idx, member, theme) work units in paper-major order.
    triples = []
    for p_idx, p in enumerate(papers):
        for name, themes in member_themes:
            for theme in themes:
                triples.append((p_idx, name, theme, p))

    primary = PAPER_CURATOR_OLLAMA_MODEL
    fallback = PAPER_CURATOR_OLLAMA_FALLBACK

    def _try(model: str, paper: Dict, theme: str) -> Optional[Dict]:
        try:
            return _judge_one(client, model, paper, theme)
        except Exception as e:
            logger.debug("ollama judge failed (model=%s): %s", model, e)
            return None

    raw_rows: List[tuple] = []
    consecutive_failures = 0
    use_model = primary
    with ThreadPoolExecutor(max_workers=2) as ex:
        results = list(ex.map(
            lambda tr: (tr, _try(use_model, tr[3], tr[2])), triples))
    for (p_idx, name, theme, paper), j in results:
        if j is None:
            j = _try(fallback, paper, theme)
            if j is None:
                consecutive_failures += 1
                if consecutive_failures >= 5:
                    # Hard-fail: ollama is down.
                    logger.error("ollama judge: too many failures, aborting")
                    return [None] * len(papers)
                raw_rows.append((p_idx, name, theme, 0, "(judge failed)"))
                continue
        raw_rows.append((p_idx, name, theme, j["score"], j["why"]))
        consecutive_failures = 0

    return _aggregate(papers, raw_rows, PAPER_CURATOR_TAG_SCORE_THRESHOLD)
