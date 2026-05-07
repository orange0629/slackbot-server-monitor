"""Ollama LLM judge for relevance + tag assignment.

Per-paper JSON-mode prompt; small batches for throughput. Returns a list of
judgments aligned to the input papers, where each judgment is:
    {"relevant": bool, "score": int, "tags": [name,...], "one_line_why": str}
On failure (parse error / Ollama down) the record is marked relevant=False.
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
    PAPER_CURATOR_USE_REMOTE,
)

logger = logging.getLogger(__name__)


SYSTEM = (
    "You are a research librarian for the Blablablab lab (PI: David Jurgens, "
    "U-Michigan School of Information). The lab works on NLP, computational "
    "social science, sociolinguistics, and human behavior with language. "
    "Decide if a paper is worth surfacing to the lab today and tag at most "
    "two members it most fits. Be strict: only mark relevant=true if a "
    "specific member would genuinely want to read it.\n"
    "REJECT (relevant=false) papers that are:\n"
    " - surveys, tutorials, reviews, 'comprehensive guides', 'practical "
    "guides', or textbook-style overviews — they look broadly relevant but "
    "are not original research the lab needs to know about today.\n"
    " - position papers, opinion pieces, or roadmaps without new results.\n"
    " - papers whose match to a member is only at the topic-keyword level "
    "(e.g. 'mentions LLMs') rather than a substantive methodological or "
    "empirical fit. A paper is only relevant if it advances or directly "
    "challenges a member's specific research direction.\n"
    "/no_think"  # Qwen3 inline directive: belt-and-suspenders with think=False
)


def _members_block(members: List[Dict]) -> str:
    lines = []
    for m in members:
        pubs = [p["title"] for p in (m.get("publications") or [])][:3]
        bio = m.get("affiliation", "") or m.get("role", "")
        themes = _interest_themes(m.get("interests"))
        lines.append(f"- {m['name']} ({bio}). Recent: " + " | ".join(pubs))
        for t in themes:
            lines.append(f"    * {t}")
    return "\n".join(lines)


def _interest_themes(raw) -> List[str]:
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    parts = [ln.strip(" -\t") for ln in str(raw).splitlines()]
    return [p for p in parts if p]


def _user_prompt(paper: Dict, members_block: str) -> str:
    return (
        f"PAPER:\n"
        f"title: {paper.get('title','')}\n"
        f"abstract: {paper.get('abstract','')[:1500]}\n"
        f"venue: {paper.get('source','')}\n\n"
        f"MEMBERS:\n{members_block}\n\n"
        "Respond with strict JSON only:\n"
        '{"relevant": bool, "score": int 0-10, '
        '"tags": [<=2 member names exactly as listed], '
        '"one_line_why": "max 10 words; specific phrase explaining the fit; '
        'omit names and filler words"}'
    )


def _chat_no_think(client, **kwargs):
    """Call client.chat with think=False; fall back if the installed ollama
    client predates that parameter (the /no_think directive in SYSTEM still
    suppresses reasoning for Qwen3 in that case)."""
    try:
        return client.chat(think=False, **kwargs)
    except TypeError:
        return client.chat(**kwargs)


def _judge_one(client, model: str, paper: Dict, members_block: str) -> Dict:
    try:
        resp = _chat_no_think(
            client,
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": _user_prompt(paper, members_block)},
            ],
            format="json",
            options={"num_ctx": 8192, "temperature": 0.2},
        )
        raw = resp["message"]["content"]
        data = json.loads(raw)
        return {
            "relevant": bool(data.get("relevant")),
            "score": int(data.get("score", 0)),
            "tags": [t for t in (data.get("tags") or []) if isinstance(t, str)][:2],
            "one_line_why": str(data.get("one_line_why", ""))[:200],
        }
    except Exception as e:
        logger.warning("LLM judge failed for paper %s: %s", paper.get("id"), e)
        raise


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
    members_block = _members_block(members)

    def _try(model: str, paper: Dict) -> Optional[Dict]:
        try:
            return _judge_one(client, model, paper, members_block)
        except Exception:
            return None

    out: List[Optional[Dict]] = [None] * len(papers)
    primary = PAPER_CURATOR_OLLAMA_MODEL
    fallback = PAPER_CURATOR_OLLAMA_FALLBACK

    consecutive_failures = 0
    use_model = primary
    with ThreadPoolExecutor(max_workers=2) as ex:
        for i, j in enumerate(ex.map(lambda p: _try(use_model, p), papers)):
            if j is None and consecutive_failures < 2:
                consecutive_failures += 1
                # retry with fallback synchronously for this one
                j = _try(fallback, papers[i])
                if j is not None:
                    use_model = fallback
                    consecutive_failures = 0
            elif j is not None:
                consecutive_failures = 0
            out[i] = j
    return out
