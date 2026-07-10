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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from config import (
    PAPER_CURATOR_OLLAMA_FALLBACK,
    PAPER_CURATOR_OLLAMA_HOST,
    PAPER_CURATOR_OLLAMA_MAX_FAILURES,
    PAPER_CURATOR_OLLAMA_MODEL,
    PAPER_CURATOR_OLLAMA_TIMEOUT,
    PAPER_CURATOR_TAG_SCORE_THRESHOLD,
    PAPER_CURATOR_USE_REMOTE,
    PAPER_CURATOR_VLLM_ENDPOINT,
    PAPER_CURATOR_VLLM_ENDPOINT_CONCURRENCY,
    PAPER_CURATOR_VLLM_ENDPOINT_MODEL,
    PAPER_CURATOR_VLLM_ENDPOINT_PROBE_TIMEOUT,
    PAPER_CURATOR_VLLM_ENDPOINT_TIMEOUT,
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


def _member_triples(papers: List[Dict], members: List[Dict]) -> List[tuple]:
    """(paper_idx, member_name, theme, paper) work units in paper-major order.

    Paper-major so a prefix-caching backend reuses the SYSTEM+PAPER prefix across
    the themes asked of one paper. Members without interests are skipped; returns
    [] when no member has any theme (caller then does bi-encoder-only)."""
    member_themes = [(m["name"], _interest_themes(m.get("interests")))
                     for m in members]
    member_themes = [(n, ts) for n, ts in member_themes if ts]
    triples: List[tuple] = []
    for p_idx, p in enumerate(papers):
        for name, themes in member_themes:
            for theme in themes:
                triples.append((p_idx, name, theme, p))
    return triples


def _run_fanout(triples, judge_one, max_workers, max_failures, label):
    """Judge every triple concurrently via judge_one(paper, theme) -> {score,why}.

    Bails out to None after `max_failures` consecutive failures — a backend that
    is down or wedged would otherwise grind through hundreds of papers at the
    per-request timeout (exactly what once hung the whole bot). Returns the list
    of (paper_idx, name, theme, score, why) rows, or None if it aborted."""
    def _judge_triple(tr):
        _, _, theme, paper = tr
        try:
            return tr, judge_one(paper, theme)
        except Exception as e:
            logger.debug("%s judge failed: %s", label, e)
            return tr, None

    raw_rows: List[tuple] = []
    consecutive = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_judge_triple, tr): tr for tr in triples}
        for fut in as_completed(futures):
            (p_idx, name, theme, _), j = fut.result()
            if j is None:
                consecutive += 1
                if consecutive >= max_failures:
                    logger.error("%s judge: %d consecutive failures (backend down "
                                 "or wedged); aborting -> bi-encoder ranking only",
                                 label, consecutive)
                    for f in futures:
                        f.cancel()
                    return None
                raw_rows.append((p_idx, name, theme, 0, "(judge failed)"))
                continue
            raw_rows.append((p_idx, name, theme, int(j["score"]), j["why"]))
            consecutive = 0
    return raw_rows


def _endpoint_model(base: str) -> Optional[str]:
    """Probe {base}/v1/models. Returns the first served model id if the endpoint
    is live, else None. Doubles as the liveness check for judge_via_endpoint."""
    import requests  # lazy import
    try:
        r = requests.get(base.rstrip("/") + "/v1/models",
                         timeout=PAPER_CURATOR_VLLM_ENDPOINT_PROBE_TIMEOUT)
        r.raise_for_status()
        data = (r.json() or {}).get("data") or []
        if data:
            return data[0].get("id")
    except Exception as e:
        logger.debug("vllm endpoint probe failed (%s): %s", base, e)
    return None


def _judge_one_endpoint(base: str, model: str, paper: Dict, theme: str) -> Dict:
    """One (paper, theme) judgment via an OpenAI-compatible /v1/chat/completions.

    No `response_format` is sent (older vLLM builds reject it); the prompt already
    demands strict JSON and we parse leniently to tolerate any wrapping prose."""
    import requests  # lazy import
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": _user_prompt(paper, theme)},
        ],
        "temperature": 0.2,
        "max_tokens": 512,
    }
    r = requests.post(base.rstrip("/") + "/v1/chat/completions", json=payload,
                     timeout=PAPER_CURATOR_VLLM_ENDPOINT_TIMEOUT)
    r.raise_for_status()
    content = (r.json()["choices"][0]["message"]["content"] or "").strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        i, k = content.find("{"), content.rfind("}")
        data = json.loads(content[i:k + 1]) if 0 <= i < k else {}
    return {"score": int(data.get("score", 0)),
            "why": str(data.get("why", ""))[:200]}


def judge_via_endpoint(papers: List[Dict],
                       members: List[Dict]) -> Optional[List[Optional[Dict]]]:
    """Judge via a live persistent vLLM HTTP server (e.g. burger:8001).

    Returns a per-paper aligned list on success, or None when the endpoint is
    unset/down so the caller falls through to the SSH path, then Ollama."""
    base = (PAPER_CURATOR_VLLM_ENDPOINT or "").strip()
    if not base:
        return None
    model = PAPER_CURATOR_VLLM_ENDPOINT_MODEL.strip() or _endpoint_model(base)
    if not model:
        return None
    triples = _member_triples(papers, members)
    if not triples:
        return [None] * len(papers)
    logger.info("vllm endpoint live: %s (model=%s); judging %d papers, %d prompts",
                base, model, len(papers), len(triples))
    rows = _run_fanout(
        triples,
        lambda paper, theme: _judge_one_endpoint(base, model, paper, theme),
        max_workers=PAPER_CURATOR_VLLM_ENDPOINT_CONCURRENCY,
        max_failures=PAPER_CURATOR_OLLAMA_MAX_FAILURES,
        label="vllm-endpoint")
    if rows is None:
        return None
    return _aggregate(papers, rows, PAPER_CURATOR_TAG_SCORE_THRESHOLD)


def judge_papers(papers: List[Dict], members: List[Dict]) -> List[Optional[Dict]]:
    """Returns a list aligned to `papers`. None = LLM failed for that paper.

    Backend preference: a live persistent vLLM HTTP endpoint
    (PAPER_CURATOR_VLLM_ENDPOINT) -> per-run SSH+vLLM (PAPER_CURATOR_USE_REMOTE)
    -> local Ollama. Final fallback is None for every paper, which the caller
    treats as 'LLM offline -> bi-encoder ranking only'."""
    if not papers:
        return []

    # Preferred: a persistent vLLM server (no GPU wait, no per-run model load).
    try:
        via_endpoint = judge_via_endpoint(papers, members)
        if via_endpoint is not None:
            return via_endpoint
    except Exception as e:
        logger.warning("vllm endpoint judge errored (%s); falling back", e)

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

    # A per-request timeout is essential: without it a wedged Ollama (e.g. a
    # large model paged onto CPU) blocks each call forever, and because the
    # scheduler runs tasks synchronously that hangs the entire bot.
    client = ollama.Client(host=PAPER_CURATOR_OLLAMA_HOST,
                           timeout=PAPER_CURATOR_OLLAMA_TIMEOUT)
    triples = _member_triples(papers, members)
    if not triples:
        return [None] * len(papers)

    primary = PAPER_CURATOR_OLLAMA_MODEL
    fallback = PAPER_CURATOR_OLLAMA_FALLBACK

    def _ollama_judge(paper: Dict, theme: str) -> Dict:
        """Primary model, then fallback model. A raise propagates to _run_fanout,
        which counts it toward the consecutive-failure abort threshold."""
        try:
            return _judge_one(client, primary, paper, theme)
        except Exception as e:
            logger.debug("ollama primary failed (%s): %s", primary, e)
        return _judge_one(client, fallback, paper, theme)

    rows = _run_fanout(triples, _ollama_judge, max_workers=2,
                       max_failures=PAPER_CURATOR_OLLAMA_MAX_FAILURES, label="ollama")
    if rows is None:
        return [None] * len(papers)

    return _aggregate(papers, rows, PAPER_CURATOR_TAG_SCORE_THRESHOLD)
