"""Orchestration: fetch → dedup → embed → rank → judge → cap → post.

Public entrypoint:
    run_curation(dry_run=False, preview_dm=None) -> bool
"""
from __future__ import annotations

import json
import logging
import os
from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from config import (
    PAPER_CURATOR_CHANNEL,
    PAPER_CURATOR_MAX_MAIN_POST,
    PAPER_CURATOR_MAX_TAGS_PER_MEMBER,
    PAPER_CURATOR_QUIET_DAY_NOTE,
    PAPER_CURATOR_TOP_K_PER_MEMBER,
    PAPER_CURATOR_TOP_K_TO_LLM,
    PAPER_CURATOR_VENUE_PRESTIGE,
    PAPER_CURATOR_WEEKDAYS,
)
from config import PAPER_CURATOR_TAG_SCORE_THRESHOLD

from . import paths

logger = logging.getLogger(__name__)

LOG_FILE = paths.CURATOR_LOG
SEEN_TTL_DAYS = 30


def run_curation(dry_run: bool = False, preview_dm: Optional[str] = None) -> bool:
    """Run one curation pass. Returns True if a post was made (or would have been)."""
    if not dry_run and preview_dm is None:
        # Weekday gate — skipped in dry-run / admin preview so a manual trigger
        # works on weekends.
        if datetime.now().weekday() not in PAPER_CURATOR_WEEKDAYS:
            logger.info("paper_curator: weekend skip")
            return False

    from . import profiles, sources, embeddings, llm, post

    profile_cache = profiles.refresh_profiles()
    # ranking_members (PI + postdocs + PhD students) drive the bi-encoder so the
    # PI's research breadth still surfaces relevant papers; taggable_members
    # (postdocs + PhD students) is the smaller pool the LLM may @-mention.
    ranking_members = profiles.load_ranking_members(profile_cache)
    taggable_members = profiles.load_taggable_members(profile_cache)
    if not ranking_members:
        logger.warning("paper_curator: no ranking members; aborting")
        return False
    # If nobody has interests the bi-encoder ranks on bio/pubs only and the
    # remote judge gets zero (paper, theme) triples, so it returns instantly
    # with no judgments and the run posts nothing. That is a misconfiguration
    # (member_interests.yml missing/unreadable — see profiles.py), not a quiet
    # day: fail loudly and DON'T mark the day complete so a restart retries.
    if not any(m.get("interests") for m in ranking_members):
        logger.error(
            "paper_curator: no member has any interests — aborting without "
            "posting (member_interests.yml missing or unreadable?). This run "
            "is NOT counted as today's digest.")
        return False
    slack_ids = profiles.load_slack_ids()

    seen, last_run = _load_seen()
    since_iso = _since_floor(last_run)

    raw = sources.fetch_all(since_iso=since_iso)
    fresh = [p for p in raw if p["id"] and p["id"] not in seen]
    # Dedup within batch by id (different sources can carry the same paper).
    by_id: Dict[str, Dict] = {}
    for p in fresh:
        by_id.setdefault(p["id"], p)
    fresh = list(by_id.values())
    logger.info("paper_curator: %d fresh papers after dedup", len(fresh))
    pre_lang = len(fresh)
    fresh = _filter_english(fresh)
    if pre_lang != len(fresh):
        logger.info("paper_curator: dropped %d non-English papers",
                    pre_lang - len(fresh))
    if not fresh:
        return _maybe_quiet_post(dry_run, preview_dm)

    paper_vecs = embeddings.embed_papers(fresh)
    member_names, member_vecs = embeddings.load_or_build_member_embeds(ranking_members)
    top_idx, sim = embeddings.rank_papers_against_members(
        paper_vecs, member_vecs, k=PAPER_CURATOR_TOP_K_TO_LLM)
    # Union the global top-k with each member's top-k so members whose interests
    # don't dominate today's score distribution still get representation.
    per_member_idx = embeddings.top_k_per_member(sim, k=PAPER_CURATOR_TOP_K_PER_MEMBER)
    # Force prestige-venue papers past the bi-encoder gate so the LLM always
    # judges them even if their abstract embeds weakly against lab interests.
    prestige_idx = [i for i, p in enumerate(fresh) if _prestige_boost(p)]
    candidate_idx: List[int] = list(dict.fromkeys(
        top_idx.tolist() + per_member_idx.tolist() + prestige_idx))
    logger.info("paper_curator: %d candidates to LLM "
                "(global=%d, +per-member=%d)",
                len(candidate_idx), len(top_idx),
                len(candidate_idx) - len(top_idx))
    candidates = [fresh[i] for i in candidate_idx]
    if not candidates:
        return _maybe_quiet_post(dry_run, preview_dm)

    # The LLM judges only against taggable_members so it can't accidentally
    # tag the PI (or anyone else outside the tag pool).
    judgments = llm.judge_papers(candidates, taggable_members)
    llm_offline = all(j is None for j in judgments) if judgments else True

    # Pair candidates with judgments; only keep relevant ones (or fall back if LLM offline).
    items: List[Dict[str, Any]] = []
    if llm_offline:
        # Fallback: no tags, top by sim but float prestige venues to the front.
        ordered = sorted(candidates, key=lambda p: _prestige_boost(p),
                         reverse=True)
        for p in ordered[:PAPER_CURATOR_MAX_MAIN_POST * 2]:
            items.append({"paper": p,
                          "judgment": {"relevant": True,
                                       "score": 5 + _prestige_boost(p),
                                       "tags": [], "one_line_why": ""}})
    else:
        member_name_set = {m["name"] for m in taggable_members}
        for p, j in zip(candidates, judgments):
            if not j:
                continue
            # Bump prestige venues before the relevance gate so a borderline
            # Science/Nature/PNAS paper can be rescued into being tagged.
            _apply_prestige(p, j, PAPER_CURATOR_TAG_SCORE_THRESHOLD)
            if not j.get("relevant"):
                continue
            j["tags"] = [t for t in j.get("tags", []) if t in member_name_set]
            items.append({"paper": p, "judgment": j})
        items.sort(key=lambda x: x["judgment"].get("score", 0), reverse=True)

    items = _apply_tag_cap(items, PAPER_CURATOR_MAX_TAGS_PER_MEMBER)
    if not items:
        return _maybe_quiet_post(dry_run, preview_dm)

    main_items = items[:PAPER_CURATOR_MAX_MAIN_POST]
    overflow_items = items[PAPER_CURATOR_MAX_MAIN_POST:]
    header_note = "(LLM offline — bi-encoder ranking only)" if llm_offline else None

    if dry_run and preview_dm is None:
        from . import post as post_mod
        blocks = post_mod.build_main_blocks(main_items, slack_ids, header_note)
        print(json.dumps({"main_blocks": blocks,
                          "overflow_count": len(overflow_items)}, indent=2))
        return True

    # Resolve channel + post.
    from main import app  # type: ignore  # late import to avoid circular at module load
    target_channel = preview_dm or PAPER_CURATOR_CHANNEL
    channel_id = _resolve_channel(app.client, target_channel)
    ts = post.post_digest(app.client, channel_id, main_items,
                          [] if preview_dm else overflow_items,
                          slack_ids, header_note=header_note)
    if not ts:
        return False

    if not preview_dm:
        for item in main_items + overflow_items:
            seen[item["paper"]["id"]] = datetime.now().date().isoformat()
        _persist_seen(seen)
        _append_log(main_items, overflow_items)

    return True


# --- Helpers --------------------------------------------------------------

def _filter_english(papers: List[Dict]) -> List[Dict]:
    """Drop papers whose title+abstract isn't detected as English. Uses
    langdetect when available; otherwise falls back to a Latin-letter ratio
    heuristic that catches obvious non-English (Cyrillic, CJK, Arabic, Greek)
    but won't catch German/Spanish — install langdetect for full coverage."""
    try:
        from langdetect import detect, DetectorFactory, LangDetectException
        DetectorFactory.seed = 0  # deterministic
    except ImportError:
        return [p for p in papers if _looks_english_ascii(p)]

    out = []
    for p in papers:
        text = ((p.get("title") or "") + ". "
                + (p.get("abstract") or "")[:600]).strip()
        if len(text) < 30:
            # Too short for reliable detection — keep, downstream judge can drop.
            out.append(p)
            continue
        try:
            if detect(text) == "en":
                out.append(p)
        except LangDetectException:
            # Detector confused; keep and let the LLM judge handle it.
            out.append(p)
    return out


def _looks_english_ascii(paper: Dict) -> bool:
    """Fallback when langdetect isn't installed: keep papers whose title is
    mostly Latin letters. Catches scripts (Cyrillic/CJK/Arabic) but not
    Latin-script non-English (German/Spanish/etc)."""
    title = paper.get("title") or ""
    if not title:
        return True
    letters = [c for c in title if c.isalpha()]
    if not letters:
        return True
    ascii_letters = sum(1 for c in letters if ord(c) < 128)
    return ascii_letters / len(letters) >= 0.85


def _prestige_boost(paper: Dict) -> int:
    """Points to add to this paper's per-member judge scores, or 0. Keyed by
    the sources.yml feed id; arxiv/osf sources carry a 'kind:detail' source so
    we match on the part before the first ':'."""
    src = (paper.get("source") or "").split(":", 1)[0].strip().lower()
    return PAPER_CURATOR_VENUE_PRESTIGE.get(src, 0)


def _apply_prestige(paper: Dict, j: Dict, threshold: int) -> None:
    """Mutate judgment `j` in place: bump every per-member score by the paper's
    prestige boost (capped at 10), then recompute relevant/tags/score/why with
    the same top-2 >= threshold rule the LLM aggregation uses. Runs uniformly
    for the remote and local judge paths since both emit `per_member`."""
    boost = _prestige_boost(paper)
    if not boost:
        return
    per_member = j.get("per_member") or {}
    if not per_member:
        # Offline/bi-encoder fallback has no per-member detail — bump the
        # flat score so prestige papers still sort ahead.
        j["score"] = min(10, (j.get("score") or 0) + boost)
        return
    for info in per_member.values():
        info["score"] = min(10, info["score"] + boost)
    ranked = sorted(
        ((name, info) for name, info in per_member.items()
         if info["score"] >= threshold),
        key=lambda x: -x[1]["score"],
    )[:2]
    if ranked:
        j["relevant"] = True
        j["tags"] = [name for name, _ in ranked]
        j["score"] = ranked[0][1]["score"]
        j["one_line_why"] = ranked[0][1]["why"]
    else:
        j["score"] = max((i["score"] for i in per_member.values()),
                         default=j.get("score", 0))


def _apply_tag_cap(items: List[Dict], max_per_member: int) -> List[Dict]:
    counts: Counter = Counter()
    for item in items:
        kept = []
        for name in item["judgment"].get("tags", []):
            if counts[name] < max_per_member:
                kept.append(name)
                counts[name] += 1
        item["judgment"]["tags"] = kept
    return items


def _since_floor(last_run: Optional[str]) -> str:
    if last_run:
        try:
            d = datetime.fromisoformat(last_run).date()
            return (d - timedelta(days=1)).isoformat()
        except ValueError:
            pass
    return (datetime.now().date() - timedelta(days=2)).isoformat()


def _load_seen() -> Tuple[Dict[str, str], Optional[str]]:
    """Load BOT_STATE['arxiv_papers'] without importing main.py (avoid circular)."""
    from filelock import FileLock
    from config import BOT_STATE_FILE
    try:
        with FileLock(BOT_STATE_FILE + ".lock"):
            with open(BOT_STATE_FILE, "r") as f:
                data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}, None
    arx = data.get("arxiv_papers", {}) or {}
    return arx.get("seen", {}) or {}, arx.get("last_run")


def _persist_seen(seen: Dict[str, str]) -> None:
    """Update BOT_STATE on disk via a merge-write that mirrors main.persist_bot_state."""
    from filelock import FileLock
    from config import BOT_STATE_FILE
    cutoff = (datetime.now().date() - timedelta(days=SEEN_TTL_DAYS)).isoformat()
    pruned = {k: v for k, v in seen.items() if v >= cutoff}
    payload = {"seen": pruned, "last_run": datetime.now().isoformat(timespec="seconds")}
    try:
        with FileLock(BOT_STATE_FILE + ".lock"):
            try:
                with open(BOT_STATE_FILE, "r") as f:
                    disk = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                disk = {}
            disk["arxiv_papers"] = payload
            tmp = BOT_STATE_FILE + f".tmp.{os.getpid()}"
            with open(tmp, "w") as f:
                json.dump(disk, f, indent=2)
            os.replace(tmp, BOT_STATE_FILE)
    except Exception as e:
        logger.error("paper_curator: failed to persist seen: %s", e)


def _append_log(main_items, overflow_items) -> None:
    try:
        paths.ensure_dirs()
        with open(LOG_FILE, "a") as f:
            for item in main_items + overflow_items:
                row = {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "id": item["paper"]["id"],
                    "title": item["paper"]["title"],
                    "source": item["paper"]["source"],
                    "tags": item["judgment"].get("tags", []),
                    "score": item["judgment"].get("score"),
                }
                f.write(json.dumps(row) + "\n")
    except Exception as e:
        logger.warning("paper_curator: log append failed: %s", e)


def digest_posted_today() -> bool:
    """True if a digest was actually posted today.

    Reads the curator log (one row per posted paper, written only on a real
    post via _append_log). Used by the scheduler to decide whether a make-up
    run is owed after a restart. A genuinely quiet day writes no row, so it
    would re-run once on the next restart — harmless: the seen-set prevents
    duplicate papers and a quiet day simply posts nothing again.
    """
    today = datetime.now().date().isoformat()
    try:
        with open(LOG_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    if json.loads(line).get("ts", "")[:10] == today:
                        return True
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return False
    except Exception as e:
        # On any unexpected read error, assume posted so we don't spam a
        # make-up run on every restart.
        logger.warning("paper_curator: could not read log to check today's "
                        "digest (%s); assuming already posted", e)
        return True
    return False


def _resolve_channel(client, channel: str) -> str:
    """Accept either a channel ID or a name; prefer the helper from paper_monitor."""
    from paper_monitor.backfill import _resolve_channel_id
    return _resolve_channel_id(client, channel)


def _maybe_quiet_post(dry_run: bool, preview_dm: Optional[str]) -> bool:
    if not PAPER_CURATOR_QUIET_DAY_NOTE:
        logger.info("paper_curator: nothing to post")
        return False
    if dry_run and preview_dm is None:
        print(":zzz: Quiet day — no papers passed filter")
        return True
    try:
        from main import app  # type: ignore
        target = preview_dm or PAPER_CURATOR_CHANNEL
        channel_id = _resolve_channel(app.client, target)
        app.client.chat_postMessage(channel=channel_id,
                                    text=":zzz: Quiet day — no papers passed filter")
    except Exception as e:
        logger.warning("paper_curator quiet-day post failed: %s", e)
    return True
