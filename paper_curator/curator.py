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
    PAPER_CURATOR_TOP_K_TO_LLM,
    PAPER_CURATOR_WEEKDAYS,
)

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
    if not fresh:
        return _maybe_quiet_post(dry_run, preview_dm)

    paper_vecs = embeddings.embed_papers(fresh)
    member_names, member_vecs = embeddings.load_or_build_member_embeds(ranking_members)
    top_idx, _sim = embeddings.rank_papers_against_members(
        paper_vecs, member_vecs, k=PAPER_CURATOR_TOP_K_TO_LLM)
    candidates = [fresh[i] for i in top_idx.tolist()]
    if not candidates:
        return _maybe_quiet_post(dry_run, preview_dm)

    # The LLM judges only against taggable_members so it can't accidentally
    # tag the PI (or anyone else outside the tag pool).
    judgments = llm.judge_papers(candidates, taggable_members)
    llm_offline = all(j is None for j in judgments) if judgments else True

    # Pair candidates with judgments; only keep relevant ones (or fall back if LLM offline).
    items: List[Dict[str, Any]] = []
    if llm_offline:
        # Fallback: no tags, just top by sim
        for p in candidates[:PAPER_CURATOR_MAX_MAIN_POST * 2]:
            items.append({"paper": p, "judgment": {"relevant": True, "score": 5,
                                                    "tags": [], "one_line_why": ""}})
    else:
        member_name_set = {m["name"] for m in taggable_members}
        for p, j in zip(candidates, judgments):
            if not j or not j.get("relevant"):
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
