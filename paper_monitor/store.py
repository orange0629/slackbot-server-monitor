"""Append-only JSONL store for resolved papers, with first-mention dedup.

Schema for each row:
  {
    "key": "<canonical dedup key>",
    "citation": "...",
    "bibtex": "...",
    "paper_url": "...",      # canonical URL (doi.org / arxiv abs / source)
    "source_url": "...",     # the URL as posted in Slack
    "title": "...",
    "authors": [...],
    "year": ...,
    "venue": "...",
    "doi": "...",
    "arxiv_id": "...",
    "openalex_id": "...",
    "s2_id": "...",
    "posted_at": "<ISO8601>",   # the first Slack ts we saw
    "posted_by": "<slack user id>",
    "channel": "<slack channel id>",
    "slack_ts": "<raw ts>",
    "resolved_via": [...],
    "status": "resolved" | "partial" | "unresolved",
    "notes": [...]
  }

Dedup key is, in priority order: doi, arxiv_id, openalex_id, s2_id,
canonical_url, source_url. First mention wins — repeats are silently dropped.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, Iterable, Optional, Set

from filelock import FileLock

from config import PAPERS_LOG_FILE

logger = logging.getLogger(__name__)


def canonical_key(res: Dict) -> str:
    for field in ("doi", "arxiv_id", "openalex_id", "s2_id"):
        v = res.get(field)
        if v:
            return f"{field}:{v}"
    return f"url:{res.get('canonical_url') or res.get('source_url')}"


def _lock_path() -> str:
    return f"{PAPERS_LOG_FILE}.lock"


def load_existing_keys(path: str = PAPERS_LOG_FILE) -> Set[str]:
    keys: Set[str] = set()
    if not os.path.exists(path):
        return keys
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = row.get("key")
            if k:
                keys.add(k)
    return keys


def slack_ts_to_iso(ts: str) -> str:
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except (TypeError, ValueError):
        return datetime.now(tz=timezone.utc).isoformat()


def to_row(res: Dict, *, slack_user: str, slack_ts: str, channel: str) -> Dict:
    return {
        "key": canonical_key(res),
        "citation": res.get("citation"),
        "bibtex": res.get("bibtex"),
        "paper_url": res.get("canonical_url") or res.get("source_url"),
        "source_url": res.get("source_url"),
        "title": res.get("title"),
        "authors": res.get("authors") or [],
        "year": res.get("year"),
        "venue": res.get("venue"),
        "doi": res.get("doi"),
        "arxiv_id": res.get("arxiv_id"),
        "openalex_id": res.get("openalex_id"),
        "s2_id": res.get("s2_id"),
        "posted_at": slack_ts_to_iso(slack_ts),
        "posted_by": slack_user,
        "channel": channel,
        "slack_ts": slack_ts,
        "resolved_via": res.get("resolved_via") or [],
        "status": res.get("status") or "unresolved",
        "notes": res.get("notes") or [],
    }


def append_row(row: Dict, path: str = PAPERS_LOG_FILE,
               existing_keys: Optional[Set[str]] = None) -> bool:
    """Append `row` if its key is new. Returns True iff written.

    `existing_keys` is an optional in-memory cache used by the backfill to
    avoid scanning the file for every URL. The live listener can pass None
    and we'll re-scan under the lock.
    """
    key = row.get("key")
    if not key:
        return False
    with FileLock(_lock_path()):
        keys = existing_keys if existing_keys is not None else load_existing_keys(path)
        if key in keys:
            return False
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        if existing_keys is not None:
            existing_keys.add(key)
        return True
