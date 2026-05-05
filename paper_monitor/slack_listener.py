"""Slack-side glue: turn an incoming `message` event into stored paper rows."""
from __future__ import annotations

import logging
from typing import Optional, Set

from config import ENABLE_PAPER_MONITOR, PAPERS_CHANNEL

from .bibtex import attach_citation_and_bibtex
from .resolver import extract_urls, resolve_url
from .store import append_row, to_row, load_existing_keys

logger = logging.getLogger(__name__)

_channel_id_cache: dict = {}  # channel_id -> name (lazy, populated by main.py's resolver)


def _matches_papers_channel(channel_id: str, channel_name: Optional[str]) -> bool:
    target = PAPERS_CHANNEL.lstrip("#")
    if channel_id == target:
        return True
    if channel_name and channel_name == target:
        return True
    return False


def ingest_message(text: str, *, slack_user: str, slack_ts: str,
                   channel_id: str, existing_keys: Optional[Set[str]] = None) -> int:
    """Resolve every URL in `text` and append new ones to the store.

    Returns the number of rows newly written. Safe to call from the live
    listener and from the offline backfill.
    """
    urls = extract_urls(text or "")
    if not urls:
        return 0
    written = 0
    for url in urls:
        try:
            res = resolve_url(url)
            attach_citation_and_bibtex(res)
            row = to_row(res, slack_user=slack_user, slack_ts=slack_ts,
                         channel=channel_id)
            if append_row(row, existing_keys=existing_keys):
                written += 1
                logger.info(f"paper_monitor: stored {row['key']!r} "
                            f"({row.get('status')}) from {url}")
            else:
                logger.info(f"paper_monitor: dedup hit for {row['key']!r}")
        except Exception as e:
            logger.error(f"paper_monitor: resolve/store failed for {url}: {e}")
    return written


def handle_paper_message(event: dict, client, *, channel_name_resolver) -> None:
    """Slack Bolt `message` event handler.

    `channel_name_resolver(client, channel_id) -> Optional[str]` is passed in
    so we can reuse main.py's `_resolve_channel_name` cache rather than making
    redundant `conversations.info` calls.
    """
    if not ENABLE_PAPER_MONITOR:
        return
    if event.get("subtype"):
        return
    if event.get("bot_id"):
        return
    channel_id = event.get("channel", "")
    if not channel_id:
        return
    name = channel_name_resolver(client, channel_id) if channel_name_resolver else None
    if not _matches_papers_channel(channel_id, name):
        return
    text = event.get("text") or ""
    user = event.get("user") or ""
    ts = event.get("ts") or ""
    ingest_message(text, slack_user=user, slack_ts=ts, channel_id=channel_id)
