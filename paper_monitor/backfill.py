"""Offline backfill: walk a Slack channel's history and ingest paper links.

Pages `conversations.history` from newest to oldest and stops at the
configured cutoff date. Thread replies are followed via
`conversations.replies` so links posted inside threads are not missed.

Usage:
    python -m paper_monitor.backfill                # uses PAPERS_CHANNEL + cutoff
    python -m paper_monitor.backfill --channel C123
    python -m paper_monitor.backfill --since 2024-01-01 --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Iterable, Optional

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from config import PAPERS_BACKFILL_CUTOFF, PAPERS_CHANNEL
from config_secret import SLACK_TOKEN, require_slack_tokens

from .bibtex import attach_citation_and_bibtex
from .resolver import extract_urls, resolve_url
from .store import append_row, load_existing_keys, to_row

logger = logging.getLogger(__name__)


def _looks_like_channel_id(s: str) -> bool:
    return (len(s) >= 9 and s[0] in ("C", "G")
            and s.isalnum() and s == s.upper())


def _resolve_channel_id(client: WebClient, channel: str) -> str:
    """Accept either a channel ID (C/G…) or a name like 'interesting-papers'.

    Name-to-ID lookup needs the `channels:read` scope. If that scope isn't
    granted, surface a clear error and tell the user to pass the ID directly.
    """
    channel = channel.lstrip("#")
    if _looks_like_channel_id(channel):
        return channel
    cursor = None
    try:
        while True:
            resp = client.conversations_list(
                cursor=cursor, limit=1000,
                types="public_channel,private_channel",
            )
            for ch in resp.get("channels", []):
                if ch.get("name") == channel:
                    return ch["id"]
            cursor = (resp.get("response_metadata") or {}).get("next_cursor") or None
            if not cursor:
                break
    except SlackApiError as e:
        err = (e.response or {}).get("error", str(e))
        raise SystemExit(
            f"could not list channels to look up name {channel!r} (Slack error: {err}).\n"
            f"  Either grant the bot the 'channels:read' scope and reinstall the app,\n"
            f"  or rerun with --channel <CHANNEL_ID> (e.g. --channel C0123456789).\n"
            f"  You can find the channel ID in Slack: right-click the channel name -> "
            f"'View channel details' -> bottom of the popup."
        )
    raise SystemExit(f"channel not found: {channel}")


def _cutoff_ts(since_iso: str) -> float:
    dt = datetime.fromisoformat(since_iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _iter_channel_messages(client: WebClient, channel_id: str,
                           oldest_ts: float) -> Iterable[dict]:
    """Yield top-level messages (newest first) and their thread replies."""
    cursor = None
    while True:
        try:
            resp = client.conversations_history(
                channel=channel_id, cursor=cursor, limit=200,
                oldest=str(oldest_ts), inclusive=True,
            )
        except SlackApiError as e:
            if e.response.get("error") == "ratelimited":
                wait = int(e.response.headers.get("Retry-After", 30))
                logger.warning(f"rate limited; sleeping {wait}s")
                time.sleep(wait)
                continue
            raise
        for msg in resp.get("messages", []):
            yield msg
            # Follow threads.
            if msg.get("thread_ts") and msg.get("reply_count", 0) > 0:
                yield from _iter_thread(client, channel_id, msg["thread_ts"])
        cursor = (resp.get("response_metadata") or {}).get("next_cursor") or None
        if not cursor:
            return


def _iter_thread(client: WebClient, channel_id: str, thread_ts: str) -> Iterable[dict]:
    cursor = None
    while True:
        try:
            resp = client.conversations_replies(
                channel=channel_id, ts=thread_ts, cursor=cursor, limit=200,
            )
        except SlackApiError as e:
            if e.response.get("error") == "ratelimited":
                wait = int(e.response.headers.get("Retry-After", 30))
                time.sleep(wait)
                continue
            logger.warning(f"thread fetch failed for {thread_ts}: {e}")
            return
        # First message in replies is the parent — skip it.
        for msg in resp.get("messages", [])[1:]:
            yield msg
        cursor = (resp.get("response_metadata") or {}).get("next_cursor") or None
        if not cursor:
            return


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--channel", default=PAPERS_CHANNEL,
                    help="channel ID or name (default: PAPERS_CHANNEL)")
    ap.add_argument("--since", default=PAPERS_BACKFILL_CUTOFF,
                    help="ISO date; messages older than this are skipped (default: PAPERS_BACKFILL_CUTOFF)")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve and print but do not append to the store")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after processing N messages (0 = no cap)")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    require_slack_tokens()  # fail fast with a clear message if .env isn't loaded
    client = WebClient(token=SLACK_TOKEN)
    # Verify auth up front so a misconfigured token gives a clear error rather
    # than a downstream 'not_authed' from conversations.list.
    try:
        who = client.auth_test()
        logger.info(f"authed as bot user '{who.get('user')}' on team '{who.get('team')}'")
    except SlackApiError as e:
        raise SystemExit(
            f"Slack auth failed: {(e.response or {}).get('error', e)}.\n"
            f"  Check SLACK_TOKEN in your .env (must be the xoxb- bot token, not xapp-)."
        )
    channel_id = _resolve_channel_id(client, args.channel)
    cutoff = _cutoff_ts(args.since)
    logger.info(f"backfilling channel={channel_id} since={args.since} (ts>={cutoff})")

    existing_keys = load_existing_keys()
    logger.info(f"loaded {len(existing_keys)} existing keys from store")

    seen_msgs = 0
    msgs_with_urls = 0
    written = 0
    for msg in _iter_channel_messages(client, channel_id, cutoff):
        seen_msgs += 1
        if args.limit and seen_msgs >= args.limit:
            break
        if msg.get("bot_id") or msg.get("subtype"):
            continue
        text = msg.get("text") or ""
        urls = extract_urls(text)
        if not urls:
            continue
        msgs_with_urls += 1
        slack_user = msg.get("user") or ""
        slack_ts = msg.get("ts") or ""
        for url in urls:
            try:
                res = resolve_url(url)
                attach_citation_and_bibtex(res)
                row = to_row(res, slack_user=slack_user, slack_ts=slack_ts,
                             channel=channel_id)
                if args.dry_run:
                    logger.info(f"[dry-run] {row['key']!r} status={row['status']} "
                                f"title={row.get('title')!r}")
                    continue
                if append_row(row, existing_keys=existing_keys):
                    written += 1
                    logger.info(f"stored {row['key']!r} ({row['status']})")
            except Exception as e:
                logger.error(f"resolve failed for {url}: {e}")

    logger.info(
        f"done: messages_seen={seen_msgs} with_urls={msgs_with_urls} "
        f"new_rows={written}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
