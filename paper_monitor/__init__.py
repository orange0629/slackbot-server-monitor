"""Watch a Slack channel for paper links, resolve citations, store as JSONL.

Public entrypoints:
- handle_paper_message(event, client): call from a Slack `message` listener.
- ingest_message(text, slack_user, slack_ts, channel_id): pure-Python helper
  used by both the live listener and the offline backfill script.
"""
from .slack_listener import handle_paper_message, ingest_message  # noqa: F401
