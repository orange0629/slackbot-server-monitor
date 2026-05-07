"""Slack block-kit formatter + post helpers."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


_MAX_WHY_WORDS = 10
# Slack section blocks cap text at 3000 chars; keep a margin for safety.
_MAX_SECTION_CHARS = 2800


def _paper_bullet(paper: Dict, judgment: Dict,
                  slack_ids: Dict[str, Optional[str]],
                  with_tags: bool) -> str:
    """Render one paper as a single mrkdwn bullet line:
        • *<url|Title>*  <@U…> <@U…> — short why (≤10 words)
    The why-suffix is included only when at least one member is tagged.
    """
    title = paper.get("title", "(no title)")
    url = paper.get("url", "") or paper.get("pdf_url", "")
    title_md = f"*<{url}|{_escape(title)}>*" if url else f"*{_escape(title)}*"

    tags = (judgment.get("tags") or []) if judgment else []
    rendered_tags: List[str] = []
    if with_tags and tags:
        for name in tags:
            sid = slack_ids.get(name)
            rendered_tags.append(f"<@{sid}>" if sid else name)
    tag_part = (" " + " ".join(rendered_tags)) if rendered_tags else ""

    why_part = ""
    if rendered_tags:
        raw_why = (judgment.get("one_line_why", "") if judgment else "").strip()
        why = _trim_words(raw_why, _MAX_WHY_WORDS)
        if why:
            why_part = f" — {_escape(why)}"

    return f"• {title_md}{tag_part}{why_part}"


def _bulletize(items: List[Dict], slack_ids: Dict[str, Optional[str]],
               with_tags: bool) -> str:
    return "\n".join(
        _paper_bullet(it["paper"], it.get("judgment", {}),
                      slack_ids, with_tags=with_tags)
        for it in items
    )


def _section_blocks(text: str) -> List[Dict]:
    """Split a long mrkdwn string into multiple section blocks at line
    boundaries so each stays under Slack's 3000-char cap."""
    blocks: List[Dict] = []
    buf: List[str] = []
    used = 0
    for line in text.split("\n"):
        # +1 for the newline we'll add
        if used + len(line) + 1 > _MAX_SECTION_CHARS and buf:
            blocks.append({"type": "section",
                           "text": {"type": "mrkdwn", "text": "\n".join(buf)}})
            buf = []
            used = 0
        buf.append(line)
        used += len(line) + 1
    if buf:
        blocks.append({"type": "section",
                       "text": {"type": "mrkdwn", "text": "\n".join(buf)}})
    return blocks


def build_main_blocks(papers_with_judgments: List[Dict],
                      slack_ids: Dict[str, Optional[str]],
                      header_note: Optional[str] = None) -> List[Dict]:
    """Header + a single bulleted list of picks. Each bullet is one paper:
       title, then any @-tagged members, then a ≤10-word rationale only when
       a member is tagged."""
    from datetime import datetime
    header_text = (f":newspaper: *Papers for "
                   f"{datetime.now().strftime('%a, %b %-d')}* — "
                   f"{len(papers_with_judgments)} picks")
    if header_note:
        header_text += f"\n_{header_note}_"
    blocks: List[Dict] = [
        {"type": "section", "text": {"type": "mrkdwn", "text": header_text}},
    ]
    bullets = _bulletize(papers_with_judgments, slack_ids, with_tags=True)
    if bullets:
        blocks.extend(_section_blocks(bullets))
    return blocks


def build_overflow_blocks(items: List[Dict],
                          slack_ids: Dict[str, Optional[str]]) -> List[Dict]:
    """Thread reply: a single bulleted list of all overflow papers, no
    @-mentions and no rationale (since rationale is only shown for tagged
    items in the new format)."""
    bullets = _bulletize(items, slack_ids, with_tags=False)
    if not bullets:
        return []
    header = "_More from today:_"
    return _section_blocks(header + "\n" + bullets)


def post_digest(client, channel_id: str,
                main_items: List[Dict], overflow_items: List[Dict],
                slack_ids: Dict[str, Optional[str]],
                header_note: Optional[str] = None) -> Optional[str]:
    """Returns the main message ts on success, else None."""
    if not main_items:
        return None
    fallback = f"Daily paper digest: {len(main_items)} picks"
    blocks = build_main_blocks(main_items, slack_ids, header_note=header_note)
    try:
        resp = client.chat_postMessage(channel=channel_id, blocks=blocks,
                                       text=fallback, unfurl_links=False,
                                       unfurl_media=False)
        ts = resp.get("ts")
    except Exception as e:
        logger.error("main digest post failed: %s", e)
        return None
    if overflow_items:
        try:
            client.chat_postMessage(
                channel=channel_id, thread_ts=ts,
                blocks=build_overflow_blocks(overflow_items, slack_ids),
                text=f"{len(overflow_items)} more papers",
                unfurl_links=False, unfurl_media=False,
            )
        except Exception as e:
            logger.warning("overflow reply failed: %s", e)
    return ts


def _trim_words(s: str, n: int) -> str:
    words = (s or "").split()
    if len(words) <= n:
        return " ".join(words)
    return " ".join(words[:n]).rstrip(",.;:") + "…"


def _escape(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
