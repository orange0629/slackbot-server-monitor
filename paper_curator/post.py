"""Slack block-kit formatter + post helpers."""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


_MAX_WHY_WORDS = 10
# Slack section blocks cap text at 3000 chars; keep a margin for safety.
_MAX_SECTION_CHARS = 2800


# --- Metadata sub-line (authors · venue · publication note) ---------------

# Friendly venue names for plain source ids (those without a "kind:id" prefix).
# Prefixed sources (arxiv:, osf:, openreview:, huggingface:) are handled in
# _friendly_venue() below. Anything not listed falls back to a title-cased id.
_VENUE_NAMES = {
    "nature": "Nature",
    "nature-human-behaviour": "Nature Human Behaviour",
    "science": "Science",
    "science-advances": "Science Advances",
    "pnas": "PNAS",
    "pnas-nexus": "PNAS Nexus",
    "jpsp": "JPSP",
    "medrxiv-health-informatics": "medRxiv",
    "biorxiv-neuroscience": "bioRxiv",
    "cognitive-science": "Cognitive Science",
    "quantitative-science-studies": "Quantitative Science Studies",
    "scientometrics": "Scientometrics",
    "political-communication": "Political Communication",
    "journal-of-communication": "Journal of Communication",
    "sociological-science": "Sociological Science",
    "journal-of-sociolinguistics": "Journal of Sociolinguistics",
    "anthology-tacl": "TACL",
    "anthology-cl": "Computational Linguistics",
}

# Preprint sources whose abstracts/notes may announce a forthcoming venue.
_PREPRINT_PREFIXES = ("arxiv:", "osf:", "huggingface")

# Acceptance/forthcoming-publication phrasing. Used to (a) lift the relevant
# clause out of an arXiv comment and (b) scan a preprint abstract when no
# comment field exists (OSF/socarxiv).
_PUB_NOTE_RE = re.compile(
    r"(?:accepted\s+(?:at|to|for|in|by)|to\s+appear\s+(?:at|in)|"
    r"to\s+be\s+published|forthcoming\s+(?:at|in)?|camera[-\s]?ready|"
    r"in\s+press|published\s+(?:at|in)|presented\s+at|appears?\s+in)\b",
    re.I,
)


def _friendly_venue(source: str) -> str:
    """Map a paper's `source` id to a human-readable venue label."""
    if not source:
        return ""
    if source.startswith("arxiv:"):
        return "arXiv"
    if source.startswith("osf:"):
        provider = source.split(":", 1)[1]
        return {"socarxiv": "SocArXiv", "psyarxiv": "PsyArXiv"}.get(
            provider, provider.title())
    if source.startswith("openreview:"):
        # venueid like "ICLR.cc/2026/Conference" -> "OpenReview (ICLR 2026)"
        venueid = source.split(":", 1)[1]
        parts = venueid.split("/")
        name = parts[0].replace(".cc", "")
        year = next((p for p in parts[1:] if p.isdigit()), "")
        label = f"{name} {year}".strip()
        return f"OpenReview ({label})" if label else "OpenReview"
    if source.startswith("huggingface"):
        return "HF Daily"
    return _VENUE_NAMES.get(source, source.replace("-", " ").title())


# Name particles that belong with the family name (e.g. "van der Berg").
_NAME_PARTICLES = {"van", "von", "der", "den", "de", "del", "della", "di",
                   "da", "do", "dos", "la", "le", "el", "al", "bin", "ibn",
                   "ter", "ten", "st", "st.", "mac", "mc"}
# Generational/honorific suffixes to skip when finding the family name.
_NAME_SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "phd", "md"}


def _last_name(full: str) -> str:
    """Best-effort family name from a free-text author string.

    Handles "Last, First" (comma form), trailing suffixes (Jr., III), and
    leading particles (van, de, von ...). Heuristic — good enough for a digest.
    """
    full = (full or "").strip()
    if not full:
        return ""
    if "," in full:
        # "Last, First" — the family name is everything before the comma.
        return full.split(",", 1)[0].strip()
    tokens = full.split()
    while len(tokens) > 1 and tokens[-1].lower().strip(".") in {
            s.strip(".") for s in _NAME_SUFFIXES}:
        tokens.pop()
    if not tokens:
        return full
    # Absorb a trailing run of particles into the family name.
    i = len(tokens) - 1
    while i > 0 and tokens[i - 1].lower() in _NAME_PARTICLES:
        i -= 1
    return " ".join(tokens[i:])


def _author_line(authors: List[str]) -> str:
    """All author last names, comma-joined (no cap, per design)."""
    names = [_last_name(a) for a in (authors or [])]
    return ", ".join(n for n in names if n)


def _pub_note(paper: Dict) -> str:
    """A short 'forthcoming at X' note for preprints, or "".

    Prefers the source's own comment field (`notes`, e.g. arXiv's
    <arxiv:comment>); for preprints lacking one (OSF/socarxiv), scans the
    abstract's opening. Returns the single clause containing the acceptance
    phrase, stripped of page/figure boilerplate.
    """
    source = paper.get("source", "") or ""
    if not source.startswith(_PREPRINT_PREFIXES):
        return ""
    notes = (paper.get("notes") or "").strip()
    if notes:
        return _extract_pub_clause(notes)
    # No comment field: scan only the abstract's opening to limit false hits.
    abstract = (paper.get("abstract") or "").strip()
    if abstract and _PUB_NOTE_RE.search(abstract[:400]):
        return _extract_pub_clause(abstract[:400])
    return ""


def _extract_pub_clause(text: str) -> str:
    """Return the sentence/clause in `text` announcing a venue, else ""."""
    if not _PUB_NOTE_RE.search(text):
        return ""
    # Split into clauses on sentence/segment boundaries and keep the first that
    # carries the acceptance phrasing — drops trailing "9 pages, 3 figures".
    for clause in re.split(r"(?<=[.;])\s+|\s*[|]\s*", text):
        clause = clause.strip().rstrip(".;")
        if clause and _PUB_NOTE_RE.search(clause):
            return clause
    return text.strip().rstrip(".;")


def _paper_bullet(paper: Dict, judgment: Dict,
                  slack_ids: Dict[str, Optional[str]],
                  with_tags: bool) -> str:
    """Render one paper as a two-line mrkdwn bullet:
        • *<url|Title>*  _Last1, Last2 · Venue · To appear at ACL 2026_
              <@U…> <@U…> — short why (≤10 words)
    Line 1 always carries the title plus an italic metadata sub-line (author
    last names, friendly venue, and a forthcoming-publication note for
    preprints). Line 2 (tags + ≤10-word rationale) appears only when at least
    one member is tagged.
    """
    title = paper.get("title", "(no title)")
    url = paper.get("url", "") or paper.get("pdf_url", "")
    title_md = f"*<{url}|{_escape(title)}>*" if url else f"*{_escape(title)}*"

    meta_parts = [p for p in (_author_line(paper.get("authors")),
                              _friendly_venue(paper.get("source", "")),
                              _pub_note(paper)) if p]
    meta_md = f"  _{_escape(' · '.join(meta_parts))}_" if meta_parts else ""
    line1 = f"• {title_md}{meta_md}"

    tags = (judgment.get("tags") or []) if judgment else []
    rendered_tags: List[str] = []
    if with_tags and tags:
        for name in tags:
            sid = slack_ids.get(name)
            rendered_tags.append(f"<@{sid}>" if sid else name)
    if not rendered_tags:
        return line1

    why_part = ""
    raw_why = (judgment.get("one_line_why", "") if judgment else "").strip()
    why = _trim_words(raw_why, _MAX_WHY_WORDS)
    if why:
        why_part = f" — {_escape(why)}"
    return f"{line1}\n        {' '.join(rendered_tags)}{why_part}"


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
