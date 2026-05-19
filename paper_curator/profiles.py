"""Lab-website scraper for member profiles.

Scrapes https://blablablab.si.umich.edu/:
  - the People panel for member name + role + personal page URL
  - the Publications panel for {title, authors, venue}
Then builds per-member representative-paper lists by matching member names
against author lists (case-insensitive last-name + first-name initial).

Caches profiles to data/profiles_cache.json. Re-scrapes only when older than
PAPER_CURATOR_PROFILE_REFRESH_DAYS or when force=True.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

from config import (
    PAPER_CURATOR_LAB_URL,
    PAPER_CURATOR_PROFILE_REFRESH_DAYS,
    PAPERS_HTTP_TIMEOUT,
)

from . import paths

logger = logging.getLogger(__name__)

CACHE_PATH = paths.PROFILES_CACHE
SLACK_IDS_PATH = paths.SLACK_IDS

USER_AGENT = "lab-paper-curator/0.1 (mailto:jurgens@umich.edu)"

# Two role sets, deliberately different:
#   RANKING_ROLES  — everyone whose research interests should influence the
#                    bi-encoder paper ranking. Includes the PI so papers
#                    relevant to broader lab themes still surface.
#   TAGGABLE_ROLES — subset of RANKING_ROLES that the LLM may @-tag in the
#                    Slack post. PI is excluded so he isn't pinged daily.
RANKING_ROLES = {"Professors", "Postdocs", "PhD Students"}
TAGGABLE_ROLES = {"Postdocs", "PhD Students"}


# ---- HTML parsing --------------------------------------------------------

def _fetch_lab_html(url: str) -> str:
    r = requests.get(url, headers={"User-Agent": USER_AGENT},
                     timeout=PAPERS_HTTP_TIMEOUT)
    r.raise_for_status()
    return r.text


def _parse_people_panel(soup: BeautifulSoup) -> List[Dict]:
    """Walk the People section's <h5> role headings and the immediately
    following member cards (<td> blocks containing a name <div>)."""
    members: List[Dict] = []
    # Find the People section by its title link.
    section = None
    for sec in soup.find_all("section"):
        title_a = sec.find("p", class_="title")
        if title_a and title_a.get_text(strip=True).lower() == "people":
            section = sec
            break
    if section is None:
        raise RuntimeError("could not find People section on lab page")

    current_role: Optional[str] = None
    for el in section.descendants:
        if getattr(el, "name", None) == "h5":
            current_role = el.get_text(strip=True)
            continue
        if not current_role:
            continue
        if getattr(el, "name", None) == "td":
            name_div = el.find("div", style=lambda v: v and "font-size: 120%" in v)
            if not name_div:
                continue
            _add_member_from_anchor(members, current_role, name_div, el)
        elif getattr(el, "name", None) == "p" and "personal-info" in (el.get("class") or []):
            # Bootstrap-card layout (e.g. PI). Anchor lives directly inside the <p>.
            _add_member_from_anchor(members, current_role, el, el)
    return members


def _add_member_from_anchor(members: List[Dict], role: str,
                            name_container, affiliation_anchor) -> None:
    link = name_container.find("a")
    if link:
        name = link.get_text(strip=True)
        profile_url = link.get("href", "") or ""
    else:
        name = name_container.get_text(strip=True)
        profile_url = ""
    if not name:
        return
    tail: List[str] = []
    for sib in affiliation_anchor.next_siblings:
        if getattr(sib, "name", None) in ("br", "a", "div", "table"):
            break
        if isinstance(sib, str):
            tail.append(sib)
        else:
            tail.append(sib.get_text(" ", strip=True))
    affiliation = re.sub(r"\s+", " ", " ".join(tail)).strip()
    members.append({
        "name": name,
        "role": role,
        "affiliation": affiliation,
        "profile_url": profile_url,
    })


def _parse_publications_panel(soup: BeautifulSoup) -> List[Dict]:
    """Each paper is <table class="table papers"> with one <tr> containing
    a paper-info <td> whose first <a> is the title and the next text line
    is the author list."""
    pubs: List[Dict] = []
    for tbl in soup.find_all("table", class_="papers"):
        info = tbl.find("td", class_="paper-info")
        if info is None:
            continue
        title_a = info.find("a")
        if title_a is None:
            continue
        title = title_a.get_text(" ", strip=True)
        url = title_a.get("href", "") or ""
        # Walk children: after the title <a>, the next strings up to the next <br/>
        # are stray whitespace; the *next* run of strings after that <br/> is the
        # author list. Easier: take all direct text and split on <br/>.
        parts: List[str] = []
        buf: List[str] = []
        for child in info.children:
            if getattr(child, "name", None) == "br":
                if buf:
                    parts.append(re.sub(r"\s+", " ", "".join(buf)).strip())
                    buf = []
            elif getattr(child, "name", None) == "a":
                buf.append(child.get_text(" ", strip=True))
            else:
                buf.append(str(child) if isinstance(child, str)
                           else child.get_text(" ", strip=True))
        if buf:
            parts.append(re.sub(r"\s+", " ", "".join(buf)).strip())
        # parts[0] = title, parts[1] = authors, parts[2] = venue (italic)
        authors_str = parts[1] if len(parts) > 1 else ""
        venue = parts[2] if len(parts) > 2 else ""
        authors = _split_authors(authors_str)
        pubs.append({
            "title": title,
            "url": url,
            "authors": authors,
            "venue": venue,
        })
    return pubs


_AUTHOR_SPLIT_RE = re.compile(r",\s*and\s+|\s+and\s+|,\s*", re.I)


def _split_authors(s: str) -> List[str]:
    s = re.sub(r"\s+", " ", (s or "")).strip().rstrip(".")
    if not s:
        return []
    return [a.strip() for a in _AUTHOR_SPLIT_RE.split(s) if a.strip()]


def _name_keys(name: str) -> Tuple[str, str]:
    """('Eleanor Lin') -> ('lin', 'e'); used for fuzzy author matching."""
    parts = re.sub(r"[^A-Za-z\s\-]", " ", name).split()
    if not parts:
        return ("", "")
    last = parts[-1].lower()
    first_initial = parts[0][:1].lower() if parts[0] else ""
    return (last, first_initial)


def _author_matches_member(author: str, member_name: str) -> bool:
    a_last, a_first = _name_keys(author)
    m_last, m_first = _name_keys(member_name)
    if not m_last or not a_last:
        return False
    return a_last == m_last and (not a_first or not m_first or a_first == m_first)


# ---- Public API ----------------------------------------------------------

def refresh_profiles(force: bool = False) -> Dict:
    """Scrape the lab site if cache is stale. Returns the loaded profile dict."""
    paths.ensure_dirs()
    if not force and os.path.exists(CACHE_PATH):
        age = time.time() - os.path.getmtime(CACHE_PATH)
        if age < PAPER_CURATOR_PROFILE_REFRESH_DAYS * 86400:
            cache = _load_cache()
            # Cheap: always re-merge interests so edits to member_interests.yml
            # take effect on the next scheduled run without requiring a re-scrape.
            # Rewrites the cache file (and bumps its mtime) only if the merged
            # interests differ from what's already on disk, so the embed cache
            # is invalidated exactly when needed.
            if _apply_interests_in_place(cache):
                _save_cache(cache)
            return cache

    try:
        html = _fetch_lab_html(PAPER_CURATOR_LAB_URL)
    except Exception as e:
        logger.warning("lab site fetch failed (%s); keeping stale cache", e)
        return _load_cache()

    soup = BeautifulSoup(html, "html.parser")
    members = _parse_people_panel(soup)
    pubs = _parse_publications_panel(soup)

    # Group publications by member.
    for m in members:
        m["publications"] = [
            {"title": p["title"], "venue": p["venue"], "url": p["url"]}
            for p in pubs
            if any(_author_matches_member(a, m["name"]) for a in p["authors"])
        ][:25]  # cap to keep embedding cheap

    # Merge in hand-curated research focuses from data/member_interests.yml.
    interests = _load_member_interests()
    for m in members:
        themes = interests.get(m["name"])
        if themes:
            m["interests"] = themes

    cache = {
        "scraped_at": datetime.now().isoformat(timespec="seconds"),
        "lab_url": PAPER_CURATOR_LAB_URL,
        "members": members,
    }
    _save_cache(cache)
    _merge_slack_id_stub(members)
    logger.info("profiles refreshed: %d members, %d publications",
                len(members), len(pubs))
    return cache


def load_taggable_members(profile_cache: Optional[Dict] = None) -> List[Dict]:
    """Members the LLM may @-tag (PhD students + postdocs)."""
    cache = profile_cache or _load_cache()
    return [m for m in cache.get("members", []) if m.get("role") in TAGGABLE_ROLES]


def load_ranking_members(profile_cache: Optional[Dict] = None) -> List[Dict]:
    """Members whose interests influence the bi-encoder ranking (PhD students,
    postdocs, AND the PI). A superset of load_taggable_members."""
    cache = profile_cache or _load_cache()
    return [m for m in cache.get("members", []) if m.get("role") in RANKING_ROLES]


def load_slack_ids() -> Dict[str, Optional[str]]:
    if not os.path.exists(SLACK_IDS_PATH):
        return {}
    try:
        with open(SLACK_IDS_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}


# ---- Cache I/O -----------------------------------------------------------

def _load_cache() -> Dict:
    if not os.path.exists(CACHE_PATH):
        return {"scraped_at": "", "lab_url": "", "members": []}
    with open(CACHE_PATH, "r") as f:
        return json.load(f)


def _save_cache(data: Dict) -> None:
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, CACHE_PATH)


def _apply_interests_in_place(cache: Dict) -> bool:
    """Merge member_interests.yml into a loaded cache. Returns True if anything
    changed (caller should re-save so the cache mtime updates and the embed
    cache invalidates)."""
    interests = _load_member_interests()
    changed = False
    for m in cache.get("members", []):
        new = interests.get(m["name"]) or []
        old = m.get("interests") or []
        # Accept legacy string form so older caches don't trigger a needless rewrite.
        if isinstance(old, str):
            old = [old.strip()] if old.strip() else []
        if new != old:
            if not new:
                m.pop("interests", None)
            else:
                m["interests"] = new
            changed = True
    return changed


def _load_member_interests() -> Dict[str, List[str]]:
    """Load the hand-curated name -> interests map. Missing/empty file -> {}.

    Each value is a list of themed lines that will become separate bi-encoder
    query vectors. Backwards-compatible:
      - list[str]  -> used as-is
      - str        -> split on blank lines / newlines into themes
    """
    path = paths.MEMBER_INTERESTS_YML
    if not os.path.exists(path):
        # A missing interests file is never expected in normal operation — it
        # silently strips every member's themes and yields an empty digest
        # (this is exactly how the 2026-05-19 stale-process incident hid).
        logger.warning(
            "member_interests.yml not found at %s — every member will have "
            "no interests and the digest will be empty", path)
        return {}
    try:
        import yaml  # lazy
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning("could not load %s: %s", path, e)
        return {}
    if not isinstance(data, dict):
        logger.warning("%s must be a top-level mapping; ignoring", path)
        return {}
    out: Dict[str, List[str]] = {}
    for k, v in data.items():
        if v is None:
            continue
        themes = _coerce_themes(v)
        if themes:
            out[str(k)] = themes
    return out


def _coerce_themes(v) -> List[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    s = str(v).strip()
    if not s:
        return []
    # Block-scalar string: each non-empty line becomes a theme.
    lines = [ln.strip(" -\t") for ln in s.splitlines()]
    themes = [ln for ln in lines if ln]
    return themes or [s]


def _merge_slack_id_stub(members: List[Dict]) -> None:
    existing = load_slack_ids()
    out: Dict[str, Optional[str]] = {}
    for m in members:
        if m.get("role") not in TAGGABLE_ROLES:
            continue
        out[m["name"]] = existing.get(m["name"])
    # Preserve any IDs that point to members not currently scraped (manual additions).
    for k, v in existing.items():
        out.setdefault(k, v)
    tmp = SLACK_IDS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, SLACK_IDS_PATH)
