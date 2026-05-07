"""Paper-source fetchers.

A `Paper` is a dict:
    {id, source, title, abstract, authors, url, pdf_url, published}
- id: canonical, stable across sources: "arxiv:<id>" | "doi:<doi>" | "urlhash:<sha1>"
- published: ISO date string (YYYY-MM-DD) when known, else "" (best-effort).

`fetch_all(since_iso)` yields papers from every enabled source. Per-source
failures are logged and skipped — one bad feed never breaks the run.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional
from xml.etree import ElementTree as ET

import requests

logger = logging.getLogger(__name__)

USER_AGENT = "lab-paper-curator/0.1 (mailto:jurgens@umich.edu)"
HTTP_TIMEOUT = 20

ARXIV_ATOM_NS = {"a": "http://www.w3.org/2005/Atom",
                 "arxiv": "http://arxiv.org/schemas/atom"}


# --- Public types ---------------------------------------------------------

def _empty_paper() -> Dict:
    return {"id": "", "source": "", "title": "", "abstract": "",
            "authors": [], "url": "", "pdf_url": "", "published": ""}


# --- Canonical IDs --------------------------------------------------------

_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?$")
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+)", re.I)


def canonical_id(*, arxiv_id: str = "", doi: str = "", url: str = "") -> str:
    if arxiv_id:
        m = _ARXIV_ID_RE.search(arxiv_id)
        if m:
            return f"arxiv:{m.group(1)}"
        return f"arxiv:{arxiv_id}"
    if doi:
        return f"doi:{doi.lower()}"
    if url:
        return f"urlhash:{hashlib.sha1(url.encode()).hexdigest()[:16]}"
    return ""


# --- Fetchers -------------------------------------------------------------

def fetch_arxiv(category: str, max_results: int = 200) -> List[Dict]:
    """Use arXiv's Atom API. Sort by submittedDate desc."""
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"cat:{category}",
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": max_results,
    }
    r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT},
                     timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    out: List[Dict] = []
    for entry in root.findall("a:entry", ARXIV_ATOM_NS):
        arxiv_url = entry.findtext("a:id", "", ARXIV_ATOM_NS).strip()
        title = (entry.findtext("a:title", "", ARXIV_ATOM_NS) or "").strip()
        abstract = (entry.findtext("a:summary", "", ARXIV_ATOM_NS) or "").strip()
        published = (entry.findtext("a:published", "", ARXIV_ATOM_NS) or "")[:10]
        authors = [a.findtext("a:name", "", ARXIV_ATOM_NS).strip()
                   for a in entry.findall("a:author", ARXIV_ATOM_NS)]
        # arxiv_url like https://arxiv.org/abs/2501.12345v1
        m = re.search(r"arxiv\.org/abs/([^v\s]+)", arxiv_url)
        arxiv_id = m.group(1) if m else ""
        pdf_url = ""
        for link in entry.findall("a:link", ARXIV_ATOM_NS):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")
        p = _empty_paper()
        p.update({
            "id": canonical_id(arxiv_id=arxiv_id),
            "source": f"arxiv:{category}",
            "title": _collapse_ws(title),
            "abstract": _collapse_ws(abstract),
            "authors": authors,
            "url": arxiv_url,
            "pdf_url": pdf_url,
            "published": published,
        })
        if p["id"] and p["title"]:
            out.append(p)
    return out


def fetch_osf(provider: str) -> List[Dict]:
    """OSF preprints API. provider is e.g. 'socarxiv', 'psyarxiv'."""
    url = "https://api.osf.io/v2/preprints/"
    params = {
        "filter[provider]": provider,
        "sort": "-date_published",
        "page[size]": 100,
    }
    r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT},
                     timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json().get("data", [])
    out: List[Dict] = []
    for item in data:
        attrs = item.get("attributes", {}) or {}
        title = (attrs.get("title") or "").strip()
        abstract = (attrs.get("description") or "").strip()
        published = (attrs.get("date_published") or "")[:10]
        doi = attrs.get("doi") or ""
        landing = (item.get("links") or {}).get("html") or ""
        pid = canonical_id(doi=doi) if doi else canonical_id(url=landing)
        p = _empty_paper()
        p.update({
            "id": pid,
            "source": f"osf:{provider}",
            "title": _collapse_ws(title),
            "abstract": _collapse_ws(abstract),
            "authors": [],  # OSF needs a second call per preprint to fetch contributors
            "url": landing,
            "pdf_url": "",
            "published": published,
        })
        if p["id"] and p["title"]:
            out.append(p)
    return out


def fetch_rss(url: str, source_id: str) -> List[Dict]:
    """Generic RSS / Atom feed parser via feedparser. Used for journals + Anthology."""
    import feedparser  # lazy import
    fp = feedparser.parse(url, agent=USER_AGENT)
    if fp.bozo and not fp.entries:
        raise RuntimeError(f"feedparser failure for {url}: {fp.bozo_exception}")
    out: List[Dict] = []
    for e in fp.entries:
        title = (e.get("title") or "").strip()
        link = e.get("link") or ""
        summary = (e.get("summary") or e.get("description") or "").strip()
        # Strip HTML tags from summary for cleaner embedding input.
        summary = re.sub(r"<[^>]+>", " ", summary)
        published = ""
        for k in ("published", "updated", "date"):
            if e.get(k):
                published = e[k][:10]
                break
        # Try DOI in id / link
        doi = ""
        for s in (e.get("id", ""), link):
            m = _DOI_RE.search(s or "")
            if m:
                doi = m.group(1)
                break
        # Try arXiv id in link (Anthology / arXiv mirrors)
        arxiv_id = ""
        m = re.search(r"arxiv\.org/abs/([^v\s/]+)", link)
        if m:
            arxiv_id = m.group(1)
        if arxiv_id:
            pid = canonical_id(arxiv_id=arxiv_id)
        elif doi:
            pid = canonical_id(doi=doi)
        else:
            pid = canonical_id(url=link or e.get("id", ""))
        authors = [a.get("name", "") for a in (e.get("authors") or [])]
        p = _empty_paper()
        p.update({
            "id": pid,
            "source": source_id,
            "title": _collapse_ws(title),
            "abstract": _collapse_ws(summary),
            "authors": [a for a in authors if a],
            "url": link,
            "pdf_url": "",
            "published": published,
        })
        if p["id"] and p["title"]:
            out.append(p)
    return out


def fetch_openreview(venueid: str, limit: int = 200) -> List[Dict]:
    """OpenReview API v2. `venueid` is e.g. 'ICLR.cc/2026/Conference' or
    'COLM/2025/Conference' — the same string OpenReview uses internally."""
    url = "https://api2.openreview.net/notes"
    params = {
        "content.venueid": venueid,
        "sort": "cdate:desc",
        "limit": min(limit, 1000),
    }
    r = requests.get(url, params=params,
                     headers={"User-Agent": USER_AGENT}, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    out: List[Dict] = []
    for n in data.get("notes", []):
        content = n.get("content", {}) or {}
        title = _or_value(content.get("title"))
        abstract = _or_value(content.get("abstract"))
        authors_raw = _or_value(content.get("authors"))
        authors = authors_raw if isinstance(authors_raw, list) else []
        forum = n.get("forum") or n.get("id") or ""
        link = f"https://openreview.net/forum?id={forum}" if forum else ""
        cdate = n.get("cdate")
        published = (datetime.fromtimestamp(cdate / 1000, tz=timezone.utc)
                     .date().isoformat()) if cdate else ""
        pid = canonical_id(url=link) if link else ""
        p = _empty_paper()
        p.update({
            "id": pid,
            "source": f"openreview:{venueid}",
            "title": _collapse_ws(title),
            "abstract": _collapse_ws(abstract),
            "authors": [a for a in authors if isinstance(a, str)],
            "url": link,
            "pdf_url": "",
            "published": published,
        })
        if p["id"] and p["title"]:
            out.append(p)
    return out


def _or_value(v):
    """OpenReview API v2 wraps content fields as {'value': X}; v1 was raw."""
    if isinstance(v, dict) and "value" in v:
        return v["value"]
    return v if v is not None else ""


def fetch_huggingface_daily(days: int = 2) -> List[Dict]:
    """HF Daily Papers — curated ~5-15 papers/day. The default endpoint
    returns the latest ~50 entries (covers several recent days at once);
    using it without a date param avoids the 400 you get for future-dated
    requests. We keep `days` as a hint for log output only — server picks
    the window."""
    url = "https://huggingface.co/api/daily_papers"
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT},
                         timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        entries = r.json()
    except Exception as e:
        logger.warning("hf daily papers failed: %s", e)
        return []
    if not isinstance(entries, list):
        return []
    out: List[Dict] = []
    for entry in entries:
        paper = entry.get("paper", {}) or {}
        arxiv_id = (paper.get("id") or "").strip()
        title = (paper.get("title") or entry.get("title") or "").strip()
        abstract = (paper.get("summary") or entry.get("summary") or "").strip()
        published = ((paper.get("publishedAt") or
                      entry.get("publishedAt") or "")[:10])
        authors = [a.get("name", "") for a in (paper.get("authors") or [])
                   if isinstance(a, dict)]
        link = (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id
                else f"https://huggingface.co/papers/{paper.get('id','')}")
        pid = canonical_id(arxiv_id=arxiv_id) if arxiv_id else canonical_id(url=link)
        p = _empty_paper()
        p.update({
            "id": pid,
            "source": "huggingface:daily",
            "title": _collapse_ws(title),
            "abstract": _collapse_ws(abstract),
            "authors": [a for a in authors if a],
            "url": link,
            "pdf_url": (f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                        if arxiv_id else ""),
            "published": published,
        })
        if p["id"] and p["title"]:
            out.append(p)
    return out


# --- Registry dispatch ----------------------------------------------------

def _load_registry(path: Optional[str] = None) -> List[Dict]:
    import yaml  # lazy import
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "data", "sources.yml")
    with open(path, "r") as f:
        return yaml.safe_load(f) or []


def fetch_all(registry_path: Optional[str] = None,
              since_iso: Optional[str] = None) -> List[Dict]:
    """Fetch papers from every enabled source. Per-source try/except — one bad
    source never breaks the run.

    `since_iso` is a YYYY-MM-DD floor: results published before that date are
    dropped post-hoc. Sources don't all support server-side date filters.
    """
    registry = _load_registry(registry_path)
    papers: List[Dict] = []
    for entry in registry:
        if not entry.get("enabled", True):
            continue
        sid = entry.get("id", "?")
        kind = entry.get("kind", "")
        try:
            if kind == "arxiv":
                got = fetch_arxiv(entry["category"], entry.get("max_results", 200))
            elif kind == "osf":
                got = fetch_osf(entry["provider"])
            elif kind in ("rss", "anthology_rss"):
                got = fetch_rss(entry["url"], sid)
            elif kind == "openreview":
                got = fetch_openreview(entry["venueid"],
                                       entry.get("limit", 200))
            elif kind == "huggingface_daily":
                got = fetch_huggingface_daily(entry.get("days", 2))
            else:
                logger.warning("unknown source kind %r for %s", kind, sid)
                continue
            logger.info("source %s: %d papers", sid, len(got))
            papers.extend(got)
        except Exception as e:
            logger.warning("source %s failed: %s", sid, e)
    if since_iso:
        papers = [p for p in papers
                  if not p["published"] or p["published"] >= since_iso]
    return papers


# --- Helpers --------------------------------------------------------------

def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()
