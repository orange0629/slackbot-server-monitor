"""Build a BibTeX entry and a human-readable citation for a Resolution.

Strategy:
  1. If a DOI is present, ask doi.org for publisher-formatted BibTeX via
     content negotiation (`Accept: application/x-bibtex`). This is the
     canonical source and avoids a lot of edge cases.
  2. Otherwise (or if doi.org refuses), assemble a BibTeX entry from the
     resolved fields.

We also produce a single-line citation string that's safe to drop into a
website list ("Authors (Year). Title. Venue.").
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

import requests

from config import PAPERS_HTTP_TIMEOUT
from .resolver import USER_AGENT

logger = logging.getLogger(__name__)


def fetch_bibtex_via_doi(doi: str) -> Optional[str]:
    if not doi:
        return None
    try:
        r = requests.get(
            f"https://doi.org/{doi}",
            headers={"Accept": "application/x-bibtex; charset=utf-8",
                     "User-Agent": USER_AGENT},
            timeout=PAPERS_HTTP_TIMEOUT,
            allow_redirects=True,
        )
    except requests.RequestException as e:
        logger.info(f"doi.org bibtex fetch failed for {doi}: {e}")
        return None
    if not r.ok:
        return None
    text = (r.text or "").strip()
    if text.startswith("@"):
        return text
    return None


def _format_authors_bib(authors: List[str]) -> str:
    return " and ".join(a for a in authors if a)


def _format_authors_cite(authors: List[str]) -> str:
    if not authors:
        return ""
    if len(authors) == 1:
        return authors[0]
    if len(authors) == 2:
        return f"{authors[0]} and {authors[1]}"
    return f"{authors[0]} et al."


def _bib_key(authors: List[str], year: Optional[int], title: Optional[str]) -> str:
    last = ""
    if authors:
        # naive last-name extraction
        first = authors[0].strip()
        parts = first.split()
        last = parts[-1] if parts else first
    last = re.sub(r"[^A-Za-z]", "", last) or "anon"
    yr = str(year) if year else "nodate"
    word = ""
    if title:
        for tok in re.findall(r"[A-Za-z]+", title):
            if len(tok) >= 4 and tok.lower() not in {"with", "from", "into", "this", "that", "what", "when", "where", "their", "using"}:
                word = tok.lower(); break
    return f"{last.lower()}{yr}{word}"


def _bib_escape(s: str) -> str:
    if s is None:
        return ""
    # Wrap in braces to preserve case; escape stray braces.
    return s.replace("\\", "").replace("{", "(").replace("}", ")")


def build_bibtex(res: Dict) -> Optional[str]:
    """Assemble a BibTeX record from resolver fields. Used when DOI fetch fails."""
    if not res.get("title"):
        return None
    entry_type = "misc"
    if res.get("doi") or res.get("venue"):
        entry_type = "article"
    if res.get("arxiv_id") and not res.get("doi"):
        entry_type = "misc"

    key = _bib_key(res.get("authors") or [], res.get("year"), res.get("title"))
    fields = []
    if res.get("authors"):
        fields.append(("author", _format_authors_bib(res["authors"])))
    fields.append(("title", _bib_escape(res["title"])))
    if res.get("year"):
        fields.append(("year", str(res["year"])))
    if res.get("venue"):
        fields.append(("journal", _bib_escape(res["venue"])))
    if res.get("doi"):
        fields.append(("doi", res["doi"]))
    if res.get("arxiv_id"):
        fields.append(("eprint", res["arxiv_id"]))
        fields.append(("archivePrefix", "arXiv"))
    if res.get("canonical_url"):
        fields.append(("url", res["canonical_url"]))

    body = ",\n  ".join(f"{k} = {{{v}}}" for k, v in fields)
    return f"@{entry_type}{{{key},\n  {body}\n}}"


def build_citation(res: Dict) -> Optional[str]:
    title = res.get("title")
    if not title:
        return None
    parts = []
    authors = _format_authors_cite(res.get("authors") or [])
    if authors:
        parts.append(authors)
    if res.get("year"):
        parts.append(f"({res['year']})")
    head = " ".join(parts)
    venue = res.get("venue")
    tail = f" {venue}." if venue else ""
    if head:
        return f"{head}. {title}.{tail}".strip()
    return f"{title}.{tail}".strip()


def attach_citation_and_bibtex(res: Dict) -> Dict:
    """Mutate `res` in place to add `citation` and `bibtex` fields."""
    res["citation"] = build_citation(res)
    bib = None
    if res.get("doi"):
        bib = fetch_bibtex_via_doi(res["doi"])
    if not bib:
        bib = build_bibtex(res)
    res["bibtex"] = bib
    return res
