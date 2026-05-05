"""Resolve a URL to a paper citation using OpenAlex + Semantic Scholar.

Best-effort pipeline:
  1. Classify the URL (arXiv / DOI / other).
  2. Look up by ID in both OpenAlex and Semantic Scholar; cross-check titles.
  3. If no ID, fetch the page; pull DOI / arXiv ID from HTML meta tags.
  4. If still no ID and the URL is a PDF, download and grep its text.
  5. Last resort: title-search OpenAlex with the page <title> or PDF first page.

Returns a `Resolution` dict (see `_empty_resolution`) — never raises on
network errors; failure is encoded as `status="unresolved"` plus a `notes`
field explaining what went wrong.
"""
from __future__ import annotations

import io
import logging
import re
from html.parser import HTMLParser
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse

import requests

from config import PAPERS_HTTP_TIMEOUT, PAPERS_PDF_MAX_BYTES

logger = logging.getLogger(__name__)

# --- Regexes ---------------------------------------------------------------

# DOIs are 10.<registrant>/<suffix>; suffix can contain a wide range of chars
# but we trim trailing punctuation that frequently rides along in URLs.
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>)]+)", re.IGNORECASE)
# arXiv: new style 2401.12345, optional version; also old-style cs.CL/0501001
_ARXIV_NEW_RE = re.compile(r"\b(\d{4}\.\d{4,5})(v\d+)?\b")
_ARXIV_OLD_RE = re.compile(r"\b([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?\b")

_URL_RE = re.compile(r"https?://[^\s<>|\]]+")
# Slack wraps links as <https://url> or <https://url|label>. Strip both forms.
_SLACK_LINK_RE = re.compile(r"<(https?://[^|>]+)(?:\|[^>]*)?>")

OPENALEX_API = "https://api.openalex.org/works"
S2_API = "https://api.semanticscholar.org/graph/v1/paper"
S2_FIELDS = "externalIds,title,authors.name,year,venue,journal,publicationVenue,publicationTypes,abstract"

USER_AGENT = "lab-paper-monitor/0.1 (mailto:jurgens@umich.edu)"


# --- Public types ----------------------------------------------------------

def _empty_resolution(url: str) -> Dict:
    return {
        "source_url": url,
        "canonical_url": None,
        "doi": None,
        "arxiv_id": None,
        "openalex_id": None,
        "s2_id": None,
        "title": None,
        "authors": [],
        "year": None,
        "venue": None,
        "citation": None,
        "bibtex": None,
        "resolved_via": [],   # ordered list of strategies that contributed
        "status": "unresolved",  # "resolved" | "partial" | "unresolved"
        "notes": [],
    }


# --- URL extraction --------------------------------------------------------

def extract_urls(text: str) -> List[str]:
    """Pull all http(s) URLs from a Slack message's `text`.

    Slack wraps links as `<url>` or `<url|label>`; we unwrap both. Bare URLs
    are picked up too, in case the message went through a non-Slack source.
    """
    urls: List[str] = []
    seen = set()
    for m in _SLACK_LINK_RE.finditer(text):
        u = m.group(1)
        if u not in seen:
            seen.add(u); urls.append(u)
    # Strip Slack-wrapped portions before scanning for bare URLs so we don't
    # double-count them.
    bare_text = _SLACK_LINK_RE.sub(" ", text)
    for m in _URL_RE.finditer(bare_text):
        u = m.group(0).rstrip(".,);]>")
        if u not in seen:
            seen.add(u); urls.append(u)
    return urls


# --- URL classification ----------------------------------------------------

def classify(url: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (doi, arxiv_id) extracted from the URL itself, if any."""
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    path = parsed.path or ""

    if "doi.org" in host:
        # /10.xxxx/yyy
        m = _DOI_RE.search(path)
        if m:
            return _clean_doi(m.group(1)), None

    if "arxiv.org" in host:
        # /abs/2401.12345  /pdf/2401.12345v2  /abs/cs.CL/0501001
        m = _ARXIV_NEW_RE.search(path) or _ARXIV_OLD_RE.search(path)
        if m:
            return None, m.group(1)

    # Some publishers embed the DOI in their URL path.
    m = _DOI_RE.search(url)
    if m:
        return _clean_doi(m.group(1)), None

    return None, None


def _clean_doi(doi: str) -> str:
    return doi.rstrip(".,);]>'\"").lower()


# --- HTTP helpers ----------------------------------------------------------

def _get(url: str, **kw) -> Optional[requests.Response]:
    headers = kw.pop("headers", {}) or {}
    headers.setdefault("User-Agent", USER_AGENT)
    try:
        return requests.get(url, timeout=PAPERS_HTTP_TIMEOUT, headers=headers,
                            allow_redirects=True, **kw)
    except requests.RequestException as e:
        logger.info(f"GET {url} failed: {e}")
        return None


# --- OpenAlex / Semantic Scholar lookups -----------------------------------

def lookup_openalex(*, doi: Optional[str] = None, arxiv_id: Optional[str] = None,
                    title: Optional[str] = None) -> Optional[Dict]:
    if doi:
        r = _get(f"{OPENALEX_API}/doi:{doi}")
    elif arxiv_id:
        # OpenAlex indexes arXiv via DOI 10.48550/arXiv.xxxx
        r = _get(f"{OPENALEX_API}/doi:10.48550/arXiv.{arxiv_id}")
    elif title:
        r = _get(OPENALEX_API, params={"search": title, "per-page": 1})
        if r is not None and r.ok:
            data = r.json()
            results = data.get("results") or []
            return results[0] if results else None
        return None
    else:
        return None
    if r is None or not r.ok:
        return None
    return r.json()


def lookup_s2(*, doi: Optional[str] = None, arxiv_id: Optional[str] = None,
              title: Optional[str] = None) -> Optional[Dict]:
    if doi:
        r = _get(f"{S2_API}/DOI:{doi}", params={"fields": S2_FIELDS})
    elif arxiv_id:
        r = _get(f"{S2_API}/arXiv:{arxiv_id}", params={"fields": S2_FIELDS})
    elif title:
        r = _get(f"{S2_API}/search",
                 params={"query": title, "limit": 1, "fields": S2_FIELDS})
        if r is not None and r.ok:
            data = r.json()
            hits = data.get("data") or []
            return hits[0] if hits else None
        return None
    else:
        return None
    if r is None or not r.ok:
        return None
    return r.json()


# --- HTML / PDF scraping ---------------------------------------------------

class _MetaTagParser(HTMLParser):
    """Collect <meta name=...> and <meta property=...> tags + <title>."""
    def __init__(self) -> None:
        super().__init__()
        self.metas: Dict[str, List[str]] = {}
        self.title: Optional[str] = None
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "meta":
            d = dict(attrs)
            key = (d.get("name") or d.get("property") or "").lower()
            val = d.get("content")
            if key and val:
                self.metas.setdefault(key, []).append(val)
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title and self.title is None:
            t = data.strip()
            if t:
                self.title = t


def scrape_html(url: str) -> Dict:
    """Return {doi, arxiv_id, title, authors[]} pulled from the page's meta tags."""
    out: Dict = {"doi": None, "arxiv_id": None, "title": None, "authors": []}
    r = _get(url)
    if r is None or not r.ok:
        return out
    ctype = (r.headers.get("Content-Type") or "").lower()
    if "html" not in ctype and "xml" not in ctype:
        return out
    p = _MetaTagParser()
    try:
        p.feed(r.text)
    except Exception as e:
        logger.info(f"HTML parse failed for {url}: {e}")
        return out

    def first(*keys: str) -> Optional[str]:
        for k in keys:
            vs = p.metas.get(k)
            if vs:
                return vs[0]
        return None

    def all_(*keys: str) -> List[str]:
        vals: List[str] = []
        for k in keys:
            vals.extend(p.metas.get(k, []))
        return vals

    doi = first("citation_doi", "dc.identifier", "prism.doi")
    if doi:
        m = _DOI_RE.search(doi)
        if m:
            out["doi"] = _clean_doi(m.group(1))

    arxiv = first("citation_arxiv_id")
    if arxiv:
        out["arxiv_id"] = arxiv.strip()

    out["title"] = first("citation_title", "dc.title", "og:title") or p.title
    out["authors"] = [a.strip() for a in all_("citation_author", "dc.creator") if a.strip()]

    # Some pages link the DOI in body text only.
    if not out["doi"]:
        m = _DOI_RE.search(r.text)
        if m:
            out["doi"] = _clean_doi(m.group(1))
    if not out["arxiv_id"]:
        m = _ARXIV_NEW_RE.search(r.text) or _ARXIV_OLD_RE.search(r.text)
        if m:
            out["arxiv_id"] = m.group(1)
    return out


def scrape_pdf(url: str) -> Dict:
    """Download a PDF and grep its text for DOI / arXiv ID. Falls back to title."""
    out: Dict = {"doi": None, "arxiv_id": None, "title": None}
    try:
        from pypdf import PdfReader
    except ImportError:
        out["_error"] = "pypdf not installed"
        return out

    r = _get(url, stream=True)
    if r is None or not r.ok:
        return out
    ctype = (r.headers.get("Content-Type") or "").lower()
    clen = int(r.headers.get("Content-Length") or 0)
    if clen and clen > PAPERS_PDF_MAX_BYTES:
        out["_error"] = f"pdf too large: {clen} bytes"
        return out
    if "pdf" not in ctype and not url.lower().endswith(".pdf"):
        # Some servers misreport content-type; trust the extension if present.
        return out
    buf = io.BytesIO()
    total = 0
    for chunk in r.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue
        total += len(chunk)
        if total > PAPERS_PDF_MAX_BYTES:
            out["_error"] = "pdf exceeds size cap during stream"
            return out
        buf.write(chunk)
    buf.seek(0)
    try:
        reader = PdfReader(buf)
    except Exception as e:
        out["_error"] = f"pdf parse failed: {e}"
        return out
    # Grab text from the first 2 pages — that's where DOIs and titles live.
    text_parts: List[str] = []
    for page in reader.pages[:2]:
        try:
            text_parts.append(page.extract_text() or "")
        except Exception:
            continue
    text = "\n".join(text_parts)
    m = _DOI_RE.search(text)
    if m:
        out["doi"] = _clean_doi(m.group(1))
    m = _ARXIV_NEW_RE.search(text) or _ARXIV_OLD_RE.search(text)
    if m:
        out["arxiv_id"] = m.group(1)
    # First non-empty line is usually the title (heuristic).
    for line in text.splitlines():
        line = line.strip()
        if len(line) > 15 and not line.lower().startswith(("arxiv", "http", "doi")):
            out["title"] = line
            break
    return out


# --- Cross-check + merge ---------------------------------------------------

def _normalize_title(t: Optional[str]) -> str:
    if not t:
        return ""
    return re.sub(r"[^a-z0-9]+", "", t.lower())


def _title_match(a: Optional[str], b: Optional[str]) -> bool:
    na, nb = _normalize_title(a), _normalize_title(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    # Allow prefix matches (S2 sometimes truncates) and high overlap.
    short, long_ = sorted([na, nb], key=len)
    if short and long_.startswith(short):
        return True
    return False


def _author_names(work: Dict, source: str) -> List[str]:
    if not work:
        return []
    if source == "openalex":
        return [a.get("author", {}).get("display_name", "")
                for a in (work.get("authorships") or []) if a.get("author")]
    if source == "s2":
        return [a.get("name", "") for a in (work.get("authors") or []) if a.get("name")]
    return []


def _venue(work: Dict, source: str) -> Optional[str]:
    if not work:
        return None
    if source == "openalex":
        host = (work.get("primary_location") or {}).get("source") or {}
        return host.get("display_name")
    if source == "s2":
        v = work.get("publicationVenue") or {}
        return v.get("name") or work.get("venue") or (work.get("journal") or {}).get("name")
    return None


def merge(oa: Optional[Dict], s2: Optional[Dict],
          *, doi: Optional[str], arxiv_id: Optional[str],
          source_url: str) -> Dict:
    res = _empty_resolution(source_url)
    res["doi"] = doi
    res["arxiv_id"] = arxiv_id

    oa_title = (oa or {}).get("title") or (oa or {}).get("display_name")
    s2_title = (s2 or {}).get("title")
    oa_authors = _author_names(oa, "openalex")
    s2_authors = _author_names(s2, "s2")
    oa_year = (oa or {}).get("publication_year")
    s2_year = (s2 or {}).get("year")

    if oa:
        res["openalex_id"] = (oa.get("id") or "").rsplit("/", 1)[-1] or None
        res["resolved_via"].append("openalex")
        if not res["doi"] and oa.get("doi"):
            res["doi"] = _clean_doi(oa["doi"].replace("https://doi.org/", ""))
    if s2:
        res["s2_id"] = s2.get("paperId")
        res["resolved_via"].append("semantic_scholar")
        if not res["doi"]:
            ext = (s2.get("externalIds") or {})
            if ext.get("DOI"):
                res["doi"] = _clean_doi(ext["DOI"])
            if not res["arxiv_id"] and ext.get("ArXiv"):
                res["arxiv_id"] = ext["ArXiv"]

    # Prefer OpenAlex for the canonical fields; fall back to S2.
    res["title"] = oa_title or s2_title
    res["authors"] = oa_authors or s2_authors
    res["year"] = oa_year or s2_year
    res["venue"] = _venue(oa, "openalex") or _venue(s2, "s2")

    # Cross-check.
    if oa_title and s2_title and not _title_match(oa_title, s2_title):
        res["notes"].append(
            f"title_mismatch: openalex={oa_title!r} s2={s2_title!r}"
        )

    if res["doi"]:
        res["canonical_url"] = f"https://doi.org/{res['doi']}"
    elif res["arxiv_id"]:
        res["canonical_url"] = f"https://arxiv.org/abs/{res['arxiv_id']}"
    else:
        res["canonical_url"] = source_url

    if res["title"] and res["authors"]:
        res["status"] = "resolved"
    elif res["title"]:
        res["status"] = "partial"
    return res


# --- Top-level entrypoint --------------------------------------------------

def resolve_url(url: str) -> Dict:
    """Best-effort: turn `url` into a populated Resolution dict."""
    doi, arxiv_id = classify(url)

    if doi or arxiv_id:
        oa = lookup_openalex(doi=doi, arxiv_id=arxiv_id)
        s2 = lookup_s2(doi=doi, arxiv_id=arxiv_id)
        res = merge(oa, s2, doi=doi, arxiv_id=arxiv_id, source_url=url)
        if res["status"] != "unresolved":
            return res
        # Fall through to scraping if both API lookups were empty.

    # No ID from the URL — scrape HTML for citation_* meta tags.
    scraped = scrape_html(url)
    doi = doi or scraped.get("doi")
    arxiv_id = arxiv_id or scraped.get("arxiv_id")
    scraped_title = scraped.get("title")

    if doi or arxiv_id:
        oa = lookup_openalex(doi=doi, arxiv_id=arxiv_id)
        s2 = lookup_s2(doi=doi, arxiv_id=arxiv_id)
        res = merge(oa, s2, doi=doi, arxiv_id=arxiv_id, source_url=url)
        if res["status"] != "unresolved":
            res["resolved_via"].insert(0, "html_meta")
            return res

    # Try PDF parsing if URL points to a PDF.
    pdf = {}
    if url.lower().endswith(".pdf") or _looks_like_pdf(url):
        pdf = scrape_pdf(url)
        doi = doi or pdf.get("doi")
        arxiv_id = arxiv_id or pdf.get("arxiv_id")
        if doi or arxiv_id:
            oa = lookup_openalex(doi=doi, arxiv_id=arxiv_id)
            s2 = lookup_s2(doi=doi, arxiv_id=arxiv_id)
            res = merge(oa, s2, doi=doi, arxiv_id=arxiv_id, source_url=url)
            if res["status"] != "unresolved":
                res["resolved_via"].insert(0, "pdf_text")
                return res

    # Last resort: title search.
    title_for_search = scraped_title or pdf.get("title")
    if title_for_search:
        oa = lookup_openalex(title=title_for_search)
        s2 = lookup_s2(title=title_for_search)
        if oa or s2:
            res = merge(oa, s2, doi=None, arxiv_id=None, source_url=url)
            res["resolved_via"].insert(0, "title_search")
            if res["title"] and not _title_match(res["title"], title_for_search):
                res["notes"].append(
                    f"title_search_low_confidence: query={title_for_search!r} "
                    f"matched={res['title']!r}"
                )
                res["status"] = "partial"
            return res

    # Give up but keep what we have for manual review.
    res = _empty_resolution(url)
    res["title"] = scraped_title or pdf.get("title")
    res["notes"].append("no_id_found_after_html_pdf_and_title_search")
    if pdf.get("_error"):
        res["notes"].append(pdf["_error"])
    return res


def _looks_like_pdf(url: str) -> bool:
    parsed = urlparse(url)
    return any(seg.endswith(".pdf") for seg in parsed.path.split("/"))
