"""Pull GIFs from the Giphy API, embed them, and append to the on-disk index.

Two entry points:
  - `run_giphy_refresh(...)` — called by the scheduled task in main.py and by
    the standalone CLI below. Idempotent: dedups against the existing index
    by giphy_id and only embeds new candidates.
  - `python -m gif_reply.giphy_refresh ...` — standalone CLI for first-time
    bulk builds (or one-off ad-hoc fetches).

Inputs come from a curated list of search queries plus the /trending feed.
Each candidate is filtered to `g`/`pg` rating, downloaded as a small mp4,
the first frame is extracted with imageio, and the frame is embedded with
the configured backend's image encoder.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
import requests

logger = logging.getLogger(__name__)


GIPHY_API = "https://api.giphy.com/v1/gifs"
USER_AGENT = "slackbot-server-monitor/gif-reply"
ALLOWED_RATINGS = {"g", "pg"}

# Reaction-GIF-shaped query list. Tuned for "expressive moods + work / lab
# slang", since that's what Slack mentions tend to be. Extend freely.
DEFAULT_QUERIES = [
    # --- Core reactions / moods ---
    "happy", "sad", "excited", "tired", "exhausted", "confused", "frustrated",
    "angry", "annoyed", "anxious", "stressed", "overwhelmed", "relieved",
    "proud", "embarrassed", "awkward", "nervous", "calm", "bored", "amused",
    "agree", "disagree", "applause", "clapping", "celebrating", "dancing",
    "thinking", "pondering", "wow", "lol", "rofl", "facepalm", "eye roll",
    "shrug", "mind blown", "head explode", "thumbs up", "thumbs down",
    "high five", "fist bump", "shocked", "surprise", "speechless", "scared",
    "worried", "crying", "sobbing", "laughing", "smiling", "winking",
    "love it", "hate it", "yes", "no", "okay", "maybe", "nope", "yep",
    "approved", "rejected", "interesting", "boring", "what", "why", "where",
    "how", "huh", "really", "no way", "for real", "seriously",

    # --- Praise / encouragement ---
    "great job", "good job", "well done", "nailed it", "winning", "victory",
    "champion", "mvp", "you got this", "keep going", "almost there",
    "congrats", "congratulations", "way to go", "impressive", "amazing",
    "awesome", "fantastic", "brilliant", "genius", "rockstar", "legend",
    "hero", "salute", "respect", "bow down", "take a bow", "standing ovation",

    # --- Fail / setbacks ---
    "fail", "epic fail", "loss", "oops", "mistake", "broken", "disaster",
    "trainwreck", "dumpster fire", "this is fine", "everything is fine",
    "panic", "running away", "hiding", "rip", "dead", "ded", "dying inside",

    # --- Science / research / lab life ---
    "science", "scientist", "lab", "laboratory", "experiment", "test tube",
    "beaker", "microscope", "chemistry", "physics", "biology", "neuroscience",
    "brain", "dna", "genetics", "research", "researcher", "phd", "phd life",
    "grad student", "graduate school", "thesis", "dissertation", "defense",
    "publish or perish", "peer review", "rejected paper", "accepted paper",
    "submission", "deadline", "conference", "poster session", "keynote",
    "talk", "presentation", "slides", "powerpoint", "whiteboard", "math",
    "equation", "statistics", "data", "data science", "machine learning",
    "artificial intelligence", "neural network", "algorithm", "code",
    "coding", "programming", "debugging", "bug", "error", "stack overflow",
    "compile", "merge conflict", "git", "rocket science", "eureka",
    "discovery", "experiment failed", "experiment worked", "hypothesis",
    "lab coat", "safety goggles", "explosion", "smoke", "boom", "magnet",

    # --- School / university / teaching ---
    "school", "classroom", "student", "students", "teacher", "professor",
    "lecture", "lecturing", "teaching", "tutoring", "homework", "studying",
    "study group", "library", "exam", "midterm", "finals", "test", "quiz",
    "pop quiz", "cramming", "all nighter", "pulling all nighter",
    "fall asleep studying", "back to school", "first day of school",
    "graduation", "graduating", "diploma", "cap and gown", "valedictorian",
    "raise hand", "raising hand", "answer question", "wrong answer",
    "right answer", "gold star", "report card", "grades", "a plus", "f minus",
    "detention", "principal", "school bus", "recess", "field trip", "campus",
    "university", "college", "dorm", "lecture hall", "office hours",
    "advisor meeting", "syllabus", "textbook", "highlighter", "notebook",
    "back to class", "snow day",

    # --- Workplace / office life ---
    "office", "office life", "cubicle", "desk", "open office", "remote work",
    "work from home", "zoom call", "video call", "video meeting",
    "conference call", "meeting", "another meeting", "this could have been an email",
    "boring meeting", "long meeting", "all hands", "stand up", "scrum",
    "kickoff", "deadline approaching", "shipping it", "ship it", "ship it now",
    "monday", "tuesday", "wednesday", "hump day", "thursday", "friday",
    "weekend", "tgif", "monday morning", "monday motivation", "case of the mondays",
    "out of office", "vacation", "pto", "sick day", "promotion", "raise",
    "fired", "you're fired", "quit", "i quit", "resign", "great resignation",
    "burnout", "burned out", "overworked", "working late", "overtime",
    "clock out", "punching out", "lunch break", "coffee break", "watercooler",
    "boss", "manager", "ceo", "intern", "new hire", "first day",
    "onboarding", "team meeting", "team building", "happy hour",
    "annual review", "performance review", "kpi", "quarterly report",
    "powerpoint slide", "spreadsheet", "excel", "email", "inbox zero",
    "too many emails", "reply all", "out of the loop", "back to the grind",
    "grinding", "hustle", "the grind",

    # --- Coffee / food / slice of life ---
    "coffee", "more coffee", "need coffee", "espresso", "tea", "matcha",
    "energy drink", "donut", "doughnut", "bagel", "breakfast", "brunch",
    "lunch", "dinner", "snack", "snacking", "hungry", "starving", "stuffed",
    "pizza", "tacos", "burger", "salad", "ramen", "sushi", "ice cream",
    "cake", "birthday cake", "cookie", "popcorn", "wine", "beer", "cheers",
    "toast", "drinking", "thirsty", "water", "stay hydrated",

    # --- Sleep / rest ---
    "sleeping", "sleepy", "yawning", "nap", "naptime", "wake up", "alarm",
    "morning person", "not a morning person", "good morning", "good night",
    "rise and shine", "snooze",

    # --- Weather / seasons / general life ---
    "rainy day", "sunny", "cold", "freezing", "hot", "summer", "winter",
    "fall", "autumn", "spring", "snow", "snowing", "vacation mode", "beach",
    "travel", "road trip", "airport", "flying", "delayed", "traffic",
    "stuck in traffic", "commute", "subway", "bus", "running late",
    "i'm late", "on my way", "almost done", "still working",

    # --- Hobbies / household ---
    "reading", "book", "bookworm", "writing", "journaling", "gardening",
    "cooking", "baking", "cleaning", "laundry", "dishes", "vacuuming",
    "moving house", "shopping", "online shopping", "package delivery",
    "exercise", "workout", "running", "yoga", "meditation", "stretching",

    # --- Pets / cute ---
    "cute animal", "cat", "kitten", "dog", "puppy", "bunny", "hamster",
    "owl", "duck", "panda", "otter", "raccoon", "support animal",

    # --- Tech / gadgets ---
    "computer", "laptop", "monitor", "keyboard", "mouse", "headphones",
    "phone", "smartphone", "tablet", "robot", "ai", "chatbot", "internet",
    "wifi", "buffering", "loading", "spinning wheel", "blue screen",
    "system error", "tech support", "have you tried turning it off and on",

    # --- Communication / chat / Slack-shaped ---
    "ping", "notification", "message", "texting", "typing", "send", "sent",
    "received", "thinking emoji", "fire emoji", "heart eyes", "pleading",
    "side eye", "pointing at screen", "nodding", "shaking head", "mic drop",
    "drop the mic", "throwing up hands", "hands up",

    # --- AI / ML / NLP ---
    "ai overlord", "skynet", "the singularity",
    "robot uprising", "sentient ai", "chatgpt", "llm", "large language model",
    "generative ai", "deepfake", "hallucinating", "hallucination",
    "prompt engineering", "prompting", "fine tuning", "training model",
    "model training", "loss going down", "loss exploding", "nan loss",
    "overfitting", "underfitting", "gradient descent", "backpropagation",
    "transformer", "attention is all you need", "embedding", "vector search",
    "rag", "retrieval", "tokens", "out of memory", "cuda out of memory",
    "gpu go brrr", "gpu", "nvidia", "training run", "epoch", "checkpoint",
    "convergence", "diverging", "self driving car", "autonomous vehicle",
    "computer vision", "image recognition", "speech recognition",
    "siri", "alexa", "smart home", "deep learning", "reinforcement learning",
    "agent", "ai agent", "copilot", "code completion", "autocomplete fail",
    "the algorithm", "blame the algorithm", "biased model", "alignment",
    "agi", "robot dance", "boston dynamics", "uncanny valley",
    "turing test", "passes the vibe check",

    # --- Social science / humanities / academia ---
    "sociology", "psychology", "anthropology", "economics", "political science",
    "linguistics", "philosophy", "history", "ethnography", "fieldwork",
    "interview", "focus group", "survey", "questionnaire", "qualitative",
    "quantitative", "regression", "p value", "p hacking", "statistical significance",
    "correlation", "correlation not causation", "causation", "cherry picking",
    "sample size", "n equals one", "small sample", "ethics board", "irb",
    "literature review", "citation needed", "citing sources", "reviewer 2",
    "reviewer two", "revise and resubmit", "minor revisions", "major revisions",
    "desk reject", "tenure", "tenure track", "publish", "publication",
    "h index", "impact factor", "open access", "preprint", "arxiv",
    "replication crisis", "replication", "reproducibility", "ethics",
    "bias", "stereotype", "demographics", "social network", "society",
    "culture", "subculture", "identity", "narrative", "framing", "discourse",

    # --- Internet lingo / meme culture ---
    "lmao", "lmaoo", "lulz", "kek", "smh", "tbh", "imo", "imho",
    "tldr", "fomo", "yolo", "irl", "afk", "brb", "ftw", "iykyk",
    "no cap", "cap", "lowkey", "highkey", "bet", "fr", "fr fr",
    "deadass", "based", "cringe", "cringey", "sus", "sussy", "vibe",
    "vibing", "vibe check", "main character", "main character energy",
    "side quest", "ratio", "ratioed", "touch grass", "go touch grass",
    "chronically online", "extremely online", "terminally online",
    "rent free", "living rent free", "ok boomer", "boomer", "zoomer",
    "millennial", "gen z", "gen alpha", "girl boss", "girlboss", "simp",
    "stan", "stan culture", "fangirl", "fanboy", "fandom", "shipping",
    "yas", "yas queen", "slay", "slayed", "ate that", "ate and left no crumbs",
    "bestie", "the girls are fighting", "not the", "literally me", "me irl",
    "this is the way", "may the force be with you", "i am inevitable",
    "thanos snap", "infinity stones", "wakanda forever", "got nerf",
    "noob", "n00b", "pwned", "owned", "rekt", "git gud", "skill issue",
    "pog", "poggers", "pogchamp", "lit", "fire", "fire fire", "savage",
    "iconic", "legendary", "extra", "salty", "shook", "shooketh",
    "wholesome", "wholesome 100", "big mood", "huge mood", "felt that",
    "i felt that", "this you", "and i oop", "and i took that personally",
    "the audacity", "the disrespect", "say it louder", "say it again",
    "we move", "anyway", "and that's on", "periodt", "period",
    "let that sink in", "let me cook", "let him cook", "she ate",
    "ok", "kk", "k", "thx", "ty", "np", "nvm", "wat", "wut", "wtf",
    "wth", "omg", "omfg", "lmfao", "rotfl",

    # --- Classic meme references (still searchable on Giphy) ---
    "doge", "shiba inu", "such wow", "many doge", "this is fine dog",
    "distracted boyfriend", "drake meme", "drake reaction", "spongebob",
    "patrick star", "kermit", "kermit drinking tea", "but that's none of my business",
    "sad pablo", "crying jordan", "leonardo dicaprio", "dicaprio cheers",
    "pikachu shocked", "surprised pikachu", "they don't know", "stonks",
    "not stonks", "not sure if", "fry meme", "ancient aliens", "history channel guy",
    "always has been", "expanding brain", "galaxy brain", "two buttons",
    "is this a pigeon", "tuxedo winnie the pooh", "math lady", "confused math lady",
    "white guy blinking", "blinking guy", "nick young confused", "michael scott no",
    "office no", "stanley", "jim halpert smirk", "leslie knope", "ron swanson",
    "i don't want to live on this planet anymore", "shut up and take my money",
    "y u no", "y tho", "but why", "but why tho", "side eye chloe",
    "ermahgerd", "much excite", "headdesk", "hide the pain harold",
    "challenge accepted",
]


@dataclass
class GiphyRecord:
    giphy_id: str
    rating: str
    title: str
    mp4_url: str
    gif_url: str
    source_query: str  # "trending" or the search query


# ---------------- API helpers ----------------

def _is_rate_limited(status_code: int, body: dict | None) -> bool:
    """Giphy signals rate limiting via 429, sometimes 403, and occasionally
    a 200 with `meta.status` set to 429 in the body."""
    if status_code in (429, 403):
        return True
    if isinstance(body, dict):
        meta = body.get("meta") or {}
        if meta.get("status") in (429, 403):
            return True
        msg = (meta.get("msg") or "").lower()
        if "rate limit" in msg or "too many" in msg:
            return True
    return False


# Rate-limit backoff schedule. We escalate quickly into long sleeps because
# the daily-cap reset on the free Giphy tier is on the order of an hour.
# First wait is 10 minutes — short retries kept hitting the same quota wall.
_RATE_LIMIT_BACKOFFS = [600, 900, 1800, 3600, 3600, 3600]  # 10m, 15m, 30m, 1h x3...


def _giphy_get(path: str, params: dict, retries: int = 3) -> dict:
    """Fetch a Giphy endpoint.

    On transient errors, retry up to `retries` times with short backoff.
    On rate-limit (429 / 403 / meta-status indicating throttling), sleep
    through the backoff schedule above and keep retrying — the caller is
    expected to be a long-running build that should pause through quota
    resets rather than fail.
    """
    url = f"{GIPHY_API}/{path}"
    last_exc: Exception | None = None
    rate_attempt = 0
    transient_attempt = 0
    while True:
        try:
            r = requests.get(url, params=params, timeout=15, headers={"User-Agent": USER_AGENT})
            body: dict | None = None
            try:
                body = r.json()
            except Exception:
                body = None
            if _is_rate_limited(r.status_code, body):
                wait = _RATE_LIMIT_BACKOFFS[min(rate_attempt, len(_RATE_LIMIT_BACKOFFS) - 1)]
                rate_attempt += 1
                logger.warning(
                    "giphy rate-limited (status=%s, attempt=%d); sleeping %ds before retry",
                    r.status_code, rate_attempt, wait,
                )
                time.sleep(wait)
                continue
            r.raise_for_status()
            return body if body is not None else {}
        except requests.RequestException as e:
            last_exc = e
            transient_attempt += 1
            if transient_attempt >= retries:
                break
            time.sleep(1 + transient_attempt)
    if last_exc:
        raise last_exc
    return {}


def _record_from_api(item: dict, source_query: str) -> GiphyRecord | None:
    images = item.get("images", {})
    # Prefer the fixed-height-small (~200px) mp4 — smaller & faster to download
    # for embedding. Fall back to original.gif. Both URLs are public Giphy CDN.
    mp4 = (
        images.get("fixed_height_small", {}).get("mp4")
        or images.get("downsized_small", {}).get("mp4")
        or images.get("original_mp4", {}).get("mp4")
        or images.get("original", {}).get("mp4")
        or ""
    )
    gif = (
        images.get("fixed_height_small", {}).get("url")
        or images.get("fixed_height", {}).get("url")
        or images.get("original", {}).get("url")
        or ""
    )
    gid = item.get("id") or ""
    if not gid or (not mp4 and not gif):
        return None
    return GiphyRecord(
        giphy_id=gid,
        rating=(item.get("rating") or "").lower(),
        title=(item.get("title") or "").strip(),
        mp4_url=mp4,
        gif_url=gif,
        source_query=source_query,
    )


def fetch_trending(api_key: str, count: int = 200, rating: str = "pg") -> list[GiphyRecord]:
    out: list[GiphyRecord] = []
    page = 50  # Giphy max per request
    for offset in range(0, count, page):
        limit = min(page, count - offset)
        data = _giphy_get("trending", {"api_key": api_key, "limit": limit, "offset": offset, "rating": rating})
        items = data.get("data", []) or []
        for it in items:
            rec = _record_from_api(it, "trending")
            if rec is not None:
                out.append(rec)
        if len(items) < limit:
            break
    return out


def fetch_search(
    api_key: str,
    query: str,
    count: int = 50,
    rating: str = "pg",
    seen_ids: set[str] | None = None,
    saturate_pages: int = 2,
) -> list[GiphyRecord]:
    """Page through Giphy search results.

    If `seen_ids` is provided, stop paging once `saturate_pages` consecutive
    pages contribute zero new ids — this avoids spending API calls on queries
    that overlap heavily with what's already indexed.
    """
    out: list[GiphyRecord] = []
    page = 50
    empty_pages = 0
    for offset in range(0, count, page):
        limit = min(page, count - offset)
        data = _giphy_get("search", {
            "api_key": api_key, "q": query, "limit": limit, "offset": offset, "rating": rating,
        })
        items = data.get("data", []) or []
        new_this_page = 0
        for it in items:
            rec = _record_from_api(it, query)
            if rec is None:
                continue
            if seen_ids is not None and rec.giphy_id in seen_ids:
                continue
            out.append(rec)
            new_this_page += 1
        if len(items) < limit:
            break
        if seen_ids is not None and new_this_page == 0:
            empty_pages += 1
            if empty_pages >= saturate_pages:
                logger.info("query %r saturated after %d empty pages; stopping", query, empty_pages)
                break
        else:
            empty_pages = 0
    return out


# ---------------- Frame extraction ----------------

def _first_frame(record: GiphyRecord) -> "PIL.Image.Image | None":
    """Download the GIF/MP4 and return its first frame as a PIL.RGB image.

    Tries GIF first (PIL handles it natively, no ffmpeg dep), falls back to
    MP4 via imageio with an explicit extension hint.
    """
    from PIL import Image

    candidates: list[tuple[str, str]] = []  # (url, ext)
    if record.gif_url:
        candidates.append((record.gif_url, ".gif"))
    if record.mp4_url:
        candidates.append((record.mp4_url, ".mp4"))
    if not candidates:
        return None

    for url, ext in candidates:
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": USER_AGENT})
            r.raise_for_status()
        except requests.RequestException as e:
            logger.warning("download failed for %s (%s): %s", record.giphy_id, ext, e)
            continue
        blob = r.content

        if ext == ".gif":
            try:
                im = Image.open(io.BytesIO(blob))
                im.seek(0)
                return im.convert("RGB")
            except Exception as e:
                logger.warning("PIL gif decode failed for %s: %s", record.giphy_id, e)
                continue

        # MP4 fallback: imageio v3 needs the extension hint when given bytes.
        try:
            import imageio.v3 as iio
            frame = iio.imread(io.BytesIO(blob), extension=ext, index=0)
            return Image.fromarray(frame).convert("RGB")
        except Exception as e:
            logger.warning("imageio mp4 decode failed for %s: %s", record.giphy_id, e)
            continue

    return None


# ---------------- Index I/O ----------------

def _load_existing_giphy_ids(index_dir: str) -> set[str]:
    meta_path = os.path.join(index_dir, "index_metadata.jsonl")
    if not os.path.exists(meta_path):
        return set()
    out: set[str] = set()
    with open(meta_path) as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            gid = row.get("giphy_id")
            if gid:
                out.add(gid)
    return out


def _atomic_extend_index(index_dir: str, backend: str, new_vectors: np.ndarray, new_entries: list[dict]) -> None:
    os.makedirs(index_dir, exist_ok=True)
    emb_path = os.path.join(index_dir, f"{backend}_embeddings.npy")
    meta_path = os.path.join(index_dir, "index_metadata.jsonl")

    if os.path.exists(emb_path):
        existing = np.load(emb_path)
        if existing.shape[0] == 0:
            combined = new_vectors
        elif existing.shape[1] != new_vectors.shape[1]:
            raise ValueError(
                f"embedding dim mismatch: existing {existing.shape[1]} vs new {new_vectors.shape[1]}"
            )
        else:
            combined = np.concatenate([existing, new_vectors], axis=0)
    else:
        combined = new_vectors

    # np.save auto-appends .npy if the path doesn't end in it, so use a
    # `.tmp.npy` suffix to keep the on-disk filename predictable for os.replace.
    tmp_emb = emb_path + ".tmp.npy"
    np.save(tmp_emb, combined.astype(np.float32, copy=False))
    os.replace(tmp_emb, emb_path)

    tmp_meta = meta_path + ".tmp"
    with open(tmp_meta, "w") as out:
        if os.path.exists(meta_path):
            with open(meta_path) as old:
                for line in old:
                    out.write(line if line.endswith("\n") else line + "\n")
        for entry in new_entries:
            out.write(json.dumps(entry) + "\n")
    os.replace(tmp_meta, meta_path)


# ---------------- Main pipeline ----------------

def run_giphy_refresh(
    api_key: str | None,
    index_dir: str,
    backend: str,
    *,
    queries: Iterable[str] | None = None,
    per_query: int = 50,
    trending_count: int = 200,
    max_new: int | None = None,
    rating: str = "pg",
    encoder=None,
    encoder_kwargs: dict | None = None,
    rebuild: bool = False,
) -> bool:
    """Returns True if any new GIFs were added."""
    if not api_key:
        logger.info("GIPHY_API_KEY not set; skipping giphy refresh")
        return False

    if rebuild and os.path.isdir(index_dir):
        for fname in (f"{backend}_embeddings.npy", "index_metadata.jsonl"):
            p = os.path.join(index_dir, fname)
            if os.path.exists(p):
                os.remove(p)
        logger.info("rebuild: cleared existing index files in %s", index_dir)

    seen = _load_existing_giphy_ids(index_dir)
    logger.info("existing index has %d giphy_ids", len(seen))

    queries = list(queries) if queries is not None else list(DEFAULT_QUERIES)

    # Live set of ids we've already accepted (existing index + this run's
    # fetches). Passed into fetch_search so it can short-circuit saturated
    # queries instead of paging through redundant results.
    live_seen: set[str] = set(seen)

    # Lazy-load encoder once up-front so we don't pay startup cost only to
    # discover the model can't load after we've burned API quota.
    if encoder is None:
        from .encoders import load_encoder
        encoder = load_encoder(backend, **(encoder_kwargs or {}))

    total_added = 0

    def _flush(records: list[GiphyRecord]) -> int:
        """Embed and atomic-extend a batch. Returns number actually added."""
        nonlocal total_added
        if not records:
            return 0
        vectors: list[np.ndarray] = []
        entries: list[dict] = []
        for rec in records:
            img = _first_frame(rec)
            if img is None:
                continue
            try:
                v = encoder.encode_image(img)
            except Exception as e:
                logger.warning("embed failed for %s: %s", rec.giphy_id, e)
                continue
            vectors.append(v.astype(np.float32, copy=False))
            entries.append({
                "gif_id": f"giphy:{rec.giphy_id}",
                "giphy_id": rec.giphy_id,
                "permalink": f"https://giphy.com/gifs/{rec.giphy_id}",
                "alt_text": rec.title or rec.source_query,
                "rating": rec.rating or "g",
            })
        if not vectors:
            return 0
        _atomic_extend_index(index_dir, backend, np.stack(vectors), entries)
        total_added += len(entries)
        logger.info("flushed %d entries (total this run: %d)", len(entries), total_added)
        return len(entries)

    def _accept(records: Iterable[GiphyRecord]) -> list[GiphyRecord]:
        out: list[GiphyRecord] = []
        for r in records:
            if r.giphy_id in live_seen:
                continue
            if r.rating not in ALLOWED_RATINGS:
                continue
            live_seen.add(r.giphy_id)
            out.append(r)
            if max_new and total_added + len(out) >= max_new:
                break
        return out

    try:
        if trending_count > 0:
            try:
                tr = fetch_trending(api_key, count=trending_count, rating=rating)
                logger.info("trending: %d raw records", len(tr))
                _flush(_accept(tr))
            except Exception as e:
                logger.warning("trending fetch failed: %s", e)
            if max_new and total_added >= max_new:
                logger.info("hit --max-new cap (%d); stopping", max_new)
                return total_added > 0

        for q in queries:
            try:
                rs = fetch_search(api_key, q, count=per_query, rating=rating, seen_ids=live_seen)
                logger.info("query %r: %d new candidates", q, len(rs))
                _flush(_accept(rs))
            except Exception as e:
                logger.warning("search %r failed: %s", q, e)
            if max_new and total_added >= max_new:
                logger.info("hit --max-new cap (%d); stopping", max_new)
                break
    except KeyboardInterrupt:
        logger.warning("interrupted by user; %d entries already saved to %s", total_added, index_dir)
        raise

    if total_added == 0:
        logger.info("no new giphy candidates after dedup")
        return False
    logger.info("done: appended %d entries to %s (backend=%s)", total_added, index_dir, backend)
    return True


# ---------------- CLI ----------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fetch GIFs from Giphy, embed, and append to the index.")
    p.add_argument("--backend", choices=["siglip", "pepe", "siglip_ft"], default="siglip")
    p.add_argument("--index-dir", default="/shared/0/projects/gif-reply-slack-bot/index")
    p.add_argument("--per-query", type=int, default=50)
    p.add_argument("--trending", type=int, default=200)
    p.add_argument("--max-new", type=int, default=0, help="cap total new GIFs (0 = unlimited)")
    p.add_argument("--rating", default="pg", choices=["g", "pg"])
    p.add_argument("--queries", nargs="*", default=None, help="override the default query list")
    p.add_argument("--queries-file", default=None, help="path to a file with one query per line")
    p.add_argument("--rebuild", action="store_true", help="wipe the existing index files before fetching")
    p.add_argument("--api-key", default=None, help="overrides GIPHY_API_KEY / .env lookup")
    p.add_argument("--siglip-model", default=None)
    p.add_argument("--pepe-checkpoint", default=None)
    p.add_argument("--ft-checkpoint", default=None, help="siglip_ft (PEPE-v2) checkpoint")
    p.add_argument("--cache-dir", default="/shared/0/projects/gif-reply-slack-bot/hf_cache")
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_arg_parser().parse_args()

    api_key = args.api_key
    if not api_key:
        try:
            from config_secret import GIPHY_API_KEY  # type: ignore
            api_key = GIPHY_API_KEY
        except Exception:
            api_key = os.environ.get("GIPHY_API_KEY")
    if not api_key:
        raise SystemExit("no GIPHY_API_KEY available (pass --api-key, set in .env, or export GIPHY_API_KEY)")

    queries = args.queries
    if args.queries_file:
        with open(args.queries_file) as f:
            queries = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    encoder_kwargs: dict = {"cache_dir": args.cache_dir}
    if args.backend == "siglip" and args.siglip_model:
        encoder_kwargs["model_name"] = args.siglip_model
    if args.backend == "pepe":
        if not args.pepe_checkpoint:
            raise SystemExit("--pepe-checkpoint is required for backend=pepe")
        encoder_kwargs = {"checkpoint_path": args.pepe_checkpoint}
    if args.backend == "siglip_ft":
        if not args.ft_checkpoint:
            raise SystemExit("--ft-checkpoint is required for backend=siglip_ft")
        encoder_kwargs = {"checkpoint_path": args.ft_checkpoint}
        if args.siglip_model:
            encoder_kwargs["model_name"] = args.siglip_model

    added = run_giphy_refresh(
        api_key=api_key,
        index_dir=args.index_dir,
        backend=args.backend,
        queries=queries,
        per_query=args.per_query,
        trending_count=args.trending,
        max_new=(args.max_new or None),
        rating=args.rating,
        rebuild=args.rebuild,
        encoder_kwargs=encoder_kwargs,
    )
    print("added new GIFs" if added else "no new GIFs added")


if __name__ == "__main__":
    main()
