"""Giphy weak-caption (alt_text→gif) augmentation source for PEPE-v2.

The currently-served index (`<index_dir>/index_metadata.jsonl`) is ~31k
Giphy-API GIFs whose entries carry `gif_id="giphy:<id>"`, `giphy_id`,
`alt_text`, `rating` — but **no media URL**. To both (a) train the SigLIP
encoder on these GIFs as a weak caption→gif stage and (b) re-embed them into a
unified serving index, we need their decoded frame tensors in the shared frame
cache.

This module:
  1. re-fetches each GIF's media URL from the Giphy get-by-id endpoint
  2. downloads + decodes → 4-frame fp16 tensor
  3. caches it under `data.cache_path_for_gif("giphy:<id>", cache_dir)` — the
     EXACT key both `reindex.py` and the stage-1 augmentation dataset resolve,
     so a mismatched key would silently drop every Giphy gif.
  4. emits a JSONL of (gif_id, alt_text) the trainer consumes like TGIF.

CLI:
  python -m gif_reply.training.giphy_frames build-giphy-frames [--workers N] [--limit N]
  python -m gif_reply.training.giphy_frames emit-examples
  python -m gif_reply.training.giphy_frames build-token-cache
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from tqdm import tqdm

from .data import DEFAULT_FRAME_CACHE_DIR, _decode_mp4_to_frames, _normalize_for_siglip, cache_path_for_gif
from .phase2 import build_token_cache_for_jsonl, download_one

logger = logging.getLogger(__name__)

DEFAULT_INDEX_METADATA = "/shared/0/projects/gif-reply-slack-bot/index/index_metadata.jsonl"
DEFAULT_GIPHY_DIR = "/shared/0/projects/gif-reply-slack-bot/phase2/giphy"


def load_index_entries(meta_path: str) -> list[dict]:
    """Read the served crawled index metadata (one JSON object per line)."""
    rows: list[dict] = []
    with open(meta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _resolve_media_url(giphy_id: str, api_key: str) -> str | None:
    """Giphy get-by-id → best mp4 (fallback gif) URL, or None if gone/expired."""
    from .. import giphy_refresh as gr

    body = gr._giphy_get(giphy_id, {"api_key": api_key})
    item = body.get("data") if isinstance(body, dict) else None
    if not item:  # 404 / removed / empty
        return None
    rec = gr._record_from_api(item, "reindex")
    if rec is None:
        return None
    return rec.mp4_url or rec.gif_url or None


def fetch_and_cache_one(giphy_id: str, api_key: str, gif_dir: str, cache_dir: str, url: str | None = None) -> str:
    """Returns 'ok' | 'cached' | 'api_fail' | 'download_fail' | 'decode_fail'.

    If `url` is provided, skips the get-by-id API roundtrip — the discoverer
    already gets media urls from search/trending results, so passing them
    through avoids burning a Giphy call per gif.
    """
    gif_id = f"giphy:{giphy_id}"
    cp = cache_path_for_gif(gif_id, cache_dir)
    if os.path.exists(cp):
        return "cached"
    if url is None:
        try:
            url = _resolve_media_url(giphy_id, api_key)
        except Exception as e:
            logger.debug(f"api fail {giphy_id}: {e}")
            return "api_fail"
    if not url:
        return "api_fail"
    # Extension is cosmetic — PyAV decodes by content. Giphy CDN urls carry
    # query strings, so don't trust splitext.
    ext = ".gif" if url.split("?", 1)[0].endswith(".gif") else ".mp4"
    raw = os.path.join(gif_dir, giphy_id[:2], giphy_id + ext)
    if not download_one(url, raw):
        return "download_fail"
    frames = _decode_mp4_to_frames(raw)  # PyAV handles mp4 and gif
    if frames is None:
        return "decode_fail"
    frames = _normalize_for_siglip(frames).to(torch.float16)
    os.makedirs(os.path.dirname(cp), exist_ok=True)
    tmp = cp + ".tmp"
    torch.save(frames, tmp)
    os.replace(tmp, cp)
    return "ok"


def crawl(giphy_ids: list[str], api_key: str, gif_dir: str, cache_dir: str, workers: int = 8) -> dict:
    counts = {"ok": 0, "cached": 0, "api_fail": 0, "download_fail": 0, "decode_fail": 0}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_and_cache_one, gid, api_key, gif_dir, cache_dir): gid for gid in giphy_ids}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="giphy"):
            try:
                r = fut.result()
            except Exception:
                r = "download_fail"
            counts[r] = counts.get(r, 0) + 1
    return counts


def write_examples(entries: list[dict], cache_dir: str, out_jsonl: str) -> int:
    """One record per (giphy gif, alt_text) that has a non-empty caption and a
    cached frame tensor. Mirrors phase2.write_examples (TGIF)."""
    n = 0
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    with open(out_jsonl, "w") as f:
        for e in entries:
            gif_id = e.get("gif_id") or ""
            if not gif_id.startswith("giphy:"):
                continue
            text = (e.get("alt_text") or "").strip()
            if not text:
                continue
            if not os.path.exists(cache_path_for_gif(gif_id, cache_dir)):
                continue
            f.write(json.dumps({
                "gif_id": gif_id,
                "text": text,
                "tag_indices": [],  # Giphy has no PEPE tag labels
                "source": "giphy",
            }) + "\n")
            n += 1
    return n


def _resolve_api_key(cli_key: str | None) -> str:
    """Same resolution order as giphy_refresh.main: --api-key, .env, env var."""
    if cli_key:
        return cli_key
    try:
        from config_secret import GIPHY_API_KEY  # type: ignore

        if GIPHY_API_KEY:
            return GIPHY_API_KEY
    except Exception:
        pass
    k = os.environ.get("GIPHY_API_KEY")
    if not k:
        raise SystemExit("no GIPHY_API_KEY available (pass --api-key, set in config_secret, or export GIPHY_API_KEY)")
    return k


def _cli():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("build-giphy-frames")
    pf.add_argument("--index-metadata", default=DEFAULT_INDEX_METADATA)
    pf.add_argument("--cache-dir", default=DEFAULT_FRAME_CACHE_DIR)
    pf.add_argument("--giphy-dir", default=DEFAULT_GIPHY_DIR)
    pf.add_argument("--api-key", default=None)
    pf.add_argument("--workers", type=int, default=8)
    pf.add_argument("--limit", type=int, default=None, help="cap unique gifs for smoke testing")

    pe = sub.add_parser("emit-examples")
    pe.add_argument("--index-metadata", default=DEFAULT_INDEX_METADATA)
    pe.add_argument("--cache-dir", default=DEFAULT_FRAME_CACHE_DIR)
    pe.add_argument("--giphy-dir", default=DEFAULT_GIPHY_DIR)

    pt = sub.add_parser("build-token-cache")
    pt.add_argument("--giphy-dir", default=DEFAULT_GIPHY_DIR)

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "build-giphy-frames":
        api_key = _resolve_api_key(args.api_key)
        entries = load_index_entries(args.index_metadata)
        giphy_ids = []
        seen: set[str] = set()
        for e in entries:
            gid = e.get("giphy_id") or (e.get("gif_id", "").split("giphy:", 1)[-1] or None)
            if gid and gid not in seen:
                seen.add(gid)
                giphy_ids.append(gid)
        if args.limit:
            giphy_ids = giphy_ids[: args.limit]
        raw_gif_dir = os.path.join(args.giphy_dir, "gifs")
        os.makedirs(raw_gif_dir, exist_ok=True)
        logger.info(f"fetching {len(giphy_ids)} unique giphy gifs with {args.workers} workers")
        counts = crawl(giphy_ids, api_key, raw_gif_dir, args.cache_dir, workers=args.workers)
        logger.info(f"counts: {counts}")

    elif args.cmd == "emit-examples":
        entries = load_index_entries(args.index_metadata)
        out = os.path.join(args.giphy_dir, "examples.jsonl")
        n = write_examples(entries, args.cache_dir, out)
        logger.info(f"wrote {n} examples → {out}")

    elif args.cmd == "build-token-cache":
        in_jsonl = os.path.join(args.giphy_dir, "examples.jsonl")
        out_pt = os.path.join(args.giphy_dir, "tokens.pt")
        n = build_token_cache_for_jsonl(in_jsonl, out_pt)
        logger.info(f"tokenized {n} captions → {out_pt}")


if __name__ == "__main__":
    _cli()
