"""Phase 2: TGIF (caption→gif) auxiliary training data.

The TGIF release at https://github.com/raingo/TGIF-Release/data/tgif-v1.0.tsv
is a 2-column TSV: <gif_url>\t<caption>. We:

  1. download the TSV (one-shot)
  2. fetch each Tumblr GIF URL in parallel (skip-if-cached)
  3. decode → 4-frame fp16 tensor in the same on-disk format as PEPE
  4. emit a JSONL of (text, frame_cache_path, source) records the trainer can
     consume alongside the PEPE examples

The output frame cache lives under
/shared/0/projects/gif-reply-slack-bot/phase2/tgif/frames/<aa>/<sha1>.pt so it
shares the same `cache_path_for_gif` layout as the PEPE cache; the trainer
treats both interchangeably.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from tqdm import tqdm

from .data import _decode_mp4_to_frames, _normalize_for_siglip

logger = logging.getLogger(__name__)

TGIF_TSV_URL = "https://raw.githubusercontent.com/raingo/TGIF-Release/master/data/tgif-v1.0.tsv"
DEFAULT_PHASE2_DIR = "/shared/0/projects/gif-reply-slack-bot/phase2/tgif"


def url_to_gif_id(url: str) -> str:
    """Stable id derived from URL — lets us reuse cache_path_for_gif layout."""
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


def cache_path(gif_id: str, frame_dir: str) -> str:
    return os.path.join(frame_dir, gif_id[:2], f"{gif_id}.pt")


def load_tgif_rows(tsv_path: str) -> list[tuple[str, str, str]]:
    """Returns [(gif_id, url, caption), ...]."""
    rows: list[tuple[str, str, str]] = []
    with open(tsv_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            url, caption = parts[0], parts[1]
            if not url.startswith("http"):
                continue
            rows.append((url_to_gif_id(url), url, caption))
    return rows


def download_one(url: str, dest: str, timeout: int = 20) -> bool:
    """Stream a single URL to disk. Returns True on success."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True
    import requests
    try:
        r = requests.get(url, stream=True, timeout=timeout, allow_redirects=True)
        if r.status_code != 200:
            return False
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp = dest + ".tmp"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
        if os.path.getsize(tmp) == 0:
            os.unlink(tmp)
            return False
        os.replace(tmp, dest)
        return True
    except Exception as e:
        logger.debug(f"download fail {url}: {e}")
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except Exception:
            pass
        return False


def download_and_cache_one(gif_id: str, url: str, gif_dir: str, frame_dir: str) -> str:
    """Returns one of: 'ok', 'cached', 'download_fail', 'decode_fail'."""
    cp = cache_path(gif_id, frame_dir)
    if os.path.exists(cp):
        return "cached"
    raw = os.path.join(gif_dir, gif_id[:2], gif_id + ".gif")
    if not download_one(url, raw):
        return "download_fail"
    frames = _decode_mp4_to_frames(raw)  # PyAV handles GIF too
    if frames is None:
        try:
            os.unlink(raw)
        except Exception:
            pass
        return "decode_fail"
    frames = _normalize_for_siglip(frames).to(torch.float16)
    os.makedirs(os.path.dirname(cp), exist_ok=True)
    tmp = cp + ".tmp"
    torch.save(frames, tmp)
    os.replace(tmp, cp)
    # Drop the raw .gif once decoded — we only need the tensor cache.
    try:
        os.unlink(raw)
    except Exception:
        pass
    return "ok"


def crawl(rows: list[tuple[str, str, str]], gif_dir: str, frame_dir: str, workers: int = 32) -> dict:
    counts = {"ok": 0, "cached": 0, "download_fail": 0, "decode_fail": 0}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(download_and_cache_one, gid, url, gif_dir, frame_dir): gid for gid, url, _ in rows}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="tgif"):
            try:
                r = fut.result()
            except Exception:
                r = "download_fail"
            counts[r] = counts.get(r, 0) + 1
    return counts


def write_examples(rows: list[tuple[str, str, str]], frame_dir: str, out_jsonl: str) -> int:
    """Emit a JSONL record per (gif, caption) pair that has a cached frame tensor."""
    n = 0
    with open(out_jsonl, "w") as f:
        for gif_id, _url, caption in rows:
            if not caption:
                continue
            if not os.path.exists(cache_path(gif_id, frame_dir)):
                continue
            f.write(json.dumps({
                "gif_id": gif_id,
                "text": caption,
                "tag_indices": [],  # TGIF has no PEPE tag labels
                "source": "tgif",
            }) + "\n")
            n += 1
    return n


def build_token_cache_for_jsonl(jsonl_path: str, out_path: str, model_name: str = "google/siglip-base-patch16-224") -> int:
    """Tokenize captions from a JSONL produced by write_examples."""
    from .data import build_token_cache
    texts: list[str] = []
    examples: list[dict] = []
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            texts.append(row["text"])
            examples.append(row)
    build_token_cache(texts, out_path, model_name=model_name)
    examples_path = out_path.replace(".pt", ".examples.pt")
    torch.save(examples, examples_path)
    return len(examples)


def _cli():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pdl = sub.add_parser("crawl")
    pdl.add_argument("--tsv", default=os.path.join(DEFAULT_PHASE2_DIR, "tgif-v1.0.tsv"))
    pdl.add_argument("--phase2-dir", default=DEFAULT_PHASE2_DIR)
    pdl.add_argument("--workers", type=int, default=32)
    pdl.add_argument("--limit", type=int, default=None)

    pe = sub.add_parser("emit-examples")
    pe.add_argument("--tsv", default=os.path.join(DEFAULT_PHASE2_DIR, "tgif-v1.0.tsv"))
    pe.add_argument("--phase2-dir", default=DEFAULT_PHASE2_DIR)

    pt = sub.add_parser("build-token-cache")
    pt.add_argument("--phase2-dir", default=DEFAULT_PHASE2_DIR)

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "crawl":
        rows = load_tgif_rows(args.tsv)
        if args.limit:
            rows = rows[: args.limit]
        gif_dir = os.path.join(args.phase2_dir, "gifs")
        frame_dir = os.path.join(args.phase2_dir, "frames")
        os.makedirs(gif_dir, exist_ok=True)
        os.makedirs(frame_dir, exist_ok=True)
        logger.info(f"crawling {len(rows)} TGIF rows with {args.workers} workers")
        counts = crawl(rows, gif_dir, frame_dir, workers=args.workers)
        logger.info(f"counts: {counts}")

    elif args.cmd == "emit-examples":
        rows = load_tgif_rows(args.tsv)
        out = os.path.join(args.phase2_dir, "examples.jsonl")
        n = write_examples(rows, os.path.join(args.phase2_dir, "frames"), out)
        logger.info(f"wrote {n} examples → {out}")

    elif args.cmd == "build-token-cache":
        in_jsonl = os.path.join(args.phase2_dir, "examples.jsonl")
        out_pt = os.path.join(args.phase2_dir, "tokens.pt")
        n = build_token_cache_for_jsonl(in_jsonl, out_pt)
        logger.info(f"tokenized {n} captions → {out_pt}")


if __name__ == "__main__":
    _cli()
