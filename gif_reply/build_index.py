"""One-shot script: build a candidate-embedding index for a chosen backend.

Usage:
    python -m gif_reply.build_index --backend siglip
    python -m gif_reply.build_index --backend pepe \
        --checkpoint /shared/2/projects/gif-reply/data/release/PEPE-model-checkpoint.pth

Inputs (defaults assume the released layout on /shared/2):
    --metadata-csv  path to gif-metadata.csv (columns: gif_id, giphy_id, ...)
    --gifs-dir      root directory of mp4/gif assets (filenames keyed by gif_id)
    --out-dir       where to write {backend}_embeddings.npy + index_metadata.jsonl

For PEPE, prefers the existing `gif-pepe-inferred-features.csv` (precomputed)
when --use-precomputed is passed, avoiding a full re-embed.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys

import numpy as np

logger = logging.getLogger(__name__)


def _open_image_first_frame(path: str):
    from PIL import Image
    im = Image.open(path)
    im.seek(0)
    return im.convert("RGB")


def build(args: argparse.Namespace) -> None:
    from .encoders import load_encoder
    from .index import Index, IndexEntry

    rows: list[dict] = []
    with open(args.metadata_csv) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    logger.info("loaded %d metadata rows", len(rows))

    if args.limit:
        rows = rows[: args.limit]

    if args.backend == "pepe" and args.use_precomputed:
        # Load gif-pepe-inferred-features.csv; first column gif_id, rest is the vector.
        feats_by_id: dict[str, np.ndarray] = {}
        with open(args.precomputed_csv) as f:
            reader = csv.reader(f)
            for r in reader:
                if not r:
                    continue
                gid = r[0]
                try:
                    vec = np.array([float(x) for x in r[1:]], dtype=np.float32)
                except ValueError:
                    continue  # header row
                feats_by_id[gid] = vec
        kept_rows, vectors = [], []
        for r in rows:
            v = feats_by_id.get(r["gif_id"])
            if v is not None:
                kept_rows.append(r)
                vectors.append(v)
        embeddings = np.stack(vectors).astype(np.float32)
        rows = kept_rows
        logger.info("using precomputed PEPE features for %d gifs", len(rows))
    else:
        encoder_kwargs = {}
        if args.backend == "pepe":
            encoder_kwargs["checkpoint_path"] = args.checkpoint
        elif args.backend == "siglip":
            encoder_kwargs["model_name"] = args.siglip_model
        encoder = load_encoder(args.backend, **encoder_kwargs)
        vectors = []
        kept_rows = []
        for i, r in enumerate(rows):
            path = os.path.join(args.gifs_dir, r.get("filename") or f"{r['gif_id']}.mp4")
            if not os.path.exists(path):
                continue
            try:
                img = _open_image_first_frame(path)
                vec = encoder.encode_image(img)
            except Exception as e:
                logger.warning("skip %s: %s", path, e)
                continue
            vectors.append(vec)
            kept_rows.append(r)
            if (i + 1) % 500 == 0:
                logger.info("embedded %d / %d", i + 1, len(rows))
        if not vectors:
            logger.error("no embeddings produced")
            sys.exit(2)
        embeddings = np.stack(vectors).astype(np.float32)
        rows = kept_rows

    entries = [
        IndexEntry(
            gif_id=r["gif_id"],
            giphy_id=r.get("giphy_id", ""),
            permalink=r.get("permalink") or (f"https://giphy.com/gifs/{r.get('giphy_id', '')}" if r.get("giphy_id") else ""),
            alt_text=r.get("alt_text") or r.get("tags") or "",
            rating=(r.get("rating") or "g").lower(),
        )
        for r in rows
    ]
    idx = Index(embeddings, entries)
    idx.save(args.out_dir, args.backend)
    logger.info("wrote %d entries to %s (backend=%s)", len(entries), args.out_dir, args.backend)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["siglip", "pepe"], required=True)
    p.add_argument("--metadata-csv", default="/shared/2/projects/gif-reply/data/release/gif-metadata.csv")
    p.add_argument("--gifs-dir", default="/shared/0/projects/gif-reply/data/gifs")
    p.add_argument("--out-dir", default="/shared/0/projects/gif-reply-slack-bot/index")
    p.add_argument("--checkpoint", default="/shared/2/projects/gif-reply/data/release/PEPE-model-checkpoint.pth")
    p.add_argument("--siglip-model", default="google/siglip-base-patch16-224")
    p.add_argument("--use-precomputed", action="store_true", help="for PEPE, load gif-pepe-inferred-features.csv instead of re-embedding")
    p.add_argument("--precomputed-csv", default="/shared/2/projects/gif-reply/data/release/gif-pepe-inferred-features.csv")
    p.add_argument("--limit", type=int, default=0, help="cap rows (0 = all)")
    args = p.parse_args()
    build(args)


if __name__ == "__main__":
    main()
