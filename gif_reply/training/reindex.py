"""Re-embed every candidate gif with a fine-tuned checkpoint and write a new index.

The output is a drop-in replacement for the existing gif_reply Index format:
  <out-dir>/siglip_ft_embeddings.npy
  <out-dir>/index_metadata.jsonl

Candidate sources (merged + de-duped by gif_id):
  - --candidates-csv: PEPE release CSV (e.g. /shared/2/.../release/gif-metadata.csv)
  - --crawled-jsonl:  the existing crawled index (index_metadata.jsonl)

Frames are loaded from --frame-cache-dir; gifs without a cached frame tensor
are skipped (with a logged count).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os

import numpy as np
import torch
from tqdm import tqdm

from .data import DEFAULT_FRAME_CACHE_DIR, cache_path_for_gif, load_meta
from .model import ModelConfig, SigLIPDualEncoder

logger = logging.getLogger(__name__)


def _load_candidates_csv(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def _load_crawled_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--backend-name", default="siglip_ft", help="prefix for the .npy file")
    p.add_argument("--candidates-csv", type=str, default=None)
    p.add_argument("--crawled-jsonl", type=str, default=None)
    p.add_argument("--frame-cache-dir", default=DEFAULT_FRAME_CACHE_DIR)
    p.add_argument("--model-name", default="google/siglip-base-patch16-224")
    p.add_argument("--freeze-fraction", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Merge candidate sources.
    all_entries: dict[str, dict] = {}
    if args.candidates_csv:
        for r in _load_candidates_csv(args.candidates_csv):
            gid = r.get("gif_id") or r.get("id")
            if not gid:
                continue
            all_entries[gid] = {
                "gif_id": gid,
                "giphy_id": r.get("giphy_id", gid),
                "permalink": r.get("permalink", ""),
                "alt_text": r.get("alt_text") or r.get("ocr_text") or "",
                "rating": r.get("rating", "g"),
            }
    if args.crawled_jsonl:
        for r in _load_crawled_jsonl(args.crawled_jsonl):
            gid = r["gif_id"]
            all_entries.setdefault(gid, r)
    logger.info(f"merged candidate set: {len(all_entries)}")

    meta = load_meta()
    cfg = ModelConfig(model_name=args.model_name, n_tags=meta["n_labels"], freeze_fraction=args.freeze_fraction)
    model = SigLIPDualEncoder(cfg)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    model.eval().to(device)

    kept_entries: list[dict] = []
    embeddings: list[np.ndarray] = []
    buf: list[torch.Tensor] = []
    buf_entries: list[dict] = []
    skipped = 0

    @torch.no_grad()
    def flush():
        if not buf:
            return
        frames = torch.stack(buf).to(device)
        emb = model.encode_frames(frames)  # (B, D), normalized
        embeddings.append(emb.float().cpu().numpy())
        kept_entries.extend(buf_entries)
        buf.clear()
        buf_entries.clear()

    for gid, entry in tqdm(all_entries.items(), desc="encode"):
        cp = cache_path_for_gif(gid, args.frame_cache_dir)
        try:
            f = torch.load(cp, map_location="cpu", weights_only=False).float()
        except Exception:
            skipped += 1
            continue
        buf.append(f)
        buf_entries.append(entry)
        if len(buf) >= args.batch_size:
            flush()
    flush()

    logger.info(f"kept: {len(kept_entries)}; skipped (missing frames): {skipped}")

    mat = np.concatenate(embeddings, axis=0).astype(np.float32) if embeddings else np.zeros((0, 0), dtype=np.float32)
    emb_path = os.path.join(args.out_dir, f"{args.backend_name}_embeddings.npy")
    meta_path = os.path.join(args.out_dir, "index_metadata.jsonl")
    tmp = emb_path + ".tmp"
    np.save(tmp, mat)
    os.replace(tmp, emb_path)
    tmp = meta_path + ".tmp"
    with open(tmp, "w") as f:
        for e in kept_entries:
            f.write(json.dumps(e) + "\n")
    os.replace(tmp, meta_path)
    logger.info(f"wrote {emb_path} {mat.shape}  and  {meta_path}")


if __name__ == "__main__":
    main()
