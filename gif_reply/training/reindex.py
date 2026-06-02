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

from .data import DEFAULT_FRAME_CACHE_DIR, PEPE_PICKLE, cache_path_for_gif, load_meta, load_pickle_compat
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


def _load_candidates_pickle(path: str) -> list[dict]:
    """Pull unique PEPE candidates from the training pickle.

    PEPE's child_gif_id is the bare 13-char Giphy id (same scheme as the frame
    cache), so these lookups hit cache_path_for_gif directly.
    """
    df = load_pickle_compat(path)
    gids = df["child_gif_id"].dropna().unique().tolist()
    return [
        {
            "gif_id": g,
            "giphy_id": g,
            "permalink": f"https://giphy.com/gifs/{g}",
            "alt_text": "",
            "rating": "g",
        }
        for g in gids
        if isinstance(g, str) and g
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--backend-name", default="siglip_ft", help="prefix for the .npy file")
    p.add_argument("--candidates-csv", type=str, default=None)
    p.add_argument("--candidates-pickle", type=str, default=None,
                   help=f"PEPE training pickle (e.g. {PEPE_PICKLE}); pulls unique child_gif_id values")
    p.add_argument("--crawled-jsonl", type=str, action="append", default=None,
                   help="path to an index_metadata.jsonl; pass multiple times to merge "
                        "(e.g. live index + discover pool)")
    p.add_argument("--frame-cache-dir", default=DEFAULT_FRAME_CACHE_DIR)
    p.add_argument("--model-name", default="google/siglip-base-patch16-224")
    p.add_argument("--freeze-fraction", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--append", action="store_true",
                   help="incremental mode: load existing <out-dir>/<backend>_embeddings.npy + "
                        "index_metadata.jsonl, skip already-indexed gif_ids, encode only the "
                        "diff, concatenate. Use this when the checkpoint is unchanged and you "
                        "are only adding new candidates.")
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
    if args.candidates_pickle:
        for r in _load_candidates_pickle(args.candidates_pickle):
            all_entries.setdefault(r["gif_id"], r)
    if args.crawled_jsonl:
        for path in args.crawled_jsonl:
            for r in _load_crawled_jsonl(path):
                gid = r["gif_id"]
                all_entries.setdefault(gid, r)
    logger.info(f"merged candidate set: {len(all_entries)}")

    emb_path = os.path.join(args.out_dir, f"{args.backend_name}_embeddings.npy")
    meta_path = os.path.join(args.out_dir, "index_metadata.jsonl")

    existing_mat: np.ndarray | None = None
    existing_entries: list[dict] = []
    if args.append and os.path.exists(emb_path) and os.path.exists(meta_path):
        existing_mat = np.load(emb_path)
        with open(meta_path) as f:
            for line in f:
                try:
                    existing_entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        existing_gids = {e.get("gif_id") for e in existing_entries if e.get("gif_id")}
        before = len(all_entries)
        all_entries = {g: e for g, e in all_entries.items() if g not in existing_gids}
        logger.info(
            f"append: {len(existing_gids)} already indexed, {before - len(all_entries)} skipped, "
            f"{len(all_entries)} new to encode"
        )
        if not all_entries:
            logger.info("nothing new to encode; exiting without rewriting index")
            return
    elif args.append:
        logger.info("append requested but no existing index at %s; falling back to full build", emb_path)

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

    new_mat = np.concatenate(embeddings, axis=0).astype(np.float32) if embeddings else np.zeros((0, 0), dtype=np.float32)
    if existing_mat is not None and existing_mat.size:
        if new_mat.size and existing_mat.shape[1] != new_mat.shape[1]:
            raise ValueError(
                f"embedding dim mismatch: existing {existing_mat.shape[1]} vs new {new_mat.shape[1]}"
            )
        mat = np.concatenate([existing_mat, new_mat], axis=0) if new_mat.size else existing_mat
        all_out_entries = existing_entries + kept_entries
        logger.info(
            f"append concat: {existing_mat.shape[0]} existing + {new_mat.shape[0]} new = {mat.shape[0]}"
        )
    else:
        mat = new_mat
        all_out_entries = kept_entries

    # np.save auto-appends ".npy" if the path doesn't already end in it, so we
    # name tmp to absorb that — tmp on disk ends up as emb_path + ".tmp.npy".
    tmp_logical = emb_path + ".tmp"
    np.save(tmp_logical, mat)
    tmp_actual = tmp_logical if tmp_logical.endswith(".npy") else tmp_logical + ".npy"
    os.replace(tmp_actual, emb_path)
    tmp = meta_path + ".tmp"
    with open(tmp, "w") as f:
        for e in all_out_entries:
            f.write(json.dumps(e) + "\n")
    os.replace(tmp, meta_path)
    logger.info(f"wrote {emb_path} {mat.shape}  and  {meta_path}")


if __name__ == "__main__":
    main()
