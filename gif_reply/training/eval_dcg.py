"""Offline DCG@30 + Recall@k harness against the PEPE dev/test split.

Usage:
    python -m gif_reply.training.eval_dcg --backend siglip --split dev
    python -m gif_reply.training.eval_dcg --backend siglip_ft --checkpoint <best.pt> --split dev

The harness:
  1. Loads the dev (or test) examples.
  2. Encodes the unique gif pool with the chosen encoder.
  3. Encodes each tweet, computes cosine to all gifs, ranks.
  4. Reports DCG@30, Recall@1/5/10/30, MRR over examples whose gold gif_id is
     in the pool.
"""
from __future__ import annotations

import argparse
import logging
import math
import os

import numpy as np
import torch

from .data import (
    DEFAULT_FRAME_CACHE_DIR, DEFAULT_TOKEN_CACHE_DIR,
    cache_path_for_gif, load_meta, load_pepe_examples,
)

logger = logging.getLogger(__name__)


def _load_encoder(backend: str, model_name: str, checkpoint: str | None, freeze_fraction: float, device):
    if backend == "siglip":
        from gif_reply.encoders import SiglipEncoder
        enc = SiglipEncoder(model_name=model_name)
        return enc, "off-the-shelf"
    if backend == "siglip_ft":
        from .model import ModelConfig, SigLIPDualEncoder
        meta = load_meta()
        cfg = ModelConfig(model_name=model_name, n_tags=meta["n_labels"], freeze_fraction=freeze_fraction)
        m = SigLIPDualEncoder(cfg)
        if checkpoint:
            ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
            m.load_state_dict(ck["model"] if "model" in ck else ck)
        m.eval().to(device)
        return m, f"checkpoint: {checkpoint}"
    raise ValueError(f"unknown backend: {backend}")


@torch.no_grad()
def encode_gif_pool(model, gif_ids: list[str], frame_cache_dir: str, batch_size: int, device) -> tuple[np.ndarray, list[str]]:
    """Returns (matrix (N, D), kept_ids)."""
    kept = []
    rows: list[torch.Tensor] = []
    buf: list[torch.Tensor] = []
    buf_ids: list[str] = []

    def flush():
        if not buf:
            return
        frames = torch.stack(buf).to(device)
        if hasattr(model, "encode_frames"):
            emb = model.encode_frames(frames)
        else:
            # Fall back: gif_reply.encoders.SiglipEncoder.encode_image takes one PIL image.
            # Use the mean of N frames by encoding each frame separately.
            from PIL import Image
            embs = []
            for f in frames:  # (N, 3, H, W) per gif
                # Reverse the SigLIP normalization to recover an image-like tensor.
                ims = []
                for fr in f:
                    arr = (fr * 0.5 + 0.5).clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
                    ims.append(Image.fromarray(arr))
                v = np.stack([model.encode_image(im) for im in ims], axis=0).mean(axis=0)
                v = v / max(np.linalg.norm(v), 1e-8)
                embs.append(v)
            emb = torch.from_numpy(np.stack(embs))
        rows.append(emb.float().cpu())
        kept.extend(buf_ids)
        buf.clear()
        buf_ids.clear()

    from tqdm import tqdm
    for gid in tqdm(gif_ids, desc="encode gifs"):
        cp = cache_path_for_gif(gid, frame_cache_dir)
        try:
            f = torch.load(cp, map_location="cpu", weights_only=False).float()
        except Exception:
            continue
        buf.append(f)
        buf_ids.append(gid)
        if len(buf) >= batch_size:
            flush()
    flush()

    if not rows:
        raise RuntimeError("no gif embeddings produced; check frame cache")
    return torch.cat(rows, dim=0).numpy(), kept


@torch.no_grad()
def encode_tweets(model, texts: list[str], device, batch_size: int = 64) -> np.ndarray:
    if hasattr(model, "encode_text") and not hasattr(model, "_base"):
        # gif_reply.encoders.SiglipEncoder
        return np.stack([model.encode_text(t) for t in texts], axis=0)
    # SigLIPDualEncoder
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(model.cfg.model_name)
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = proc(text=chunk, return_tensors="pt", padding="max_length", truncation=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        emb = model.encode_text(enc["input_ids"], enc["attention_mask"])
        out.append(emb.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def metrics_from_ranks(ranks: np.ndarray) -> dict:
    n = len(ranks)
    out = {
        "n": n,
        "recall@1": float((ranks < 1).mean()),
        "recall@5": float((ranks < 5).mean()),
        "recall@10": float((ranks < 10).mean()),
        "recall@30": float((ranks < 30).mean()),
        "mrr": float((1.0 / (ranks + 1)).mean()),
        # DCG@30: 1/log2(rank+2) if rank<30 else 0; ideal DCG=1 since 1 relevant.
        "dcg@30": float(np.where(ranks < 30, 1.0 / np.log2(ranks + 2), 0.0).mean()),
    }
    return out


@torch.no_grad()
def evaluate_retrieval(
    model,
    split: str,
    frame_cache_dir: str,
    device,
    max_tweets: int = 3000,
    batch_size: int = 64,
) -> dict:
    """Recall@k / DCG@30 / MRR for an in-memory SigLIPDualEncoder.

    Reused by train.py for checkpoint selection. Caller is responsible for
    model.eval() / restoring train mode. Deterministic tweet subsample.
    """
    was_training = model.training
    model.eval()
    examples, _ = load_pepe_examples(split)
    if max_tweets and len(examples) > max_tweets:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(examples), size=max_tweets, replace=False)
        examples = [examples[i] for i in idx]
    unique_gifs = sorted(set(e.gif_id for e in examples))
    gif_mat, kept_ids = encode_gif_pool(model, unique_gifs, frame_cache_dir, batch_size, device)
    id_to_row = {g: i for i, g in enumerate(kept_ids)}
    eligible = [e for e in examples if e.gif_id in id_to_row]
    txt_mat = encode_tweets(model, [e.text for e in eligible], device, batch_size=batch_size)
    txt_mat = txt_mat / np.maximum(np.linalg.norm(txt_mat, axis=1, keepdims=True), 1e-8)
    gif_mat = gif_mat / np.maximum(np.linalg.norm(gif_mat, axis=1, keepdims=True), 1e-8)
    sims = txt_mat @ gif_mat.T
    gold_rows = np.array([id_to_row[e.gif_id] for e in eligible])
    gold_scores = sims[np.arange(len(eligible)), gold_rows]
    ranks = (sims > gold_scores[:, None]).sum(axis=1)
    if was_training:
        model.train()
    return metrics_from_ranks(ranks)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True, choices=["siglip", "siglip_ft"])
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--model-name", default="google/siglip-base-patch16-224")
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    p.add_argument("--frame-cache-dir", default=DEFAULT_FRAME_CACHE_DIR)
    p.add_argument("--token-cache-dir", default=DEFAULT_TOKEN_CACHE_DIR)
    p.add_argument("--max-tweets", type=int, default=5000, help="cap tweets for speed")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--freeze-fraction", type=float, default=0.5)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, src = _load_encoder(args.backend, args.model_name, args.checkpoint, args.freeze_fraction, device)
    logger.info(f"backend={args.backend} ({src})  device={device}")

    examples, _ = load_pepe_examples(args.split)
    if args.max_tweets and len(examples) > args.max_tweets:
        # Deterministic sample.
        rng = np.random.default_rng(0)
        idx = rng.choice(len(examples), size=args.max_tweets, replace=False)
        examples = [examples[i] for i in idx]

    unique_gifs = sorted(set(e.gif_id for e in examples))
    logger.info(f"tweets: {len(examples)}  unique gifs: {len(unique_gifs)}")

    gif_mat, kept_ids = encode_gif_pool(model, unique_gifs, args.frame_cache_dir, args.batch_size, device)
    id_to_row = {g: i for i, g in enumerate(kept_ids)}
    logger.info(f"gif pool: {gif_mat.shape}; kept {len(kept_ids)}/{len(unique_gifs)}")

    eligible = [e for e in examples if e.gif_id in id_to_row]
    logger.info(f"eligible tweets: {len(eligible)}")
    txt_mat = encode_tweets(model, [e.text for e in eligible], device, batch_size=args.batch_size)
    logger.info(f"tweet matrix: {txt_mat.shape}")

    # Cosine since both sides are L2-normalized; if siglip backend returns unnorm, normalize defensively.
    txt_mat = txt_mat / np.maximum(np.linalg.norm(txt_mat, axis=1, keepdims=True), 1e-8)
    gif_mat = gif_mat / np.maximum(np.linalg.norm(gif_mat, axis=1, keepdims=True), 1e-8)
    sims = txt_mat @ gif_mat.T  # (T, G)

    gold_rows = np.array([id_to_row[e.gif_id] for e in eligible])
    # Rank of the gold gif among all gifs: how many have a strictly higher score.
    gold_scores = sims[np.arange(len(eligible)), gold_rows]
    ranks = (sims > gold_scores[:, None]).sum(axis=1)
    m = metrics_from_ranks(ranks)
    logger.info(f"metrics: {m}")
    print(m)


if __name__ == "__main__":
    main()
