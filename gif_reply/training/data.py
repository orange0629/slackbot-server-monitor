"""Dataset + collate for PEPE-style (tweet, gif, tags) training.

Reads /shared/2/projects/gif-reply/data/processed/dataset/bertweet-normalize/
finalized-split-dataset/tweet-gif-reply.pickle and the matching .meta label map.

Per training example:
  - input_ids, attention_mask: SigLIP-tokenized tweet text
  - frames: float tensor (N_FRAMES, 3, 224, 224) — N evenly-spaced frames per gif
  - tag_vec: float tensor (N_TAGS,) — multi-hot

Frame tensors are cached per-gif under FRAME_CACHE_DIR so we decode each MP4
exactly once across all epochs and across all dataloader workers.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import logging
import os
import pickle
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# --- Constants from the PEPE release ---
PEPE_PICKLE = "/shared/2/projects/gif-reply/data/processed/dataset/bertweet-normalize/finalized-split-dataset/tweet-gif-reply.pickle"
PEPE_PICKLE_META = PEPE_PICKLE + ".meta"
PEPE_GIFS_ROOT = "/shared/2/projects/gif-reply/data/processed/dataset/gifs"

# --- Cache locations (override via env or CLI) ---
DEFAULT_FRAME_CACHE_DIR = "/shared/0/projects/gif-reply-slack-bot/training_cache/frames"
DEFAULT_TOKEN_CACHE_DIR = "/shared/0/projects/gif-reply-slack-bot/training_cache/tokens"

# Match PEPE's frame-sampling formula (utils/__init__.py:66 in the original tree).
N_FRAMES = 4
FRAME_SIZE = 224


def gif_id_to_mp4_path(gif_id: str, root: str = PEPE_GIFS_ROOT) -> str:
    """PEPE's path scheme: gifs/<id[0]>/<id[1]>/<id[2]>/<id[3:]>.mp4."""
    if len(gif_id) < 4:
        raise ValueError(f"gif_id too short: {gif_id!r}")
    return os.path.join(root, gif_id[0], gif_id[1], gif_id[2], gif_id[3:] + ".mp4")


def load_meta(meta_path: str = PEPE_PICKLE_META) -> dict:
    with open(meta_path, "rb") as f:
        return pickle.load(f)


def load_pickle_compat(path: str):
    """Load the PEPE training pickle. The pickle was saved with an older pandas
    that referenced `pandas.core.indexes.numeric`, which moved in pandas 2.x.
    Provide a shim before unpickling so we can read it on newer pandas.
    """
    import pandas as pd  # noqa
    # Inject compatibility shim: newer pandas removed the old internal module.
    try:
        import pandas.core.indexes.numeric  # noqa: F401
    except ImportError:
        import pandas.core.indexes.range as _range_mod
        import types
        shim = types.ModuleType("pandas.core.indexes.numeric")
        # Older code only needs Int64Index/Float64Index/UInt64Index aliases; fall
        # back to RangeIndex.__bases__[0] which is `Index`.
        shim.Int64Index = pd.Index
        shim.Float64Index = pd.Index
        shim.UInt64Index = pd.Index
        sys.modules["pandas.core.indexes.numeric"] = shim
    with open(path, "rb") as f:
        return pickle.load(f)


# --- Frame decoding ---

def _decode_with_pyav(mp4_path: str) -> np.ndarray | None:
    try:
        import av
    except ImportError:
        return None
    try:
        with av.open(mp4_path) as container:
            stream = container.streams.video[0]
            frames = []
            for frame in container.decode(stream):
                frames.append(frame.to_ndarray(format="rgb24"))
            if not frames:
                return None
            return np.stack(frames, axis=0)  # (T, H, W, 3)
    except Exception as e:
        logger.warning(f"pyav decode failed {mp4_path}: {e}")
        return None


def _decode_with_cv2(mp4_path: str) -> np.ndarray | None:
    try:
        import cv2
    except ImportError:
        return None
    try:
        cap = cv2.VideoCapture(mp4_path)
        out = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap.release()
        if not out:
            return None
        return np.stack(out, axis=0)
    except Exception as e:
        logger.warning(f"cv2 decode failed {mp4_path}: {e}")
        return None


def _decode_mp4_to_frames(mp4_path: str, n_frames: int = N_FRAMES, size: int = FRAME_SIZE) -> torch.Tensor | None:
    """Decode an MP4 → (n_frames, 3, size, size) float tensor in [0, 1].

    Tries PyAV first, then OpenCV, then imageio. Returns None on failure.
    """
    frames = _decode_with_pyav(mp4_path)
    if frames is None:
        frames = _decode_with_cv2(mp4_path)
    if frames is None:
        try:
            try:
                import imageio.v3 as iio
            except ImportError:
                import imageio as iio
            frames = iio.imread(mp4_path, plugin="FFMPEG")
        except Exception as e:
            logger.warning(f"all decoders failed {mp4_path}: {e}")
            return None
    if frames is None or len(frames) == 0:
        return None
    n_total = frames.shape[0]
    if n_total < n_frames:
        # Repeat last frame to pad.
        idx = list(range(n_total)) + [n_total - 1] * (n_frames - n_total)
    else:
        # Match PEPE: i * (n_total // n_frames). Clamp to last index.
        step = max(n_total // n_frames, 1)
        idx = [min(i * step, n_total - 1) for i in range(n_frames)]
    sel = frames[idx]  # (n_frames, H, W, 3)
    # Resize via torch (avoid PIL roundtrip per-frame).
    t = torch.from_numpy(sel).float().permute(0, 3, 1, 2) / 255.0  # (N, 3, H, W)
    t = torch.nn.functional.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t


def _normalize_for_siglip(frames: torch.Tensor) -> torch.Tensor:
    """SigLIP uses CLIP-like normalization: mean/std = 0.5/0.5 per channel."""
    mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    return (frames - mean) / std


def gif_uid(gif_id: str) -> int:
    """Stable signed-int64 hash of a gif_id.

    Used only for in-batch equality (multi-positive masking). It must be
    consistent across datasets (PEPE and TGIF) and across processes/workers,
    so we hash the string rather than use a per-dataset running index.
    Collisions across the ~10^5 gif pool are astronomically unlikely at 64 bit.
    """
    h = hashlib.blake2b(gif_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little", signed=True)


def cache_path_for_gif(gif_id: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, gif_id[:2], f"{gif_id}.pt")


def decode_and_cache(gif_id: str, cache_dir: str, gifs_root: str = PEPE_GIFS_ROOT) -> bool:
    """Decode → normalize → save. Returns True on success."""
    out = cache_path_for_gif(gif_id, cache_dir)
    if os.path.exists(out):
        return True
    mp4 = gif_id_to_mp4_path(gif_id, gifs_root)
    if not os.path.exists(mp4):
        return False
    frames = _decode_mp4_to_frames(mp4)
    if frames is None:
        return False
    frames = _normalize_for_siglip(frames).to(torch.float16)  # half-precision on disk
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    torch.save(frames, tmp)
    os.replace(tmp, out)
    return True


def build_frame_cache(gif_ids: Iterable[str], cache_dir: str, workers: int = 8) -> dict:
    """Decode every gif to disk. Idempotent; skips existing cache files."""
    gif_ids = list(dict.fromkeys(gif_ids))  # dedup
    results = {"ok": 0, "skip": 0, "fail": 0}
    os.makedirs(cache_dir, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(decode_and_cache, g, cache_dir): g for g in gif_ids}
        from tqdm import tqdm
        for fut in tqdm(as_completed(futs), total=len(futs), desc="decode"):
            try:
                ok = fut.result()
            except Exception:
                ok = False
            if ok:
                results["ok"] += 1
            else:
                results["fail"] += 1
    return results


# --- Tokenization cache ---

def build_token_cache(texts: list[str], cache_path: str, model_name: str = "google/siglip-base-patch16-224") -> None:
    """Tokenize once, store input_ids (and attention_mask if the processor emits one)."""
    if os.path.exists(cache_path):
        return
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(model_name)
    # SigLIP uses padding="max_length" and does NOT return attention_mask.
    # We tokenize in chunks to keep peak memory bounded for the 1.25M train split.
    chunks_ids: list[torch.Tensor] = []
    chunks_mask: list[torch.Tensor] = []
    chunk_size = 4096
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        enc = proc(text=chunk, return_tensors="pt", padding="max_length", truncation=True)
        chunks_ids.append(enc["input_ids"])
        if "attention_mask" in enc:
            chunks_mask.append(enc["attention_mask"])
    payload = {"input_ids": torch.cat(chunks_ids, dim=0)}
    if chunks_mask:
        payload["attention_mask"] = torch.cat(chunks_mask, dim=0)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp = cache_path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, cache_path)


# --- Dataset ---

@dataclass
class Example:
    text: str
    gif_id: str
    tag_indices: list[int]  # indices into label_to_id


class GifReplyDataset(Dataset):
    """Returns dicts with input_ids, attention_mask, frames, tag_vec.

    Loads frames lazily from the on-disk cache; missing frames are skipped at
    __getitem__ time by returning a sentinel that the collate function drops.
    """

    def __init__(
        self,
        examples: list[Example],
        token_cache_path: str,
        frame_cache_dir: str,
        n_tags: int,
        n_frames: int | None = None,
    ):
        self.examples = examples
        self.frame_cache_dir = frame_cache_dir
        self.n_tags = n_tags
        # Cache holds N_FRAMES frames per gif; optionally use fewer (memory/speed).
        self.n_frames = n_frames
        cache = torch.load(token_cache_path, map_location="cpu", weights_only=False)
        self.input_ids = cache["input_ids"]
        # SigLIP doesn't emit an attention_mask. Derive an all-ones mask of the
        # right shape so collate has a tensor to stack — the model ignores it.
        self.attention_mask = cache.get(
            "attention_mask",
            torch.ones_like(self.input_ids),
        )
        if len(self.input_ids) != len(examples):
            raise ValueError(f"token cache size {len(self.input_ids)} != examples {len(examples)}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        gif_id = ex["gif_id"] if isinstance(ex, dict) else ex.gif_id
        tag_indices = ex["tag_indices"] if isinstance(ex, dict) else ex.tag_indices
        cache_path = cache_path_for_gif(gif_id, self.frame_cache_dir)
        try:
            frames = torch.load(cache_path, map_location="cpu", weights_only=False).float()
        except Exception:
            return None  # collate drops Nones
        if self.n_frames is not None and frames.shape[0] > self.n_frames:
            # Evenly subsample the cached frames (cache is N_FRAMES, ordered).
            step = max(frames.shape[0] // self.n_frames, 1)
            sel = [min(i * step, frames.shape[0] - 1) for i in range(self.n_frames)]
            frames = frames[sel]
        tag_vec = torch.zeros(self.n_tags, dtype=torch.float32)
        if tag_indices:
            tag_vec[torch.tensor(tag_indices, dtype=torch.long)] = 1.0
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "frames": frames,
            "tag_vec": tag_vec,
            "gif_uid": torch.tensor(gif_uid(gif_id), dtype=torch.long),
        }


def collate(batch):
    """Drop Nones from missing-cache items, stack the rest."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "frames": torch.stack([b["frames"] for b in batch]),  # (B, N_FRAMES, 3, H, W)
        "tag_vec": torch.stack([b["tag_vec"] for b in batch]),
        "gif_uid": torch.stack([b["gif_uid"] for b in batch]),  # (B,) for multi-positive
    }


# --- Loading the PEPE pickle into Examples ---

def load_pepe_examples(split: str, pickle_path: str = PEPE_PICKLE, meta_path: str = PEPE_PICKLE_META) -> tuple[list[Example], dict]:
    """Returns (examples, meta). split ∈ {"train", "dev", "test"}."""
    meta = load_meta(meta_path)
    label_to_id = meta["label_to_id"]
    df = load_pickle_compat(pickle_path)
    sub = df[df["set"] == split]
    examples: list[Example] = []
    for text, gif_id, tags in zip(sub["parent_text"].tolist(), sub["child_gif_id"].tolist(), sub["all_tags"].tolist()):
        if not isinstance(text, str) or not isinstance(gif_id, str):
            continue
        idxs = [label_to_id[t] for t in (tags or []) if t in label_to_id]
        examples.append(Example(text=text, gif_id=gif_id, tag_indices=idxs))
    return examples, meta


# --- CLI ---

def _cli():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("build-frame-cache")
    pf.add_argument("--cache-dir", default=DEFAULT_FRAME_CACHE_DIR)
    pf.add_argument("--workers", type=int, default=16)
    pf.add_argument("--limit", type=int, default=None, help="cap unique gifs for smoke testing")

    pt = sub.add_parser("build-token-cache")
    pt.add_argument("--split", required=True, choices=["train", "dev", "test"])
    pt.add_argument("--cache-dir", default=DEFAULT_TOKEN_CACHE_DIR)
    pt.add_argument("--model-name", default="google/siglip-base-patch16-224")

    pi = sub.add_parser("inspect")
    pi.add_argument("--n", type=int, default=5)

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "build-frame-cache":
        # Collect unique gif_ids across all splits.
        df = load_pickle_compat(PEPE_PICKLE)
        gif_ids = df["child_gif_id"].dropna().unique().tolist()
        if args.limit:
            random.seed(0)
            gif_ids = random.sample(gif_ids, min(args.limit, len(gif_ids)))
        logger.info(f"decoding {len(gif_ids)} unique gifs into {args.cache_dir}")
        results = build_frame_cache(gif_ids, args.cache_dir, workers=args.workers)
        logger.info(f"done: {results}")

    elif args.cmd == "build-token-cache":
        examples, _ = load_pepe_examples(args.split)
        out = os.path.join(args.cache_dir, f"{args.split}.pt")
        logger.info(f"tokenizing {len(examples)} {args.split} texts → {out}")
        build_token_cache([e.text for e in examples], out, model_name=args.model_name)
        # Save as plain dicts so unpickling doesn't depend on the dataclass'
        # __module__ (which differs between `python -m ...` and library import).
        as_dicts = [{"text": e.text, "gif_id": e.gif_id, "tag_indices": e.tag_indices} for e in examples]
        meta_out = os.path.join(args.cache_dir, f"{args.split}.examples.pt")
        torch.save(as_dicts, meta_out)
        logger.info(f"saved {meta_out}")

    elif args.cmd == "inspect":
        meta = load_meta()
        print(f"label space: {meta['n_labels']} tags; first 5: {meta['id_to_label'][:5]}")
        df = load_pickle_compat(PEPE_PICKLE)
        print(f"rows: {len(df)}; cols: {list(df.columns)}")
        print(df.head(args.n))
        print("split counts:", df["set"].value_counts().to_dict())


if __name__ == "__main__":
    _cli()
