"""Fair head-to-head: legacy PEPE vs prior SigLIP finetune vs PEPE-v2.

The comparison pins everything that *can* be pinned so the model is the only
variable:
  - same dev queries (deterministic rng(0) subsample, identical to eval_dcg)
  - same candidate pool (intersection of what every arm can embed)
  - same metric (eval_dcg.metrics_from_ranks: cosine on L2-normalized rows,
    rank = #gifs scoring strictly higher than gold)

Arms:
  - prior-finetune-768d : SigLIPDualEncoder @ siglip_ft_p2_sigmoid/best.pt
  - pepe_v2-768d        : SigLIPDualEncoder @ pepe_v2/best.pt
  - legacy-PEPE-512d    : precomputed gif vectors + original PEPE text encoder
                          (best-effort; degrades with a NOTE if the legacy
                          model tree can't be imported)

The legacy arm is dimensionally/architecturally asymmetric by construction —
that asymmetry IS the thing being compared. The unambiguous claim is
prior-finetune vs pepe_v2 (fully symmetric code path).

Usage:
  python -m gif_reply.training.compare_pepe --split dev --max-tweets 200   # smoke
  python -m gif_reply.training.compare_pepe --split dev --max-tweets 3000  # full
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import sys

import numpy as np
import torch

from .data import DEFAULT_FRAME_CACHE_DIR, load_pepe_examples
from .eval_dcg import _load_encoder, encode_gif_pool, encode_tweets, metrics_from_ranks

logger = logging.getLogger(__name__)

PEPE_FEATURES_CSV = "/shared/2/projects/gif-reply/data/release/gif-pepe-inferred-features.csv"


def _subsample(examples, max_tweets):
    if max_tweets and len(examples) > max_tweets:
        rng = np.random.default_rng(0)  # identical to eval_dcg.evaluate_retrieval
        idx = rng.choice(len(examples), size=max_tweets, replace=False)
        examples = [examples[i] for i in idx]
    return examples


def _siglip_arm(checkpoint, examples, unique_gifs, model_name, freeze_fraction,
                frame_cache_dir, batch_size, device) -> dict[str, np.ndarray] | None:
    """Returns {gif_id: gifvec} plus a stashed text matrix keyed by '__txt__'/
    '__txt_ids__'. None on failure."""
    try:
        model, src = _load_encoder("siglip_ft", model_name, checkpoint, freeze_fraction, device)
    except Exception as e:
        logger.warning(f"siglip arm load failed ({checkpoint}): {e}")
        return None
    logger.info(f"siglip arm ({src})")
    gif_mat, kept_ids = encode_gif_pool(model, unique_gifs, frame_cache_dir, batch_size, device)
    gif_vecs = {g: gif_mat[i] for i, g in enumerate(kept_ids)}
    # Encode every dev tweet; we restrict to the common pool later.
    txt = encode_tweets(model, [e.text for e in examples], device, batch_size=batch_size)
    return {"gifs": gif_vecs, "txt": txt}


def _legacy_pepe_arm(examples, unique_gifs, device) -> dict | None:
    """Precomputed 512-d gif vectors + original PEPE text encoder. None if the
    legacy model tree can't be imported."""
    needed = set(unique_gifs)
    gif_vecs: dict[str, np.ndarray] = {}
    try:
        with open(PEPE_FEATURES_CSV, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                gid = r.get("gif_id")
                if gid not in needed:
                    continue
                raw = r.get("gif_feature") or ""
                try:
                    vec = np.asarray(json.loads(raw), dtype=np.float32)
                except Exception:
                    vec = np.asarray(ast.literal_eval(raw), dtype=np.float32)
                gif_vecs[gid] = vec
    except FileNotFoundError:
        logger.warning(f"legacy PEPE features csv not found: {PEPE_FEATURES_CSV}")
        return None
    if not gif_vecs:
        logger.warning("legacy PEPE arm: no precomputed vectors for the dev pool")
        return None
    try:
        from gif_reply.encoders import PepeEncoder
        import config

        enc = PepeEncoder(checkpoint_path=config.GIF_REPLY_PEPE_CHECKPOINT)
    except Exception as e:
        logger.warning(f"legacy PEPE arm: text encoder unavailable ({e}); skipping arm")
        return None
    txt = np.stack([enc.encode_text(e.text) for e in examples], axis=0)
    return {"gifs": gif_vecs, "txt": txt}


def _rank_metrics(arm: dict, examples, common_ids: list[str]) -> dict:
    """Restrict the arm to common_ids + tweets whose gold is in common, then
    compute eval_dcg metrics — identical math across all arms."""
    id_to_row = {g: i for i, g in enumerate(common_ids)}
    gif_mat = np.stack([arm["gifs"][g] for g in common_ids], axis=0).astype(np.float32)
    elig_idx = [i for i, e in enumerate(examples) if e.gif_id in id_to_row]
    txt_mat = arm["txt"][elig_idx].astype(np.float32)
    txt_mat = txt_mat / np.maximum(np.linalg.norm(txt_mat, axis=1, keepdims=True), 1e-8)
    gif_mat = gif_mat / np.maximum(np.linalg.norm(gif_mat, axis=1, keepdims=True), 1e-8)
    sims = txt_mat @ gif_mat.T
    gold_rows = np.array([id_to_row[examples[i].gif_id] for i in elig_idx])
    gold_scores = sims[np.arange(len(elig_idx)), gold_rows]
    ranks = (sims > gold_scores[:, None]).sum(axis=1)
    m = metrics_from_ranks(ranks)
    m["pool_size"] = len(common_ids)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    p.add_argument("--max-tweets", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--model-name", default="google/siglip-base-patch16-224")
    p.add_argument("--freeze-fraction", type=float, default=0.5)
    p.add_argument("--frame-cache-dir", default=DEFAULT_FRAME_CACHE_DIR)
    p.add_argument("--prior-checkpoint",
                   default="/shared/0/projects/gif-reply-slack-bot/models/siglip_ft_p2_sigmoid/best.pt")
    p.add_argument("--pepe-v2-checkpoint",
                   default="/shared/0/projects/gif-reply-slack-bot/models/pepe_v2/best.pt")
    p.add_argument("--skip-legacy", action="store_true",
                   help="don't attempt the asymmetric legacy-PEPE arm")
    p.add_argument("--out-json", default=None, help="write the results table here")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    examples, _ = load_pepe_examples(args.split)
    examples = _subsample(examples, args.max_tweets)
    unique_gifs = sorted(set(e.gif_id for e in examples))
    logger.info(f"split={args.split} tweets={len(examples)} unique_gifs={len(unique_gifs)}")

    arms: dict[str, dict] = {}
    notes: list[str] = []

    prior = _siglip_arm(args.prior_checkpoint, examples, unique_gifs, args.model_name,
                        args.freeze_fraction, args.frame_cache_dir, args.batch_size, device)
    if prior:
        arms["prior-finetune-768d"] = prior
    else:
        notes.append("prior-finetune arm unavailable (checkpoint load failed)")

    v2 = _siglip_arm(args.pepe_v2_checkpoint, examples, unique_gifs, args.model_name,
                     args.freeze_fraction, args.frame_cache_dir, args.batch_size, device)
    if v2:
        arms["pepe_v2-768d"] = v2
    else:
        notes.append("pepe_v2 arm unavailable (checkpoint not found — train it first)")

    if not args.skip_legacy:
        legacy = _legacy_pepe_arm(examples, unique_gifs, device)
        if legacy:
            arms["legacy-PEPE-512d"] = legacy
        else:
            notes.append("legacy-PEPE arm degraded out (see warnings above) — "
                         "report covers the symmetric SigLIP arms only")

    if len(arms) < 1:
        logger.error("no arms available; nothing to compare")
        sys.exit(1)

    # Fairness: common pool = gifs every arm can embed; intersect across arms.
    common = set(unique_gifs)
    for arm in arms.values():
        common &= set(arm["gifs"].keys())
    common_ids = sorted(common)
    if not common_ids:
        logger.error("empty common gif pool across arms; cannot compare fairly")
        sys.exit(1)
    logger.info(f"common pool across {len(arms)} arm(s): {len(common_ids)} gifs")

    results = {name: _rank_metrics(arm, examples, common_ids) for name, arm in arms.items()}

    cols = ["recall@1", "recall@5", "recall@10", "recall@30", "mrr", "dcg@30", "n", "pool_size"]
    print("\n=== PEPE vs PEPE-v2 retrieval (pinned queries/pool/metric) ===")
    print(f"{'arm':<22}" + "".join(f"{c:>11}" for c in cols))
    for name, m in results.items():
        row = f"{name:<22}"
        for c in cols:
            v = m[c]
            row += f"{v:>11}" if isinstance(v, int) else f"{v:>11.4f}"
        print(row)
    if notes:
        print("\nNOTES:")
        for nt in notes:
            print(f"  - {nt}")
    print("\nClean apples-to-apples claim = prior-finetune-768d vs pepe_v2-768d "
          "(identical code path). legacy-PEPE-512d differs in dim + encoder by construction.")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({"results": results, "notes": notes,
                       "split": args.split, "n_arms": len(arms)}, f, indent=2)
        logger.info(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
