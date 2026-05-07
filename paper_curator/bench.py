"""Benchmark Ollama models for the paper curator's relevance-judge step.

Usage:
    # default: bench the configured PAPER_CURATOR_OLLAMA_MODEL + fallback
    python -m paper_curator.bench

    # custom model list (skip the expensive ones you already timed)
    python -m paper_curator.bench --models gemma4:e2b gemma4:e4b qwen3.5:4b

    # smaller trial count for quick smoke tests
    python -m paper_curator.bench --models qwen3.5:4b --n 5

Past results (Mon May 5, this CPU box, 20 trials):
    qwen3.6:35b-a3b   244.7 s/paper
    gemma4:26b        727.2 s/paper
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import List

from config import (
    PAPER_CURATOR_OLLAMA_FALLBACK,
    PAPER_CURATOR_OLLAMA_HOST,
    PAPER_CURATOR_OLLAMA_MODEL,
)

SAMPLE_PAPERS = [
    {"title": "Beyond Consensus: Perspectivist Modeling of Annotator Disagreement",
     "abstract": "We argue that aggregating annotator labels obscures meaningful disagreement, "
                 "and propose a perspectivist framework that models per-annotator beliefs jointly."},
    {"title": "Quantum coherence in photosynthetic complexes",
     "abstract": "We provide spectroscopic evidence for long-lived quantum coherence in pigment-protein complexes."},
]


def _bench_one(client, model: str, papers: List[dict], members_block: str,
               warmup: bool = True, num_thread: int = 0) -> float:
    """Returns total seconds. Raises on failure.

    `num_thread`: passed through Ollama's options dict. 0 = let Ollama decide
    (currently defaults to physical core count, but this varies by version)."""
    from tqdm import tqdm

    base_options = {"num_ctx": 4096, "temperature": 0.2}
    if num_thread:
        base_options["num_thread"] = num_thread

    if warmup:
        kwargs = dict(model=model,
                      messages=[{"role": "user", "content": "ping"}],
                      options={**base_options, "num_ctx": 1024})
        try:
            client.chat(think=False, **kwargs)
        except TypeError:
            client.chat(**kwargs)

    label = f"{model} t={num_thread or 'auto'}"
    t0 = time.time()
    bar = tqdm(papers, desc=label, unit="paper",
               bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}")
    for p in bar:
        kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content":
                    "You are a research librarian. Reply with strict JSON.\n/no_think"},
                {"role": "user", "content":
                    f"PAPER:\ntitle: {p['title']}\nabstract: {p['abstract']}\n\n"
                    f"MEMBERS:\n{members_block}\n\n"
                    'Respond as: {"relevant": bool, "score": int, "tags": [str], "one_line_why": str}'},
            ],
            format="json",
            options=base_options,
        )
        try:
            resp = client.chat(think=False, **kwargs)
        except TypeError:
            resp = client.chat(**kwargs)
        _ = resp["message"]["content"]
        bar.set_postfix_str(f"{(time.time()-t0)/(bar.n+1):.2f}s/paper")
    bar.close()
    return time.time() - t0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=[PAPER_CURATOR_OLLAMA_MODEL, PAPER_CURATOR_OLLAMA_FALLBACK],
                    help="Ollama model tags to benchmark")
    ap.add_argument("--n", type=int, default=20,
                    help="number of paper trials per model (default: 20)")
    ap.add_argument("--no-warmup", action="store_true",
                    help="skip the untimed warmup call (less accurate but quicker)")
    ap.add_argument("--out", default=None,
                    help="optional path to append JSON results (one row per run)")
    ap.add_argument("--threads", nargs="+", type=int, default=[0],
                    help="thread counts to sweep (default: [0] = let Ollama decide). "
                         "Example: --threads 8 16 24 40 — runs each model at each thread count.")
    args = ap.parse_args(argv)

    try:
        import ollama
    except ImportError:
        print("ollama python client not installed", file=sys.stderr)
        return 2
    client = ollama.Client(host=PAPER_CURATOR_OLLAMA_HOST)
    members_block = ("- Alice (NLP): annotation, perspectivism, evaluation\n"
                     "- Bob (CSS): online communities, sociolinguistics")

    papers = (SAMPLE_PAPERS * ((args.n // len(SAMPLE_PAPERS)) + 1))[:args.n]

    results = {}
    for model in args.models:
        for nt in args.threads:
            key = f"{model} (t={nt or 'auto'})"
            print(f"=== {key} ===")
            try:
                elapsed = _bench_one(client, model, papers, members_block,
                                     warmup=not args.no_warmup, num_thread=nt)
                results[key] = elapsed
                print(f"  total: {elapsed:.1f}s  ({elapsed/len(papers):.2f}s/paper)")
            except Exception as e:
                print(f"  FAILED: {e}")
                results[key] = None

    print("\n=== summary ===")
    print(json.dumps(results, indent=2))
    valid = {k: v for k, v in results.items() if v is not None}
    if valid:
        winner = min(valid, key=valid.get)
        print(f"\nFastest: {winner}  ({valid[winner]/len(papers):.2f}s/paper)")

    if args.out:
        row = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n": len(papers),
            "warmup": not args.no_warmup,
            "results": results,
        }
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"\nappended results to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
