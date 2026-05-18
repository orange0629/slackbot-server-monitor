"""Remote-side script: runs on a GPU box (e.g. burger), reads papers + members
from a JSON file, batches a single vLLM generate(), writes per-paper judgments.

Self-contained — only imports stdlib + vllm. Do NOT import any other
paper_curator submodule here; the file is shipped via scp to a host that may
not have the rest of the codebase.

Strategy
--------
For each (paper, member, theme) triple we issue one short prompt asking
"would someone whose research focus is THEME care about this paper?". The
prompt is structured paper-first / theme-last so vLLM's prefix cache reuses
the SYSTEM + PAPER prefix across every theme asked of the same paper
(`enable_prefix_caching=True`). We aggregate per-paper into the same output
schema the previous version produced.

Usage (on the remote host):
    python remote_judge.py \\
        --input  /tmp/paper_curator_remote/in.json \\
        --output /tmp/paper_curator_remote/out.json \\
        --model  Qwen/Qwen3-4B-Instruct \\
        --gpu-id 2 \\
        --tag-threshold 7

Input JSON schema (unchanged):
    {
      "papers":  [{"id": str, "title": str, "abstract": str, "source": str}, ...],
      "members": [{"name": str, "role": str, "affiliation": str,
                   "interests": [str, ...],
                   "publications": [{"title": str}, ...]}, ...]
    }

Output JSON schema (per-paper, aggregated):
    {
      "judgments": [
        {"id": str, "relevant": bool, "score": int, "tags": [str],
         "one_line_why": str,
         "per_member": {name: {"theme": str, "score": int, "why": str}}},
        ...
      ],
      "model": str,
      "elapsed_sec": float
    }
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


SYSTEM = (
    "You judge whether a research paper would be useful today to someone "
    "whose research focus is the one stated. Be VERY strict — when in doubt, "
    "score lower. Most papers should score 0-5; reserve 8+ for unambiguous "
    "matches the focus-holder would clearly want to read.\n"
    "Score guide (0-10):\n"
    " 9-10 = the paper IS the focus area: directly studies it as its central "
    "contribution, and the focus-holder would reliably read it.\n"
    " 8    = clear and substantive match; the focus-holder would almost "
    "certainly read it. Use only when the methodological or empirical core "
    "of the paper directly engages the focus, not just the topic area.\n"
    " 5-7  = related but not a clear match (adjacent subfield, partial "
    "overlap, or applies the focus to an unrelated problem).\n"
    " 0-4  = tangential, keyword-only match (e.g. 'mentions LLMs' without "
    "engaging the focus), or a survey/tutorial/review/comprehensive guide/"
    "position paper.\n"
    "If you are not confident the focus-holder would actively want to read "
    "this paper, score <= 7.\n"
    "Quote a specific phrase or claim from the paper in your `why`."
)


def _interest_themes(raw):
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    parts = [ln.strip(" -\t") for ln in str(raw).splitlines()]
    return [p for p in parts if p]


def _user_prompt(paper, theme):
    """Paper-first / theme-last so vLLM's prefix cache reuses the SYSTEM + PAPER
    portion across every theme asked of the same paper."""
    return (
        f"PAPER\n"
        f"title: {paper.get('title','')}\n"
        f"abstract: {paper.get('abstract','')[:1500]}\n"
        f"venue: {paper.get('source','')}\n"
        f"\n---\n\n"
        f"RESEARCH FOCUS: {theme}\n\n"
        "Respond with strict JSON only:\n"
        '{"score": 0-10, "why": "max 10 words; quote a specific phrase from '
        'the paper"}'
    )


def _parse_one(raw):
    try:
        s = raw.strip()
        i = s.find("{")
        if i > 0:
            s = s[i:]
        j = s.rfind("}")
        if j >= 0:
            s = s[: j + 1]
        data = json.loads(s)
        return {
            "score": int(data.get("score", 0)),
            "why": str(data.get("why", ""))[:200],
        }
    except Exception as e:
        return {"score": 0, "why": f"(parse failure: {e})"}


def _aggregate(papers, members, raw_rows, threshold):
    """raw_rows: list of (p_idx, member_name, theme, score, why).
    Returns list of per-paper aggregated judgments."""
    by_paper = {}
    for p_idx, name, theme, score, why in raw_rows:
        slot = by_paper.setdefault(p_idx, {})
        # Keep only the highest-scoring theme per (paper, member).
        prev = slot.get(name)
        if prev is None or score > prev["score"]:
            slot[name] = {"theme": theme, "score": score, "why": why}

    out = []
    for p_idx, paper in enumerate(papers):
        per_member = by_paper.get(p_idx, {})
        # Filter members by threshold; sort by score desc; take top 2.
        candidates = sorted(
            ((name, info) for name, info in per_member.items()
             if info["score"] >= threshold),
            key=lambda x: -x[1]["score"],
        )
        top = candidates[:2]
        tags = [name for name, _ in top]
        if top:
            best_why = top[0][1]["why"]
            best_score = top[0][1]["score"]
        else:
            best_why = ""
            best_score = max((info["score"] for info in per_member.values()),
                             default=0)
        out.append({
            "id": paper["id"],
            "relevant": bool(top),
            "score": best_score,
            "tags": tags,
            "one_line_why": best_why,
            "per_member": per_member,
        })
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu-id", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=80,
                    help="Generation cap per (paper, theme) judgment")
    ap.add_argument("--gpu-mem-util", type=float, default=0.85,
                    help="vLLM gpu_memory_utilization fraction")
    ap.add_argument("--tag-threshold", type=int, default=7,
                    help="Per-theme score below which a member is NOT tagged")
    args = ap.parse_args(argv)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    with open(args.input, "r") as f:
        payload = json.load(f)
    papers = payload.get("papers", [])
    members = payload.get("members", [])

    if not papers or not members:
        with open(args.output, "w") as f:
            json.dump({"judgments": [], "model": args.model,
                       "elapsed_sec": 0.0}, f)
        return 0

    # Pre-compute themes per member; skip members without any interests.
    member_themes = [(m["name"], _interest_themes(m.get("interests")))
                     for m in members]
    member_themes = [(n, ts) for n, ts in member_themes if ts]

    # Build prompts in paper-major order so the SYSTEM + PAPER prefix is
    # asked many times back-to-back (max prefix-cache reuse).
    triples = []  # (p_idx, member_name, theme)
    msg_pairs = []  # parallel list of [system, user] message dicts
    for p_idx, p in enumerate(papers):
        for name, themes in member_themes:
            for theme in themes:
                triples.append((p_idx, name, theme))
                msg_pairs.append([
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": _user_prompt(p, theme)},
                ])

    if not triples:
        with open(args.output, "w") as f:
            json.dump({"judgments": _aggregate(papers, members, [],
                                                args.tag_threshold),
                       "model": args.model, "elapsed_sec": 0.0}, f)
        return 0

    from vllm import LLM, SamplingParams  # heavy import; only on remote

    t0 = time.time()
    llm = LLM(model=args.model,
              gpu_memory_utilization=args.gpu_mem_util,
              enable_prefix_caching=True,
              dtype="auto",
              enforce_eager=False,
              trust_remote_code=True)
    sampling = SamplingParams(temperature=0.2, top_p=0.95,
                              max_tokens=args.max_tokens)

    tokenizer = llm.get_tokenizer()
    prompts = []
    for msgs in msg_pairs:
        try:
            prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = msgs[0]["content"] + "\n\n" + msgs[1]["content"]
        prompts.append(prompt)

    outputs = llm.generate(prompts, sampling)
    raw_rows = []
    for (p_idx, name, theme), out in zip(triples, outputs):
        text = out.outputs[0].text if out.outputs else ""
        parsed = _parse_one(text)
        raw_rows.append((p_idx, name, theme, parsed["score"], parsed["why"]))

    judgments = _aggregate(papers, members, raw_rows, args.tag_threshold)
    elapsed = time.time() - t0
    with open(args.output, "w") as f:
        json.dump({"judgments": judgments, "model": args.model,
                   "elapsed_sec": elapsed,
                   "n_prompts": len(prompts),
                   "tag_threshold": args.tag_threshold}, f)
    print(f"remote_judge: {len(prompts)} (paper, theme) prompts across "
          f"{len(papers)} papers in {elapsed:.1f}s "
          f"({elapsed/max(len(prompts),1):.2f}s/prompt)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
