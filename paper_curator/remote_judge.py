"""Remote-side script: runs on a GPU box (e.g. burger), reads papers + members
from a JSON file, batches a single vLLM generate(), writes judgments.

Self-contained — only imports stdlib + vllm. Do NOT import any other
paper_curator submodule here; the file is shipped via scp to a host that may
not have the rest of the codebase.

Usage (on the remote host):
    python remote_judge.py \\
        --input  /tmp/paper_curator_remote/in.json \\
        --output /tmp/paper_curator_remote/out.json \\
        --model  Qwen/Qwen3-4B-Instruct \\
        --gpu-id 2

Input JSON schema:
    {
      "papers":  [{"id": str, "title": str, "abstract": str, "source": str}, ...],
      "members": [{"name": str, "role": str, "affiliation": str,
                   "publications": [{"title": str}, ...]}, ...]
    }

Output JSON schema:
    {
      "judgments": [
        {"id": str, "relevant": bool, "score": int, "tags": [str], "one_line_why": str},
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
    "You are a research librarian for the Blablablab lab (PI: David Jurgens, "
    "U-Michigan School of Information). The lab works on NLP, computational "
    "social science, sociolinguistics, and human behavior with language. "
    "Decide if a paper is worth surfacing to the lab today and tag at most "
    "two members it most fits. Be strict: only mark relevant=true if a "
    "specific member would genuinely want to read it.\n"
    "REJECT (relevant=false) papers that are:\n"
    " - surveys, tutorials, reviews, 'comprehensive guides', 'practical "
    "guides', or textbook-style overviews — they look broadly relevant but "
    "are not original research the lab needs to know about today.\n"
    " - position papers, opinion pieces, or roadmaps without new results.\n"
    " - papers whose match to a member is only at the topic-keyword level "
    "(e.g. 'mentions LLMs') rather than a substantive methodological or "
    "empirical fit. A paper is only relevant if it advances or directly "
    "challenges a member's specific research direction."
)


def _members_block(members):
    lines = []
    for m in members:
        pubs = [p["title"] for p in (m.get("publications") or [])][:3]
        bio = m.get("affiliation", "") or m.get("role", "")
        themes = _interest_themes(m.get("interests"))
        lines.append(f"- {m['name']} ({bio}). Recent: " + " | ".join(pubs))
        for t in themes:
            lines.append(f"    * {t}")
    return "\n".join(lines)


def _interest_themes(raw):
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    parts = [ln.strip(" -\t") for ln in str(raw).splitlines()]
    return [p for p in parts if p]


def _user_prompt(paper, members_block):
    return (
        f"PAPER:\n"
        f"title: {paper.get('title','')}\n"
        f"abstract: {paper.get('abstract','')[:1500]}\n"
        f"venue: {paper.get('source','')}\n\n"
        f"MEMBERS:\n{members_block}\n\n"
        "Respond with strict JSON only:\n"
        '{"relevant": bool, "score": int 0-10, '
        '"tags": [<=2 member names exactly as listed], '
        '"one_line_why": "max 10 words; specific phrase explaining the fit; '
        'omit names and filler words"}'
    )


def _parse_judgment(raw, paper_id, member_names):
    try:
        # vLLM may emit leading whitespace; find the first '{'.
        s = raw.strip()
        i = s.find("{")
        if i > 0:
            s = s[i:]
        # And trim anything after the last '}'.
        j = s.rfind("}")
        if j >= 0:
            s = s[: j + 1]
        data = json.loads(s)
        return {
            "id": paper_id,
            "relevant": bool(data.get("relevant")),
            "score": int(data.get("score", 0)),
            "tags": [t for t in (data.get("tags") or [])
                     if isinstance(t, str) and t in member_names][:2],
            "one_line_why": str(data.get("one_line_why", ""))[:200],
        }
    except Exception as e:
        return {
            "id": paper_id,
            "relevant": False,
            "score": 0,
            "tags": [],
            "one_line_why": f"(parse failure: {e})",
        }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu-id", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85,
                    help="vLLM gpu_memory_utilization fraction")
    args = ap.parse_args(argv)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    with open(args.input, "r") as f:
        payload = json.load(f)
    papers = payload.get("papers", [])
    members = payload.get("members", [])
    member_names = {m["name"] for m in members}
    members_block = _members_block(members)

    if not papers:
        with open(args.output, "w") as f:
            json.dump({"judgments": [], "model": args.model, "elapsed_sec": 0.0}, f)
        return 0

    from vllm import LLM, SamplingParams  # heavy import; only on remote

    t0 = time.time()
    llm = LLM(model=args.model,
              gpu_memory_utilization=args.gpu_mem_util,
              dtype="auto",
              enforce_eager=False,
              trust_remote_code=True)
    sampling = SamplingParams(temperature=0.2, top_p=0.95,
                              max_tokens=args.max_tokens)

    # Build chat-template prompts via the tokenizer wrapped in the LLM.
    tokenizer = llm.get_tokenizer()
    prompts = []
    for p in papers:
        msgs = [
            {"role": "system", "content": SYSTEM + "\n/no_think"},
            {"role": "user", "content": _user_prompt(p, members_block)},
        ]
        # Qwen3 tokenizer honors enable_thinking=False to suppress the
        # <think>...</think> reasoning trace at the template level.
        try:
            prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            # Older transformers / non-Qwen tokenizers: fall back without the kwarg.
            # The "/no_think" directive already in SYSTEM still suppresses thinking
            # for Qwen3 in that case.
            prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = SYSTEM + "\n\n" + _user_prompt(p, members_block)
        prompts.append(prompt)

    outputs = llm.generate(prompts, sampling)
    judgments = []
    for paper, out in zip(papers, outputs):
        text = out.outputs[0].text if out.outputs else ""
        judgments.append(_parse_judgment(text, paper["id"], member_names))

    elapsed = time.time() - t0
    with open(args.output, "w") as f:
        json.dump({"judgments": judgments, "model": args.model,
                   "elapsed_sec": elapsed}, f)
    print(f"remote_judge: {len(judgments)} papers in {elapsed:.1f}s "
          f"({elapsed/len(judgments):.2f}s/paper)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
