"""Local-side dispatcher for the GPU box (burger).

Uses the shared `/shared/6/projects/food-bot/` filesystem (visible from both
this host and burger) so no scp is needed. We just:
  1. Write inputs to runs/<ts>/in.json on the shared FS.
  2. SSH to burger and exec remote_judge.py via flock — it reads inputs and
     writes outputs on the same shared path.
  3. Read outputs back locally.

Returns a list of judgments aligned to the input papers, or None on any
failure (caller falls back to local Ollama / bi-encoder-only).
"""
from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config import (
    PAPER_CURATOR_REMOTE_BATCH_SIZE,
    PAPER_CURATOR_REMOTE_GPU_POLL_INTERVAL,
    PAPER_CURATOR_REMOTE_GPU_WAIT_TIMEOUT,
    PAPER_CURATOR_REMOTE_HOST,
    PAPER_CURATOR_REMOTE_MAX_GPU_UTIL,
    PAPER_CURATOR_REMOTE_MIN_GPU_FREE_GB,
    PAPER_CURATOR_REMOTE_MODEL,
    PAPER_CURATOR_REMOTE_PYTHON,
    PAPER_CURATOR_REMOTE_SALVAGE_GRACE,
    PAPER_CURATOR_REMOTE_TIMEOUT,
    PAPER_CURATOR_TAG_SCORE_THRESHOLD,
)

from . import paths

logger = logging.getLogger(__name__)

SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10",
]

RUNS_KEEP = 50  # garbage-collect older per-run dirs to keep things tidy


def _ssh(cmd: str, timeout: int = 30) -> Optional[str]:
    full = ["ssh", *SSH_OPTS, PAPER_CURATOR_REMOTE_HOST, cmd]
    try:
        out = subprocess.check_output(full, stderr=subprocess.PIPE, timeout=timeout)
        return out.decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        logger.warning("ssh timeout for: %s", cmd[:80])
        return None
    except subprocess.CalledProcessError as e:
        logger.warning("ssh failed (%d): %s | stderr=%s",
                       e.returncode, cmd[:80],
                       e.stderr.decode("utf-8", errors="replace")[:300])
        return None


# ---------- GPU pick ------------------------------------------------------

def _probe_free_gpu() -> Optional[tuple]:
    """One-shot probe: returns (idx, free_gb, util) for the most-free qualifying
    GPU, or None if nvidia-smi failed or no GPU meets the thresholds."""
    raw = _ssh("nvidia-smi --query-gpu=index,memory.free,utilization.gpu "
               "--format=csv,noheader,nounits")
    if not raw:
        return None
    candidates = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx, free_mib, util = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        free_gb = free_mib / 1024.0
        if (free_gb >= PAPER_CURATOR_REMOTE_MIN_GPU_FREE_GB
                and util <= PAPER_CURATOR_REMOTE_MAX_GPU_UTIL):
            candidates.append((idx, free_gb, util))
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[1])
    return candidates[0]


def find_free_gpu(
    wait_timeout: int = PAPER_CURATOR_REMOTE_GPU_WAIT_TIMEOUT,
    poll_interval: int = PAPER_CURATOR_REMOTE_GPU_POLL_INTERVAL,
) -> Optional[int]:
    """Pick a remote GPU index meeting free-VRAM + idle thresholds, polling up
    to ``wait_timeout`` seconds if none is currently free. Returns the GPU index
    or None if nvidia-smi is unreachable or the wait expires."""
    deadline = time.time() + max(0, wait_timeout)
    first = True
    while True:
        pick = _probe_free_gpu()
        if pick is not None:
            logger.info("remote GPU %d selected (%.1f GB free, %d%% util)", *pick)
            return pick[0]
        remaining = deadline - time.time()
        if remaining <= 0:
            logger.info("gave up waiting for remote GPU (need %sGB free, util<=%s%%)",
                        PAPER_CURATOR_REMOTE_MIN_GPU_FREE_GB,
                        PAPER_CURATOR_REMOTE_MAX_GPU_UTIL)
            return None
        if first:
            logger.info("no remote GPU meets thresholds (need %sGB free, util<=%s%%); "
                        "polling every %ds for up to %ds",
                        PAPER_CURATOR_REMOTE_MIN_GPU_FREE_GB,
                        PAPER_CURATOR_REMOTE_MAX_GPU_UTIL,
                        poll_interval, wait_timeout)
            first = False
        time.sleep(min(poll_interval, remaining))


# ---------- Shared-FS layout ---------------------------------------------

def _sync_remote_script() -> str:
    """Copy paper_curator/remote_judge.py into the shared bin/ on every change."""
    paths.ensure_dirs()
    src = os.path.join(os.path.dirname(__file__), "remote_judge.py")
    dst = os.path.join(paths.BIN_DIR, "remote_judge.py")
    if (not os.path.exists(dst)
            or os.path.getmtime(src) > os.path.getmtime(dst)):
        shutil.copy2(src, dst)
        logger.info("synced remote_judge.py -> %s", dst)
    return dst


def _new_run_dir() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(paths.RUNS_DIR, ts)
    os.makedirs(path, exist_ok=True)
    return path


def _gc_old_runs(keep: int = RUNS_KEEP) -> None:
    try:
        entries = sorted(os.listdir(paths.RUNS_DIR))
        for name in entries[:-keep]:
            target = os.path.join(paths.RUNS_DIR, name)
            if os.path.isdir(target):
                shutil.rmtree(target, ignore_errors=True)
    except Exception as e:
        logger.warning("runs gc skipped: %s", e)


def _hf_env_prefix() -> str:
    """Build a `KEY=VAL ...` shell prefix so vLLM on the remote uses the shared
    HF cache (same as bge-small here) — no duplicate model downloads."""
    return " ".join(f"{k}={shlex.quote(v)}" for k, v in paths.hf_env().items())


# ---------- Per-batch dispatch -------------------------------------------

def _wait_for_output(out_path: str, grace: int, poll: int = 15) -> bool:
    """Poll for the remote's output file after an SSH timeout.

    The remote judge holds a flock and writes out.json atomically to the
    shared filesystem, so it keeps running even after the local SSH client
    disconnects. Returns True once the file exists, False if `grace` seconds
    elapse first."""
    deadline = time.time() + max(0, grace)
    while True:
        if os.path.exists(out_path):
            return True
        remaining = deadline - time.time()
        if remaining <= 0:
            return os.path.exists(out_path)
        time.sleep(min(poll, remaining))


def _build_payload(papers: List[Dict], members: List[Dict]) -> Dict:
    return {
        "papers": [{"id": p["id"], "title": p.get("title", ""),
                    "abstract": p.get("abstract", ""),
                    "source": p.get("source", "")} for p in papers],
        "members": [{"name": m["name"], "role": m.get("role", ""),
                     "affiliation": m.get("affiliation", ""),
                     # interests is List[str] — the remote judge accepts either
                     # list or legacy string form via _interest_themes.
                     "interests": m.get("interests") or [],
                     "publications": m.get("publications", [])}
                    for m in members],
    }


def _run_batch(papers: List[Dict], members: List[Dict], gpu_id: int,
               script_path: str, label: str) -> Optional[List[Dict]]:
    """Dispatch one batch to the remote judge. Returns the raw per-paper
    judgment list from out.json, or None if the batch could not be run or
    recovered. `label` is an 'i/n' string used only in log lines."""
    run_dir = _new_run_dir()
    in_path = os.path.join(run_dir, "in.json")
    out_path = os.path.join(run_dir, "out.json")
    err_path = os.path.join(run_dir, "stderr.log")

    with open(in_path, "w") as f:
        json.dump(_build_payload(papers, members), f)

    # flock -w (not -n): if the previous batch's remote is still draining its
    # vLLM teardown it still holds the lock + GPU; wait briefly rather than
    # fail. The lock also serializes GPU use across our own batches.
    cmd = (
        f"{_hf_env_prefix()} "
        f"flock -w 60 {shlex.quote(paths.LOCK_PATH)} "
        f"{shlex.quote(PAPER_CURATOR_REMOTE_PYTHON)} "
        f"{shlex.quote(script_path)} "
        f"--input {shlex.quote(in_path)} "
        f"--output {shlex.quote(out_path)} "
        f"--model {shlex.quote(PAPER_CURATOR_REMOTE_MODEL)} "
        f"--gpu-id {gpu_id} "
        f"--tag-threshold {int(PAPER_CURATOR_TAG_SCORE_THRESHOLD)} "
        f"2> {shlex.quote(err_path)}"
    )
    full = ["ssh", *SSH_OPTS, PAPER_CURATOR_REMOTE_HOST, cmd]

    t0 = time.time()
    timed_out = False
    try:
        subprocess.check_output(full, stderr=subprocess.PIPE,
                                timeout=PAPER_CURATOR_REMOTE_TIMEOUT)
    except subprocess.TimeoutExpired:
        # The SSH client gave up, but the remote keeps running — salvage it.
        timed_out = True
        logger.info("remote judge %s: SSH timed out at %ds — polling shared "
                    "FS up to %ds for the still-running remote's output",
                    label, PAPER_CURATOR_REMOTE_TIMEOUT,
                    PAPER_CURATOR_REMOTE_SALVAGE_GRACE)
    except subprocess.CalledProcessError as e:
        logger.warning("remote judge %s ssh failed (%d); stderr=%s",
                       label, e.returncode,
                       e.stderr.decode("utf-8", errors="replace")[:300])
        return None
    except Exception as e:
        logger.warning("remote judge %s ssh error: %s", label, e)
        return None

    if timed_out and not _wait_for_output(out_path,
                                          PAPER_CURATOR_REMOTE_SALVAGE_GRACE):
        logger.warning("remote judge %s produced no output within the grace "
                       "period; stderr at %s", label, err_path)
        return None
    if not os.path.exists(out_path):
        logger.warning("remote judge %s produced no output at %s; stderr at %s",
                       label, out_path, err_path)
        return None

    try:
        with open(out_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning("remote judge %s: could not read %s: %s",
                       label, out_path, e)
        return None

    judgments = data.get("judgments", [])
    elapsed = data.get("elapsed_sec")
    logger.info("remote judge %s done: %d/%d papers, %.1fs vllm (%.1fs wall)%s",
                label, len(judgments), len(papers),
                elapsed if elapsed is not None else -1,
                time.time() - t0,
                " [salvaged after SSH timeout]" if timed_out else "")
    return judgments


# ---------- Public entrypoint --------------------------------------------

def judge_remotely(papers: List[Dict], members: List[Dict]) -> Optional[List[Dict]]:
    """Returns judgments aligned to `papers`, or None if no batch succeeded.

    Large candidate sets (backlog / recovery days, where a missed stretch of
    runs balloons the set to hundreds of papers and ~10k judge prompts) are
    split into batches of PAPER_CURATOR_REMOTE_BATCH_SIZE papers, each its own
    remote invocation. This keeps every remote call well under REMOTE_TIMEOUT,
    and a batch that still fails is skipped — its papers fall back to the
    'missing from remote output' default — so one bad batch doesn't discard
    the judgments from the others."""
    if not papers:
        return []

    gpu_id = find_free_gpu()
    if gpu_id is None:
        return None

    script_path = _sync_remote_script()

    size = PAPER_CURATOR_REMOTE_BATCH_SIZE or len(papers)
    batches = [papers[i:i + size] for i in range(0, len(papers), size)]
    if len(batches) > 1:
        logger.info("remote judge: %d papers split into %d batches of <=%d",
                    len(papers), len(batches), size)

    judged: List[Dict] = []
    n_ok = 0
    for i, batch in enumerate(batches, 1):
        label = f"batch {i}/{len(batches)}"
        rows = _run_batch(batch, members, gpu_id, script_path, label)
        if rows is None:
            logger.warning("remote judge: %s failed — its %d papers will be "
                           "left unjudged", label, len(batch))
            continue
        judged.extend(rows)
        n_ok += 1

    if n_ok == 0:
        return None
    if n_ok < len(batches):
        logger.warning("remote judge: %d/%d batches succeeded; %d papers "
                       "unjudged", n_ok, len(batches),
                       len(papers) - len(judged))

    by_id = {j["id"]: j for j in judged}
    aligned: List[Dict] = []
    for p in papers:
        j = by_id.get(p["id"])
        if j is None:
            aligned.append({"relevant": False, "score": 0, "tags": [],
                            "one_line_why": "(missing from remote output)"})
        else:
            row = {k: j[k] for k in
                   ("relevant", "score", "tags", "one_line_why")}
            if "per_member" in j:
                row["per_member"] = j["per_member"]
            aligned.append(row)
    _gc_old_runs()
    return aligned
