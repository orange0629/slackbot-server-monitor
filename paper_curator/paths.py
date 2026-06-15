"""Single source of truth for paper_curator on-disk layout.

Everything except `sources.yml` (which is source-controlled config) lives
under `PAPER_CURATOR_DATA_DIR` so this host and burger share state via the
same shared filesystem.
"""
from __future__ import annotations

import os

from config import PAPER_CURATOR_DATA_DIR, PAPER_CURATOR_HF_HOME

ROOT = PAPER_CURATOR_DATA_DIR

BIN_DIR = os.path.join(ROOT, "bin")
RUNS_DIR = os.path.join(ROOT, "runs")
PROFILES_DIR = os.path.join(ROOT, "profiles")
LOGS_DIR = os.path.join(ROOT, "logs")
CONFIG_DIR = os.path.join(ROOT, "config")  # group-editable; members self-serve here
HF_CACHE_DIR = PAPER_CURATOR_HF_HOME  # separate shared volume for HF model weights

LOCK_PATH = os.path.join(ROOT, "lock")
PROFILES_CACHE = os.path.join(PROFILES_DIR, "profiles_cache.json")
PROFILE_EMBEDS = os.path.join(PROFILES_DIR, "profile_embeds.npz")
SLACK_IDS = os.path.join(PROFILES_DIR, "group_slack_ids.json")
CURATOR_LOG = os.path.join(LOGS_DIR, "paper_curator_log.jsonl")
# Last successful arxiv crawl, reused only as a fallback when a later crawl
# fails outright (see sources.fetch_arxiv). Lives at ROOT so both hosts share it.
ARXIV_CACHE = os.path.join(ROOT, "arxiv_cache.json")

# Repo-side config artifacts (not runtime state) — stay in the package.
SOURCES_YML = os.path.join(os.path.dirname(__file__), "data", "sources.yml")

# member_interests.yml lives on the shared volume under a group-editable
# (setgid, mode 2775) config dir so lab members can update their own interests
# without a code change/deploy. It is re-merged on every run (see profiles.py).
MEMBER_INTERESTS_YML = os.path.join(CONFIG_DIR, "member_interests.yml")


def ensure_dirs() -> None:
    """Create the runtime layout under DATA_DIR if missing. Cheap to call repeatedly.
    HF_CACHE_DIR is NOT created here — it lives on a separate shared volume
    that's expected to already exist; we don't want to mkdir on a missing mount."""
    for d in (BIN_DIR, RUNS_DIR, PROFILES_DIR, LOGS_DIR, CONFIG_DIR):
        os.makedirs(d, exist_ok=True)


def hf_env() -> dict:
    """Env vars to point sentence-transformers / huggingface_hub / vLLM at the shared cache.

    `HF_TOKEN_PATH` is redirected away from `$HF_HOME/token` (the default).
    That default lives on the shared /shared/4/models volume as a multi-user
    file, and we've seen huggingface_hub crash reading it with a transient
    PermissionError mid-load (incident 2026-05-25). The model we use
    (bge-small-en-v1.5) is public, so we don't need a token at all —
    pointing at a non-existent file makes `_get_token_from_file` return None
    cleanly (FileNotFoundError is swallowed; PermissionError is not).
    `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` is belt-and-suspenders for the same goal.
    """
    return {
        "HF_HOME": HF_CACHE_DIR,
        "HUGGINGFACE_HUB_CACHE": os.path.join(HF_CACHE_DIR, "hub"),
        "TRANSFORMERS_CACHE": os.path.join(HF_CACHE_DIR, "hub"),
        "SENTENCE_TRANSFORMERS_HOME": os.path.join(HF_CACHE_DIR, "sentence_transformers"),
        "HF_TOKEN_PATH": os.path.join(ROOT, "hf_token_unused"),
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    }


def apply_hf_env() -> None:
    """Set the cache env vars in the current process. Must run BEFORE importing
    sentence-transformers / vllm / transformers — otherwise they snapshot the
    default ~/.cache path."""
    for k, v in hf_env().items():
        os.environ.setdefault(k, v)
