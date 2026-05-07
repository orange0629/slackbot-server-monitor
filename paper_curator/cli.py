"""CLI for paper_curator.

Examples:
    python -m paper_curator.cli --refresh-profiles --print-profiles
    python -m paper_curator.cli --dry-run
    python -m paper_curator.cli                     # wet run; posts to PAPER_CURATOR_CHANNEL
"""
from __future__ import annotations

import argparse
import json
import logging
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-profiles", action="store_true",
                    help="force-refresh the lab profile cache before running")
    ap.add_argument("--print-profiles", action="store_true",
                    help="print active profiles (PI + postdocs + PhD students) and exit")
    ap.add_argument("--print-all-profiles", action="store_true",
                    help="print the FULL scraped cache including alumni / "
                         "undergrads / external collaborators")
    ap.add_argument("--dry-run", action="store_true",
                    help="print main blocks as JSON instead of posting to Slack")
    ap.add_argument("--remote-test", action="store_true",
                    help="just probe the remote GPU host: SSH reachable? free GPU? "
                         "vLLM importable? prints a short report and exits.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.remote_test:
        return _remote_test()

    from . import profiles
    if args.refresh_profiles or args.print_profiles or args.print_all_profiles:
        cache = profiles.refresh_profiles(force=args.refresh_profiles)
        if args.print_all_profiles:
            print(json.dumps(cache, indent=2))
            return 0
        if args.print_profiles:
            filtered = {
                "scraped_at": cache.get("scraped_at"),
                "lab_url": cache.get("lab_url"),
                "members": profiles.load_ranking_members(cache),
            }
            print(json.dumps(filtered, indent=2))
            return 0

    from .curator import run_curation
    ok = run_curation(dry_run=args.dry_run)
    return 0 if ok else 1


def _remote_test() -> int:
    """Probe burger: SSH reach, free GPU pick, vLLM importable on the remote."""
    from . import remote_dispatch as rd
    from config import (PAPER_CURATOR_REMOTE_HOST, PAPER_CURATOR_REMOTE_MODEL,
                        PAPER_CURATOR_REMOTE_PYTHON)
    print(f"host:   {PAPER_CURATOR_REMOTE_HOST}")
    print(f"python: {PAPER_CURATOR_REMOTE_PYTHON}")
    print(f"model:  {PAPER_CURATOR_REMOTE_MODEL}")
    out = rd._ssh("echo SSH_OK && hostname && uname -r", timeout=15)
    if out is None:
        print("FAIL: SSH unreachable (check passwordless setup for the bot user)")
        return 1
    print("SSH OK:", out.strip().replace("\n", " | "))
    gpu = rd.find_free_gpu()
    print(f"free GPU pick: {gpu}")
    py_check = rd._ssh(
        f"{PAPER_CURATOR_REMOTE_PYTHON} -c 'import vllm, sys; "
        f"print(\"vllm\", vllm.__version__, \"py\", sys.version.split()[0])'",
        timeout=20)
    if py_check is None:
        print("FAIL: vLLM not importable with PAPER_CURATOR_REMOTE_PYTHON; "
              "install via `pip install vllm` in that env.")
        return 1
    print("remote env:", py_check.strip())
    return 0 if gpu is not None else 1


if __name__ == "__main__":
    sys.exit(main())
