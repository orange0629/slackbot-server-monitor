"""Auto-fill group_slack_ids.json by matching scraped member names against
the workspace user list (uses the bot's existing SLACK_TOKEN).

Match strategy (per member, first hit wins):
  1. exact match on user.real_name
  2. exact match on user.profile.display_name
  3. name_keys() match: same lowercase last name + first-initial — same fuzzy
     matcher used to attribute publications to authors.

Existing IDs in the stub are PRESERVED. Anything still null after matching
is left for you to fill in by hand.

Usage:
    python -m paper_curator.find_slack_ids                # dry-run; prints proposed mapping
    python -m paper_curator.find_slack_ids --write        # write back to group_slack_ids.json
    python -m paper_curator.find_slack_ids --write --overwrite  # also overwrite existing IDs
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Dict, List, Optional

from . import paths, profiles

logger = logging.getLogger(__name__)


def _all_workspace_users(client) -> List[Dict]:
    out: List[Dict] = []
    cursor = None
    while True:
        resp = client.users_list(limit=1000, cursor=cursor)
        for u in resp.get("members", []):
            if u.get("deleted") or u.get("is_bot"):
                continue
            out.append(u)
        cursor = (resp.get("response_metadata") or {}).get("next_cursor")
        if not cursor:
            break
    return out


def _match(member_name: str, users: List[Dict]) -> Optional[Dict]:
    target = member_name.strip().lower()
    # 1. real_name exact
    for u in users:
        rn = (u.get("real_name") or u.get("profile", {}).get("real_name") or "").strip().lower()
        if rn and rn == target:
            return u
    # 2. display_name exact
    for u in users:
        dn = (u.get("profile", {}).get("display_name") or "").strip().lower()
        if dn and dn == target:
            return u
    # 3. fuzzy: same last name + first initial
    m_last, m_first = profiles._name_keys(member_name)
    if not m_last:
        return None
    for u in users:
        rn = u.get("real_name") or u.get("profile", {}).get("real_name") or ""
        u_last, u_first = profiles._name_keys(rn)
        if u_last == m_last and (not u_first or not m_first or u_first == m_first):
            return u
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="persist results to group_slack_ids.json (default: dry run)")
    ap.add_argument("--overwrite", action="store_true",
                    help="also replace IDs that are already set in the stub")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    # Importing main is cheap-ish; we only need its `app` for the SLACK_TOKEN.
    from main import app  # type: ignore

    cache = profiles.refresh_profiles()
    members = profiles.load_ranking_members(cache)
    existing = profiles.load_slack_ids()

    logger.info("fetching workspace users...")
    try:
        users = _all_workspace_users(app.client)
    except Exception as e:
        logger.error("users.list failed: %s — does the bot have users:read scope?", e)
        return 1
    logger.info("workspace has %d non-bot users", len(users))

    proposed: Dict[str, Optional[str]] = dict(existing)
    matched, unmatched, skipped = 0, [], 0
    for m in members:
        name = m["name"]
        cur = existing.get(name)
        if cur and not args.overwrite:
            skipped += 1
            continue
        u = _match(name, users)
        if not u:
            unmatched.append(name)
            proposed.setdefault(name, None)
            continue
        sid = u["id"]
        rn = u.get("real_name") or u.get("profile", {}).get("real_name") or ""
        dn = u.get("profile", {}).get("display_name") or ""
        proposed[name] = sid
        matched += 1
        print(f"  {name:<22} -> {sid}   ({rn} / @{dn})")

    print()
    print(f"matched:   {matched}")
    print(f"skipped:   {skipped} (already set; pass --overwrite to replace)")
    print(f"unmatched: {len(unmatched)}")
    for n in unmatched:
        print(f"  - {n}")

    if args.write:
        with open(paths.SLACK_IDS, "w") as f:
            json.dump(proposed, f, indent=2)
        print(f"\nwrote {paths.SLACK_IDS}")
    else:
        print("\n(dry run; pass --write to persist)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
