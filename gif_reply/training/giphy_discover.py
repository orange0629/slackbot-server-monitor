"""Slow, persistent Giphy discoverer that grows the served candidate pool.

Designed to run inside a screen session for days at a time, paging deeper into
Giphy search results across cycles so we accumulate genuinely-new content
rather than re-hitting the same top-50 every run.

Output:
  - Frame tensors cached under `--cache-dir`, keyed `giphy:<id>` — the same
    scheme `reindex.py` and the stage-1 augmentation dataset both resolve.
  - Metadata appended to `<pool-dir>/index_metadata.jsonl` in the same shape
    as the live served index (gif_id, giphy_id, permalink, alt_text, rating)
    plus `source_query` and `discovered_at` for audit.
  - Per-query offset state in `<pool-dir>/discover_state.json` so each cycle
    moves forward through Giphy's paginated results.

Seed queries come from `/v1/gifs/categories` (top + sub) + `/v1/trending/searches`,
cached on disk and refreshed weekly.

Auto-reindex: once `unflushed_since_reindex` crosses `--reindex-threshold`,
ssh's the Slurm submit host and sbatches `reindex_pool.sbatch` (after checking
no in-flight job with the same name).

CLI:
  python -m gif_reply.training.giphy_discover --loop --interval 3600 \\
      --queries-per-cycle 20 --max-per-cycle 300 --reindex-threshold 2000
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field

from tqdm import tqdm

from .. import giphy_refresh as gr
from .data import DEFAULT_FRAME_CACHE_DIR
from .giphy_frames import _resolve_api_key, fetch_and_cache_one

logger = logging.getLogger(__name__)


DEFAULT_POOL_DIR = "/shared/0/projects/gif-reply-slack-bot/discover_pool"
DEFAULT_LIVE_INDEX_METADATA = "/shared/0/projects/gif-reply-slack-bot/index/index_metadata.jsonl"

# Auto-reindex (append mode) defaults — runs locally on CPU since incremental
# encode of K hundred~thousand gifs takes <5 min and avoids the Slurm queue.
# A from-scratch rebuild (new checkpoint) should still go through
# slurm/reindex_pool.sbatch on GPU.
DEFAULT_REINDEX_CHECKPOINT = "/shared/0/projects/gif-reply-slack-bot/models/pepe_v2/best.pt"
DEFAULT_REINDEX_OUT_DIR = "/shared/0/projects/gif-reply-slack-bot/index_pepe_v2_pool"
DEFAULT_REINDEX_CANDIDATES_PICKLE = "/shared/2/projects/gif-reply/data/processed/dataset/bertweet-normalize/finalized-split-dataset/tweet-gif-reply.pickle"

SEED_QUERY_FILE = "seed_queries.json"
STATE_FILE = "discover_state.json"
POOL_JSONL = "index_metadata.jsonl"

# Giphy paginated endpoints cap offset around 4999.
GIPHY_MAX_OFFSET = 4999
SEED_REFRESH_SECONDS = 7 * 24 * 3600  # weekly


# ---------------- Seed queries ----------------

def _fetch_categories(api_key: str) -> list[str]:
    """Top-level + sub categories from /v1/gifs/categories."""
    data = gr._giphy_get("categories", {"api_key": api_key})
    out: list[str] = []
    for c in data.get("data") or []:
        if name := (c.get("name") or "").strip():
            out.append(name)
        for sc in c.get("subcategories") or []:
            if sn := (sc.get("name") or "").strip():
                out.append(sn)
    return out


def _fetch_trending_searches(api_key: str) -> list[str]:
    """Trending search strings from /v1/trending/searches.

    Lives at a different path prefix than `/v1/gifs/*`, so call requests
    directly rather than via `gr._giphy_get` (which hardcodes `/v1/gifs/`).
    """
    import requests

    try:
        r = requests.get(
            "https://api.giphy.com/v1/trending/searches",
            params={"api_key": api_key},
            timeout=15,
            headers={"User-Agent": gr.USER_AGENT},
        )
    except requests.RequestException as e:
        logger.warning("trending/searches fetch failed: %s", e)
        return []
    if r.status_code != 200:
        return []
    try:
        body = r.json()
    except ValueError:
        return []
    return [s.strip() for s in (body.get("data") or []) if isinstance(s, str) and s.strip()]


def load_or_refresh_seeds(pool_dir: str, api_key: str, force: bool = False) -> list[str]:
    """Build/refresh the seed query list, cached on disk and refreshed weekly."""
    path = os.path.join(pool_dir, SEED_QUERY_FILE)
    if not force and os.path.exists(path):
        if time.time() - os.path.getmtime(path) < SEED_REFRESH_SECONDS:
            with open(path) as f:
                payload = json.load(f)
            return list(payload.get("queries") or [])

    cats = _fetch_categories(api_key)
    trending = _fetch_trending_searches(api_key)
    seen: set[str] = set()
    queries: list[str] = []
    for q in cats + trending:
        k = q.lower()
        if k in seen:
            continue
        seen.add(k)
        queries.append(q)
    payload = {
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_categories": len(cats),
        "n_trending": len(trending),
        "queries": queries,
    }
    os.makedirs(pool_dir, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)
    logger.info("seeds: %d categories + %d trending = %d unique queries",
                len(cats), len(trending), len(queries))
    return queries


# ---------------- Persistent state ----------------

@dataclass
class QueryState:
    next_offset: int = 0
    cycles_empty: int = 0  # consecutive cycles producing 0 new ids
    last_run: str = ""


@dataclass
class DiscoverState:
    queries: dict[str, QueryState] = field(default_factory=dict)
    unflushed_since_reindex: int = 0
    total_added: int = 0
    last_reindex_submitted_at: str = ""
    last_reindex_jobid: str = ""

    @classmethod
    def load(cls, pool_dir: str) -> "DiscoverState":
        path = os.path.join(pool_dir, STATE_FILE)
        if not os.path.exists(path):
            return cls()
        with open(path) as f:
            raw = json.load(f)
        qs = {k: QueryState(**v) for k, v in (raw.get("queries") or {}).items()}
        return cls(
            queries=qs,
            unflushed_since_reindex=raw.get("unflushed_since_reindex", 0),
            total_added=raw.get("total_added", 0),
            last_reindex_submitted_at=raw.get("last_reindex_submitted_at", ""),
            last_reindex_jobid=raw.get("last_reindex_jobid", ""),
        )

    def save(self, pool_dir: str) -> None:
        os.makedirs(pool_dir, exist_ok=True)
        path = os.path.join(pool_dir, STATE_FILE)
        payload = {
            "queries": {k: asdict(v) for k, v in self.queries.items()},
            "unflushed_since_reindex": self.unflushed_since_reindex,
            "total_added": self.total_added,
            "last_reindex_submitted_at": self.last_reindex_submitted_at,
            "last_reindex_jobid": self.last_reindex_jobid,
        }
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)


# ---------------- Pool I/O ----------------

def load_pool_seen_ids(pool_jsonl: str, live_jsonl: str | None) -> set[str]:
    """All giphy_ids we've already accepted (live index + discover pool)."""
    out: set[str] = set()
    for path in (live_jsonl, pool_jsonl):
        if not path or not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                gid = row.get("giphy_id") or (
                    row.get("gif_id", "").split("giphy:", 1)[-1] or None
                )
                if gid:
                    out.add(gid)
    return out


def append_pool(pool_jsonl: str, entries: list[dict]) -> None:
    if not entries:
        return
    os.makedirs(os.path.dirname(pool_jsonl), exist_ok=True)
    with open(pool_jsonl, "a") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


# ---------------- Reindex trigger ----------------

def run_reindex_local(
    *,
    checkpoint: str,
    out_dir: str,
    candidates_pickle: str,
    crawled_jsonls: list[str],
    batch_size: int = 64,
    timeout: int = 7200,
) -> bool:
    """Run reindex.py --append synchronously in a subprocess.

    Runs on whichever device torch picks (CPU on the bot host). Returns True
    on success. The reindex itself is idempotent — if the index is up to date,
    it exits early without rewriting.
    """
    cmd = [
        sys.executable, "-m", "gif_reply.training.reindex",
        "--checkpoint", checkpoint,
        "--out-dir", out_dir,
        "--backend-name", "siglip_ft",
        "--append",
        "--candidates-pickle", candidates_pickle,
        "--batch-size", str(batch_size),
    ]
    for j in crawled_jsonls:
        cmd += ["--crawled-jsonl", j]
    logger.info("starting local CPU reindex (this blocks the crawl loop until done)")
    logger.info("  cmd: %s", " ".join(cmd))
    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.error("reindex timed out after %ds", timeout)
        return False
    elapsed = time.time() - t0
    if result.returncode != 0:
        logger.error("reindex failed (rc=%d) after %.1fs", result.returncode, elapsed)
        for line in result.stderr.strip().splitlines()[-10:]:
            logger.error("  %s", line)
        return False
    # Echo the meaningful tail of reindex's own logs (kept/skipped/wrote lines).
    tail = result.stdout.strip().splitlines()[-6:]
    for line in tail:
        logger.info("  reindex: %s", line)
    logger.info("reindex done in %.1fs", elapsed)
    return True


# ---------------- Crawl ----------------

def pick_queries_this_cycle(seeds: list[str], state: DiscoverState, n: int) -> list[str]:
    """Round-robin biased toward least-recently-advanced queries.

    Sort key: (cycles_empty asc, next_offset asc, name) — never-seen queries
    first (no entry → defaults to (0,0)), then queries that haven't saturated.
    """
    def key(q: str):
        s = state.queries.get(q)
        if s is None:
            return (0, 0, q)
        return (s.cycles_empty, s.next_offset, q)
    return sorted(seeds, key=key)[:n]


def _flush_records(
    records: list[gr.GiphyRecord],
    api_key: str,
    seen: set[str],
    pool_jsonl: str,
    cache_dir: str,
    gif_dir: str,
    state: DiscoverState,
    *,
    workers: int,
    label: str,
) -> dict[str, int | float]:
    """Download frames in parallel for `records`, append accepted ones to the pool.

    Returns a small stats dict for per-source logging:
      {kept, cached, dl_fail, decode_fail, api_fail, elapsed}
    """
    stats = {"kept": 0, "cached": 0, "dl_fail": 0, "decode_fail": 0, "api_fail": 0, "elapsed": 0.0}
    if not records:
        return stats
    t0 = time.time()
    rec_by_id = {r.giphy_id: r for r in records}
    kept: list[dict] = []

    def _job(rec: gr.GiphyRecord) -> tuple[str, str]:
        url = rec.mp4_url or rec.gif_url
        status = fetch_and_cache_one(rec.giphy_id, api_key, gif_dir, cache_dir, url=url)
        return rec.giphy_id, status

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_job, r) for r in records]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=label, leave=False):
            try:
                gid, status = fut.result()
            except Exception:
                stats["dl_fail"] += 1
                continue
            if status == "ok":
                stats["kept"] += 1
            elif status == "cached":
                stats["cached"] += 1
            elif status == "download_fail":
                stats["dl_fail"] += 1
                continue
            elif status == "decode_fail":
                stats["decode_fail"] += 1
                continue
            elif status == "api_fail":
                stats["api_fail"] += 1
                continue
            else:
                continue
            rec = rec_by_id[gid]
            kept.append({
                "gif_id": f"giphy:{gid}",
                "giphy_id": gid,
                "permalink": f"https://giphy.com/gifs/{gid}",
                "alt_text": (rec.title or rec.source_query or "").strip(),
                "rating": rec.rating or "g",
                "source_query": rec.source_query,
                "discovered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })

    append_pool(pool_jsonl, kept)
    for e in kept:
        seen.add(e["giphy_id"])
    state.unflushed_since_reindex += len(kept)
    state.total_added += len(kept)
    stats["elapsed"] = time.time() - t0
    return stats


def crawl_one_cycle(
    api_key: str,
    seeds: list[str],
    state: DiscoverState,
    seen: set[str],
    pool_jsonl: str,
    cache_dir: str,
    gif_dir: str,
    *,
    queries_per_cycle: int,
    per_query: int,
    max_per_cycle: int,
    rating: str,
    workers: int,
    inter_call_sleep: float,
) -> int:
    """One pass over a slice of seeds. Returns count of new gifs added this cycle.

    `inter_call_sleep` is paused between every Giphy API call (trending +
    each search). This is the main throttle for staying under hourly/daily
    quota; the per-cycle interval only controls how *batches* of calls space
    out, not the in-cycle burst rate.
    """
    cycle_added = 0
    queries = pick_queries_this_cycle(seeds, state, queries_per_cycle)
    logger.info("queries this cycle (%d): %s", len(queries), ", ".join(queries))

    # Trending tap — always offset=0, the dedupe set handles redundancy.
    # Trending content rotates frequently enough that fresh hits show up cycle-over-cycle.
    try:
        data = gr._giphy_get("trending", {
            "api_key": api_key, "limit": 50, "offset": 0, "rating": rating,
        })
        items = data.get("data") or []
        recs: list[gr.GiphyRecord] = []
        for it in items:
            rec = gr._record_from_api(it, "trending")
            if rec is None or rec.giphy_id in seen or rec.rating not in gr.ALLOWED_RATINGS:
                continue
            recs.append(rec)
        s = _flush_records(
            recs, api_key, seen, pool_jsonl, cache_dir, gif_dir, state,
            workers=workers, label="trending",
        )
        cycle_added += s["kept"] + s["cached"]
        logger.info(
            "  trending: api=%d items, %d new vs seen → kept=%d cached=%d "
            "(dl_fail=%d decode_fail=%d api_fail=%d) in %.1fs",
            len(items), len(recs), s["kept"], s["cached"],
            s["dl_fail"], s["decode_fail"], s["api_fail"], s["elapsed"],
        )
    except Exception as e:
        logger.warning("trending fetch failed: %s", e)

    for q in queries:
        if max_per_cycle and cycle_added >= max_per_cycle:
            logger.info("  max-per-cycle (%d) hit; remaining queries deferred to next cycle", max_per_cycle)
            break
        qs = state.queries.setdefault(q, QueryState())
        if qs.next_offset > GIPHY_MAX_OFFSET:
            logger.info("  [%s] saturated (offset>%d); skipping", q, GIPHY_MAX_OFFSET)
            continue
        if inter_call_sleep > 0:
            time.sleep(inter_call_sleep)
        try:
            data = gr._giphy_get("search", {
                "api_key": api_key, "q": q, "limit": per_query,
                "offset": qs.next_offset, "rating": rating,
            })
        except Exception as e:
            logger.warning("  [%s] search failed: %s", q, e)
            continue
        items = data.get("data") or []
        recs = []
        for it in items:
            rec = gr._record_from_api(it, q)
            if rec is None or rec.giphy_id in seen or rec.rating not in gr.ALLOWED_RATINGS:
                continue
            recs.append(rec)
        s = _flush_records(
            recs, api_key, seen, pool_jsonl, cache_dir, gif_dir, state,
            workers=workers, label=q,
        )
        added = s["kept"] + s["cached"]
        cycle_added += added
        logger.info(
            "  [%s] offset=%d → %d items, %d new vs seen → kept=%d cached=%d "
            "(dl_fail=%d decode_fail=%d api_fail=%d) in %.1fs",
            q, qs.next_offset, len(items), len(recs),
            s["kept"], s["cached"], s["dl_fail"], s["decode_fail"], s["api_fail"], s["elapsed"],
        )
        qs.next_offset += len(items)
        qs.last_run = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        qs.cycles_empty = qs.cycles_empty + 1 if added == 0 else 0
        if not items:
            # Page came back empty — mark exhausted so future cycles skip.
            qs.next_offset = GIPHY_MAX_OFFSET + 1

    return cycle_added


def main():
    p = argparse.ArgumentParser(
        description="Slow, persistent Giphy discoverer for PEPE-v2 pool reindex.",
    )
    p.add_argument("--pool-dir", default=DEFAULT_POOL_DIR)
    p.add_argument("--live-index-metadata", default=DEFAULT_LIVE_INDEX_METADATA,
                   help="existing served index seeded into the dedupe set")
    p.add_argument("--cache-dir", default=DEFAULT_FRAME_CACHE_DIR)
    p.add_argument("--api-key", default=None)
    p.add_argument("--queries-per-cycle", type=int, default=32,
                   help="seed queries advanced one page per cycle")
    p.add_argument("--per-query", type=int, default=50, help="page size per query (Giphy max 50)")
    p.add_argument("--max-per-cycle", type=int, default=500,
                   help="hard cap on new gifs added per cycle; 0 = unlimited")
    p.add_argument("--rating", default="pg", choices=["g", "pg"])
    p.add_argument("--workers", type=int, default=4, help="parallel CDN download workers (not API-rate-limited)")
    p.add_argument("--inter-call-sleep", type=float, default=30.0,
                   help="seconds to sleep between every Giphy API call within a cycle "
                        "(set to 0 to disable; this is the main quota throttle)")
    p.add_argument("--loop", action="store_true",
                   help="keep cycling forever; Ctrl-C is safe — state flushes each cycle")
    p.add_argument("--interval", type=int, default=3600, help="seconds between cycles in --loop (default 1h)")
    p.add_argument("--reindex-threshold", type=int, default=500,
                   help="auto-run reindex.py --append once this many new gifs accumulate; 0 disables")
    p.add_argument("--reindex-checkpoint", default=DEFAULT_REINDEX_CHECKPOINT)
    p.add_argument("--reindex-out-dir", default=DEFAULT_REINDEX_OUT_DIR)
    p.add_argument("--reindex-candidates-pickle", default=DEFAULT_REINDEX_CANDIDATES_PICKLE)
    p.add_argument("--reindex-batch-size", type=int, default=64)
    p.add_argument("--reindex-timeout", type=int, default=7200,
                   help="seconds to allow the local CPU reindex before timing out")
    p.add_argument("--refresh-seeds", action="store_true",
                   help="force-refresh the seed query cache (otherwise refreshed weekly)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    api_key = _resolve_api_key(args.api_key)
    os.makedirs(args.pool_dir, exist_ok=True)
    pool_jsonl = os.path.join(args.pool_dir, POOL_JSONL)
    gif_dir = os.path.join(args.pool_dir, "raw_gifs")
    os.makedirs(gif_dir, exist_ok=True)

    seeds = load_or_refresh_seeds(args.pool_dir, api_key, force=args.refresh_seeds)
    if not seeds:
        raise SystemExit("no seed queries available — Giphy /categories returned nothing")

    state = DiscoverState.load(args.pool_dir)
    seen = load_pool_seen_ids(pool_jsonl, args.live_index_metadata)
    logger.info("starting: %d seeds, %d already-seen ids, %d total added previously",
                len(seeds), len(seen), state.total_added)

    cycle_num = 0
    run_t0 = time.time()
    try:
        while True:
            cycle_num += 1
            t0 = time.time()
            n_saturated = sum(1 for q in seeds
                              if (qs := state.queries.get(q)) and qs.next_offset > GIPHY_MAX_OFFSET)
            logger.info("==== cycle %d starting (seeds: %d total, %d saturated) ====",
                        cycle_num, len(seeds), n_saturated)
            added = crawl_one_cycle(
                api_key, seeds, state, seen, pool_jsonl,
                args.cache_dir, gif_dir,
                queries_per_cycle=args.queries_per_cycle,
                per_query=args.per_query,
                max_per_cycle=args.max_per_cycle,
                rating=args.rating,
                workers=args.workers,
                inter_call_sleep=args.inter_call_sleep,
            )
            state.save(args.pool_dir)
            elapsed = time.time() - t0
            pool_size = sum(1 for _ in open(pool_jsonl)) if os.path.exists(pool_jsonl) else 0
            run_elapsed_h = (time.time() - run_t0) / 3600
            rate = state.total_added / run_elapsed_h if run_elapsed_h > 0 else 0.0
            logger.info(
                "==== cycle %d done: +%d in %.1fs | pool=%d | unflushed=%d/%d | run total=%d @ %.1f gifs/h ====",
                cycle_num, added, elapsed, pool_size,
                state.unflushed_since_reindex, args.reindex_threshold,
                state.total_added, rate,
            )

            if args.reindex_threshold and state.unflushed_since_reindex >= args.reindex_threshold:
                logger.info("reindex threshold hit (%d unflushed >= %d); running locally on CPU",
                            state.unflushed_since_reindex, args.reindex_threshold)
                ok = run_reindex_local(
                    checkpoint=args.reindex_checkpoint,
                    out_dir=args.reindex_out_dir,
                    candidates_pickle=args.reindex_candidates_pickle,
                    crawled_jsonls=[args.live_index_metadata, pool_jsonl],
                    batch_size=args.reindex_batch_size,
                    timeout=args.reindex_timeout,
                )
                if ok:
                    state.unflushed_since_reindex = 0
                    state.last_reindex_submitted_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    state.last_reindex_jobid = "local"
                    state.save(args.pool_dir)

            if not args.loop:
                break
            logger.info("sleeping %ds until next cycle", args.interval)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        logger.warning("interrupted; state saved at %s", os.path.join(args.pool_dir, STATE_FILE))
        state.save(args.pool_dir)


if __name__ == "__main__":
    main()
