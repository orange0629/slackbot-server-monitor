# Lab Server Slack Bot

A Slack bot that watches the lab's GPU servers, SLURM cluster, and shared
`/home` partition, and lets users self-serve common questions ("which GPU is
free?", "how is my disk usage?", "what's running on SLURM?") from Slack.

The bot DMs you when something needs your attention, and exposes a `/food-bot`
slash command for on-demand checks.

---

## For end users

### The slash command

Type `/food-bot` anywhere in Slack to get a private menu of buttons. Nothing
you click is visible to other people — every response is ephemeral.

Buttons available to everyone:

| Button | What it does |
|---|---|
| **GPU Usage** | Pick a server (or *all_servers*) and get a per-GPU breakdown: VRAM used / total, utilization %, and which user / PID is on each GPU. Picking a single server also runs a quick diagnostic check for underutilized GPUs on that box. |
| **Find Free GPU** | Ranks the top 10 GPUs across all servers by absolute free VRAM. Use this when you just want a place to start a job. |
| **Slurm Status** | Shows the live `squeue` for the cluster, plus per-user job counts grouped by state (RUNNING / PENDING / etc.). Wait/run times are tracked from the moment the bot first saw the job. |
| **Slurm Job History** | Then choose **My Slurm History** or **All Slurm History** to see recently-finished jobs (job id, node, GPUs, CPUs, memory, wait time, run time, end time). |
| **My Home Usage** | Shows `df -h /home` plus *your* most recent `/home` footprint in GB, taken from the latest nightly disk scan. |

Admin-only buttons (only shown to users in `ADMIN_USERS`):

| Button | What it does |
|---|---|
| **All Home Usages** | Like *My Home Usage* but lists every user, sorted largest first. |
| **Slurm Usage Report** | Drop-down for *Last 7 Days / This Month / Last Month / This Year / All Time* — totals jobs, wait hours, run hours, and GPU-hours per user. |
| **Bot Running Logs** | Tail of `monitor.log`. |
| **Scanning Schedules** | Status of every scheduled task: schedule, next run, last run, whether it's enabled. |

### Direct messages you might receive from the bot

You'll only ever receive DMs about *your own* activity. Specifically:

- **New SLURM job detected.** When the bot first sees a job under your
  username, it DMs you once with a *Don't track* button. Click it to silence
  notifications about that one job. The follow-up message also offers *Mute
  for 24h* if you'd rather silence all SLURM notifications for a day.
- **SLURM job state changes & completion.** If you didn't opt out, you'll get
  a DM when the job starts running and another when it ends (with the run
  duration).
- **Underutilized GPU.** If a GPU is holding a lot of VRAM (over 50% by
  default) but barely doing any compute (under 8% utilization) for two
  consecutive checks, the top user gets DMed. After an alert you're on a 12h
  cooldown so you won't get re-pinged repeatedly.
- **GPU not in your SLURM allocation.** On SLURM-managed nodes, if one of
  your processes is running on a GPU that wasn't allocated to your job, the
  bot DMs you with the mismatch. 12h cooldown applies.
- **Home directory over the threshold.** The nightly disk scan DMs you if
  your `/home` usage exceeds the threshold (default 80 GB).
- **Partition full.** If `/home` itself crosses the partition threshold
  (default 90%), admins are alerted (with a 6h cooldown).

### Daily paper digest

Every weekday at 09:00 the bot posts a curated literature scan to
`#interesting-papers` — a bulleted list of ~10 fresh papers from arXiv +
preprint servers + top journals + OpenReview + HuggingFace Daily, ranked
against scraped lab-member research interests. If a paper looks aimed at
*you* you'll be `@mentioned` after the title with a brief reason.

Things to know:

- Mentions are **advisory**, not assignments — they're the bot's guess at
  who'd find each paper most relevant. Each member is capped at 2 mentions
  per post.
- Overflow papers (anything not in the main 10) appear as a thread reply
  under the digest, with no `@mentions`.
- The full paper list, source registry, member interests, and tuning knobs
  are documented in `paper_curator/README.md`.

### Monthly report

On the 1st of each month the bot posts a summary to the configured channel:
last month's per-user job/GPU-hour totals, a "top GPU user" callout, and
totals across the cluster. On January 1st there's also a year-in-review.

### GIF replies

If the bot has GIF reply enabled in your channel, `@mention` it with any text
and it will reply in-thread with a contextual GIF. Examples:

> `@food-bot what a great day`
> `@food-bot the build is broken again`

A few things to know:

- Replies are **threaded** under your message and visible to the channel.
- The bot remembers the last 50 GIFs it has used in the channel and avoids
  repeating them.
- Per-user rate limit (default 5/hour) and per-channel rate limit (default
  20/hour). Going over is silently dropped.
- **Mute it for yourself.** `/food-bot` → *Mute GIF Reply (24h)* — the bot
  won't reply to your `@mention`s for a day. Same shape as the SLURM/GPU mutes.
- **Toggle it for a channel** (admins only). `/food-bot` from inside the
  channel → *Toggle GIF Reply Here*. The override is sticky and survives
  restarts.
- Off-topic / NSFW queries and GIFs rated above PG are filtered out before
  posting; if nothing safe scores high enough, the bot just stays quiet.

### Tips

- The bot assumes your **Slack username matches your Linux username**. If
  they differ, the bot can't match your DMs to your processes.
- Buttons that fan out across servers ("all_servers", "Find Free GPU") take a
  few seconds — you'll see a *Loading…* message first.
- Notifications you mute stay muted for the configured window (24h for SLURM,
  12h for GPU alerts) and then resume automatically.

---

## For admins / operators

### Deployment

```bash
pip install slack_bolt slack_sdk filelock tqdm
cp .env.example .env       # then fill in SLACK_TOKEN and SLACK_APP_TOKEN
python main.py
```

### Secrets / environment

All secrets live in a single **`.env`** file at the repo root (gitignored).
`config_secret.py` parses it on import — no `python-dotenv` dependency. Real
process-environment variables take precedence over `.env`, so the same code
works under systemd / Docker / a shell where you `export` the keys directly.

| Key | Required | What it's for |
|---|---|---|
| `SLACK_TOKEN` | yes | Slack bot OAuth token (`xoxb-…`). |
| `SLACK_APP_TOKEN` | yes | Slack app-level token for Socket Mode (`xapp-…`). |
| `GIPHY_API_KEY` | no | Enables the `giphy_refresh` scheduled task. Without it the task no-ops. |
| `HF_TOKEN` | no | Only needed if `GIF_REPLY_SIGLIP_MODEL` is changed to a gated HuggingFace checkpoint. |

`.env.example` is committed as a template. Copy it to `.env` and fill in the
values. The bot will refuse to start if `SLACK_TOKEN` or `SLACK_APP_TOKEN`
isn't set.

The process forks into two workers (Slack handler + scheduler) and runs in
the foreground. Run it under your supervisor of choice (`tmux`, `systemd`, …).

The host running the bot needs **passwordless SSH** to every server in
`AVAILABLE_SERVERS` and to `SLURM_SERVER`. All `nvidia-smi` / `squeue` /
`scontrol` calls go over SSH from this single host.

### Configuration (`config.py`)

Most settings are self-explanatory toggles. Common things you'll want to
tune:

- `SLACK_CHANNEL` — channel for monthly reports and admin alerts.
- `ADMIN_USERS` / `EXCLUDED_USERS` — who sees admin buttons; whose `/home`
  usage is ignored by the disk-scan alerter.
- `AVAILABLE_SERVERS` — every host the GPU checks SSH into.
- `SLURM_SERVER` — the host where `squeue` / `scontrol` are run.
- `SLURM_SUPPORTED_SERVERS` — subset of `AVAILABLE_SERVERS` whose GPU
  allocations are managed by SLURM. Only these get the GPU-allocation
  mismatch check.
- `SCAN_METHOD` — `"DU"` (default), `"NCDU"`, or `"FIND"`. See
  `disk_scan.py`.
- Thresholds: `USER_THRESHOLD_GB`, `PARTITION_USAGE_THRESHOLD`,
  `GPU_VRAM_THRESHOLD_PERCENT`, `GPU_UTILIZATION_THRESHOLD_PERCENT`.

### Files written at runtime

| File | Contents |
|---|---|
| `monitor.log` | Rotating log (5 × 5 MB). |
| `usage_log.jsonl` | Append-only `/home` scan snapshots. |
| `slurm_usage_log.jsonl` | One record per finished SLURM job; powers history & reports. |
| `bot_state.json` | Shared state between the two processes (tracked SLURM jobs, GPU two-strike flags, user mute expirations). |
| `scheduler_state.json` | `next_time` / `last_run_time` per scheduled task so restarts don't re-fire everything. |
| `ncdu_cache.json` | Only when `SCAN_METHOD = "NCDU"`. |

### Scheduled tasks

| Task | Cadence | Effect |
|---|---|---|
| `disk_scan` | daily at 04:00 | Scans `/home`, DMs users over the threshold. |
| `gpu_check` | every 30 min | Two-strike underutilized-GPU alerts + GPU-vs-SLURM allocation check. |
| `partition_check` | hourly (6h cooldown after an alert) | Alerts admins when `/home` is too full. |
| `slurm_poll` | every 45 s | New / state-change / completion DMs to job owners. |
| `monthly_slurm_report` | daily at 09:00, sends only on the 1st | Posts the monthly summary to `SLACK_CHANNEL`. |
| `paper_curator` | weekdays at 09:00 | Pulls fresh papers, ranks them against lab-member interests, and posts the daily digest to `PAPER_CURATOR_CHANNEL`. See `paper_curator/README.md`. |
| `giphy_refresh` | every 24h (when `ENABLE_GIF_REPLY` and `GIPHY_API_KEY` are set) | Pulls new GIFs from Giphy and appends them to the on-disk index. No-ops without an API key. |

Disable any of them by flipping the corresponding `ENABLE_*` flag in
`config.py`.

---

## GIF reply

The bot can post a contextual GIF as a threaded reply when it's `@mention`ed
in an allowlisted channel. The retrieval engine is **CPU-only and inline** —
no extra service to run. A modest amount of one-time setup is required.

### How it works

1. A candidate set of GIFs is embedded once, offline, into a numpy matrix on
   disk (`{backend}_embeddings.npy` + `index_metadata.jsonl`). The release
   data on `/shared/2/projects/gif-reply` provides ~hundreds of thousands of
   candidates plus precomputed PEPE features.
2. At Slack-message time, the bot encodes the user's text with a frozen
   query-side text encoder (default: `google/siglip-base-patch16-224`) and
   takes the cosine top-k.
3. Safety filters drop NSFW / banned candidates; rate limits and the
   per-channel "recent gifs" cache prevent repetition and spam.
4. The bot posts a Slack image block whose URL is the Giphy direct media URL
   (`https://media.giphy.com/media/{giphy_id}/giphy.gif`), threaded under
   the triggering message.

### One-time setup

```bash
# 1. Install the extra dependencies (CPU-only torch is fine).
pip install torch transformers pillow numpy pytest

# 2. Build the candidate index. Two backends are supported:

# (a) Frozen SigLIP — recommended default. Embeds the released GIFs with
#     a current multimodal model. Run on a node with reasonable RAM; the
#     image embedding loop is the slow part.
python -m gif_reply.build_index --backend siglip

# (b) PEPE — uses precomputed features from the original release, so no
#     model needs to load. Fastest path to a PEPE index.
python -m gif_reply.build_index --backend pepe --use-precomputed
```

Both backends write into `GIF_REPLY_INDEX_DIR`
(default: `/shared/0/projects/gif-reply-slack-bot/index/`). The HuggingFace
model cache is stored under `/shared/0/projects/gif-reply-slack-bot/hf_cache/`
so the SigLIP weights aren't re-downloaded on every host.

### Turning it on

In `config.py`:

```python
ENABLE_GIF_REPLY = True
GIF_REPLY_BACKEND = "siglip"            # or "pepe"
GIF_REPLY_CHANNELS = ["C0123456789"]    # static channel ID allowlist
```

Restart the bot. From inside an allowlisted channel, `@mention` the bot with
any text and it will post a GIF in thread. Admins can also toggle individual
channels at runtime via `/food-bot` → *Toggle GIF Reply Here*; runtime
overrides are persisted in `bot_state.json` under the new `gif_reply` key
and merge on top of the static `GIF_REPLY_CHANNELS` list.

### Configuration knobs (`config.py`)

| Setting | Default | What it does |
|---|---|---|
| `ENABLE_GIF_REPLY` | `False` | Master kill switch. |
| `GIF_REPLY_BACKEND` | `"siglip"` | `"siglip"` or `"pepe"`. |
| `GIF_REPLY_CHANNELS` | `[]` | Static channel-ID allowlist. |
| `GIF_REPLY_RATE_LIMIT_PER_USER_HOUR` | `5` | Per-user cap; further mentions are silently dropped. |
| `GIF_REPLY_RATE_LIMIT_PER_CHANNEL_HOUR` | `20` | Per-channel cap. |
| `GIF_REPLY_RECENT_HISTORY` | `50` | Recent-gif memory per channel — these are excluded from retrieval. |
| `GIF_REPLY_INDEX_DIR` | `/shared/0/projects/gif-reply-slack-bot/index` | Where the embedding `.npy` and metadata jsonl live. |
| `GIF_REPLY_DATA_DIR` | `/shared/0/projects/gif-reply-slack-bot` | Parent dir; contains `hf_cache/`, `giphy_downloads/`, and the safety word lists. |
| `GIF_REPLY_SIGLIP_MODEL` | `google/siglip-base-patch16-224` | HF model id for the SigLIP backend. |
| `GIF_REPLY_PEPE_CHECKPOINT` | `/shared/2/projects/gif-reply/data/release/PEPE-model-checkpoint.pth` | Path to the PEPE weights. |
| `GIF_REPLY_GIPHY_REFRESH_HOURS` | `24` | Cadence of the background Giphy refresh task. |

### Safety

Two filter sources, both optional files under `GIF_REPLY_DATA_DIR`:

- `offensive-words.txt` — one banned word per line. Matched as whole words
  (so `ass` won't hit `class`) against both the user's query and a
  candidate's alt text. Reuse the file from
  `/shared/2/projects/gif-reply/data/processed/offensive-words.txt` to start.
- `banned-giphy-gifs.txt` — one Giphy id per line. Drops specific GIFs
  regardless of score.

In addition, only GIFs rated `g` or `pg` by Giphy are considered. If
nothing safe scores high enough, the bot stays quiet.

### Mutes & rate limits at runtime

Both live in `bot_state.json`, written by the same merge-write protocol the
SLURM/GPU code uses, so they survive restarts and don't get clobbered between
processes:

- `user_mutes.gif_reply` — username → expiry ISO timestamp. Set by the
  *Mute GIF Reply (24h)* button on `/food-bot`.
- `gif_reply.user_calls` / `gif_reply.channel_calls` — rolling 1-hour
  windows of call timestamps used by the rate limiter.
- `gif_reply.channel_overrides` — channel id → `True`/`False`, set by the
  admin-only *Toggle GIF Reply Here* button.
- `gif_reply.recent_per_channel` — channel id → list of recently-served
  `gif_id`s, capped at `GIF_REPLY_RECENT_HISTORY`.

### Building or refreshing the index from Giphy

`gif_reply.giphy_refresh` is the full Giphy → embedding → on-disk-index
pipeline. It runs both as the scheduled background task (`giphy_refresh` in
`scheduled_tasks`) and as a standalone CLI for first-time bulk builds.

**First-time build (recommended path):**

```bash
# Set GIPHY_API_KEY in .env, then:
python -m gif_reply.giphy_refresh \
    --backend siglip \
    --index-dir /shared/0/projects/gif-reply-slack-bot/index \
    --trending 200 \
    --per-query 50
```

This pulls the Giphy `/trending` feed plus 50 GIFs for each of the ~60
default queries (mood/reaction-shaped — see `DEFAULT_QUERIES` in the
module), filters to `g`/`pg` rating, downloads the small fixed-height
preview, extracts the first frame, embeds with SigLIP, and writes
`siglip_embeddings.npy` + `index_metadata.jsonl`. Expect a few thousand
GIFs after dedup and rating filter.

**CLI flags:**

| Flag | Purpose |
|---|---|
| `--backend {siglip,pepe}` | Encoder backend. Index files are backend-keyed. |
| `--rebuild` | Wipe the existing `{backend}_embeddings.npy` + `index_metadata.jsonl` before fetching. |
| `--queries q1 q2 ...` | Override the default query list inline. |
| `--queries-file path.txt` | One query per line. Overrides defaults if set. |
| `--per-query N` / `--trending N` | Cap how many GIFs to pull per query / from trending. |
| `--max-new N` | Cap *total* new GIFs added in this run. |
| `--rating {g,pg}` | Strictest rating to allow through Giphy's filter. |
| `--api-key …` | Overrides the `.env` lookup, e.g. for ad-hoc runs. |

**Idempotency:** every run dedups against the existing
`index_metadata.jsonl` by `giphy_id`, so re-running is safe and cheap. The
scheduled task uses this same path with the default flags.

**Failure modes:** download / decode failures for individual GIFs are
logged at WARNING and the GIF is skipped — the rest of the batch still
makes it into the index. Atomic writes keep the index consistent across
crashes.

### Tests

```bash
python -m pytest tests/
```

Index, safety, and rate-limit logic run on every checkout. Encoder smoke
tests are skipped automatically when `torch` / `transformers` aren't
installed, and the PEPE smoke test additionally skips when the model class
can't be imported (the original training tree depends on a project-local
`opt` config — see the next section).

### Switching backends or scaling later

- **SigLIP variants.** Point `GIF_REPLY_SIGLIP_MODEL` at a larger checkpoint
  (e.g. `google/siglip-large-patch16-384`) and rerun `build_index`. Index
  files are backend-keyed, so you can keep both around.
- **Full PEPE inference.** The training code at
  `/shared/2/projects/gif-reply/src/models/CLIP-variant-multitask/` requires
  a project-local `opt` config to import the model class, so we don't load
  it at runtime today. The cleanest path is to vendor a minimum
  `gif_reply/pepe_model.py` lifted from
  `/shared/2/projects/gif-reply/src/application/pepe-deploy/retrieval.py`
  (which was the actual production inference code) and load
  `OscarCLIPModel-epoch-12.pth` or `CLIPModel-epoch-6.pth` from
  `application/deploy/data/` rather than the full
  `PEPE-model-checkpoint.pth`. Until that's done, the PEPE backend is best
  used with `--use-precomputed` to build the GIF-side index from the
  released features.

---

## Paper monitor

Watches a Slack channel (default `#interesting-papers`) for paper links,
resolves each to a citation using **both** OpenAlex and Semantic Scholar
(cross-checked), and appends a row to `papers_log.jsonl`. The output is a
running reading list you can later read with pandas and publish on a website.

### What gets resolved

Best-effort, in this order:

1. **arXiv / DOI URLs** — looked up directly in OpenAlex and Semantic Scholar.
2. **Other landing pages (publishers, blogs)** — the page is fetched and
   parsed for `citation_doi` / `citation_arxiv_id` / Dublin Core meta tags;
   any DOI / arXiv ID found is then looked up in both APIs.
3. **PDFs** — downloaded (capped at `PAPERS_PDF_MAX_BYTES`, default 25 MB)
   and the first two pages are scanned for a DOI or arXiv ID.
4. **Title-only fallback** — if no ID is recoverable, the page `<title>` (or
   PDF first-line title) is used to search OpenAlex / Semantic Scholar.
5. **Give up gracefully** — anything still unresolved is written with
   `status: "unresolved"` and a `notes` field so you can fix it manually.

Cross-check: when both APIs return, titles are normalized and compared.
Mismatches are kept (not dropped) but flagged in `notes` as
`title_mismatch`.

### Storage schema (`papers_log.jsonl`)

One JSON object per line. Dedup is **first-mention-wins** keyed on (in
priority order) `doi`, `arxiv_id`, `openalex_id`, `s2_id`, then canonical
URL. Repeats — including the same paper posted in a thread later — are
silently skipped.

```json
{
  "key": "doi:10.1145/3442188.3445922",
  "citation": "Bender et al. (2021). On the Dangers of Stochastic Parrots ...",
  "bibtex": "@article{bender2021dangers, ...}",
  "paper_url": "https://doi.org/10.1145/3442188.3445922",
  "source_url": "https://dl.acm.org/doi/10.1145/3442188.3445922",
  "title": "...", "authors": ["..."], "year": 2021, "venue": "FAccT",
  "doi": "10.1145/3442188.3445922", "arxiv_id": null,
  "openalex_id": "W3134614001", "s2_id": "0...",
  "posted_at": "2024-08-12T15:42:11+00:00",
  "posted_by": "U01ABC...", "channel": "C09XYZ...", "slack_ts": "1723477331.123456",
  "resolved_via": ["openalex", "semantic_scholar"],
  "status": "resolved", "notes": []
}
```

Read it with `pandas.read_json("papers_log.jsonl", lines=True)`.

### Slack admin checklist (one-time)

The bot already replies in `#interesting-papers` for GIF replies, so most of
this is likely already done — but verify each item before relying on the
monitor:

- [ ] **Bot scope** `channels:history` is enabled on the Slack app
      (`api.slack.com/apps` → your app → *OAuth & Permissions* → *Bot Token Scopes*).
- [ ] **Event subscription** `message.channels` is enabled
      (*Event Subscriptions* → *Subscribe to bot events*).
- [ ] **Channel membership** the bot is a member of `#interesting-papers`
      (run `/invite @food-bot` from inside the channel if not).
- [ ] **App reinstalled** to the workspace if you added scopes above
      (the OAuth page will show a yellow "reinstall" banner if needed).
- [ ] **Restart the bot** so `ENABLE_PAPER_MONITOR=True` takes effect.

A quick post-deploy smoke test: paste an arXiv link into
`#interesting-papers`, then check that a new line appears in
`papers_log.jsonl` (or look for `paper_monitor: stored ...` in
`monitor.log`).

### Configuration (`config.py`)

| Setting | Default | What it does |
|---|---|---|
| `ENABLE_PAPER_MONITOR` | `True` | Master kill switch. |
| `PAPERS_CHANNEL` | `"interesting-papers"` | Channel name (no `#`) or channel ID. |
| `PAPERS_LOG_FILE` | `"papers_log.jsonl"` | Where rows are appended. |
| `PAPERS_BACKFILL_CUTOFF` | `"2024-01-01"` | Backfill stops at this date (UTC). |
| `PAPERS_HTTP_TIMEOUT` | `15` | Seconds for OpenAlex / S2 / page fetches. |
| `PAPERS_PDF_MAX_BYTES` | `25 MB` | Skip PDFs larger than this. |

### Backfilling (offline)

Walks the channel's history through the cutoff and runs every URL through
the same pipeline as the live listener. Safe to re-run: the dedup key keeps
repeated runs idempotent.

```bash
pip install pypdf                                  # for PDF parsing
python -m paper_monitor.backfill                   # uses PAPERS_CHANNEL + cutoff
python -m paper_monitor.backfill --dry-run         # resolve & log but don't write
python -m paper_monitor.backfill --since 2025-01-01
python -m paper_monitor.backfill --channel C0123456789
```

The backfill caches existing keys in memory for the run, so it stays fast
even with a large `papers_log.jsonl`. It's runnable manually only — it does
not run on the scheduler.

---

## Paper curator

Daily agentic literature scan. Pulls fresh papers from arXiv + OSF + journal
RSS + OpenReview + HuggingFace Daily, ranks them with a CPU bi-encoder
against scraped lab-member interests, judges the top 30 with an LLM (vLLM
on a remote GPU box, with a local Ollama fallback), and posts a curated
bulleted digest to `#interesting-papers` Mon–Fri at 09:00.

```
:newspaper: Papers for Wed, May 6 — 10 picks
• *<url|Where Paths Split: Localized Control of Moral Reasoning>* @shivani @nancy — moral reasoning calibration in LLMs
• *<url|SHIELD: Distilled SLMs for Clinical De-id>* @kenan — clinical de-identification, distilled SLMs
…
```

### Architecture (high level)

1. `sources.fetch_all()` — ~1500–2500 candidate papers/day across ~20 sources.
2. Dedup against `BOT_STATE.arxiv_papers.seen` (last 30 days).
3. Bi-encoder embeds papers + member interests; **each themed line in
   `member_interests.yml` becomes its own query vector**, so a paper
   strongly matching one theme keeps its full cosine score.
4. Top 30 go to the LLM judge (remote vLLM → local Ollama → bi-encoder-only
   fallback chain).
5. Sort, apply per-member tag cap, format as bulleted block-kit, post.
6. Append paper IDs to `seen`; append per-paper rows to
   `paper_curator_log.jsonl`.

### Configuration knobs (`config.py`)

| Setting | Default | What it does |
|---|---|---|
| `ENABLE_PAPER_CURATOR` | `True` | Master kill switch. |
| `PAPER_CURATOR_CHANNEL` | `"interesting-papers"` | Destination channel name or ID. |
| `PAPER_CURATOR_POST_TIME` | `"09:00"` | Local server time. |
| `PAPER_CURATOR_WEEKDAYS` | `[0,1,2,3,4]` | `datetime.weekday()` ints; default Mon–Fri. |
| `PAPER_CURATOR_TOP_K_TO_LLM` | `30` | Bi-encoder shortlist size sent to the LLM. |
| `PAPER_CURATOR_MAX_MAIN_POST` | `10` | Picks in the main post; the rest go in a thread reply. |
| `PAPER_CURATOR_MAX_TAGS_PER_MEMBER` | `2` | Per-member `@mention` cap per post. |
| `PAPER_CURATOR_BIENCODER` | `"BAAI/bge-small-en-v1.5"` | Sentence-Transformers ranking model. |
| `PAPER_CURATOR_DRY_RUN` | `False` | True = scheduled fires print blocks instead of posting. |
| `PAPER_CURATOR_USE_REMOTE` | `True` | Use remote vLLM first, fall back to Ollama. |
| `PAPER_CURATOR_REMOTE_HOST` | `"burger.si.umich.edu"` | SSH host with vLLM + a free GPU. |
| `PAPER_CURATOR_REMOTE_PYTHON` | `"/opt/anaconda/bin/python"` | Interpreter on the remote with vllm installed. |
| `PAPER_CURATOR_REMOTE_MODEL` | `"Qwen/Qwen3.5-4B"` | HF model id for the remote judge. |
| `PAPER_CURATOR_OLLAMA_MODEL` | `"qwen3.6:35b-a3b"` | Local fallback primary. |
| `PAPER_CURATOR_OLLAMA_FALLBACK` | `"gemma4:26b"` | Local fallback's fallback. |
| `PAPER_CURATOR_DATA_DIR` | `"/shared/6/projects/food-bot"` | Runtime artifacts (must be visible from both this host and the remote). |
| `PAPER_CURATOR_HF_HOME` | `"/shared/4/models"` | Shared HF cache for vLLM + bge-small. |

Full configuration reference, including all GPU-pick thresholds and
profile-cache flags, is in `paper_curator/README.md`.

### Files you'll edit

| File | Purpose |
|---|---|
| `paper_curator/data/sources.yml` | Source registry. Add/remove journals + arXiv categories + OpenReview venues here. Per-source `enabled: true/false` flag. |
| `paper_curator/data/member_interests.yml` | Hand-curated research focuses, keyed by exact scraped name. Each line becomes one bi-encoder query vector. |
| `<DATA_DIR>/profiles/group_slack_ids.json` | Member name → Slack `U…` ID. Auto-fillable via `python -m paper_curator.find_slack_ids --write`. |

### One-time setup

- [ ] Bot is in `PAPER_CURATOR_CHANNEL` (`/invite @<bot>`).
- [ ] Passwordless SSH from this host to `PAPER_CURATOR_REMOTE_HOST`.
- [ ] `vllm` importable with `PAPER_CURATOR_REMOTE_PYTHON` on the remote.
- [ ] `<DATA_DIR>` mounted at the same path on both hosts.
- [ ] `<HF_HOME>` mounted on the remote (and ideally locally too).
- [ ] `python -m paper_curator.find_slack_ids --write` to populate the
      `name → Slack ID` mapping.
- [ ] `python -m paper_curator.cli --remote-test` to verify SSH + GPU pick
      + vLLM import.
- [ ] `python -m paper_curator.cli --dry-run` for a full end-to-end check
      (prints blocks to stdout instead of posting).

### Manual triggers

- **Slack admin button** — `/food-bot` → *Run Paper Curator (Admin Only)*.
  DMs the digest to the clicking admin (no public channel post). Safe to
  spam-test.
- **CLI** — `python -m paper_curator.cli` (full options below):

  ```bash
  python -m paper_curator.cli --refresh-profiles --print-profiles  # show active members
  python -m paper_curator.cli --dry-run                            # full pipeline, print only
  python -m paper_curator.cli --remote-test                        # probe burger
  python -m paper_curator.cli                                      # live run
  ```

### Files written at runtime

| Path | Contents |
|---|---|
| `<DATA_DIR>/profiles/profiles_cache.json` | Last lab-site scrape + merged interests. |
| `<DATA_DIR>/profiles/profile_embeds.npz` | Cached bi-encoder vectors (keyed on cache mtime). |
| `<DATA_DIR>/profiles/group_slack_ids.json` | Member → Slack ID map. |
| `<DATA_DIR>/runs/<UTC-ts>/{in,out}.json` | Per-run remote-judge payloads (last 50 kept). |
| `<DATA_DIR>/logs/paper_curator_log.jsonl` | One row per posted paper (id, title, source, tags, score). |
| `<DATA_DIR>/bin/remote_judge.py` | Synced from the package on each run. |
| `bot_state.json` → `arxiv_papers` | `{seen: {id: date}, last_run: iso}` — drives 30-day dedup. |

### Where to look first when something breaks

- `monitor.log` — every run prints `paper_curator: 686 fresh papers after
  dedup`, `remote GPU 2 selected`, `remote judge done: 30/30 papers …`.
- `<DATA_DIR>/runs/<latest>/stderr.log` — remote vLLM stderr if the GPU
  job crashed.
- `paper_curator_log.jsonl` — auditable history of what got posted and
  who got tagged.

See `paper_curator/README.md` for the full troubleshooting guide.

