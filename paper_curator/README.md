# Paper Curator

Daily agentic literature scan for the Blablablab. Every weekday at 09:00 the
bot pulls fresh papers from arXiv + OSF preprints + journal RSS + OpenReview +
HuggingFace Daily, ranks them against scraped lab-member interests with a
bi-encoder, judges the top 30 with an LLM (vLLM on a GPU box, with a local
Ollama fallback), and posts a curated bulleted digest to `#interesting-papers`
with @-mentions for any member the picks fit best.

```
:newspaper: Papers for Wed, May 6 — 10 picks
• *<url|Where Paths Split: Localized Control of Moral Reasoning>* @shivani @nancy — moral reasoning calibration in LLMs
• *<url|SHIELD: Distilled SLMs for Clinical De-id>* @kenan — clinical de-identification, distilled SLMs
• *<url|Some broadly relevant paper>*
…
```

---

## Contents

- [Quick start](#quick-start)
- [Pipeline](#pipeline)
- [Configuration reference](#configuration-reference)
- [Source registry — `data/sources.yml`](#source-registry--datasourcesyml)
- [Member interests — `data/member_interests.yml`](#member-interests--datamember_interestsyml)
- [Slack ID mapping](#slack-id-mapping)
- [Manual triggers](#manual-triggers)
- [Remote vLLM dispatch](#remote-vllm-dispatch)
- [File layout](#file-layout)
- [Logs & observability](#logs--observability)
- [Tuning recipes](#tuning-recipes)
- [Troubleshooting](#troubleshooting)

---

## Quick start

1. **Confirm config flags in `config.py`:**

   ```python
   ENABLE_PAPER_CURATOR = True
   PAPER_CURATOR_CHANNEL = "interesting-papers"
   PAPER_CURATOR_POST_TIME = "09:00"
   PAPER_CURATOR_USE_REMOTE = True   # use burger.si.umich.edu vLLM
   ```

2. **Invite the bot** to the destination channel:

   ```
   /invite @<bot-name>
   ```

3. **(Re)start the scheduler process:**

   ```bash
   python main.py
   ```

4. **Smoke-test from Slack:**

   ```
   /food-bot → Run Paper Curator (Admin Only)
   ```

   The digest gets DM'd to the clicking admin (channel post is *not*
   triggered by this button — it's a safe preview).

5. **Wait for auto-fire** at 09:00 the next weekday for the first real post.

---

## Pipeline

```
   sources.fetch_all()              ── 1500–2500 fresh paper records/day
            │
            ▼
   dedup against BOT_STATE.arxiv_papers.seen   (last 30 days, JSON file)
            │
            ▼
   bi-encoder embed (BAAI/bge-small-en-v1.5, CPU)
   member representation = N themed query vectors + 1 pub-mean vector
   paper score = max cosine across all rows
            │
            ▼
   top PAPER_CURATOR_TOP_K_TO_LLM (default 30) → LLM judge
            │
            ▼
   remote vLLM on burger (Qwen3.5-4B) ── primary
   local Ollama qwen3.6:35b-a3b       ── fallback if remote fails
   bi-encoder-only with header note   ── if both fail
            │
            ▼
   keep relevant=true ; cap @-mentions at PAPER_CURATOR_MAX_TAGS_PER_MEMBER
            │
            ▼
   sort by score → top 10 = main post, rest = thread reply
            │
            ▼
   chat_postMessage to PAPER_CURATOR_CHANNEL
   append paper IDs to BOT_STATE.arxiv_papers.seen
   append per-paper rows to logs/paper_curator_log.jsonl
```

The full pipeline lives in `paper_curator.curator.run_curation()`.

### Why this design

- **Two-stage ranking.** A small CPU bi-encoder is cheap enough to score
  thousands of candidates; the expensive LLM only sees ~30. This keeps the
  per-day inference cost trivial.
- **Themed query vectors.** Each line in `member_interests.yml` becomes its
  own embedding row. A paper that's a strong fit on *one* of a member's
  themes keeps its full cosine score (no dilution from averaging across
  unrelated themes).
- **Three-tier LLM fallback.** Remote vLLM is fastest; local Ollama is the
  no-network safety net; bi-encoder-only ranking still posts something
  useful when both LLMs are unreachable.
- **Two-process state via JSON.** `bot_state.json` is shared between the
  Slack handler and the scheduler the same way every other feature in this
  repo shares state — top-level key, file lock, merge-write.

---

## Configuration reference

All flags live in the project root `config.py`. Editing requires a
`main.py` restart for the scheduler process to see the new values.

### Master switch + posting

| Flag | Default | Effect |
|---|---|---|
| `ENABLE_PAPER_CURATOR` | `True` | Master kill switch. False = scheduled task is skipped and the admin button no-ops. |
| `PAPER_CURATOR_CHANNEL` | `"interesting-papers"` | Channel name (no `#`) or channel ID where the digest posts. |
| `PAPER_CURATOR_POST_TIME` | `"09:00"` | Local server time for the daily fire. |
| `PAPER_CURATOR_WEEKDAYS` | `[0,1,2,3,4]` | `datetime.weekday()` ints; default Mon–Fri. |
| `PAPER_CURATOR_DRY_RUN` | `False` | Scheduled runs print blocks to log instead of posting. Manual button is unaffected. |
| `PAPER_CURATOR_QUIET_DAY_NOTE` | `False` | Post a `:zzz: Quiet day…` message when nothing passes filter. False = silent. |

### Ranking + filtering

| Flag | Default | Effect |
|---|---|---|
| `PAPER_CURATOR_TOP_K_TO_LLM` | `30` | After bi-encoder ranking, top-K go to the LLM. Bigger = more LLM cost; smaller = more chance of missing a match. |
| `PAPER_CURATOR_MAX_MAIN_POST` | `10` | Picks shown in the main post. The rest go in a thread reply. |
| `PAPER_CURATOR_MAX_TAGS_PER_MEMBER` | `2` | Per-member @-mention cap **per post**. If hit, the LLM may judge a paper relevant but no one gets pinged on it. |
| `PAPER_CURATOR_BIENCODER` | `"BAAI/bge-small-en-v1.5"` | Sentence-Transformers model for ranking. Anything compatible with `SentenceTransformer(...)` works. |

### Profile cache

| Flag | Default | Effect |
|---|---|---|
| `PAPER_CURATOR_LAB_URL` | `"https://blablablab.si.umich.edu/"` | Page to scrape for member names + publications. |
| `PAPER_CURATOR_PROFILE_REFRESH_DAYS` | `7` | Re-scrape the lab site this often. `member_interests.yml` is *always* re-merged on every run regardless. |

### Local Ollama fallback

| Flag | Default | Effect |
|---|---|---|
| `PAPER_CURATOR_OLLAMA_HOST` | `"http://localhost:11434"` | Ollama server URL. |
| `PAPER_CURATOR_OLLAMA_MODEL` | `"qwen3.6:35b-a3b"` | Primary local model. Confirm with `python -m paper_curator.bench`. |
| `PAPER_CURATOR_OLLAMA_FALLBACK` | `"gemma4:26b"` | Fallback model if the primary fails twice in a row. |

### Remote vLLM dispatch

| Flag | Default | Effect |
|---|---|---|
| `PAPER_CURATOR_USE_REMOTE` | `True` | Try remote vLLM first; fall back to local Ollama on any failure. |
| `PAPER_CURATOR_REMOTE_HOST` | `"burger.si.umich.edu"` | SSH host with a free GPU + vLLM installed. |
| `PAPER_CURATOR_REMOTE_PYTHON` | `"/opt/anaconda/bin/python"` | Python interpreter on the remote with vllm in its env. |
| `PAPER_CURATOR_REMOTE_MODEL` | `"Qwen/Qwen3.5-4B"` | HF model id. vLLM downloads on first run into the shared HF cache. |
| `PAPER_CURATOR_REMOTE_MIN_GPU_FREE_GB` | `16` | Skip a GPU with less free VRAM than this. |
| `PAPER_CURATOR_REMOTE_MAX_GPU_UTIL` | `10` | Skip a GPU busier than this %. |
| `PAPER_CURATOR_REMOTE_TIMEOUT` | `600` | Seconds for the remote run before giving up. |

### Shared filesystem layout

| Flag | Default | Effect |
|---|---|---|
| `PAPER_CURATOR_DATA_DIR` | `"/shared/6/projects/food-bot"` | Runtime artifacts: profiles cache, embeds, run inputs/outputs, log. Must be visible from both this host and the remote vLLM box. |
| `PAPER_CURATOR_HF_HOME` | `"/shared/4/models"` | Shared HuggingFace cache. vLLM on burger and bge-small here both read/write to the same path so models download once. |

---

## Source registry — `data/sources.yml`

Each entry is a YAML object with `id`, `kind`, `enabled`, plus kind-specific
fields. Per-source failures are caught and logged — one bad feed never
breaks the run.

### Supported `kind` values

| `kind` | Required fields | Notes |
|---|---|---|
| `arxiv` | `category`, `max_results` | arXiv Atom API by category, e.g. `cs.CL`. |
| `osf` | `provider` | OSF preprint server: `socarxiv`, `psyarxiv`, etc. |
| `rss` | `url` | Generic RSS/Atom feed via `feedparser`. |
| `anthology_rss` | `url` | Same as `rss` but flags it as ACL Anthology (currently all broken upstream). |
| `openreview` | `venueid`, `limit` | OpenReview v2 API by venue id, e.g. `ICLR.cc/2026/Conference`. |
| `huggingface_daily` | `days` | HF Daily Papers list (curated). |

### Adding a new source

1. Find a feed URL or API endpoint.
2. Add an entry to `data/sources.yml`:

   ```yaml
   - id: my-new-source
     kind: rss
     url: https://example.org/feed.xml
     enabled: true
   ```

3. Re-run; check the logs for `source my-new-source: N papers` or a
   `failed:` warning. No code changes needed for `rss`/`arxiv`/`osf`/
   `openreview`/`huggingface_daily` kinds.

### What's currently enabled (and why)

**arXiv** (`cs.CL`, `cs.AI`, `cs.LG`, `cs.CY`, `cs.SI`, `cs.HC`, `cs.MA`)
covers core NLP/ML, society/ethics, social networks, HCI, and multi-agent.
`cs.DL` and `physics.soc-ph` are added but disabled — flip on for more
science-of-science / opinion-dynamics coverage.

**OSF**: SocArxiv (sociology, political sci) and PsyArxiv (psychology).

**Curated**: HuggingFace Daily Papers (~50/run, mostly arxiv-overlap which
dedupes naturally — value-add is HF's curation acting as a quality signal
plus coverage of cs.CV/cs.MM cross-disciplinary picks).

**Conferences**: OpenReview ICLR + NeurIPS submissions.

**Top journals**: Nature, Nature Human Behaviour, Science, Science Advances,
PNAS, JPSP. Tier-2 journals: medRxiv health-informatics, bioRxiv
neuroscience, Cognitive Science, Scientometrics, Political Communication,
Sociological Science, Journal of Sociolinguistics.

**Disabled (broken upstream)**: All ACL Anthology per-venue feeds (TACL,
CL, ACL, EMNLP, NAACL all return malformed XML — Anthology-wide problem),
Quantitative Science Studies, Journal of Communication. Left in YAML with
`enabled: false` and a comment so they're easy to flip back on if upstream
fixes the feed.

---

## Member interests — `data/member_interests.yml`

Hand-curated research focuses, keyed by member name **exactly as it
appears on the lab website**. Each value is a list of themed lines:

```yaml
Junghwan Kim:
  - Authorship attribution, verification, and classification
  - LLM-generated text detection and stylometry
  - AI-text watermarking and fingerprinting
  - Human-AI co-authorship; adversarial robustness of detectors
```

### What each line does

Each line becomes:

1. **A separate query vector** in the bi-encoder's per-member representation.
   A paper matching one theme keeps its full cosine score, undiluted by other
   themes.
2. **A bullet in the LLM's MEMBERS prompt block**, so the model can reason
   about specific topical fits when picking @-mentions.

### Editing rules

- **Names must match scraped profiles** (`profiles_cache.json`). Mismatches
  are silently dropped. Run `python -m paper_curator.cli --print-profiles`
  to see exact names.
- **Each line should be one coherent theme**: a tight cluster of related
  terms. 3–7 lines per person works well; more = diminishing returns.
- **Be concrete.** "Multilingual moral reasoning in code-mixed
  Hindi-English dialogue" beats "NLP".
- **Negatives are read by the LLM** — e.g. `Not interested in pure-vision
  papers` will steer it away.
- **Backward compatible**: a single string value (or a `|` block scalar)
  is still accepted; non-empty lines are split into themes.

### Cache invalidation

After editing, run:

```bash
python -m paper_curator.cli --refresh-profiles --print-profiles
```

`refresh_profiles` re-merges interests into the cache even when the scrape
itself is skipped. If interests changed, it rewrites `profiles_cache.json`
— the bi-encoder embed cache (`profile_embeds.npz`) is keyed on that file's
mtime, so embeddings rebuild on the next run.

---

## Slack ID mapping

Stored in `<DATA_DIR>/profiles/group_slack_ids.json`. Maps each scraped
member name to their Slack user ID (`U…`) so the digest can render
`<@USERID>` mentions that ping the right person.

### Auto-fill

```bash
python -m paper_curator.find_slack_ids                # dry-run, prints proposed mapping
python -m paper_curator.find_slack_ids --write        # persist to JSON
python -m paper_curator.find_slack_ids --write --overwrite   # also replace existing IDs
```

Match strategy: exact `real_name` → exact `display_name` → fuzzy
last-name + first-initial (same as the publication-author matcher).
Requires the bot to have the `users:read` scope.

### Manual fill

Edit `group_slack_ids.json` directly:

```json
{
  "Shivani Kumar": "U072JRB66JJ",
  "Some Person Without Slack": null
}
```

Members with `null` IDs render as plain text (no @-mention).

---

## Manual triggers

### Slack admin button

`/food-bot` shows a button **"Run Paper Curator (Admin Only)"** to anyone
in `ADMIN_USERS`. Clicking it spawns a background thread that calls
`run_curation(dry_run=False, preview_dm=f"@{user}")` — the digest gets DM'd
to the clicking admin instead of posted publicly. Safe to spam-test.

### CLI

```bash
# Refresh the lab profile cache, print the active member list (PI + postdocs + PhD students).
python -m paper_curator.cli --refresh-profiles --print-profiles

# Dump the FULL scraped cache including alumni / undergrads.
python -m paper_curator.cli --print-all-profiles

# End-to-end run, print blocks as JSON instead of posting.
python -m paper_curator.cli --dry-run

# Live run; posts to PAPER_CURATOR_CHANNEL. Use sparingly.
python -m paper_curator.cli

# Probe the remote GPU host: SSH reach, free GPU, vLLM importable.
python -m paper_curator.cli --remote-test

# Verbose logging.
python -m paper_curator.cli -v
```

### Bench (Ollama models)

```bash
python -m paper_curator.bench --models qwen3.6:35b-a3b,gemma4:26b --n 20
```

Flags: `--models`, `--n` (samples), `--threads` (sweep), `--out`,
`--no-warmup`. Pin the fastest model in `PAPER_CURATOR_OLLAMA_MODEL`.

---

## Remote vLLM dispatch

The remote LLM judge runs on `burger.si.umich.edu` (or whatever
`PAPER_CURATOR_REMOTE_HOST` points to) via SSH + a shared NFS filesystem.
No `scp` involved.

### How a run works

1. **Local side** (`paper_curator.remote_dispatch`):
   - SSH probes `nvidia-smi` to find a GPU meeting the free-VRAM and
     idle-utilization thresholds.
   - Copies `paper_curator/remote_judge.py` to `<DATA_DIR>/bin/` if its
     mtime changed (cheap rsync alternative).
   - Writes the input payload (papers + members) to
     `<DATA_DIR>/runs/<UTC-ts>/in.json`.
   - SSHes a one-shot command:
     ```bash
     HF_HOME=/shared/4/models ... flock -n /shared/6/projects/food-bot/lock \
       /opt/anaconda/bin/python /shared/6/projects/food-bot/bin/remote_judge.py \
       --input  in.json --output out.json --model Qwen/Qwen3.5-4B --gpu-id N
     ```
   - Reads `out.json` back from the shared FS once SSH returns.

2. **Remote side** (`remote_judge.py`, runs on burger):
   - Imports vllm, instantiates `LLM(...)` with the chosen GPU.
   - Builds chat-template prompts via the tokenizer with
     `enable_thinking=False` to suppress Qwen3's `<think>` blocks.
   - Single batched `llm.generate(prompts, sampling)` for all papers.
   - Parses each output as JSON; writes
     `{judgments, model, elapsed_sec}` to `out.json`.

### Setup checklist

- [ ] Passwordless SSH from the bot host to `PAPER_CURATOR_REMOTE_HOST`.
- [ ] `vllm` importable with `PAPER_CURATOR_REMOTE_PYTHON` on the remote.
- [ ] `<DATA_DIR>` mounted with the same path on both hosts.
- [ ] `<HF_HOME>` mounted on both hosts (or just on the remote — it's
      written through the env vars `HF_HOME` / `HUGGINGFACE_HUB_CACHE` /
      `TRANSFORMERS_CACHE` / `SENTENCE_TRANSFORMERS_HOME`).
- [ ] At least one GPU on the remote with ≥ `MIN_GPU_FREE_GB` free.

### Verifying

```bash
python -m paper_curator.cli --remote-test
```

Expected output:

```
host:   burger.si.umich.edu
python: /opt/anaconda/bin/python
model:  Qwen/Qwen3.5-4B
SSH OK: SSH_OK | burger.si.umich.edu | <kernel>
free GPU pick: 0
remote env: vllm 0.x.x py 3.12.x
```

If any step fails, the dispatcher logs a warning and `judge_papers` falls
back to local Ollama transparently.

### Thinking-mode suppression

Qwen3 models can emit `<think>...</think>` reasoning traces that break JSON
parsing. We suppress this at three layers, belt-and-suspenders:

1. **Ollama**: `client.chat(think=False, ...)` in `llm.py:_chat_no_think`.
2. **System prompt**: `/no_think` directive included in `SYSTEM`.
3. **vLLM**: `tokenizer.apply_chat_template(..., enable_thinking=False)`
   in `remote_judge.py`.

---

## File layout

```
paper_curator/
  __init__.py
  curator.py            # orchestration: fetch → rank → judge → post → persist
  sources.py            # paper-source fetchers (arxiv, osf, rss, openreview, hf)
  profiles.py           # lab-website scraper + interest merger
  embeddings.py         # bi-encoder; per-theme query vectors
  llm.py                # local Ollama judge with remote-first dispatch
  remote_dispatch.py    # local-side: SSH, GPU pick, shared-FS payload
  remote_judge.py       # remote-side: vLLM batched generate (self-contained)
  post.py               # block-kit formatter (header + bulleted list)
  paths.py              # single source of truth for on-disk paths
  cli.py                # python -m paper_curator.cli
  bench.py              # python -m paper_curator.bench
  find_slack_ids.py     # python -m paper_curator.find_slack_ids
  data/
    sources.yml          # source registry (in source control)
    member_interests.yml # hand-curated focuses (in source control)

# Runtime data (PAPER_CURATOR_DATA_DIR; default /shared/6/projects/food-bot/):
  bin/remote_judge.py            # synced from the package on each run
  runs/<UTC-ts>/in.json,out.json # one dir per remote-judge invocation
  profiles/profiles_cache.json   # last lab-site scrape + merged interests
  profiles/profile_embeds.npz    # cached bi-encoder vectors
  profiles/group_slack_ids.json  # member name → Slack user ID
  logs/paper_curator_log.jsonl   # one row per posted paper
  lock                           # flock target for serializing remote runs

# Shared HF cache (PAPER_CURATOR_HF_HOME; default /shared/4/models/):
  hub/                  # HF Hub model snapshots (vLLM + bge-small)
  sentence_transformers/  # ST-specific pooling configs
```

---

## Logs & observability

- **`<DATA_DIR>/logs/paper_curator_log.jsonl`** — append-only, one row per
  posted paper:
  ```json
  {"ts": "2026-05-06T09:00:12", "id": "arxiv:2605.03609",
   "title": "...", "source": "arxiv:cs.AI", "tags": ["Nancy Xu"], "score": 9}
  ```
  Useful for "why was this picked?" questions and for auditing tag
  distribution over time.

- **`<DATA_DIR>/runs/<ts>/`** — `in.json`, `out.json`, `stderr.log` for
  every remote vLLM run. Auto-GCed; the most recent 50 runs are kept.

- **`bot_state.json` → `arxiv_papers`** — `{seen: {id: date}, last_run:
  iso}`. Pruned to the last 30 days each time it's persisted.

- **Main process log** (whatever `main.py` is logging to) — every run
  prints lines like `paper_curator: 686 fresh papers after dedup`,
  `remote GPU 2 selected (23.5 GB free, 0% util)`, `remote judge done:
  30/30 papers, 194.8s vllm`.

- **Slack admin DM** — on profile-scrape failures with no usable cache, the
  bot DMs every user in `ADMIN_USERS` so you find out before the next
  scheduled fire.

---

## Tuning recipes

| Want… | Knob |
|---|---|
| **Fewer / more main picks** | `PAPER_CURATOR_MAX_MAIN_POST` (default 10) |
| **More LLM judgements (deeper coverage)** | `PAPER_CURATOR_TOP_K_TO_LLM` (default 30) — costs more vLLM time |
| **Spread tags across more people** | Lower `PAPER_CURATOR_MAX_TAGS_PER_MEMBER` to 1 |
| **Fewer "this person never gets tagged" days** | Raise the cap to 3, or expand their `member_interests.yml` themes |
| **A specific member to never get pinged** | Remove them from `group_slack_ids.json` (set their value to `null`) — the LLM may still tag the name; Slack just won't @-mention. |
| **Run on weekends too** | `PAPER_CURATOR_WEEKDAYS = [0,1,2,3,4,5,6]` |
| **Different post time** | `PAPER_CURATOR_POST_TIME` |
| **A canary period** | `PAPER_CURATOR_CHANNEL = "bot-testing"` (or your DM ID) for a week before flipping back |
| **Logs only, no posts** | `PAPER_CURATOR_DRY_RUN = True` (scheduled fires print to stdout instead) |
| **Add a journal** | New entry in `data/sources.yml` with `kind: rss` |
| **Add a member's research focuses** | New entry in `data/member_interests.yml` keyed by their exact scraped name |
| **Bypass remote vLLM** | `PAPER_CURATOR_USE_REMOTE = False` — falls back to local Ollama directly |
| **Different remote model** | `PAPER_CURATOR_REMOTE_MODEL = "Qwen/Qwen3.5-1.5B"` (no other code changes; vLLM downloads on first run) |

---

## Troubleshooting

### "0 papers after dedup"

- Check that arXiv didn't 429 you — there's a per-source `failed: 429` log
  line. arXiv asks for ≥3s between requests; we do that, but their backoff
  state can persist across processes. Wait 10 minutes and re-run.
- Inspect `BOT_STATE.arxiv_papers.seen` — if it grew faster than expected
  (e.g. someone manually triggered the curator dozens of times), all of
  today's papers may already be marked seen.

### "no remote GPU meets thresholds"

- Run `ssh burger.si.umich.edu nvidia-smi` to see what's busy.
- Drop `PAPER_CURATOR_REMOTE_MIN_GPU_FREE_GB` (e.g. to 12) if your model is
  smaller than expected.
- Drop `PAPER_CURATOR_REMOTE_MAX_GPU_UTIL` (e.g. to 30) if every GPU has
  some background load.
- The dispatcher will fall back to local Ollama on its own — this isn't a
  hard failure.

### "remote judge errored … falling back to Ollama"

Check `<DATA_DIR>/runs/<latest>/stderr.log` on either host. Common causes:

- `ImportError: vllm` — `vllm` not installed in the env that
  `PAPER_CURATOR_REMOTE_PYTHON` points at.
- `transformers` / `huggingface-hub` version skew — vLLM pins both;
  `pip install -U transformers huggingface_hub` on the remote.
- OOM during weight load — model is too big for the chosen GPU. Drop to a
  smaller `PAPER_CURATOR_REMOTE_MODEL` or raise `MIN_GPU_FREE_GB`.

### "<think>" appearing in rationales

Means thinking-mode suppression failed. Check (in priority order):

1. The remote tokenizer doesn't support `enable_thinking=False` — the code
   already falls back to a `TypeError` branch using just `/no_think` in
   the system prompt, which works for Qwen3 models.
2. The model isn't a Qwen3 derivative and doesn't honor `/no_think`. Pick
   a different model or strip `<think>...</think>` blocks in
   `_parse_judgment`.

### "feedparser failure for X: syntax error"

The upstream feed is serving malformed XML. Set `enabled: false` for that
source in `data/sources.yml` and add a comment with the date you
verified, so it's easy to flip back later.

### Member listed but never gets tagged

1. Verify their name appears in `group_slack_ids.json` with a non-null
   value. (Without an ID the LLM may tag them but Slack can't ping them.)
2. Verify their scraped name matches their key in `member_interests.yml`
   exactly — `Ben Litterer` vs `Benjamin Litterer` is a silent miss.
3. Look at `paper_curator_log.jsonl` — if they get tagged some days but
   not others, that's just a low-overlap day, not a bug.
4. If they consistently get zero, expand their themes in
   `member_interests.yml` (especially with adjacent vocab — e.g. add
   "stylometry" alongside "authorship attribution").

### Profile scrape returns 0 members

Lab site HTML changed. The two layouts currently handled:
- `<td>` blocks with a `font-size: 120%` name `<div>` (most members).
- `<p class="personal-info">` blocks (PI card).

If a third layout appears, extend `_parse_people_panel` and
`_add_member_from_anchor` in `profiles.py`.

### Slack post fails

The bot must be in the destination channel. Run `/invite @<bot>` in the
channel. If you see `not_in_channel` errors and the bot already appears in
member lists, it may be archived — unarchive or pick a new channel via
`PAPER_CURATOR_CHANNEL`.
