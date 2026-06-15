# SLURM cluster operator's guide

A project-agnostic reference for running jobs on this cluster. Captures the
conventions, gotchas, and patterns that took time to learn — none of which
are specific to any particular workload.

## Cluster shape

- **Login node** vs **compute nodes** are distinct environments. Most ML /
  GPU-aware Python packages (CUDA, vLLM, torch with GPU support, MCP tooling
  installed via `npx`/`uvx`, etc.) are present **only on compute nodes**.
  Imports that touch them will fail on the login node.
  - Practical rule: do not run `python -c "import torch"`-style smoke
    checks from the login shell. Wrap them in `srun` or submit a small
    `sbatch` job.
  - The login node is fine for editing files, planning, lightweight CPU
    Python, `git`, and submitting jobs.
- **Shared filesystem.** A path like `/shared/<group>/projects/<name>/` is
  visible from every node and is the canonical place for large artifacts,
  shared tool installs (Node, npm cache, model snapshots, datasets), and
  log directories. Compute nodes typically do **not** ship Node, so any
  workload that shells out to `npx` / `uvx` should source binaries from
  the shared install rather than the system path.
- **GPU partitions and types.** This cluster exposes a single `gpu`
  partition. GPU types (e.g. `A100`, `A6000`) are selected via
  `--gres=gpu:<type>:<count>` rather than `--constraint`. Different GPU
  types have very different idle capacity at different times of day —
  alternating shards across types (e.g. even shards → A100, odd → A6000)
  can roughly double effective throughput for embarrassingly parallel
  workloads.

## SBATCH directives we standardize on

```bash
#!/bin/bash
#SBATCH --job-name=<short-name>
#SBATCH --partition=gpu
#SBATCH --time=HH:MM:SS               # tight upper bound; jobs are killed at expiry
#SBATCH --nice=<priority>             # see "Nice values" below
#SBATCH --cpus-per-task=8             # 8 is a reasonable default; bump for data-heavy jobs
#SBATCH --mem=64G                     # host RAM; GPU VRAM is implicit from --gres
#SBATCH --requeue                     # re-queue automatically on preemption / node failure
#SBATCH --output=/shared/.../slurm_logs/%x-%j.out
```

`--gres=gpu:<type>:<count>` is passed at submit time (`sbatch --gres=...`)
rather than baked into the script, because shard-level routing decides the
GPU type per-job.

### Nice values (priority lanes)

`--nice` adds to the job's scheduling priority — **higher value = lower
priority**. We use it to create informal lanes inside the single `gpu`
partition:

| Nice | Use case |
| ---- | -------- |
| 0      | Post-processing / fast-turnaround jobs (filter, export, score). Skips ahead of de-prioritized rollout queue. |
| 100    | Generator jobs (paraphrasing, prompt synthesis). Slightly de-prioritized vs. interactive work but ahead of bulk rollouts. |
| 10000  | Bulk long-running rollout jobs. Yields readily to anything important. |
| 50000  | Backfill: only runs when the cluster would otherwise be idle. |

Important: when both a `#SBATCH --nice=N` directive and a CLI
`sbatch --nice=M` flag are given, **the CLI flag wins**. Use this to
override the bake-in default per submission.

### `--requeue`

Long jobs (multi-hour to multi-day) should set `--requeue`. Combined with
idempotent work (skip-if-output-exists logic in the inner script), this
makes preemption recoverable without operator intervention.

## Submitting jobs

### Shell pattern

The repo convention is a two-file split:

- `batch/run_<thing>_job.sh` — the SBATCH script. Reads required
  parameters from environment variables (`: "${VAR:?VAR is required}"`)
  and exits non-zero if anything is missing.
- `batch/run_<thing>.sh` (or `submit_<thing>.sh`) — a thin login-side
  wrapper that loops over inputs and calls `sbatch --export=ALL,VAR=...
  run_<thing>_job.sh` per shard.

This keeps the SBATCH script a pure executor and the submit script a pure
fan-out, which is much easier to reason about than a single 200-line file
that does both.

### Passing environment to the job

Use `--export="ALL,KEY1=val1,KEY2=val2,..."`. The `ALL` token forwards the
submitting shell's environment too.

**Comma trap.** `--export` itself is comma-separated, so any value that
contains a comma (JSON config blobs, CSV-ish flags) will be silently split
into separate variables. Workaround: `export THE_VAR="..."` in the parent
shell **before** the `sbatch` call, then rely on `ALL` to propagate it
without re-listing it inside `--export`.

### Reading status

```bash
squeue -u $USER                              # my jobs
squeue -u $USER --format='%.18i %.20j %.2t %.10M %.6D %R'  # nicer format
sacct -u $USER --starttime=now-24hours \
      --format=JobID,JobName%30,State,ExitCode,Elapsed,ReqMem,NodeList
scontrol show job <jobid>                    # everything about one job
```

## Gotchas worth memorizing

### 1. SLURM scripts are **frozen at submit time**

`sbatch` snapshots the script file. Editing it on disk afterward does
**not** affect already-queued jobs. If you find a bug in a queued
batch:

```bash
scancel <jobid>          # or scancel -u $USER <jobname-pattern>
sbatch ...               # resubmit the fixed script
```

This catches everyone at least once.

### 2. `InvalidAccount` flags are usually transient — but don't trust them

`squeue` may show jobs with reason `InvalidAccount` even though they
eventually run. The flag itself is harmless on this cluster. **However**:
do not let "oh, that's just the InvalidAccount thing" short-circuit a
real diagnosis when jobs aren't progressing. If a job is stuck, check
`scontrol show job` for the actual reason (resources, priority, time
limits) before blaming the flag.

### 3. Compute nodes lack Node.js (and similar runtimes)

Workloads that spawn helper subprocesses via `npx`, `uvx`, `bun`, etc.
need those binaries on `PATH`. The system `PATH` on compute nodes does
not include them. The repo convention:

```bash
SHARED_NODE_BIN="/shared/.../tools/node/current/bin"
SHARED_NPM_CACHE="/shared/.../tools/npm-cache"
UVX_BIN_DIR="$HOME/.cargo/bin"
export PATH="$SHARED_NODE_BIN:$UVX_BIN_DIR:$PATH"
export NPM_CONFIG_CACHE="$SHARED_NPM_CACHE"
```

Front-load this in **every** SBATCH script that may shell out to those
tools, and `command -v npx >/dev/null || exit 1` to fail fast — a missing
runtime usually manifests as silent empty output, not a clean error.

### 4. Don't run imports on the login node

`import torch`, `import vllm`, `import transformers` (with GPU paths),
and similar are missing or partially broken on the login node. To
sanity-check an environment, use `srun --pty bash` for an interactive
compute-node shell, or wrap the check in a one-shot `sbatch`.

### 5. Prefer idempotent inner scripts

Combine `--requeue` with output-existence checks in the worker:

```bash
[[ -f "$OUT_PATH" ]] && { echo "[skip] already done"; exit 0; }
```

This makes preemption, manual `scancel`s, and re-submitting a partially
completed batch all safe. It is the single highest-leverage habit on
this cluster.

### 6. Logs directory must exist before submission

`#SBATCH --output=/path/%x-%j.out` does not create the parent directory.
The submit wrapper should `mkdir -p "$LOGS_DIR"` before any `sbatch`
call, or jobs fail to start with no obvious log.

### 7. Time limits are hard ceilings

Jobs are killed at `--time` expiry with no grace period. Pad estimates
generously — re-queueing a 47-hour job at hour 47:55 is painful. For
unknown-runtime work, use `--requeue` plus checkpoint logic instead of
guessing a generous bound.

## Useful patterns

### Per-shard fan-out

```bash
for shard in $SHARDS_DIR/shard_*.jsonl; do
  sbatch \
    --job-name="work-$(basename "$shard" .jsonl)" \
    --gres="gpu:A100:1" \
    --export="ALL,SHARD_PATH=$shard" \
    run_worker_job.sh
done
```

Each shard becomes one independent job. SLURM handles concurrency via
fair-share / partition limits; nothing in the submitter coordinates.

### Subset / range filters in the submitter

Useful when you've already submitted a batch and want to re-run a
subset (failed shards, new shards, specific indices):

```bash
SHARD_MIN=42 bash submit.sh         # only shards with index >= 42
SHARD_MAX=80 bash submit.sh         # only shards with index <= 80
SKIP_INDICES="3,7,19" bash submit.sh
LIMIT_SHARDS=3 bash submit.sh        # smoke-test first 3
```

Pattern: parse the integer index out of `shard_<NNN>.jsonl` filenames
and gate `sbatch` on it.

### Cancelling many jobs at once

```bash
scancel -u $USER --name=synth-rollout              # by job name
scancel -u $USER --state=PENDING                   # only pending
squeue -u $USER -h -o '%i %j' | awk '/pattern/ {print $1}' | xargs -r scancel
```

## Anti-patterns

- **Polling `squeue` in a tight loop.** SLURM's controller is shared;
  rapid polling impacts everyone. 30s+ between polls is plenty.
- **Per-job conda activation.** Slow and brittle. Front-load `PATH` /
  `PYTHONPATH` instead, or bake the env into a wrapper like `py.sh`.
- **Writing temp files to the job's working directory.** This may be
  the login node's home or a slow shared FS. Use `$SLURM_TMPDIR` (node-
  local scratch) for transient I/O, then copy results back.
- **Building one giant SBATCH script.** Split into a thin worker that
  reads env vars and a fan-out wrapper that loops `sbatch`. Easier to
  rerun a single shard for debugging.
- **`scancel` without scoping.** A bare `scancel` with a typo can nuke
  unrelated work. Always include `-u $USER` and a name/state filter.
- **Editing a queued script.** See gotcha 1 — re-submit instead.

## Quick reference

```bash
# inspect
squeue -u $USER
sacct -u $USER --starttime=now-24hours
scontrol show job <jobid>
seff <jobid>                        # post-hoc efficiency report

# control
sbatch script.sh
scancel <jobid>
scontrol hold <jobid>               # pause a pending job
scontrol release <jobid>

# interactive
srun --partition=gpu --gres=gpu:A100:1 --time=1:00:00 --pty bash
```
