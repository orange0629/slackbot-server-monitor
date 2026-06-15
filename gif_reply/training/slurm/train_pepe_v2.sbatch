#!/bin/bash
#SBATCH --job-name=gif_pepe_v2
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --nice=0
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/shared/0/projects/gif-reply-slack-bot/training_logs/%x-%j.out
#SBATCH --error=/shared/0/projects/gif-reply-slack-bot/training_logs/%x-%j.err

# PEPE-v2: same two-stage recipe as train_phase2.sbatch, but stage 1 mixes
# TWO weak-caption augmentation sources alongside PEPE:
#   - TGIF   (caption -> gif)
#   - Giphy  (alt_text -> gif, the ~31k currently-served crawled pool)
# so the embedding space covers the UNION of PEPE 115k + Giphy. Stage 2 is
# PEPE-only fine-tune. Warm-starts from Phase-1 best.pt with --reset-best so
# the only changed variable vs the prior run is the added Giphy data.
#
# Submit:
#   sbatch --gres=gpu:A6000:1 gif_reply/training/slurm/train_pepe_v2.sbatch

MODEL_DIR="${MODEL_DIR:-/shared/0/projects/gif-reply-slack-bot/models/pepe_v2}"

set -euo pipefail
mkdir -p /shared/0/projects/gif-reply-slack-bot/training_logs
mkdir -p "$MODEL_DIR"

cd /home/jurgens/projects/slackbot-server-monitor

# Idempotent resume: prefer this run's own latest.pt, else Phase 1 best.pt.
if [[ -f "$MODEL_DIR/latest.pt" ]]; then
  RESUME="$MODEL_DIR/latest.pt"
  echo "[resume] $RESUME"
else
  RESUME=/shared/0/projects/gif-reply-slack-bot/models/siglip_ft/best.pt
  if [[ ! -f "$RESUME" ]]; then
    echo "Phase 1 best.pt not found at $RESUME; refusing to start" >&2
    exit 1
  fi
  echo "[warm-start from Phase 1] $RESUME"
fi

TGIF_DIR=/shared/0/projects/gif-reply-slack-bot/phase2/tgif
GIPHY_DIR=/shared/0/projects/gif-reply-slack-bot/phase2/giphy
SHARED_FRAMES=/shared/0/projects/gif-reply-slack-bot/training_cache/frames

for d in "$TGIF_DIR" "$GIPHY_DIR"; do
  if [[ ! -f "$d/examples.jsonl" || ! -f "$d/tokens.pt" ]]; then
    echo "augmentation data not ready ($d/examples.jsonl or $d/tokens.pt missing)" >&2
    exit 1
  fi
done

# TGIF frames live in $TGIF_DIR/frames (sha1-keyed); Giphy frames live in the
# shared PEPE cache (giphy:<id>-keyed) — each extra source carries its own
# --stage1-extra-frame-dir, so they don't need to share a directory.
python -m gif_reply.training.train \
  --output-dir "$MODEL_DIR" \
  --resume "$RESUME" --reset-best \
  --loss sigmoid --multi-positive --grad-cache \
  --gradient-checkpointing --n-frames 2 \
  --batch-size 96 --grad-accum 8 --num-workers 8 \
  --lr 3e-6 --head-lr 3e-5 --weight-decay 5e-3 --tag-loss-weight 0.05 \
  --freeze-fraction 0.75 \
  --select-metric recall@10 --retrieval-max-tweets 3000 \
  --stage1-epochs 1 --stage1-freeze-fraction 1.0 \
  --stage1-extra-examples-jsonl "$TGIF_DIR/examples.jsonl" \
  --stage1-extra-tokens-pt "$TGIF_DIR/tokens.pt" \
  --stage1-extra-frame-dir "$TGIF_DIR/frames" \
  --stage1-extra-examples-jsonl "$GIPHY_DIR/examples.jsonl" \
  --stage1-extra-tokens-pt "$GIPHY_DIR/tokens.pt" \
  --stage1-extra-frame-dir "$SHARED_FRAMES" \
  --stage2-epochs 2
