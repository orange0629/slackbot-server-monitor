#!/bin/bash
#SBATCH --job-name=gif_ft_p2
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --nice=10000
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/shared/0/projects/gif-reply-slack-bot/training_logs/%x-%j.out
#SBATCH --error=/shared/0/projects/gif-reply-slack-bot/training_logs/%x-%j.err

# Single-process TWO-STAGE Phase-2 continuation:
#   Stage 1: warm heads/projection on TGIF+PEPE (towers frozen).
#   Stage 2: PEPE-only fine-tune (bottom-half frozen).
# Both run in one `python -m` call. SigLIP sigmoid loss + multi-positive +
# GradCache (contrastive batch = batch_size * grad_accum). Checkpoints are
# selected on PEPE-dev Recall@10, not dev loss.
#
# Submit (depends on Phase 1):
#   sbatch --gres=gpu:A6000:1 --dependency=afterok:<phase1_jobid> \
#          gif_reply/training/slurm/train_phase2.sbatch

set -euo pipefail
mkdir -p /shared/0/projects/gif-reply-slack-bot/training_logs
mkdir -p /shared/0/projects/gif-reply-slack-bot/models/siglip_ft_p2

cd /home/jurgens/projects/slackbot-server-monitor

# Idempotent resume: prefer this run's own latest.pt, else Phase 1 best.pt.
if [[ -f /shared/0/projects/gif-reply-slack-bot/models/siglip_ft_p2/latest.pt ]]; then
  RESUME=/shared/0/projects/gif-reply-slack-bot/models/siglip_ft_p2/latest.pt
  echo "[resume] $RESUME"
else
  RESUME=/shared/0/projects/gif-reply-slack-bot/models/siglip_ft/best.pt
  if [[ ! -f "$RESUME" ]]; then
    echo "Phase 1 best.pt not found at $RESUME; refusing to start" >&2
    exit 1
  fi
  echo "[warm-start from Phase 1] $RESUME"
fi

PHASE2_DIR=/shared/0/projects/gif-reply-slack-bot/phase2/tgif
EXTRA_JSONL=$PHASE2_DIR/examples.jsonl
EXTRA_TOKENS=$PHASE2_DIR/tokens.pt
EXTRA_FRAMES=$PHASE2_DIR/frames

if [[ ! -f "$EXTRA_JSONL" || ! -f "$EXTRA_TOKENS" ]]; then
  echo "Phase 2 data not ready ($EXTRA_JSONL or $EXTRA_TOKENS missing)" >&2
  exit 1
fi

# --grad-cache makes the effective contrastive batch 96 * 8 = 768 at the
# memory cost of one micro-batch; --n-frames 2 + gradient checkpointing let
# the per-step micro-batch grow. Tune --batch-size to the A6000's 48 GB.
python -m gif_reply.training.train \
  --output-dir /shared/0/projects/gif-reply-slack-bot/models/siglip_ft_p2 \
  --resume "$RESUME" --reset-best \
  --loss sigmoid --multi-positive --grad-cache \
  --gradient-checkpointing --n-frames 2 \
  --batch-size 96 --grad-accum 8 --num-workers 8 \
  --lr 3e-6 --head-lr 3e-5 --weight-decay 5e-3 --tag-loss-weight 0.05 \
  --freeze-fraction 0.75 \
  --select-metric recall@10 --retrieval-max-tweets 3000 \
  --stage1-epochs 1 --stage1-freeze-fraction 1.0 \
  --stage1-extra-examples-jsonl "$EXTRA_JSONL" \
  --stage1-extra-tokens-pt "$EXTRA_TOKENS" \
  --stage1-extra-frame-dir "$EXTRA_FRAMES" \
  --stage2-epochs 2
