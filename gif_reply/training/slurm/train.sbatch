#!/bin/bash
#SBATCH --job-name=gif_ft
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --nice=10000
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/shared/0/projects/gif-reply-slack-bot/training_logs/%x-%j.out
#SBATCH --error=/shared/0/projects/gif-reply-slack-bot/training_logs/%x-%j.err

# Submit with the GPU type as a CLI flag (per slurm_cluster_guide.md):
#   sbatch --gres=gpu:A6000:1 gif_reply/training/slurm/train.sbatch
#   sbatch --gres=gpu:A100:1  gif_reply/training/slurm/train.sbatch

set -euo pipefail
mkdir -p /shared/0/projects/gif-reply-slack-bot/training_logs
mkdir -p /shared/0/projects/gif-reply-slack-bot/models/siglip_ft

cd /home/jurgens/projects/slackbot-server-monitor

# Idempotent: if a previous run produced latest.pt, resume from it.
RESUME_FLAG=""
if [[ -f /shared/0/projects/gif-reply-slack-bot/models/siglip_ft/latest.pt ]]; then
  RESUME_FLAG="--resume /shared/0/projects/gif-reply-slack-bot/models/siglip_ft/latest.pt"
  echo "[resume] $RESUME_FLAG"
fi

python -m gif_reply.training.train --output-dir /shared/0/projects/gif-reply-slack-bot/models/siglip_ft --batch-size 64 --grad-accum 8 --epochs 5 --lr 1e-5 --head-lr 1e-4 --tag-loss-weight 0.1 --num-workers 8 $RESUME_FLAG
