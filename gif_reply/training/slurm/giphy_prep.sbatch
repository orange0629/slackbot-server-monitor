#!/bin/bash
#SBATCH --job-name=giphy_prep
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --nice=10000
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/shared/0/projects/gif-reply-slack-bot/training_logs/%x-%j.out
#SBATCH --error=/shared/0/projects/gif-reply-slack-bot/training_logs/%x-%j.err

# Build the Giphy weak-caption augmentation source for PEPE-v2:
#   1. refetch + decode + cache frames for the ~31k served Giphy gifs
#      (network + CPU decode; NO GPU — submitted without --gres)
#   2. emit alt_text->gif examples.jsonl
#   3. tokenize captions -> tokens.pt
#
# Idempotent: crawl skips already-cached gifs, so a --requeue or a manual
# resubmission after a --time kill simply continues. Giphy free-tier rate
# limits can stretch step 1 across the backoff schedule; that's expected.
#
# Submit (no GPU):
#   sbatch gif_reply/training/slurm/giphy_prep.sbatch

set -euo pipefail
mkdir -p /shared/0/projects/gif-reply-slack-bot/training_logs
cd /home/jurgens/projects/slackbot-server-monitor

echo "[giphy_prep] step 1/3: build-giphy-frames"
python -m gif_reply.training.giphy_frames build-giphy-frames --workers 8

echo "[giphy_prep] step 2/3: emit-examples"
python -m gif_reply.training.giphy_frames emit-examples

echo "[giphy_prep] step 3/3: build-token-cache"
python -m gif_reply.training.giphy_frames build-token-cache

echo "[giphy_prep] done"
