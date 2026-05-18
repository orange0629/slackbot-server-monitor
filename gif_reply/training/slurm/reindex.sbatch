#!/bin/bash
#SBATCH --job-name=gif_reindex
#SBATCH --partition=gpu
#SBATCH --time=4:00:00
#SBATCH --nice=0
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/shared/0/projects/gif-reply-slack-bot/training_logs/%x-%j.out
#SBATCH --error=/shared/0/projects/gif-reply-slack-bot/training_logs/%x-%j.err

# Submit with the GPU type as a CLI flag (per slurm_cluster_guide.md):
#   sbatch --gres=gpu:A6000:1 gif_reply/training/slurm/reindex.sbatch

set -euo pipefail
mkdir -p /shared/0/projects/gif-reply-slack-bot/index_v2

cd /home/jurgens/projects/slackbot-server-monitor

OUT=/shared/0/projects/gif-reply-slack-bot/index_v2/siglip_ft_embeddings.npy
if [[ -f "$OUT" ]]; then
  echo "[skip] $OUT already exists"
  exit 0
fi

python -m gif_reply.training.reindex --checkpoint /shared/0/projects/gif-reply-slack-bot/models/siglip_ft/best.pt --out-dir /shared/0/projects/gif-reply-slack-bot/index_v2 --backend-name siglip_ft --candidates-csv /shared/2/projects/gif-reply/data/release/gif-metadata.csv --crawled-jsonl /shared/0/projects/gif-reply-slack-bot/index/index_metadata.jsonl --batch-size 64
