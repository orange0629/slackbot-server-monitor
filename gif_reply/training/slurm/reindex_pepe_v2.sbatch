#!/bin/bash
#SBATCH --job-name=gif_reindex_pepe_v2
#SBATCH --partition=gpu
#SBATCH --time=4:00:00
#SBATCH --nice=0
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/shared/0/projects/gif-reply-slack-bot/training_logs/%x-%j.out
#SBATCH --error=/shared/0/projects/gif-reply-slack-bot/training_logs/%x-%j.err

# Re-embed the UNION pool (PEPE 115k pulled from the training pickle +
# the ~31k served Giphy crawl) with the PEPE-v2 checkpoint. PEPE candidates
# come from --candidates-pickle (child_gif_id is the bare 13-char Giphy id
# that matches the shared frame cache). Giphy gifs only contribute if their
# frames were cached by gif_reply.training.giphy_frames (keyed giphy:<id>).
#
# Submit:
#   sbatch --gres=gpu:A6000:1 gif_reply/training/slurm/reindex_pepe_v2.sbatch

set -euo pipefail
mkdir -p /shared/0/projects/gif-reply-slack-bot/index_pepe_v2

cd /home/jurgens/projects/slackbot-server-monitor

OUT=/shared/0/projects/gif-reply-slack-bot/index_pepe_v2/siglip_ft_embeddings.npy
if [[ -f "$OUT" ]]; then
  echo "[skip] $OUT already exists"
  exit 0
fi

python -m gif_reply.training.reindex --checkpoint /shared/0/projects/gif-reply-slack-bot/models/pepe_v2/best.pt --out-dir /shared/0/projects/gif-reply-slack-bot/index_pepe_v2 --backend-name siglip_ft --candidates-pickle /shared/2/projects/gif-reply/data/processed/dataset/bertweet-normalize/finalized-split-dataset/tweet-gif-reply.pickle --crawled-jsonl /shared/0/projects/gif-reply-slack-bot/index/index_metadata.jsonl --batch-size 64
