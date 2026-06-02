#!/bin/bash
#SBATCH --job-name=gif_reindex_pool
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --nice=0
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/shared/0/projects/gif-reply-slack-bot/training_logs/%x-%j.out
#SBATCH --error=/shared/0/projects/gif-reply-slack-bot/training_logs/%x-%j.err

# Incremental pool reindex. With --append, reindex.py:
#   1. Loads the existing siglip_ft_embeddings.npy + index_metadata.jsonl
#      from --out-dir.
#   2. Computes (merged candidates) - (already-indexed gif_ids).
#   3. Encodes ONLY the new diff on the GPU with the frozen PEPE-v2 checkpoint.
#   4. Concatenates onto existing matrix + atomic-rewrites both files.
#
# Safe to run every K new gifs (the discoverer's --reindex-threshold gates this).
# Falls back to a full build if the index dir is empty.
#
# A full rebuild is only required when the PEPE-v2 checkpoint itself changes,
# in which case run reindex_pepe_v2.sbatch (no --append) instead.
#
# Submit (the discoverer does this automatically; also runnable by hand):
#   sbatch --gres=gpu:A6000:1 /home/jurgens/projects/slackbot-server-monitor/gif_reply/training/slurm/reindex_pool.sbatch

set -euo pipefail
mkdir -p /shared/0/projects/gif-reply-slack-bot/index_pepe_v2_pool

cd /home/jurgens/projects/slackbot-server-monitor

python -m gif_reply.training.reindex --checkpoint /shared/0/projects/gif-reply-slack-bot/models/pepe_v2/best.pt --out-dir /shared/0/projects/gif-reply-slack-bot/index_pepe_v2_pool --backend-name siglip_ft --append --candidates-pickle /shared/2/projects/gif-reply/data/processed/dataset/bertweet-normalize/finalized-split-dataset/tweet-gif-reply.pickle --crawled-jsonl /shared/0/projects/gif-reply-slack-bot/index/index_metadata.jsonl --crawled-jsonl /shared/0/projects/gif-reply-slack-bot/discover_pool/index_metadata.jsonl --batch-size 64
