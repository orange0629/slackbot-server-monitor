# config.py

# Slack settings
SLACK_CHANNEL = "#tech-support"  # or channel ID like "C12345678"

# Log and cache paths
USAGE_LOG_FILE = "usage_log.jsonl"
MONITOR_LOG_FILE = "monitor.log"
SCHEDULER_STATE_FILE = "scheduler_state.json" # For periodic monitoring
BOT_STATE_FILE = "bot_state.json" # Shared bot state between processes

# Exclude these usernames from monitoring
EXCLUDED_USERS = ["root", "sysadmin", "lucyli", "jurgens"]

# Admin users
ADMIN_USERS = ["jurgens", "leczhang", "elealin", "leczhang0629"]

# Home usage monitoring
ENABLE_HOME_MONITORING = True
SCAN_METHOD = "DU" # Scanning method: "NCDU" or "DU" or "FIND"
HOME_DIR = "/home"
NCDU_CACHE_PATH = "ncdu_cache.json"
USER_THRESHOLD_GB = 80 # Alert thresholds in GB
PARTITION_USAGE_THRESHOLD = 90  # in %
ENABLE_LEADERBOARD = True  # Enable or disable the leaderboard feature

# GPU usage monitoring
ENABLE_GPU_MONITORING = True
GPU_VRAM_THRESHOLD_PERCENT = 50
GPU_UTILIZATION_THRESHOLD_PERCENT = 8
AVAILABLE_SERVERS = ["sushi", "lackerli", "burger", "taco", "bagel"]

# === SLURM Trigger & Event Queue ===
ENABLE_SLURM_MONITORING = True
SLURM_SERVER = "curry"
SLURM_SUPPORTED_SERVERS = ["sushi", "lackerli", "taco"]  # Servers managed by SLURM (for GPU-SLURM alignment check)
SLURM_USAGE_LOG_FILE = "slurm_usage_log.jsonl"

# === GIF reply ===
ENABLE_GIF_REPLY = True  # master kill switch
GIF_REPLY_BACKEND = "siglip"  # "siglip" or "pepe"
GIF_REPLY_CHANNELS = []  # mention-only allowlist (channel IDs); merged with runtime overrides in bot_state.json
GIF_REPLY_ALWAYS_REPLY_CHANNELS = ["general", "interesting-papers", "gif-testing"]  # channel names; bot replies to every top-level user message AND on @mention
GIF_REPLY_RATE_LIMIT_PER_USER_HOUR = 5
GIF_REPLY_RATE_LIMIT_PER_CHANNEL_HOUR = 20
GIF_REPLY_RECENT_HISTORY = 50  # avoid repeating the last N gifs per channel
GIF_REPLY_SAMPLE_TOP_K = 10  # softmax-sample from the top-K safe candidates (1 = deterministic argmax)
GIF_REPLY_SAMPLE_TEMPERATURE = 0.05  # cosine scores are small; low temp keeps the sample close to top results
GIF_REPLY_INDEX_DIR = "/shared/0/projects/gif-reply-slack-bot/index"
GIF_REPLY_DATA_DIR = "/shared/0/projects/gif-reply-slack-bot"  # parent for cached gifs, downloads, hf cache
GIF_REPLY_SIGLIP_MODEL = "google/siglip-base-patch16-224"
GIF_REPLY_PEPE_CHECKPOINT = "/shared/2/projects/gif-reply/data/release/PEPE-model-checkpoint.pth"
GIF_REPLY_GIPHY_REFRESH_HOURS = 24

# === Paper monitor ===
ENABLE_PAPER_MONITOR = True
PAPERS_CHANNEL = "interesting-papers"  # channel name (no '#') or channel ID
PAPERS_LOG_FILE = "papers_log.jsonl"  # append-only; one row per unique paper
PAPERS_BACKFILL_CUTOFF = "2024-01-01"  # backfill stops at this date (UTC, inclusive)
PAPERS_HTTP_TIMEOUT = 15  # seconds, for OpenAlex / Semantic Scholar / page fetches
PAPERS_PDF_MAX_BYTES = 25 * 1024 * 1024  # skip PDFs larger than 25 MB
