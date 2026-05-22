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

# === Paper curator (daily agentic digest) ===
ENABLE_PAPER_CURATOR = True
PAPER_CURATOR_CHANNEL = "interesting-papers"     # name (no '#') or channel ID
PAPER_CURATOR_POST_TIME = "09:00"                # local server time
PAPER_CURATOR_WEEKDAYS = [0, 1, 2, 3, 4]         # Mon–Fri (datetime.weekday())
PAPER_CURATOR_TOP_K_TO_LLM = 30                  # global top-k by max-sim across members
PAPER_CURATOR_TOP_K_PER_MEMBER = 5               # plus this many per-member best matches
PAPER_CURATOR_MAX_MAIN_POST = 10
PAPER_CURATOR_MAX_TAGS_PER_MEMBER = 2
# Per-(member, theme) score threshold. The judge asks "would someone whose
# focus is THEME care about this paper?" for each interest line. A member is
# only @-tagged if at least one of their themes scores >= this on a 0-10 scale.
# Higher = stricter (fewer tags, more silence on weak fits).
PAPER_CURATOR_TAG_SCORE_THRESHOLD = 9

# Prestige-venue boost. Keyed by paper["source"] (the sources.yml feed id).
# Papers from these venues are (1) force-included past the bi-encoder gate so
# the LLM always judges them, and (2) given this many points added to every
# per-member judge score (0-10 scale, capped at 10) before the tag threshold
# and final sort are applied — so they surface higher and tag more readily.
# Tier 1 (flagship) gets a bigger bump than the tier-2 sister journals.
PAPER_CURATOR_VENUE_PRESTIGE = {
    "science": 3,
    "nature": 3,
    "pnas": 3,
    "science-advances": 2,
    "nature-human-behaviour": 2,
    "pnas-nexus": 2,
}

PAPER_CURATOR_BIENCODER = "BAAI/bge-small-en-v1.5"
PAPER_CURATOR_OLLAMA_HOST = "http://localhost:11434"
PAPER_CURATOR_OLLAMA_MODEL = "qwen3.6:35b-a3b"   # confirm via paper_curator.bench
PAPER_CURATOR_OLLAMA_FALLBACK = "gemma4:26b"
PAPER_CURATOR_PROFILE_REFRESH_DAYS = 7
PAPER_CURATOR_LAB_URL = "https://blablablab.si.umich.edu/"
PAPER_CURATOR_DRY_RUN = False                    # if True, scheduled runs print blocks instead of posting
PAPER_CURATOR_QUIET_DAY_NOTE = False             # if False, no message on empty days

# Remote vLLM judge (offload LLM step to a GPU box over SSH).
# When PAPER_CURATOR_USE_REMOTE is True, paper_curator dispatches the relevance
# judgment to PAPER_CURATOR_REMOTE_HOST instead of using local Ollama. Falls back
# to Ollama on any remote failure (no free GPU, ssh down, vllm crash, etc.).
PAPER_CURATOR_USE_REMOTE = True
PAPER_CURATOR_REMOTE_HOST = "burger.si.umich.edu"
PAPER_CURATOR_REMOTE_PYTHON = "/opt/anaconda/bin/python"  # interpreter on the remote that has vllm installed
PAPER_CURATOR_REMOTE_MODEL = "Qwen/Qwen3.5-4B"  # HF id; vLLM downloads + caches on first run
PAPER_CURATOR_REMOTE_MIN_GPU_FREE_GB = 16        # require this much free VRAM on the chosen GPU
PAPER_CURATOR_REMOTE_MAX_GPU_UTIL = 10           # require utilization <= this percent
PAPER_CURATOR_REMOTE_TIMEOUT = 1800              # seconds for the full remote run; must
                                                 # exceed model-load + generation wall time,
                                                 # which scales with the candidate count
                                                 # (a large backlog day is ~15 min)
# If no GPU meets the thresholds at dispatch time, poll instead of giving up.
PAPER_CURATOR_REMOTE_GPU_WAIT_TIMEOUT = 3 * 60 * 60  # max seconds to wait for a free GPU
PAPER_CURATOR_REMOTE_GPU_POLL_INTERVAL = 120         # seconds between polls while waiting
# Cap papers per remote vLLM invocation. Backlog/recovery days can balloon the
# candidate set (a missed week => hundreds of papers, ~10k judge prompts);
# splitting into batches keeps each remote call well under REMOTE_TIMEOUT and
# lets a failed batch be skipped without losing the rest.
PAPER_CURATOR_REMOTE_BATCH_SIZE = 120
# If the SSH client times out mid-run, the remote keeps going (it holds a flock
# and writes to the shared FS). Poll this many extra seconds for its output
# before giving up on the batch.
PAPER_CURATOR_REMOTE_SALVAGE_GRACE = 600

# Shared root for paper_curator runtime artifacts (logs, embeddings cache,
# scraped profiles, per-run inputs/outputs). Visible from both this host and
# burger.
PAPER_CURATOR_DATA_DIR = "/shared/6/projects/food-bot"

# Shared HuggingFace model cache. Lives on a separate volume optimized for
# large model weights so vLLM downloads on burger and sentence-transformers
# downloads here both land in one place.
PAPER_CURATOR_HF_HOME = "/shared/4/models"
