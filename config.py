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
ADMIN_USERS = ["jurgens", "leczhang", "elealin"]

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
SLURM_USAGE_LOG_FILE = "slurm_usage_log.jsonl"

# === LLM Agent (via @mention) ===
ENABLE_LLM_AGENT = True
LLM_MODEL = "Qwen/Qwen3-8B"
LLM_AGENT_JOB_DIR = "/tmp/slack_agent_jobs"
LLM_SLURM_PARTITION = "gpu"
LLM_SLURM_GPUS = 1
LLM_SLURM_CPUS = 4
LLM_SLURM_MEM_GB = 32
LLM_SLURM_TIME_LIMIT = "6:00:00"
LLM_SYSTEM_PROMPT = (
    "You are a helpful AI assistant embedded in a Slack workspace. "
    "Answer questions concisely and helpfully based on the conversation context provided. "
    "Use markdown formatting where appropriate."
)
LLM_MAX_NEW_TOKENS = 1024
