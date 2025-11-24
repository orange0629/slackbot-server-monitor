# config.py

# Slack settings
SLACK_TOKEN = "" # Start with "xoxb-" for bot tokens
SLACK_APP_TOKEN = "" # Start with "xapp-" for app tokens
SLACK_CHANNEL = "#tech-support"  # or channel ID like "C12345678"

# Log and cache paths
USAGE_LOG_FILE = "usage_log.jsonl"
MONITOR_LOG_FILE = "monitor.log"
SCHEDULER_STATE_FILE = "scheduler_state.json" # For periodic monitoring

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
