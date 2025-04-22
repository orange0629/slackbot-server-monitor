import os
import subprocess
import json
import logging
import pwd
from datetime import datetime
from tqdm import tqdm
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from config import (
    SLACK_TOKEN, SLACK_CHANNEL, SCAN_METHOD, HOME_DIR, NCDU_CACHE_PATH,
    USER_THRESHOLD_GB, PARTITION_USAGE_THRESHOLD,
    EXCLUDED_USERS, USAGE_LOG_FILE, MONITOR_LOG_FILE, ENABLE_HOME_MONITORING, ENABLE_LEADERBOARD,
    ENABLE_GPU_MONITORING, GPU_UTILIZATION_THRESHOLD_PERCENT, GPU_VRAM_THRESHOLD_PERCENT,
)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(MONITOR_LOG_FILE),
        logging.StreamHandler()
    ]
)

client = WebClient(token=SLACK_TOKEN)

def send_slack_alert(message, recipient):
    try:
        client.chat_postMessage(channel=recipient, text=message)
        logging.info(f"Sent alert to {recipient}: {message}")
    except SlackApiError as e:
        logging.error(f"Slack API error: {e.response['error']}")

def append_usage_log(username, usage_gb):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "username": username,
        "usage_gb": round(usage_gb, 2)
    }
    with open(USAGE_LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    logging.info(f"Logged usage: {username} - {usage_gb:.2f} GB")

def get_username_from_pid(pid):
    try:
        return pwd.getpwuid(os.stat(f"/proc/{pid}").st_uid).pw_name
    except Exception:
        return None

# --- Scan with 'find' ---
def scan_with_find():
    usage = {}
    for username in os.listdir(HOME_DIR):
        if username in EXCLUDED_USERS:
            logging.info(f"Skipping excluded user: {username}")
            continue
        user_path = os.path.join(HOME_DIR, username)
        if not os.path.isdir(user_path):
            continue
        try:
            output = subprocess.check_output(
                f'find {user_path} -type f -printf "%s\\n" | awk \'{{sum+=$1}} END {{print sum/1024/1024/1024}}\'',
                shell=True,
                text=True
            )
            usage[username] = float(output.strip())
        except Exception as e:
            logging.error(f"Error scanning {user_path}: {e}")
    return usage

# --- Scan with 'du' ---
def scan_with_du():
    usage = {}
    for username in tqdm(os.listdir(HOME_DIR)):
        if username in EXCLUDED_USERS:
            logging.info(f"Skipping excluded user: {username}")
            continue
        user_path = os.path.join(HOME_DIR, username)
        if not os.path.isdir(user_path):
            continue
        result = subprocess.run(
            ['du', '-s', user_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True
        )
        try:
            size_kb = int(result.stdout.strip().split()[0])
            usage[username] = size_kb / 1024 / 1024
            logging.info(f"Finish scanning {username}: {usage[username]:.2f} GB")
        except (IndexError, ValueError):
            logging.warning(f"Failed to parse du output for {username}")
    return usage

# --- Scan with 'ncdu' ---
def run_ncdu_scan(output_path, target_dir):
    try:
        subprocess.run(["ncdu", "-o", output_path, "-0", target_dir], check=True)
        logging.info(f"ncdu scan completed. Output saved to {output_path}")
    except Exception as e:
        logging.error(f"ncdu scan failed: {e}")

def scan_with_ncdu():
    run_ncdu_scan(NCDU_CACHE_PATH, HOME_DIR)
    usage = {}
    try:
        with open(NCDU_CACHE_PATH, "rb") as f:
            raw = f.read()
        json_data = json.loads(raw.split(b'\x00', 1)[1].decode())
        for entry in json_data.get("items", []):
            name = entry.get("name")
            if not name or name in EXCLUDED_USERS:
                logging.info(f"Skipping excluded user: {name}")
                continue
            if entry.get("asize"):
                usage_gb = entry["asize"] / 1024 / 1024 / 1024
                usage[name] = usage_gb
        logging.info("Parsed ncdu cache successfully.")
    except Exception as e:
        logging.error(f"Failed to parse ncdu cache: {e}")
    return usage

# --- Dispatch to scan method ---
def get_all_user_usages():
    if SCAN_METHOD == "NCDU":
        return scan_with_ncdu()
    elif SCAN_METHOD == "DU":
        return scan_with_du()
    elif SCAN_METHOD == "FIND":
        return scan_with_find()
    else:
        raise ValueError(f"Unknown scan method: {SCAN_METHOD}")

# --- Check /home partition usage ---
def check_partition_usage(path):
    try:
        output = subprocess.check_output(["df", path], text=True).splitlines()
        if len(output) >= 2:
            usage_percent = int(output[1].split()[4].replace('%', ''))
            return usage_percent
    except Exception as e:
        logging.error(f"df error: {e}")
    return 0

# --- Check GPU usage ---
def check_gpu_usage_and_alert():
    try:
        # Step 1: Get per-GPU summary info
        gpu_info_lines = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True
        ).strip().splitlines()

        gpu_map = {}  # uuid -> info
        for line in gpu_info_lines:
            gpu_index, uuid, mem_total, mem_used, util = line.split(",")
            gpu_index = int(gpu_index.strip())
            uuid = uuid.strip()
            gpu_map[uuid] = {
                "gpu_index": gpu_index,
                "mem_total": int(mem_total),
                "mem_used": int(mem_used),
                "util": int(util),
                "processes": []  # (username, pid, used_memory)
            }

        # Step 2: Query running GPU processes
        try:
            proc_lines = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"],
                text=True
            ).strip().splitlines()
        except subprocess.CalledProcessError:
            logging.warning("No compute processes found, skipping GPU-level alerts.")
            return

        for line in proc_lines:
            fields = [f.strip() for f in line.split(",")]
            if len(fields) < 3:
                continue
            pid, uuid, used_memory = fields
            pid = int(pid)
            used_memory = int(used_memory)
            uuid = uuid.strip()

            username = get_username_from_pid(pid)
            if not username or uuid not in gpu_map:
                continue
            gpu_map[uuid]["processes"].append((username, pid, used_memory))

        # Step 3: For each GPU, if low util and high mem, alert top user
        for uuid, info in gpu_map.items():
            gpu_index = info["gpu_index"]
            util = info["util"]
            mem_total = info["mem_total"]
            mem_used = info["mem_used"]
            processes = info["processes"]

            if util < GPU_UTILIZATION_THRESHOLD_PERCENT and mem_used > mem_total * (GPU_VRAM_THRESHOLD_PERCENT / 100):
                if not processes:
                    continue

                # Find the process with highest used_memory
                top_user_proc = max(processes, key=lambda x: x[2])  # (username, pid, used_memory)
                username, pid, vram = top_user_proc

                message = (
                    f":warning: GPU {gpu_index} on `{os.uname().nodename}` is underutilized (utilization {util}%) "
                    f"but VRAM usage is {mem_used}/{mem_total} MiB. "
                    f"Top user: `{username}` (PID {pid}) using {vram} MiB. "
                    f"Please check if the job is active."
                )
                send_slack_alert(message, recipient=f"@{username}")

    except Exception as e:
        logging.error(f"GPU check failed: {e}")


# --- Main script ---
def main():
    if ENABLE_GPU_MONITORING:
        logging.info("=== GPU usage monitoring started ===")
        check_gpu_usage_and_alert()
        logging.info("=== GPU usage monitoring completed ===\n")
    
    if ENABLE_HOME_MONITORING:
        logging.info("=== Disk usage monitoring started ===")

        usage_dict = get_all_user_usages()

        for username, usage in usage_dict.items():
            append_usage_log(username, usage)
            if usage > USER_THRESHOLD_GB:
                send_slack_alert(f":warning: User `{username}` is using {usage:.2f} GB in `/home`, exceeding the threshold of {USER_THRESHOLD_GB} GB.", f"@{username}")
                # send_slack_alert(f":warning: User `{username}` is using {usage:.2f} GB in `/home`, exceeding the threshold of {USER_THRESHOLD_GB} GB.", SLACK_CHANNEL)

        home_usage_percent = check_partition_usage(HOME_DIR)
        if home_usage_percent >= PARTITION_USAGE_THRESHOLD:
            if usage_dict and ENABLE_LEADERBOARD:
                # 排序 usage_dict
                sorted_usage = sorted(usage_dict.items(), key=lambda x: x[1], reverse=True)
                top_users = sorted_usage[:3]

                leaderboard_lines = [f"🏆 *Top 3 /home users by disk usage:*"]
                for i, (user, gb) in enumerate(top_users, 1):
                    leaderboard_lines.append(f"{i}. `{user}` - {gb:.2f} GB")

                leaderboard_text = "\n".join(leaderboard_lines)

                full_message = (
                    f":warning: `/home` partition usage is at {home_usage_percent}%, "
                    f"exceeding the threshold of {PARTITION_USAGE_THRESHOLD}%.\n\n"
                    f"{leaderboard_text}"
                )
                send_slack_alert(full_message, SLACK_CHANNEL)
            else:
                send_slack_alert(f":warning: `/home` partition usage is at {home_usage_percent}%, exceeding the threshold of {PARTITION_USAGE_THRESHOLD}%.", SLACK_CHANNEL)
        else:
            logging.info(f"/home partition usage is OK: {home_usage_percent}%")
        logging.info("=== Disk usage monitoring completed ===\n")

if __name__ == "__main__":
    main()
