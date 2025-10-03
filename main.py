import os
import subprocess
import json
import logging
import pwd
from tqdm import tqdm
from multiprocessing import Process
import time
from collections import defaultdict
from filelock import FileLock
from datetime import datetime, timedelta
from slack_sdk.errors import SlackApiError
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from typing import Dict, List, Optional, Union, Any, Set, Tuple, Callable
from config import (
    SLACK_TOKEN, SLACK_APP_TOKEN_DICT, SLACK_CHANNEL, SCAN_METHOD, HOME_DIR, NCDU_CACHE_PATH,
    USER_THRESHOLD_GB, PARTITION_USAGE_THRESHOLD, AVAILABLE_SERVERS,
    EXCLUDED_USERS, ADMIN_USERS, USAGE_LOG_FILE, MONITOR_LOG_FILE, ENABLE_HOME_MONITORING, ENABLE_LEADERBOARD,
    ENABLE_GPU_MONITORING, GPU_UTILIZATION_THRESHOLD_PERCENT, GPU_VRAM_THRESHOLD_PERCENT, SCHEDULER_STATE_FILE,
)
import disk_scan
import re
import concurrent.futures

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(MONITOR_LOG_FILE),
        logging.StreamHandler()
    ]
)

app = App(token=SLACK_TOKEN)

def send_slack_alert(message: str, recipient: str) -> None:
    try:
        app.client.chat_postMessage(channel=recipient, text=message)
        for admin in ADMIN_USERS:
            if "@" in recipient:
                app.client.chat_postMessage(channel=f"@{admin}", text=f"Sent alert to {recipient}: {message}")
        # app.client.chat_postMessage(channel="@leczhang", text=f"Sent alert to {recipient}: {message}")
        logging.info(f"Sent alert to {recipient}: {message}")
    except SlackApiError as e:
        logging.error(f"Slack API error when sending {message} to {recipient}: {e.response['error']}")

def append_usage_log(log_entry: Dict[str, Any]) -> None:
    with open(USAGE_LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    logging.info("Logged usage snapshot.")

def run_remote_command(cmd: str, server: str = "localhost") -> str:
    try:
        if server in ["localhost", None]:
            return subprocess.check_output(cmd, shell=True, text=True).strip()
        else:
            ssh_cmd = (
                f"ssh -o StrictHostKeyChecking=no "
                f"{server} \"{cmd}\""
            )
            return subprocess.check_output(ssh_cmd, shell=True, text=True).strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to run command on {server}: {e}")
        return ""

def get_username_from_pid(pid: int, server: str = "localhost") -> str:
    try:
        return run_remote_command(f"ps -o user= -p {pid}", server)
    except:
        return "unknown"


def get_all_usernames(server: str = "localhost") -> dict:
    try:
        output = run_remote_command("ps -e -o pid=,user=", server)
    except subprocess.CalledProcessError:
        logging.error(f"Failed to get process list from {server}")
        return {}

    pid_user_map = {}
    for line in output.strip().splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            pid, user = parts
            pid_user_map[int(pid)] = user
    return pid_user_map


def get_gpu_snapshot(server: str = "localhost") -> Dict[str, Dict[str, Any]]:
    # Step 1: Get per-GPU summary info
    gpu_info_lines = run_remote_command(
        "nvidia-smi --query-gpu=index,uuid,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits",
        server
    ).splitlines()

    gpu_map = {}  # uuid -> info
    for line in gpu_info_lines:
        gpu_index, uuid, mem_total, mem_used, util = line.split(",")
        gpu_map[uuid.strip()] = {
            "gpu_index": int(gpu_index.strip()),
            "mem_total": int(mem_total),
            "mem_used": int(mem_used),
            "util": int(util),
            "processes": []  # (username, pid, used_memory)
        }

    # Step 2: Query running GPU processes
    try:
        proc_lines = run_remote_command(
            "nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits",
            server
        ).splitlines()
    except subprocess.CalledProcessError:
        logging.warning("No compute processes found, skipping GPU-level alerts.")
        return ""

    pid_user_map = get_all_usernames(server)

    for line in proc_lines:
        fields = [f.strip() for f in line.split(",")]
        if len(fields) < 3:
            continue
        pid, uuid, used_memory = fields
        pid, uuid, used_memory = int(pid), uuid.strip(), int(used_memory)

        username = pid_user_map.get(pid, "unknown")
        if not username or uuid not in gpu_map:
            continue
        gpu_map[uuid]["processes"].append((username, pid, used_memory))

    return gpu_map


def generate_usage_report(gpu_map: Dict[str, Dict[str, Any]]) -> str:
    lines = []
    user_summary = defaultdict(lambda: {"gpus": set(), "mem": 0})

    for uuid, info in gpu_map.items():
        gpu_index = info["gpu_index"]
        for username, pid, used_mem in info["processes"]:
            user_summary[username]["gpus"].add(gpu_index)
            user_summary[username]["mem"] += used_mem

    lines.append("=" * 40)
    lines.append("📊 Per-user Summary")
    lines.append("-" * 40)
    for user, d in sorted(user_summary.items()):
        lines.append(f"{user:>12}: {len(d['gpus'])} GPU(s), {d['mem']:>6} MiB")
    lines.append("")

    lines.append("=" * 40)
    lines.append("💻 GPU Details")
    lines.append("-" * 40)
    for uuid, info in gpu_map.items():
        lines.append(f"GPU {info['gpu_index']}: {info['mem_used']:>5}/{info['mem_total']} MiB  "
                     f"{info['util']:>3}% util")
        if not info["processes"]:
            lines.append("   (no active processes)")
        else:
            for username, pid, used_mem in info["processes"]:
                lines.append(f"   PID {pid:>6}  {username:<10} {used_mem:>5} MiB")

    return "\n".join(lines)


# --- Dispatch to scan method ---
def get_all_user_usages(scan_method: str = "DU") -> Dict[str, Any]:
    logging.info(f"Running disk usage scan using method: {scan_method}")
    if scan_method == "LOG":
        usage_log = disk_scan.read_last_jsonl_line(file_path=USAGE_LOG_FILE)
    elif scan_method == "NCDU":
        usage_log = disk_scan.scan_with_ncdu(home_dir=HOME_DIR, excluded_users=[], ncdu_cache_path=NCDU_CACHE_PATH)
        append_usage_log(usage_log)      
    elif scan_method == "DU":
        usage_log = disk_scan.scan_with_du(home_dir=HOME_DIR, excluded_users=[])
        append_usage_log(usage_log)
    elif scan_method == "FIND":
        usage_log = disk_scan.scan_with_find(home_dir=HOME_DIR, excluded_users=[])
        append_usage_log(usage_log)
    else:
        raise ValueError(f"Unknown scan method: {scan_method}")
    logging.info(f"Completed disk usage scan. Users found: {len(usage_log.get('usages', {}))}")
    return usage_log

# --- Check /home partition usage ---
def check_partition_usage(path: str) -> int:
    try:
        output = subprocess.check_output(["df", path], text=True).splitlines()
        if len(output) >= 2:
            usage_percent = int(output[1].split()[4].replace('%', ''))
            logging.info(f"Partition usage for {path}: {usage_percent}%")
            return usage_percent
    except Exception as e:
        logging.error(f"df error: {e}")
    return 0

def check_partition_usage_and_alert(skip_alert: bool = False) -> bool:
    home_usage_percent = check_partition_usage(HOME_DIR)
    if home_usage_percent >= PARTITION_USAGE_THRESHOLD:
        if not skip_alert:
            df_output = subprocess.check_output(["df", "-h", HOME_DIR], text=True)
            send_slack_alert(f":warning: `/home` partition usage is at {home_usage_percent}%, exceeding the threshold of {PARTITION_USAGE_THRESHOLD}%.\n```{df_output.strip()}```", SLACK_CHANNEL)
        return True
    else:
        logging.info(f"/home partition usage is OK: {home_usage_percent}%")
        return False
    

def check_home_usage_and_alert() -> None:
    logging.info("=== Disk usage monitoring started ===")
    usage_log = get_all_user_usages(scan_method=SCAN_METHOD)
    usage_dict = usage_log.get("usages", {})

    for username, usage in usage_dict.items():
        if usage > USER_THRESHOLD_GB and username not in EXCLUDED_USERS:
            send_slack_alert(f":warning: User `{username}` is using {usage:.2f} GB in `/home`, exceeding the threshold of {USER_THRESHOLD_GB} GB.", f"@{username}")
    logging.info("=== Disk usage monitoring completed ===\n")


# --- Check GPU usage ---
def check_gpu_usage_and_alert(local_only: bool = False, skip_alert: bool = False) -> str:
    logging.info("=== GPU usage monitoring started ===")
    try:
        concatenated_message = ""
        server_list = AVAILABLE_SERVERS if not local_only else ["localhost"]
        for server in server_list:
            gpu_map = get_gpu_snapshot(server=server)
            # For each GPU, if low util and high mem, alert top user
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
                        f":warning: GPU {gpu_index} on `{server}` is underutilized (utilization {util}%) "
                        f"but VRAM usage is {mem_used}/{mem_total} MiB. "
                        f"Top user: `{username}` (PID {pid}) using {vram} MiB. "
                        f"Please check if the job is active."
                    )
                    concatenated_message += f"{message}\n"
                    if not skip_alert:
                        send_slack_alert(message, recipient=f"@{username}")

    except Exception as e:
        logging.error(f"GPU check failed: {e}")
    logging.info("=== GPU usage monitoring completed ===\n")
    return concatenated_message

scheduled_tasks = {
    "disk_scan": {
        "desc": "Scan `/home` disk usage of every user",
        "type": "fixed",  # Runs at fixed times of day
        "times": ["04:00"],  # Multiple daily times
        "next_time": None,  # Only the soonest upcoming run
        "enabled": ENABLE_HOME_MONITORING,
        "function": check_home_usage_and_alert,
        "cool_down_interval": None,
        "last_run_time": None,
    },
    "gpu_check": {
        "desc": "Check for underutilized GPUs",
        "type": "interval",  # Runs every X time
        "interval": timedelta(hours=12),
        "next_time": None,
        "enabled": ENABLE_GPU_MONITORING,
        "function": check_gpu_usage_and_alert,
        "cool_down_interval": None,
        "last_run_time": None,
    },
    "partition_check": {
        "desc": "Check `/home` partition usage",
        "type": "interval",
        "interval": timedelta(hours=1),
        "next_time": None,
        "enabled": True,
        "function": check_partition_usage_and_alert,
        "cool_down_interval": timedelta(hours=6),
        "last_run_time": None,
    }
}

def write_scheduler_state_to_file() -> None:
    state = {
        name: {
            "next_time": task["next_time"].isoformat() if task["next_time"] else None,
            "last_run_time": task["last_run_time"].isoformat() if task["last_run_time"] else None,
        }
        for name, task in scheduled_tasks.items()
    }
    try:
        with FileLock(SCHEDULER_STATE_FILE + ".lock"):
            with open(SCHEDULER_STATE_FILE, "w") as f:
                json.dump(state, f, indent=4)
    except Exception as e:
        logging.error(f"Failed to write scheduler state: {e}")


def read_scheduler_state_from_file() -> None:
    try:
        with FileLock(SCHEDULER_STATE_FILE + ".lock"):
            with open(SCHEDULER_STATE_FILE, "r") as f:
                state = json.load(f)
    except Exception as e:
        logging.warning(f"Could not read scheduler state: {e}")
        return

    for name, times in state.items():
        task = scheduled_tasks.get(name)
        if not task:
            continue
        for key in ["next_time", "last_run_time"]:
            if key in times and times[key]:
                try:
                    task[key] = datetime.fromisoformat(times[key])
                except ValueError:
                    logging.warning(f"Invalid {key} format for task `{name}`")


def get_soonest_time_from_list(times: List[str]) -> datetime:
    """Get the next datetime for any of the time strings today or tomorrow."""
    now = datetime.now()
    candidates = []
    for t_str in times:
        hour, minute = map(int, t_str.split(":"))
        run_time = datetime.combine(now.date(), datetime.min.time()) + timedelta(hours=hour, minutes=minute)
        if run_time < now:
            run_time += timedelta(days=1)
        candidates.append(run_time)
    return min(candidates)

def start_monitor_scheduler() -> None:
    logging.info("Starting monitoring scheduler...")
    now = datetime.now()
    read_scheduler_state_from_file()

    for name, task in scheduled_tasks.items():
        if not task["enabled"]:
            continue

    while True:
        now = datetime.now()
        read_scheduler_state_from_file()

        for name, task in scheduled_tasks.items():
            if not task["enabled"]:
                continue
            
            # Initialize if not initialized
            if not task["next_time"]:
                if task["type"] == "fixed":
                    task["next_time"] = get_soonest_time_from_list(task["times"])
                    logging.info(f"Next `{name}` scheduled at: {task['next_time']}")
                elif task["type"] == "interval":
                    task["next_time"] = now
                    logging.info(f"Next `{name}` scheduled at now!")
                write_scheduler_state_to_file()

            if now >= task["next_time"]:
                logging.info(f"Running task: {name}")
                alerted = False
                try:
                    task["last_run_time"] = now
                    alerted = task["function"]()
                    write_scheduler_state_to_file()
                except Exception as e:
                    logging.error(f"Error running {name}: {e}")

                if alerted and task.get("cool_down_interval"):
                    task["next_time"] = now + task["cool_down_interval"]
                    logging.info(f"Task `{name}` is on cooldown until: {task['next_time']}")
                    write_scheduler_state_to_file()
                elif task["type"] == "fixed":
                    task["next_time"] = get_soonest_time_from_list(task["times"])
                    logging.info(f"Next `{name}` scheduled at: {task['next_time']}")
                    write_scheduler_state_to_file()
                elif task["type"] == "interval":
                    task["next_time"] = now + task["interval"]
                    logging.info(f"Next `{name}` scheduled at: {task['next_time']}")
                    write_scheduler_state_to_file()

        time.sleep(60)

# @app.command("/nvidia-smi")
# def handle_gpu_command(ack, respond, command):
#     ack()
#     text = command.get("text", "").strip()
#     requested_server = text.split()[0] if text else ""
#     current_hostname = os.uname().nodename
#     logging.info(f"Received /nvidia-smi command: '{text}' from Slack.")

#     valid_hosts = ["sushi", "taco", "lackerli", "burger", "bagel", "curry"]
#     if requested_server not in valid_hosts:
#         respond(response_type="ephemeral", text="❗ Please specify a valid server, e.g., `/nvidia-smi sushi`")
#         return

#     if requested_server == current_hostname:
#         try:
#             output = subprocess.check_output(["nvidia-smi"], text=True)
#         except Exception as e:
#             output = f"Error: {str(e)}"
#         respond(response_type="ephemeral", text=f"✅ GPU status on `{current_hostname}`:\n```{output}```")

@app.command("/food-bot")
def handle_food_bot_command(ack, respond, command):
    ack()
    current_hostname = os.uname().nodename
    slack_user = command.get("user_name", "unknown")

    base_elements = [
        {
            "type": "button",
            "text": {"type": "plain_text", "text": "GPU Usage"},
            "action_id": "gpu_usage"
        },
        {
            "type": "button",
            "text": {"type": "plain_text", "text": "Find Free GPU"},
            "action_id": "find_free_gpu"
        },
        {
            "type": "button",
            "text": {"type": "plain_text", "text": "My Home Usage"},
            "action_id": "home_usage"
        },
        {
            "type": "button",
            "text": {"type": "plain_text", "text": "Scanning Schedules"},
            "action_id": "check_schedules"
        }
    ]

    # If admin, add button for all users
    if slack_user in ADMIN_USERS:
        base_elements.append({
            "type": "button",
            "text": {"type": "plain_text", "text": "All Home Usages (Admin Only)"},
            "action_id": "all_home_usage"
        })

    respond(
        response_type="ephemeral", 
        blocks=[
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*👋 Greetings! How can I help you today?*"}
            },
            {
                "type": "actions",
                "elements": base_elements
            }
        ]
    )


@app.action("gpu_usage")
def handle_gpu_usage(ack, body, respond):
    ack()
    slack_user = body.get("user", {}).get("username", "unknown")
    logging.info(f"{slack_user} requested GPU Usage")

    respond(
        response_type="ephemeral",
        blocks=[
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Select a server to check its GPU usage:*"
                }
            },
            {
                "type": "actions",
                "block_id": "gpu_server_selection",
                "elements": [
                    *[
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": server},
                            "action_id": f"gpu_server_{server}",
                            "value": server
                        }
                        for server in AVAILABLE_SERVERS + ["all_servers"]
                    ]
                ]
            }
        ]
    )


def generate_all_servers_gpu_summary(servers: List[str]) -> str:
    user_summary = defaultdict(lambda: {"gpus": set(), "mem": 0})
    per_server_gpu_status = {}

    def process_server(server: str):
        try:
            gpu_map = get_gpu_snapshot(server)
            per_server_gpu_status[server] = gpu_map
            for uuid, info in gpu_map.items():
                for username, pid, used_mem in info["processes"]:
                    user_summary[username]["gpus"].add((server, info["gpu_index"]))
                    user_summary[username]["mem"] += used_mem
        except Exception as e:
            logging.warning(f"Error retrieving GPU info from {server}: {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(servers)) as executor:
        executor.map(process_server, servers)

    lines = ["📊 *Summary of All Servers:*", ""]

    if not user_summary:
        lines.append("⚠️ No active GPU usage detected.")
    else:
        lines.append("*👤 Per-User Summary:*")
        for user, d in sorted(user_summary.items(), key=lambda x: -len(x[1]["gpus"])):
            gpu_count = len(d["gpus"])
            total_mem = d["mem"]
            servers_used = sorted(set(s for s, _ in d["gpus"]))
            lines.append(f"• `{user}`: {gpu_count} GPU(s) across {len(servers_used)} server(s) {servers_used}, using {total_mem} MiB")
        lines.append("")

    lines.append("*💻 Per-Server GPU Usage:*")
    lines.append("```")
    lines.append("=" * 40)
    for server in sorted(per_server_gpu_status.keys()):
        lines.append(f"\n🔹 `{server}`:")
        gpu_map = per_server_gpu_status[server]
        for uuid, info in sorted(gpu_map.items(), key=lambda x: x[1]["gpu_index"]):
            status_line = f"  • GPU {info['gpu_index']}: {info['mem_used']:>5}/{info['mem_total']} MiB | Util: {info['util']:>2}%"
            lines.append(status_line)
            if not info["processes"]:
                lines.append("      (idle)")
            else:
                for username, pid, used_mem in info["processes"]:
                    lines.append(f"      PID {pid:<6} {username:<10} {used_mem:>5} MiB")
    lines.append("=" * 40)
    lines.append("```")
    return "\n".join(lines)


@app.action(re.compile(r"gpu_server_(.+)"))
def handle_gpu_server_selection(ack, body, respond, context, action):
    ack()
    selected_server = action["value"]
    slack_user = body.get("user", {}).get("username", "unknown")
    logging.info(f"{slack_user} selected GPU server `{selected_server}`")
    respond(
        response_type="ephemeral",
        text=":loading: Loading your request. This may take a few seconds..."
    )

    if selected_server == "all_servers":
        summary = generate_all_servers_gpu_summary(AVAILABLE_SERVERS)
    else:
        try:
            output = generate_usage_report(get_gpu_snapshot(selected_server))
            header = f"📊 GPU status on `{selected_server}`:\n```{output}```"

            extra_message = check_gpu_usage_and_alert(local_only=True, skip_alert=True)
            diag = f"\n\n🔍 *GPU diagnostic info:*\n{extra_message or 'No issues detected.'}"

            summary = header + diag
        except Exception as e:
            summary = f"❗ Error retrieving GPU info from `{selected_server}`: {str(e)}"
            logging.error(summary)

    respond(response_type="ephemeral", text=summary)

@app.action("find_free_gpu")
def handle_find_freest_gpu(ack, body, respond):
    ack()
    slack_user = body.get("user", {}).get("username", "unknown")
    logging.info(f"{slack_user} requested freest GPU (top 10)")
    respond(
        response_type="ephemeral",
        text=":loading: Loading your request. This may take a few seconds..."
    )

    def get_all_gpu_frees_on_server(server: str) -> List[Tuple[str, int, int, int]]:
        """
        Returns a list of (server, gpu_index, free_mem_MiB, util_percent)
        """
        try:
            gpu_map = get_gpu_snapshot(server)
            result = []
            for info in gpu_map.values():
                free_mem = info["mem_total"] - info["mem_used"]
                result.append((server, info["gpu_index"], free_mem, info["util"]))
            return result
        except Exception as e:
            logging.warning(f"Error checking {server}: {e}")
            return []

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(AVAILABLE_SERVERS)) as executor:
            all_results = list(executor.map(get_all_gpu_frees_on_server, AVAILABLE_SERVERS))

        flattened_results = [gpu for server_result in all_results for gpu in server_result]
        if not flattened_results:
            respond(response_type="ephemeral", text="❗ Could not retrieve GPU info from any server.")
            return

        top_10 = sorted(flattened_results, key=lambda x: x[2], reverse=True)[:10]

        message_lines = ["🎯 *Top 10 Freest GPUs (by absolute free VRAM)*:"]
        for i, (server, gpu_index, free_mem, util) in enumerate(top_10, start=1):
            message_lines.append(
                f"{i:2d}. `{server}` GPU {gpu_index} → {free_mem:>5} MiB free | Util: {util:>2}%"
            )

        respond(response_type="ephemeral", text="\n".join(message_lines))

    except Exception as e:
        logging.error(f"Error while ranking freest GPUs: {e}")
        respond(response_type="ephemeral", text=f"❗ Error occurred while ranking freest GPUs: {str(e)}")

@app.action("home_usage")
def handle_home_usage(ack, body, respond):
    ack()
    slack_user = body.get("user", {}).get("username", "unknown")
    logging.info(f"{slack_user} requested their home usage")
    try:
        # 1. Get the overall usage of the /home partition
        df_output = subprocess.check_output(["df", "-h", HOME_DIR], text=True)
        home_disk_usage = "📊 *Overall `/home` Disk Usage:*\n```" + df_output.strip() + "```"

        # 2. Try to read the most recent record from the log file
        usage_log = get_all_user_usages(scan_method="LOG")
        timestamp = usage_log.get("timestamp", "N/A")
        usage_dict = usage_log.get("usages", {})

        # 3. Map the Slack username to the system username (assuming they are the same)
        user_usage_gb = usage_dict.get(slack_user)
        if user_usage_gb is not None:
            personal_usage = f"👤 *Your `/home` Usage (`{slack_user}`):* {user_usage_gb:.2f} GB according to disk scan at `{timestamp}`."
        else:
            personal_usage = f"❗ Could not find usage record for user `{slack_user}` in the latest scan at {timestamp}."
    except Exception as e:
        logging.error(f"Error while generating home usage report for `{slack_user}`: {e}")
        respond(response_type="ephemeral", text=f"Error retrieving home usage: {str(e)}")
    respond(response_type="ephemeral", text=f"{home_disk_usage}\n\n{personal_usage}")


@app.action("all_home_usage")
def handle_all_home_usage(ack, body, respond):
    ack()
    slack_user = body.get("user", {}).get("username", "unknown")
    logging.info(f"{slack_user} requested all home usages (admin only)")
    try:
        # 1. Overall /home usage (like `df -h /home`)
        df_output = subprocess.check_output(["df", "-h", HOME_DIR], text=True)
        home_disk_usage = "📊 *Overall `/home` Disk Usage:*\n```" + df_output.strip() + "```"

        # 2. Load latest usage log
        usage_log = get_all_user_usages(scan_method="LOG")
        timestamp = usage_log.get("timestamp", "unknown")
        usage_dict = usage_log.get("usages", {})

        # 3. Format all user usages
        if not usage_dict:
            respond(response_type="ephemeral", text=f"{home_disk_usage}\n\n" + "⚠️ No usage records found.")
            return

        sorted_users = sorted(usage_dict.items(), key=lambda x: x[1], reverse=True)
        usage_lines = [f"📋 *All `/home` Usages (as of `{timestamp}`):*"]
        for user, usage in sorted_users:
            usage_lines.append(f"• `{user}`: {usage:.2f} GB")

        # 4. Combine both parts
        full_message = f"{home_disk_usage}\n\n" + "\n".join(usage_lines)
        respond(response_type="ephemeral", text=full_message)

    except Exception as e:
        logging.error(f"Error while generating full home usage report: {e}")
        respond(response_type="ephemeral", text=f"Error generating report: {str(e)}")


@app.action("check_schedules")
def handle_check_schedules(ack, body, respond):
    ack()
    slack_user = body.get("user", {}).get("username", "unknown")
    logging.info(f"{slack_user} requested check schedules")
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"*⏰ Scheduled Monitor Tasks (current time: `{now_str}`):*"]
    read_scheduler_state_from_file()

    for name, task in scheduled_tasks.items():
        if not task["enabled"]:
            lines.append(f"• *{task['desc']}* → ❌ *Disabled*")
            continue

        cooldown = task.get("cool_down_interval")
        cooldown_str = f"(cool-down {cooldown})" if cooldown else ""
        schedule_desc = (
            f"(daily at {', '.join(task['times'])}) {cooldown_str}" if task["type"] == "fixed"
            else f"(every {task['interval']}) {cooldown_str}"
        )

        next_time = task.get("next_time")
        last_run = task.get("last_run_time")
        if next_time is None:
            next_time_str = "N/A"
        elif next_time <= datetime.now():
            next_time_str = "Running"
        else:
            next_time_str = next_time.strftime("%Y-%m-%d %H:%M:%S")
        last_run_str = last_run.strftime("%Y-%m-%d %H:%M:%S") if last_run else "Never"

        lines.append(
            f"• *{task['desc']}*\n"
            f"  ├─ Schedule: {schedule_desc}\n"
            f"  ├─ Next Run: `{next_time_str}`\n"
            f"  └─ Last Run: `{last_run_str}`"
        )

    respond(response_type="ephemeral", text="\n".join(lines))


def start_slack_bot() -> None:
    handler = SocketModeHandler(app, SLACK_APP_TOKEN_DICT[os.uname().nodename])
    handler.start()

if __name__ == "__main__":
    Process(target=start_slack_bot).start()
    Process(target=start_monitor_scheduler).start()
