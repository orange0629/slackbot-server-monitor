import os
import subprocess
import json
import logging
import pwd
from tqdm import tqdm
from multiprocessing import Process
import time
from filelock import FileLock
from datetime import datetime, timedelta
from slack_sdk.errors import SlackApiError
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from config import (
    SLACK_TOKEN, SLACK_APP_TOKEN_DICT, SLACK_CHANNEL, SCAN_METHOD, HOME_DIR, NCDU_CACHE_PATH,
    USER_THRESHOLD_GB, PARTITION_USAGE_THRESHOLD,
    EXCLUDED_USERS, ADMIN_USERS, USAGE_LOG_FILE, MONITOR_LOG_FILE, ENABLE_HOME_MONITORING, ENABLE_LEADERBOARD,
    ENABLE_GPU_MONITORING, GPU_UTILIZATION_THRESHOLD_PERCENT, GPU_VRAM_THRESHOLD_PERCENT, SCHEDULER_STATE_FILE
)
import disk_scan

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

def send_slack_alert(message, recipient):
    try:
        app.client.chat_postMessage(channel=recipient, text=message)
        app.client.chat_postMessage(channel="@leczhang", text=f"Sent alert to {recipient}: {message}")
        logging.info(f"Sent alert to {recipient}: {message}")
    except SlackApiError as e:
        logging.error(f"Slack API error when sending {message} to {recipient}: {e.response['error']}")

def append_usage_log(log_entry):
    with open(USAGE_LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    logging.info("Logged usage snapshot.")

def get_username_from_pid(pid):
    try:
        return pwd.getpwuid(os.stat(f"/proc/{pid}").st_uid).pw_name
    except Exception:
        return None

# --- Dispatch to scan method ---
def get_all_user_usages(scan_method="DU"):
    logging.info(f"Running disk usage scan using method: {scan_method}")
    if scan_method == "LOG":
        usage_log = disk_scan.read_last_jsonl_line(file_path=USAGE_LOG_FILE)
    elif scan_method == "NCDU":
        usage_log = disk_scan.scan_with_ncdu(home_dir=HOME_DIR, excluded_users=EXCLUDED_USERS, ncdu_cache_path=NCDU_CACHE_PATH)
        append_usage_log(usage_log)      
    elif scan_method == "DU":
        usage_log = disk_scan.scan_with_du(home_dir=HOME_DIR, excluded_users=EXCLUDED_USERS)
        append_usage_log(usage_log)
    elif scan_method == "FIND":
        usage_log = disk_scan.scan_with_find(home_dir=HOME_DIR, excluded_users=EXCLUDED_USERS)
        append_usage_log(usage_log)
    else:
        raise ValueError(f"Unknown scan method: {scan_method}")
    logging.info(f"Completed disk usage scan. Users found: {len(usage_log.get('usages', {}))}")
    return usage_log

# --- Check /home partition usage ---
def check_partition_usage(path):
    try:
        output = subprocess.check_output(["df", path], text=True).splitlines()
        if len(output) >= 2:
            usage_percent = int(output[1].split()[4].replace('%', ''))
            logging.info(f"Partition usage for {path}: {usage_percent}%")
            return usage_percent
    except Exception as e:
        logging.error(f"df error: {e}")
    return 0

def check_partition_usage_and_alert(skip_alert=False):
    home_usage_percent = check_partition_usage(HOME_DIR)
    if home_usage_percent >= PARTITION_USAGE_THRESHOLD:
        if not skip_alert:
            send_slack_alert(f":warning: `/home` partition usage is at {home_usage_percent}%, exceeding the threshold of {PARTITION_USAGE_THRESHOLD}%.", SLACK_CHANNEL)
        return True
    else:
        logging.info(f"/home partition usage is OK: {home_usage_percent}%")
        return False
    

def check_home_usage_and_alert():
    logging.info("=== Disk usage monitoring started ===")
    usage_log = get_all_user_usages(scan_method=SCAN_METHOD)
    usage_dict = usage_log.get("usages", {})

    for username, usage in usage_dict.items():
        if usage > USER_THRESHOLD_GB:
            send_slack_alert(f":warning: User `{username}` is using {usage:.2f} GB in `/home`, exceeding the threshold of {USER_THRESHOLD_GB} GB.", f"@{username}")
            # send_slack_alert(f":warning: User `{username}` is using {usage:.2f} GB in `/home`, exceeding the threshold of {USER_THRESHOLD_GB} GB.", SLACK_CHANNEL)

    home_usage_percent = check_partition_usage(HOME_DIR)
    if home_usage_percent >= PARTITION_USAGE_THRESHOLD:
        if usage_dict and ENABLE_LEADERBOARD:
            # Rank usage_dict
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


# --- Check GPU usage ---
def check_gpu_usage_and_alert(skip_alert=False):
    logging.info("=== GPU usage monitoring started ===")
    try:
        concatenated_message = ""
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
            return ""

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
        "times": ["00:00"],  # Multiple daily times
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
        "cool_down_interval": timedelta(hours=24),
        "last_run_time": None,
    }
}

def write_scheduler_state_to_file():
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


def read_scheduler_state_from_file():
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


def get_soonest_time_from_list(times):
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

def start_monitor_scheduler():
    logging.info("Starting monitoring scheduler...")
    now = datetime.now()
    read_scheduler_state_from_file()

    for name, task in scheduled_tasks.items():
        if not task["enabled"]:
            continue
        if task["type"] == "fixed":
            task["next_time"] = get_soonest_time_from_list(task["times"])
        elif task["type"] == "interval":
            task["next_time"] = now

    while True:
        now = datetime.now()
        read_scheduler_state_from_file()

        for name, task in scheduled_tasks.items():
            if not task["enabled"]:
                continue

            if now >= task["next_time"]:
                logging.info(f"Running task: {name}")
                alerted = False
                try:
                    alerted = task["function"]()
                    task["last_run_time"] = now
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
                "text": {"type": "mrkdwn", "text": f"*👋 Greetings from `{current_hostname}`! How can I help you today?*"}
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
    logging.info(f"{slack_user} requested GPU usage")
    current_hostname = os.uname().nodename
    try:
        output = subprocess.check_output(["nvidia-smi"], text=True)
    except Exception as e:
        output = f"Error: {str(e)}"
    
    nvidia_output = f"📊 GPU status on `{current_hostname}`:\n```{output}```"
    
    extra_message = check_gpu_usage_and_alert(skip_alert=True)
    if extra_message:
        nvidia_output += f"\n\n🔍 *GPU diagnostic info:*\n{extra_message}"
    else:
        nvidia_output += f"\n\n🔍 *GPU diagnostic info:*\nNo issues detected."
    
    respond(response_type="ephemeral", text=nvidia_output)


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
    lines = ["*⏰ Scheduled Monitor Tasks:*"]
    read_scheduler_state_from_file()

    for name, task in scheduled_tasks.items():
        if not task["enabled"]:
            lines.append(f"• *{task['desc']}* → ❌ *Disabled*")
            continue

        cooldown = task.get("cool_down_interval")
        cooldown_str = f"cool-down {cooldown}" if cooldown else ""
        schedule_desc = (
            f"(daily at {', '.join(task['times'])}) ({cooldown_str})" if task["type"] == "fixed"
            else f"(every {task['interval']}){cooldown_str}"
        )

        next_time = task.get("next_time")
        last_run = task.get("last_run_time")
        next_time_str = next_time.strftime("%Y-%m-%d %H:%M:%S") if next_time else "N/A"
        last_run_str = last_run.strftime("%Y-%m-%d %H:%M:%S") if last_run else "Never"

        lines.append(
            f"• *{task['desc']}*\n"
            f"  ├─ Schedule: {schedule_desc}\n"
            f"  ├─ Next Run: `{next_time_str}`\n"
            f"  └─ Last Run: `{last_run_str}`"
        )

    respond(response_type="ephemeral", text="\n".join(lines))


def start_slack_bot():
    handler = SocketModeHandler(app, SLACK_APP_TOKEN_DICT[os.uname().nodename])
    handler.start()

if __name__ == "__main__":
    Process(target=start_slack_bot).start()
    Process(target=start_monitor_scheduler).start()
