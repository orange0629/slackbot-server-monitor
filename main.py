import os
import subprocess
import json
import logging
from logging.handlers import RotatingFileHandler
import pwd
from tqdm import tqdm
from multiprocessing import Process
import time
from collections import defaultdict, deque
from filelock import FileLock
from datetime import datetime, timedelta
from slack_sdk.errors import SlackApiError
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from typing import Dict, List, Optional, Union, Any, Set, Tuple, Callable, Iterable
from config import (
    SLACK_CHANNEL, SCAN_METHOD, HOME_DIR, NCDU_CACHE_PATH,
    USER_THRESHOLD_GB, PARTITION_USAGE_THRESHOLD, AVAILABLE_SERVERS,
    EXCLUDED_USERS, ADMIN_USERS, USAGE_LOG_FILE, MONITOR_LOG_FILE, ENABLE_HOME_MONITORING, ENABLE_LEADERBOARD,
    ENABLE_GPU_MONITORING, GPU_UTILIZATION_THRESHOLD_PERCENT, GPU_VRAM_THRESHOLD_PERCENT, SCHEDULER_STATE_FILE,
    ENABLE_SLURM_MONITORING, SLURM_SERVER, SLURM_SUPPORTED_SERVERS, SLURM_USAGE_LOG_FILE, BOT_STATE_FILE,
)
from config_secret import SLACK_TOKEN, SLACK_APP_TOKEN
import disk_scan
import re
import concurrent.futures

# Logging setup
MAX_MONITOR_LOG_BYTES = 5 * 1024 * 1024  # 5 MB
MONITOR_LOG_BACKUP_COUNT = 5

file_handler = RotatingFileHandler(
    MONITOR_LOG_FILE,
    maxBytes=MAX_MONITOR_LOG_BYTES,
    backupCount=MONITOR_LOG_BACKUP_COUNT,
    encoding="utf-8"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        file_handler,
        logging.StreamHandler()
    ]
)

app = App(token=SLACK_TOKEN)

def send_slack_alert(message: str, recipient: str, notify_admin: bool = True) -> None:
    try:
        app.client.chat_postMessage(channel=recipient, text=message)
        for admin in ADMIN_USERS:
            if "@" in recipient and notify_admin:
                app.client.chat_postMessage(channel=f"@{admin}", text=f"Sent alert to {recipient}: {message}")
        # app.client.chat_postMessage(channel="@leczhang", text=f"Sent alert to {recipient}: {message}")
        logging.info(f"Sent alert to {recipient}: {message}")
    except SlackApiError as e:
        logging.error(f"Slack API error when sending {message} to {recipient}: {e.response['error']}")

def append_usage_log(log_entry: Dict[str, Any]) -> None:
    with open(USAGE_LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    logging.info("Logged usage snapshot.")


def read_recent_log_lines(file_path: str, max_lines: int = 50) -> str:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            recent_lines = deque(f, maxlen=max_lines)
    except FileNotFoundError:
        logging.warning(f"Log file not found at {file_path}")
        return ""
    except Exception as e:
        logging.error(f"Failed to read recent log lines from {file_path}: {e}")
        return ""

    joined = "".join(recent_lines).strip()
    if not joined:
        return ""

    max_chars = 3500  # keep messages within Slack limits
    if len(joined) > max_chars:
        return joined[-max_chars:]
    return joined


def run_remote_command(cmd: str, server: str = "localhost", timeout: int = 10) -> Optional[str]:
    try:
        if server in ["localhost", None]:
            return subprocess.check_output(cmd, shell=True, text=True, timeout=timeout).strip()
        else:
            ssh_cmd = (
                f"ssh -o StrictHostKeyChecking=no "
                f"{server} \"{cmd}\""
            )
            return subprocess.check_output(ssh_cmd, shell=True, text=True, timeout=timeout).strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to run command on {server}: {e}")
        return None
    except subprocess.TimeoutExpired:
        logging.error(f"Command {cmd} on {server} timed out after {timeout} seconds")
        return None


def get_username_from_pid(pid: int, server: str = "localhost") -> str:
    try:
        result = run_remote_command(f"ps -o user= -p {pid}", server)
        return result if result is not None else "unknown"
    except:
        return "unknown"


def get_all_usernames(server: str = "localhost") -> dict:
    try:
        output = run_remote_command("ps -e -o pid=,user=", server)
    except subprocess.CalledProcessError:
        logging.error(f"Failed to get process list from {server}")
        return {}

    if not output:
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
    gpu_info_raw = run_remote_command(
        "nvidia-smi --query-gpu=index,uuid,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits",
        server
    )
    if not gpu_info_raw:
        return {}
    gpu_info_lines = gpu_info_raw.splitlines()

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
        proc_raw = run_remote_command(
            "nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits",
            server
        )
        proc_lines = proc_raw.splitlines() if proc_raw else []
    except subprocess.CalledProcessError:
        logging.warning("No compute processes found, skipping GPU-level alerts.")
        return {}

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
    """
    GPU underutilization check.
    - skip_alert=True: diagnostic mode — just returns message string, no state changes.
    - skip_alert=False (scheduled): two-strike alerting with per-user merging.
    """
    logging.info("=== GPU usage monitoring started ===")
    read_bot_state_from_file()
    concatenated_message = ""
    try:
        server_list = AVAILABLE_SERVERS if not local_only else ["localhost"]
        current_flags: Set[str] = set()
        user_alerts: Dict[str, List[str]] = defaultdict(list)

        for server in server_list:
            gpu_map = get_gpu_snapshot(server=server)
            for uuid, info in gpu_map.items():
                gpu_index = info["gpu_index"]
                util = info["util"]
                mem_total = info["mem_total"]
                mem_used = info["mem_used"]
                processes = info["processes"]

                if util < GPU_UTILIZATION_THRESHOLD_PERCENT and mem_used > mem_total * (GPU_VRAM_THRESHOLD_PERCENT / 100):
                    if not processes:
                        continue

                    top_user_proc = max(processes, key=lambda x: x[2])
                    username, pid, vram = top_user_proc

                    if skip_alert:
                        # Diagnostic mode: just collect messages, no state changes
                        concatenated_message += (
                            f":warning: GPU {gpu_index} on `{server}` is underutilized (utilization {util}%) "
                            f"but VRAM usage is {mem_used}/{mem_total} MiB. "
                            f"Top user: `{username}` (PID {pid}) using {vram} MiB.\n"
                        )
                    else:
                        # Skip users in GPU cooldown
                        if is_user_muted(username, "gpu_check"):
                            continue
                        # Two-strike logic
                        flag_key = f"{server}|{gpu_index}|{username}"
                        current_flags.add(flag_key)
                        detail = (
                            f"    •  GPU {gpu_index} on `{server}`: utilization {util}%, "
                            f"VRAM {mem_used}/{mem_total} MiB (PID {pid}, {vram} MiB)"
                        )
                        if flag_key in BOT_STATE["gpu_flags"]:
                            user_alerts[username].append(detail)
                        else:
                            BOT_STATE["gpu_flags"][flag_key] = datetime.now().isoformat()
                            logging.info(f"GPU Low Utilization flag (first strike): {flag_key}")

        if not skip_alert:
            # Clear flags for entries no longer underutilized
            stale_keys = [k for k in BOT_STATE["gpu_flags"] if k not in current_flags]
            for k in stale_keys:
                del BOT_STATE["gpu_flags"][k]

            # Send merged alerts per user, then set 12h cooldown
            for username, details in user_alerts.items():
                message = (
                    f":warning: Underutilized GPU alert for `{username}`. Please check if the job(s) are active:\n" + "\n".join(details)
                )
                send_slack_alert(message, recipient=f"@{username}")
                set_user_mute(username, "gpu_check", timedelta(hours=12))

    except Exception as e:
        logging.error(f"GPU check failed: {e}")

    # GPU-SLURM alignment check (also runs in diagnostic mode for info)
    try:
        alignment_msg = _check_gpu_slurm_alignment()
        if alignment_msg:
            concatenated_message += alignment_msg
    except Exception as e:
        logging.error(f"GPU-SLURM alignment check failed: {e}")

    persist_bot_state(["gpu_flags", "user_mutes"])
    logging.info("=== GPU usage monitoring completed ===\n")
    return concatenated_message



# --- Bot state (shared between processes via file persistence) ---
BOT_STATE: Dict[str, Any] = {
    "slurm_jobs": {},   # job_id(str) -> {"user": str, "name": str, "state": str, "last_seen": str, "opt_out": bool, "queued_time": str, "running_time": str|None}
    "gpu_flags": {},    # "server|gpu_idx|user" -> first_flag_iso (two-strike tracking)
    "user_mutes": {
        "slurm_notification": {},  # username -> expiry_iso (24h mute for slurm alerts)
        "gpu_check": {},           # username -> expiry_iso (12h cooldown after GPU alert)
        "slurm_gpu_mismatch": {},  # username -> expiry_iso (12h cooldown after mismatch alert)
    },
}


def _load_json_file(path: str) -> Optional[Dict[str, Any]]:
    """Load a JSON file. Return None if missing or corrupted."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        # The file may be partially written due to a crash.
        logging.warning(f"Json file is corrupted (JSONDecodeError): {path}")
        return None
    except Exception as e:
        logging.warning(f"Failed to read json file {path}: {e}")
        return None

def _atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    """Atomically write JSON by writing to a temp file then os.replace."""
    tmppath = f"{path}.tmp.{os.getpid()}"
    with open(tmppath, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmppath, path)

def persist_bot_state(updated_keys: Iterable[str]) -> None:
    """
    Merge-write BOT_STATE into BOT_STATE_FILE safely across processes.

    Behavior:
      1. Acquire a file lock.
      2. Load the latest on-disk state.
      3. Only overwrite top-level keys listed in updated_keys.
      4. If those keys are unchanged compared to disk, do not write.
      5. Otherwise, atomically write the merged JSON.

    This prevents lost updates between processes, and avoids unnecessary writes.
    """
    keys = list(dict.fromkeys(updated_keys))  # stable de-dup
    try:
        with FileLock(BOT_STATE_FILE + ".lock"):
            disk = _load_json_file(BOT_STATE_FILE) or {}

            # Check whether any of the requested keys actually changed.
            changed = False
            for k in keys:
                disk_val = disk.get(k, {})
                mem_val = BOT_STATE.get(k, {})
                if disk_val != mem_val:
                    changed = True
                    break

            if not changed:
                return

            # Merge: only override the specified top-level keys.
            for k in keys:
                disk[k] = BOT_STATE.get(k, {})

            _atomic_write_json(BOT_STATE_FILE, disk)

    except Exception as e:
        logging.error(f"Failed to persist bot state: {e}")


def read_bot_state_from_file() -> None:
    try:
        with FileLock(BOT_STATE_FILE + ".lock"):
            with open(BOT_STATE_FILE, "r") as f:
                data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return
    except Exception as e:
        logging.warning(f"Could not read bot state: {e}")
        return
    BOT_STATE["slurm_jobs"] = data.get("slurm_jobs", {})
    BOT_STATE["gpu_flags"] = data.get("gpu_flags", {})
    for mute_type in BOT_STATE["user_mutes"]:
        BOT_STATE["user_mutes"][mute_type] = data.get("user_mutes", {}).get(mute_type, {})


def is_user_muted(username: str, mute_type: str) -> bool:
    mutes = BOT_STATE["user_mutes"].get(mute_type, {})
    expiry = mutes.get(username)
    if not expiry:
        return False
    if datetime.now() < datetime.fromisoformat(expiry):
        return True
    # Expired, clean up
    del mutes[username]
    return False


def set_user_mute(username: str, mute_type: str, duration: timedelta) -> str:
    expiry = (datetime.now() + duration).isoformat()
    BOT_STATE["user_mutes"][mute_type][username] = expiry
    return expiry

def _fmt_duration(start_iso: str, end: datetime = None) -> str:
    """Format elapsed time between an ISO timestamp and now (or end) as e.g. '2h 15m'."""
    if end is None:
        end = datetime.now()
    secs = (end - datetime.fromisoformat(start_iso)).total_seconds()
    h, m = int(secs // 3600), int((secs % 3600) // 60)
    return f"{h}h {m}m" if h else f"{m}m"


def parse_squeue(server: str = "localhost") -> Optional[List[Dict[str, str]]]:
    """
    Returns list of jobs with fields: id, user, state, name, node_list, num_gpus.
    Uses a stable, machine-parsable format: %i|%u|%T|%j|%N|%b
    Returns None if the command failed (timeout, SSH error, etc.).
    Returns [] if the command succeeded but there are no jobs.
    """
    cmd = r"export PATH=$PATH:/usr/local/slurm/bin; squeue -h -o '%i|%u|%T|%j|%N|%b'"
    out = run_remote_command(cmd, server)
    if out is None:
        return None
    jobs = []
    if not out:
        return jobs
    for line in out.splitlines():
        parts = [p.strip() for p in line.split("|", 5)]
        if len(parts) != 6:
            continue
        jid, usr, state, name, node_list, tres = parts
        # Parse GPU count from tres-per-node (e.g. "gpu:2" or "gres/gpu:4")
        num_gpus = 0
        for part in tres.split(","):
            if "gpu" in part.lower() and ":" in part:
                try:
                    num_gpus = int(part.split(":")[-1])
                except ValueError:
                    pass
        jobs.append({"id": jid, "user": usr, "state": state, "name": name, "node_list": node_list, "num_gpus": num_gpus})
    return jobs


def _parse_gres_gpu_indices(gres_str: str) -> Set[int]:
    """Parse GPU indices from GRES string like 'gpu:A6000:1(IDX:6)' or 'gpu:2(IDX:0-1,4)'."""
    indices = set()
    m = re.search(r"\(IDX:([\d,\-]+)\)", gres_str)
    if not m:
        return indices
    for part in m.group(1).split(","):
        if "-" in part:
            lo, hi = part.split("-", 1)
            indices.update(range(int(lo), int(hi) + 1))
        else:
            indices.add(int(part))
    return indices


def parse_scontrol_jobs() -> List[Dict[str, Any]]:
    """
    Run `scontrol show jobid -d` on SLURM_SERVER to get all jobs with
    per-node GPU allocation details, CPU count, and memory.
    Returns list of dicts per job:
        {"job_id", "user", "num_cpus", "mem_mb",
         "nodes": [{"node": str, "gpu_indices": set[int]}, ...]}
    Only includes RUNNING jobs.
    """
    cmd = r"export PATH=$PATH:/usr/local/slurm/bin; scontrol show jobid -d"
    out = run_remote_command(cmd, SLURM_SERVER, timeout=15)
    if not out:
        return []

    results = []
    job_blocks = re.split(r"\n\s*\n", out)
    for block in job_blocks:
        if not block.strip():
            continue
        state_m = re.search(r"JobState=(\S+)", block)
        if not state_m or state_m.group(1) != "RUNNING":
            continue
        user_m = re.search(r"UserId=(\w+)", block)
        job_id_m = re.search(r"JobId=(\d+)", block)
        if not user_m or not job_id_m:
            continue

        # Parse CPU count
        num_cpus = 0
        cpus_m = re.search(r"NumCPUs=(\d+)", block)
        if cpus_m:
            num_cpus = int(cpus_m.group(1))

        # Parse memory (e.g. mem=515546M or mem=64G)
        mem_mb = 0
        mem_m = re.search(r"\bmem=(\d+)([MmGg])", block)
        if mem_m:
            val = int(mem_m.group(1))
            unit = mem_m.group(2).upper()
            mem_mb = val * 1024 if unit == "G" else val

        # Parse per-node GPU allocations
        nodes = []
        for line in block.splitlines():
            node_m = re.search(r"Nodes=(\S+)", line)
            gres_m = re.search(r"GRES=(\S*gpu\S*)", line)
            if node_m and gres_m:
                gpu_indices = _parse_gres_gpu_indices(gres_m.group(1))
                if gpu_indices:
                    nodes.append({"node": node_m.group(1), "gpu_indices": gpu_indices})

        results.append({
            "job_id": job_id_m.group(1),
            "user": user_m.group(1),
            "num_cpus": num_cpus,
            "mem_mb": mem_mb,
            "nodes": nodes,
        })
    
    return results


def _find_gpu_slurm_mismatches(servers: List[str] = None) -> Dict[Tuple[str, int, str], List[Tuple[int, int]]]:
    """
    Single pass: snapshot GPUs on servers, query scontrol, return mismatches.
    Returns {(server, gpu_index, username): [(pid, vram), ...]} for GPU
    processes that have no matching SLURM allocation.
    """
    if servers is None:
        servers = SLURM_SUPPORTED_SERVERS

    # Collect GPU processes concurrently across all servers
    gpu_processes: Dict[Tuple[str, int, str], List[Tuple[int, int]]] = defaultdict(list)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(servers)) as executor:
        future_to_server = {executor.submit(get_gpu_snapshot, server=s): s for s in servers}
        for future in concurrent.futures.as_completed(future_to_server):
            server = future_to_server[future]
            try:
                gpu_map = future.result()
            except Exception:
                continue
            for uuid, info in gpu_map.items():
                for username, pid, vram in info["processes"]:
                    gpu_processes[(server, info["gpu_index"], username)].append((pid, vram))

    if not gpu_processes:
        return {}

    # Collect SLURM allocations
    slurm_set: Set[Tuple[str, int, str]] = set()
    for alloc in parse_scontrol_jobs():
        for node_info in alloc["nodes"]:
            for idx in node_info["gpu_indices"]:
                slurm_set.add((node_info["node"], idx, alloc["user"]))

    return {key: procs for key, procs in gpu_processes.items() if key not in slurm_set}


def _check_gpu_slurm_alignment() -> str:
    """
    Detect GPU processes without matching SLURM jobs.
    Runs the check twice (with a short delay) and only acts on mismatches
    that appear in both passes. Returns diagnostic message.
    """
    try:
        first = _find_gpu_slurm_mismatches()
        if not first:
            return ""

        logging.info(f"GPU-SLURM mismatch candidates (first pass): {list(first.keys())}")
        time.sleep(1)

        # Re-check only the affected servers
        affected_servers = list({s for s, _, _ in first})
        second = _find_gpu_slurm_mismatches(servers=affected_servers)

        # Keep only mismatches present in both passes
        confirmed = {key: second[key] for key in first if key in second}
        if not confirmed:
            logging.info("GPU-SLURM mismatches not confirmed on re-check, clearing.")
            return ""

        logging.info(f"Confirmed GPU-SLURM mismatches: {list(confirmed.keys())}")

        # Alert per user
        user_details: Dict[str, List[str]] = defaultdict(list)
        for (server, gpu_idx, user), procs in confirmed.items():
            proc_info = ", ".join(f"PID {pid}; VRAM {vram} MiB" for pid, vram in procs)
            user_details[user].append(f"    •  GPU {gpu_idx} on `{server}` ({proc_info})")

        diagnostic = ""
        for username, details in user_details.items():
            msg = (
                f":warning: GPU usage without *SLURM* detected for `{username}`. Please ensure your jobs are submitted through *SLURM*.:\n"
                + "\n".join(details)
            )
            diagnostic += msg + "\n"
            if not is_user_muted(username, "slurm_gpu_mismatch"):
                send_slack_alert(msg, recipient=f"@{username}")
                set_user_mute(username, "slurm_gpu_mismatch", timedelta(hours=6))

        return diagnostic

    except Exception as e:
        logging.error(f"GPU-SLURM alignment check failed: {e}")
        return ""


def _slack_dm_when_new_slurm_job_detected(job: Dict[str, str]) -> None:
    job_id = job["id"]
    text = (
        f"👋 Detected your new Slurm job `{job_id}` ({job['name']}) now *{job['state']}*. "
        f"You will be notified on state changes. "
        f"If you do not want notifications for this job, click Don't track."
    )
    logging.info(f"Detected new Slurm job `{job_id}` ({job['name']}) now *{job['state']}*. Sending DM to user `{job['user']}`.")
    try:
        app.client.chat_postMessage(
            channel=f"@{job["user"]}",
            text=text,
            blocks=[
                {"type": "section", "text": {"type": "mrkdwn", "text": text}},
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "Don't track"},
                            "style": "danger",
                            "action_id": "slurm_dont_track",
                            "value": f"{job_id}|{job['user']}"
                        },
                    ]
                }
            ]
        )
    except SlackApiError as e:
        logging.error(f"Slack error prompting tracking for job {job_id}: {e.response.get('error')}")



def _log_slurm_job_usage(rec: Dict[str, Any], jid: str) -> None:
    now = datetime.now()
    running_time = rec.get("running_time")
    queued_time = rec.get("queued_time", now.isoformat())
    if running_time:
        run_hours = (now - datetime.fromisoformat(running_time)).total_seconds() / 3600
        wait_hours = (datetime.fromisoformat(running_time) - datetime.fromisoformat(queued_time)).total_seconds() / 3600
    else:
        run_hours = 0.0
        wait_hours = 0.0
    entry = {
        "user": rec["user"], "job_id": jid, "job_name": rec["name"],
        "queued_time": queued_time, "running_time": running_time,
        "end_time": now.isoformat(),
        "run_duration_hours": round(run_hours, 2),
        "wait_duration_hours": round(wait_hours, 2),
        "node_list": rec.get("node_list", ""),
        "num_gpus": rec.get("num_gpus", 0),
        "num_cpus": rec.get("num_cpus", 0),
        "mem_mb": rec.get("mem_mb", 0),
    }
    with open(SLURM_USAGE_LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def generate_slurm_usage_report(start_date: datetime, end_date: datetime, title: str = None) -> str:
    if title is None:
        title = f"📊 *Slurm Usage Report ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})*"

    user_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"run_hours": 0.0, "wait_hours": 0.0, "jobs": 0, "gpu_hours": 0.0})
    try:
        with open(SLURM_USAGE_LOG_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                end_time = datetime.fromisoformat(entry["end_time"])
                if start_date <= end_time < end_date:
                    user = entry["user"]
                    user_stats[user]["run_hours"] += entry.get("run_duration_hours", 0)
                    user_stats[user]["wait_hours"] += entry.get("wait_duration_hours", 0)
                    user_stats[user]["jobs"] += 1
                    user_stats[user]["gpu_hours"] += entry.get("run_duration_hours", 0) * entry.get("num_gpus", 1)
    except FileNotFoundError:
        return f"No usage data found (file `{SLURM_USAGE_LOG_FILE}` does not exist).", {}

    if not user_stats:
        return f"No Slurm job records for the selected period.", {}

    lines = [title, ""]
    lines.append("```")
    lines.append(f"{'User':<12} {'# Jobs':>6} {'Total Wait Time(h)':>19} {'Total Run Time(h)':>18} {'Total GPU Hours(h)':>19}")
    lines.append("-" * 76)
    for user, stats in sorted(user_stats.items(), key=lambda x: -x[1]["gpu_hours"]):
        lines.append(f"{user:<12} {stats['jobs']:>6} {stats['wait_hours']:>19.1f} {stats['run_hours']:>18.1f} {stats['gpu_hours']:>19.1f}")
    lines.append("```")
    return "\n".join(lines), user_stats


def _get_report_date_range(period: str) -> Tuple[datetime, datetime, str]:
    """Return (start_date, end_date, title) for a named period."""
    now = datetime.now()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "last_7_days":
        start = today - timedelta(days=7)
        end = today + timedelta(days=1)
        title = f"📊 *Slurm Usage Report — Last 7 Days ({start.strftime('%m/%d')} – {today.strftime('%m/%d')})*"
    elif period == "this_month":
        start = today.replace(day=1)
        end = (start + timedelta(days=32)).replace(day=1)
        title = f"📊 *Slurm Usage Report — {start.strftime('%B %Y')}*"
    elif period == "last_month":
        first_this = today.replace(day=1)
        end = first_this
        start = (first_this - timedelta(days=1)).replace(day=1)
        title = f"📊 *Slurm Usage Report — {start.strftime('%B %Y')}*"
    elif period == "this_year":
        start = today.replace(month=1, day=1)
        end = today.replace(month=12, day=31) + timedelta(days=1)
        title = f"📊 *Slurm Usage Report — {start.year}*"
    elif period == "all_time":
        start = datetime(2000, 1, 1)
        end = datetime(9999, 12, 31)
        title = "📊 *Slurm Usage Report — All Time*"
    else:
        start = today.replace(day=1)
        end = (start + timedelta(days=32)).replace(day=1)
        title = f"📊 *Slurm Usage Report — {start.strftime('%B %Y')}*"
    return start, end, title


def send_monthly_slurm_report() -> None:
    """Auto-send a fun Slurm usage report on the 1st of each month to the channel."""
    now = datetime.now()
    if now.day != 1:
        return
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # Generate last month's report
    end = today
    start = (today - timedelta(days=1)).replace(day=1)
    month_name = start.strftime("%B %Y")
    title = f"📊 *Monthly Slurm Usage Report — {month_name}*"
    report, user_stats = generate_slurm_usage_report(start, end, title)

    if not user_stats:
        return

    # Add fun flavor text
    lines = [report, ""]
    top_user = max(user_stats.items(), key=lambda x: x[1]["gpu_hours"])
    lines.append(f"🏆 Top GPU user: *{top_user[0]}* with {top_user[1]['gpu_hours']:.1f} GPU hours!")
    total_jobs = sum(s["jobs"] for s in user_stats.values())
    total_gpu_h = sum(s["gpu_hours"] for s in user_stats.values())
    lines.append(f"🔢 Total: {total_jobs} jobs, {total_gpu_h:.1f} GPU hours across {len(user_stats)} users")

    # Yearly easter egg: January 1st
    if now.month == 1:
        last_year = now.year - 1
        year_start = datetime(last_year, 1, 1)
        year_end = datetime(now.year, 1, 1)
        year_title = f"🎆 *{last_year} Year in Review*"
        year_report, year_stats = generate_slurm_usage_report(year_start, year_end, year_title)
        if year_stats:
            lines.append("")
            lines.append(f"🎉 Happy New Year! Here's a look back at {last_year}:")
            lines.append(year_report)
            year_top = max(year_stats.items(), key=lambda x: x[1]["gpu_hours"])
            year_total_gpu = sum(s["gpu_hours"] for s in year_stats.values())
            lines.append(f"👑 {last_year} GPU champion: *{year_top[0]}* with {year_top[1]['gpu_hours']:.1f} GPU hours!")
            lines.append(f"⚡ The cluster crunched {year_total_gpu:.0f} GPU hours total in {last_year}. Here's to an even more productive {now.year}!")

    try:
        app.client.chat_postMessage(channel=SLACK_CHANNEL, text="\n".join(lines))
        logging.info(f"Sent monthly Slurm report for {month_name} to {SLACK_CHANNEL}")
    except Exception as e:
        logging.error(f"Failed to send monthly Slurm report: {e}")


def poll_slurm_and_alert() -> None:
    """
    1) see new jobs → create records; DM once with opt-out buttons
    2) see state changes → notify (update running_time/node when RUNNING)
    3) see disappearances → log usage, notify completion, remove record
    Default behavior: tracked unless user clicked 'Don't track'.
    """
    try:
        read_bot_state_from_file()
        now = datetime.now()
        now_iso = now.isoformat()
        squeue_result = parse_squeue(server=SLURM_SERVER)
        if squeue_result is None:
            logging.warning("Skipping SLURM poll cycle: squeue command failed or timed out")
            return
        live = {j["id"]: j for j in squeue_result}
        tracked = BOT_STATE["slurm_jobs"]

        # 1) handle new jobs
        newly_running_jids = []
        for jid, job in live.items():
            if jid not in tracked:
                is_running = job["state"] == "RUNNING"
                tracked[jid] = {
                    "user": job["user"],
                    "name": job["name"],
                    "state": job["state"],
                    "last_seen": now_iso,
                    "opt_out": False,
                    "queued_time": now_iso,
                    "running_time": now_iso if is_running else None,
                    "node_list": job.get("node_list", ""),
                    "num_gpus": job.get("num_gpus", 0),
                    "num_cpus": 0,
                    "mem_mb": 0,
                }
                if is_running:
                    newly_running_jids.append(jid)
                if not is_user_muted(job["user"], "slurm_notification"):
                    _slack_dm_when_new_slurm_job_detected(job)
            else:
                # 2) handle state changes
                old_state = tracked[jid]["state"]
                new_state = job["state"]
                tracked[jid]["last_seen"] = now_iso
                # Update node_list whenever we see it (becomes available when running)
                if job.get("node_list"):
                    tracked[jid]["node_list"] = job["node_list"]
                if new_state != old_state:
                    if new_state == "RUNNING" and not tracked[jid].get("running_time"):
                        tracked[jid]["running_time"] = now_iso
                        newly_running_jids.append(jid)
                    if not tracked[jid].get("opt_out", False) and not is_user_muted(tracked[jid]["user"], "slurm_notification"):
                        extra = f" after waiting {_fmt_duration(tracked[jid]['queued_time'], now)}" if new_state == "RUNNING" and tracked[jid].get("queued_time") else ""
                        send_slack_alert(f"🔄 Your job `{jid}` ({tracked[jid]["name"]}) changed state: *{old_state}* → *{new_state}*{extra}", f"@{tracked[jid]["user"]}", notify_admin=False)
                tracked[jid]["state"] = new_state

        # Enrich newly running jobs with CPU/mem from scontrol
        if newly_running_jids:
            try:
                scontrol_data = {j["job_id"]: j for j in parse_scontrol_jobs()}
                for jid in newly_running_jids:
                    if jid in scontrol_data:
                        tracked[jid]["num_cpus"] = scontrol_data[jid]["num_cpus"]
                        tracked[jid]["mem_mb"] = scontrol_data[jid]["mem_mb"]
            except Exception as e:
                logging.warning(f"Failed to enrich jobs with scontrol data: {e}")

        # 3) handle gone jobs (in tracked but not live)
        gone_ids: List[str] = [jid for jid in tracked.keys() if jid not in live]
        for jid in gone_ids:
            rec = tracked.get(jid)
            if rec:
                _log_slurm_job_usage(rec, jid)
                if not rec.get("opt_out", False) and not is_user_muted(rec["user"], "slurm_notification"):
                    run_extra = f" after running {_fmt_duration(rec['running_time'], now)}" if rec.get("running_time") else ""
                    send_slack_alert(f"✅ Your job `{jid}` ({rec["name"]}) is now finished{run_extra}", f"@{rec["user"]}", notify_admin=False)
                del tracked[jid]

        persist_bot_state(["slurm_jobs"])
    except Exception as e:
        logging.error(f"Slurm poll failed: {e}")


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
        "interval": timedelta(minutes=30),
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
    },
    "slurm_poll": {
        "desc": "Poll Slurm squeue for new jobs and changes",
        "type": "interval",
        "interval": timedelta(seconds=45),
        "next_time": None,
        "enabled": ENABLE_SLURM_MONITORING,
        "function": poll_slurm_and_alert,
        "cool_down_interval": None,
        "last_run_time": None,
    },
    "monthly_slurm_report": {
        "desc": "Send monthly Slurm usage report to channel",
        "type": "fixed",
        "times": ["09:00"],
        "next_time": None,
        "enabled": ENABLE_SLURM_MONITORING,
        "function": send_monthly_slurm_report,
        "cool_down_interval": None,
        "last_run_time": None,
    },
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
                if "interval" in task and task["interval"] > timedelta(minutes=10): # Don't log too frequently
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
                    if "interval" in task and task["interval"] > timedelta(minutes=10): # Don't log too frequently
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
            "text": {"type": "plain_text", "text": "Slurm Status"},
            "action_id": "slurm_status"
        },
        {
            "type": "button",
            "text": {"type": "plain_text", "text": "Slurm Job History"},
            "action_id": "slurm_history"
        },
        {
            "type": "button",
            "text": {"type": "plain_text", "text": "My Home Usage"},
            "action_id": "home_usage"
        },
        {
            "type": "button",
            "text": {"type": "plain_text", "text": "Find Free GPU"},
            "action_id": "find_free_gpu"
        },
    ]

    # If admin, add button for all users
    if slack_user in ADMIN_USERS:
        base_elements.extend([
            {
                "type": "button",
                "text": {"type": "plain_text", "text": "All Home Usages (Admin Only)"},
                "action_id": "all_home_usage"
            },
            {
                "type": "static_select",
                "placeholder": {"type": "plain_text", "text": "Slurm Usage Report (Admin Only)"},
                "action_id": "slurm_usage_report",
                "options": [
                    {"text": {"type": "plain_text", "text": "Last 7 Days"}, "value": "last_7_days"},
                    {"text": {"type": "plain_text", "text": "This Month"}, "value": "this_month"},
                    {"text": {"type": "plain_text", "text": "Last Month"}, "value": "last_month"},
                    {"text": {"type": "plain_text", "text": "This Year"}, "value": "this_year"},
                    {"text": {"type": "plain_text", "text": "All Time"}, "value": "all_time"},
                ]
            },
            {
                "type": "button",
                "text": {"type": "plain_text", "text": "Bot Running Logs (Admin Only)"},
                "action_id": "recent_monitor_log"
            },
            {
                "type": "button",
                "text": {"type": "plain_text", "text": "Scanning Schedules (Admin Only)"},
                "action_id": "check_schedules"
            },
            # {
            #     "type": "button",
            #     "text": {"type": "plain_text", "text": "Bot State (Debug Only)"},
            #     "action_id": "bot_state"
            # }
        ])

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


@app.action("recent_monitor_log")
def handle_recent_monitor_log(ack, body, respond):
    ack()
    slack_user = body.get("user", {}).get("username", "unknown")
    if slack_user not in ADMIN_USERS:
        logging.warning(f"Unauthorized log access attempt by `{slack_user}`")
        respond(response_type="ephemeral", text="❗ You are not authorized to view monitor logs.")
        return

    logging.info(f"{slack_user} requested recent monitor log entries")
    try:
        log_snippet = read_recent_log_lines(MONITOR_LOG_FILE)
        if not log_snippet:
            respond(response_type="ephemeral", text="ℹ️ Monitor log is empty or unavailable.")
            return

        respond(
            response_type="ephemeral",
            text=f"🗒️ *Most recent monitor log entries:*\n```{log_snippet}```"
        )
    except Exception as e:
        logging.error(f"Error retrieving recent monitor logs: {e}")
        respond(response_type="ephemeral", text=f"❗ Error retrieving monitor logs: {str(e)}")


@app.action("slurm_usage_report")
def handle_slurm_usage_report(ack, body, action, respond):
    ack()
    slack_user = body.get("user", {}).get("username", "unknown")
    if slack_user not in ADMIN_USERS:
        respond(response_type="ephemeral", text="❗ You are not authorized to view this report.")
        return
    period = action.get("selected_option", {}).get("value", "this_month")
    logging.info(f"{slack_user} requested Slurm usage report ({period})")
    start, end, title = _get_report_date_range(period)
    report, _ = generate_slurm_usage_report(start, end, title)
    respond(response_type="ephemeral", text=report)


@app.action("bot_state")
def handle_bot_state(ack, body, respond):
    ack()
    slack_user = body.get("user", {}).get("username", "unknown")
    if slack_user not in ADMIN_USERS:
        respond(response_type="ephemeral", text="❗ You are not authorized to view bot state.")
        return
    logging.info(f"{slack_user} requested BOT_STATE")
    read_bot_state_from_file()
    state_str = json.dumps(BOT_STATE, indent=2, default=str)
    # Truncate if too long for Slack
    if len(state_str) > 3500:
        state_str = state_str[:3500] + "\n... (truncated)"
    respond(response_type="ephemeral", text=f"🤖 *BOT_STATE:*\n```{state_str}```")


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


@app.action("slurm_dont_track")
def handle_slurm_dont_track(ack, body, action, respond):
    ack()
    parts = action.get("value", "").split("|", 1)
    jid = parts[0]
    username = parts[1] if len(parts) > 1 else body.get("user", {}).get("username", "unknown")
    read_bot_state_from_file()
    rec = BOT_STATE["slurm_jobs"].get(jid)
    if rec:
        rec["opt_out"] = True
        persist_bot_state(["slurm_jobs"])
    logging.info(f"Slurm job `{jid}` opted out from tracking by user {username}")
    respond(
        replace_original=True,
        text=f"👌 Got it. I will not track job `{jid}`. Want to also mute trackers of new jobs for 24h?",
        blocks=[
            {"type": "section", "text": {"type": "mrkdwn", "text": f"👌 Got it. I will not track job `{jid}`. Want to also mute trackers of new jobs for 24h?"}},
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Mute for 24h"},
                        "style": "danger",
                        "action_id": "slurm_mute_24h",
                        "value": username
                    }
                ]
            }
        ]
    )


@app.action("slurm_mute_24h")
def handle_slurm_mute_24h(ack, body, action, respond):
    ack()
    username = action.get("value")
    read_bot_state_from_file()
    expiry = set_user_mute(username, "slurm_notification", timedelta(hours=24))
    persist_bot_state(["user_mutes"])
    logging.info(f"User `{username}` muted slurm notifications for 24h (until {expiry})")
    respond(replace_original=True, text=f"👌 Got it. You will not receive slurm notifications until {expiry[:16].replace('T', ' ')}.")


@app.action("slurm_status")
def handle_slurm_status(ack, body, respond):
    ack()
    respond(response_type="ephemeral", text=":loading: Fetching Slurm status...")
    slack_user = body.get("user", {}).get("username", "unknown")
    logging.info(f"{slack_user} requested Slurm status")
    try:
        jobs = parse_squeue(server=SLURM_SERVER)
        if jobs is None:
            respond(response_type="ephemeral", text="⚠️ Failed to reach the SLURM server. Please try again later.")
            return
        if not jobs:
            respond(response_type="ephemeral", text="✅ Slurm is empty. No queued or running jobs.")
            return

        scontrol_map = {}
        try:
            scontrol_map = {j["job_id"]: j for j in parse_scontrol_jobs()}
        except Exception:
            pass
        read_bot_state_from_file()
        now = datetime.now()

        by_user = defaultdict(list)
        for j in jobs:
            by_user[j["user"]].append(j)

        lines = []
        lines.append("🧮 *Per-user job counts*")
        for u in sorted(by_user.keys()):
            states = defaultdict(int)
            for j in by_user[u]:
                states[j["state"]] += 1
            summary = ", ".join(f"{k}:{v}" for k, v in sorted(states.items()))
            lines.append(f"• `{u}` — {len(by_user[u])} job(s) [{summary}]")
        lines.append("")
        lines.append("📋 *Current squeue*")
        lines.append("```")
        lines.append(f"{'JOBID':<8} {'USER':<10} {'STATE':<10} {'NODE':<10} {'GPU':>3} {'CPU':>4} {'MEM':>8} {'TIME':>10}  NAME")
        lines.append("-" * 85)
        for j in jobs[:200]:
            sc = scontrol_map.get(j["id"], {})
            num_cpus = sc.get("num_cpus", 0)
            mem_mb = sc.get("mem_mb", 0)
            mem_str = f"{mem_mb}M" if mem_mb else "-"
            node = j.get("node_list", "") or "-"
            tracked = BOT_STATE["slurm_jobs"].get(j["id"], {})
            if j["state"] == "RUNNING" and tracked.get("running_time"):
                time_str = _fmt_duration(tracked["running_time"], now)
            elif tracked.get("queued_time"):
                time_str = f"w {_fmt_duration(tracked['queued_time'], now)}"
            else:
                time_str = "-"
            lines.append(
                f"{j['id']:<8} {j['user']:<10} {j['state']:<10} {node:<10} {j.get('num_gpus', 0):>3} {num_cpus:>4} {mem_str:>8} {time_str:>10}  {j['name']}"
            )
        if len(jobs) > 200:
            lines.append(f"... ({len(jobs) - 200} more)")
        lines.append("```")
        respond(response_type="ephemeral", text="\n".join(lines))
    except Exception as e:
        logging.error(f"Slurm status error: {e}")
        respond(response_type="ephemeral", text=f"❗ Error retrieving Slurm status: {str(e)}")


@app.action("slurm_history")
def handle_slurm_history(ack, body, respond):
    ack()
    slack_user = body.get("user", {}).get("username", "unknown")
    logging.info(f"{slack_user} requested Slurm job history")
    respond(
        response_type="ephemeral",
        blocks=[
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "*Which job history would you like to see?*"}
            },
            {
                "type": "actions",
                "block_id": "slurm_history_selection",
                "elements": [
                    {"type": "button", "text": {"type": "plain_text", "text": "My Slurm History"}, "action_id": "slurm_history_mine", "value": "mine"},
                    {"type": "button", "text": {"type": "plain_text", "text": "All Slurm History"}, "action_id": "slurm_history_all", "value": "all"},
                ]
            }
        ]
    )


@app.action(re.compile(r"slurm_history_(mine|all)"))
def handle_slurm_history_view(ack, body, action, respond):
    ack()
    mine_only = action.get("value") == "mine"
    slack_user = body.get("user", {}).get("username", "unknown")
    respond(replace_original=True, text=":loading: Fetching job history...")
    logging.info(f"{slack_user} requested Slurm job history ({'mine' if mine_only else 'all'})")
    try:
        entries = []
        with open(SLURM_USAGE_LOG_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))
    except FileNotFoundError:
        respond(response_type="ephemeral", text="No job history found yet.")
        return

    if mine_only:
        entries = [e for e in entries if e.get("user") == slack_user]

    entries.sort(key=lambda e: e.get("end_time", ""), reverse=True)
    entries = entries[:25]

    label = "Mine" if mine_only else "All"
    lines = [f"📜 *Slurm Job History — {label} (last 25 finished jobs)*", ""]
    lines.append("```")
    lines.append(f"{'JOBID':<8} {'USER':<10} {'NAME':<16} {'NODE':<10} {'GPU':>3} {'CPU':>4} {'MEM':>8} {'WAIT':>7} {'RUN':>7}  ENDED")
    lines.append("-" * 100)
    for e in entries:
        mem_mb = e.get("mem_mb", 0)
        mem_str = f"{mem_mb}M" if mem_mb else "-"
        wait_h = e.get("wait_duration_hours", 0)
        run_h = e.get("run_duration_hours", 0)
        wait_str = f"{wait_h:.1f}h" if wait_h >= 1 else f"{wait_h * 60:.0f}m"
        run_str = f"{run_h:.1f}h" if run_h >= 1 else f"{run_h * 60:.0f}m"
        end_str = e.get("end_time", "")[:16].replace("T", " ")
        name = e.get("job_name", "")[:15]
        node = e.get("node_list", "") or "-"
        lines.append(
            f"{e.get('job_id', '?'):<8} {e.get('user', '?'):<10} {name:<16} {node:<10} {e.get('num_gpus', 0):>3} {e.get('num_cpus', 0):>4} {mem_str:>8} {wait_str:>7} {run_str:>7}  {end_str}"
        )
    lines.append("```")
    respond(response_type="ephemeral", text="\n".join(lines))


def start_slack_bot() -> None:
    handler = SocketModeHandler(app, SLACK_APP_TOKEN) # SLACK_APP_TOKEN_DICT[os.uname().nodename])
    handler.start()

if __name__ == "__main__":
    Process(target=start_slack_bot).start()
    Process(target=start_monitor_scheduler).start()
