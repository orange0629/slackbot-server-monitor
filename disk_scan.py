import os
import subprocess
import json
import logging
from tqdm import tqdm
from datetime import datetime

# --- Scan with 'find' ---
def scan_with_find(home_dir, excluded_users=[]):
    usage = {}
    for username in os.listdir(home_dir):
        if username in excluded_users:
            logging.info(f"Skipping excluded user: {username}")
            continue
        user_path = os.path.join(home_dir, username)
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
    usage_log = {
        "timestamp": datetime.now().isoformat(),
        "usages": {user: round(usage, 2) for user, usage in usage.items()}
        }
    return usage_log


# --- Scan with 'du' ---
def scan_with_du(home_dir, excluded_users=[]):
    usage = {}
    for username in tqdm(os.listdir(home_dir)):
        if username in excluded_users:
            logging.info(f"Skipping excluded user: {username}")
            continue
        user_path = os.path.join(home_dir, username)
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
    usage_log = {
        "timestamp": datetime.now().isoformat(),
        "usages": {user: round(usage, 2) for user, usage in usage.items()}
        }
    return usage_log

# --- Scan with 'ncdu' ---
def get_dsize_from_ncdu_node(node):
    if isinstance(node, list):
        total = node[0].get("dsize", 0)
        for child in node[1:]:
            total += get_dsize_from_ncdu_node(child)
        return total
    elif isinstance(node, dict):
        return node.get("dsize", 0)
    return 0

def scan_with_ncdu(home_dir, excluded_users=[], ncdu_cache_path="ncdu_cache.json"):
    try:
        subprocess.run(["ncdu", "-o", ncdu_cache_path, home_dir], check=True)
        logging.info(f"ncdu scan completed. Output saved to {ncdu_cache_path}")
    except Exception as e:
        logging.error(f"ncdu scan failed: {e}")
    usage = {}
    try:
        with open(ncdu_cache_path, "r") as f:
            content = json.load(f)
        home_items = content[3]

        for entry in home_items:
            if not isinstance(entry, list) or not entry:
                continue
            metadata = entry[0]
            username = metadata.get("name")
            if not username or username in excluded_users:
                continue
            total_dsize = get_dsize_from_ncdu_node(entry)
            usage_gb = total_dsize / 1024 / 1024 / 1024
            usage[username] = usage_gb
            logging.info(f"Finish scanning {username}: {usage[username]:.2f} GB")

        logging.info("Parsed ncdu JSON cache successfully.")
    except Exception as e:
        logging.error(f"Failed to parse ncdu cache: {e}")

    usage_log = {
        "timestamp": datetime.now().isoformat(),
        "usages": {user: round(gb, 2) for user, gb in usage.items()}
    }
    return usage_log

def read_last_jsonl_line(file_path):
    try:
        with open(file_path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            if end == 0:
                return {}

            # Step backward and collect lines until a non-empty one is found
            while end > 0:
                end -= 1
                f.seek(end)
                if f.read(1) == b'\n':
                    line = f.readline().decode().strip()
                    if line:
                        return json.loads(line)

            # Catch the first (and maybe only) line if no newline at end
            f.seek(0)
            line = f.readline().decode().strip()
            if line:
                return json.loads(line)

        return {}

    except Exception as e:
        logging.error(f"Error reading log from {file_path}: {e}")
        return {}

