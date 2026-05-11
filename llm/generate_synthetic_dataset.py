#!/usr/bin/env python3
"""
generate_synthetic_dataset.py
=============================
Plan-B synthetic dataset generator for Phi-3 fine-tuning.

Instead of parsing raw TTY logs, this script:
  1. Extracts attacker commands from Processed_Data.csv
     (source_cowrie == 1, event_type contains 'command').
  2. Matches each command against Cowrie's local response files
     (honeyfs/, txtcmds/, and hardcoded dynamic responses).
  3. Writes a JSONL dataset:
     {"instruction": "<command>", "output": "<simulated response>"}
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
COWRIE_DIR = BASE_DIR / "cowrie_responses"
HONEYFS_DIR = COWRIE_DIR / "honeyfs"
TXTCMDS_DIR = COWRIE_DIR / "txtcmds"
CSV_PATH = BASE_DIR.parent / "Processed_Data.csv"
OUTPUT_PATH = BASE_DIR / "synthetic_finetune_dataset.jsonl"
COMBINED_OUTPUT_PATH = BASE_DIR / "combined_finetune_dataset.jsonl"
TTY_DATASET_PATH = BASE_DIR / "fine_tune_dataset.jsonl"

HOSTNAME = "svr04"
KERNEL_NAME = "Linux"
KERNEL_VERSION = "3.2.0-4-amd64"
KERNEL_BUILD = "#1 SMP Debian 3.2.68-1+deb7u1"
HARDWARE = "x86_64"
OPERATING_SYSTEM = "GNU/Linux"


def _read_file(path: Path) -> str | None:
    for enc in ("utf-8", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, OSError):
            continue
    return None


VIRTUAL_FILES: dict[str, str] = {
    "/proc/uptime": "3852041.18 7662292.32",
    "/proc/loadavg": "0.08 0.02 0.01 1/120 523",
    "/proc/stat": "cpu  12345 678 9012 3456789 0 0 0 0 0 0",
    "/proc/self/status": "Name:\tbash\nState:\tS (sleeping)\nPid:\t420\nUid:\t0\t0\t0\t0\nGid:\t0\t0\t0\t0",
    "/etc/hosts.deny": "",
}


def _handle_file_reader(cmd_name: str, args: list[str]) -> str | None:
    raw = " ".join(args)
    raw = re.sub(r'\s*\d+\s*>\s*&?\S*', '', raw)   # 2>/dev/null, 2>&1
    raw = re.sub(r'\s*>\s*>?\s*\S+', '', raw)       # > file, >> file
    raw = raw.strip()
    try:
        cleaned_args = shlex.split(raw)
    except ValueError:
        cleaned_args = raw.split()

    file_args: list[str] = []
    skip_next = False
    for i, a in enumerate(cleaned_args):
        if skip_next:
            skip_next = False
            continue
        if a.startswith("-"):
            if a in ("-n", "-c"):
                skip_next = True
            continue
        if a.isdigit():
            continue
        file_args.append(a)

    if not file_args:
        return None  

    outputs: list[str] = []
    for fpath in file_args:
        rel = fpath.lstrip("/")
        local = HONEYFS_DIR / rel
        if local.is_file():
            content = _read_file(local)
            if content is not None:
                outputs.append(content.rstrip("\n"))
            else:
                outputs.append(f"{cmd_name}: {fpath}: Permission denied")
        elif fpath in VIRTUAL_FILES:
            outputs.append(VIRTUAL_FILES[fpath])
        else:
            outputs.append(f"{cmd_name}: {fpath}: No such file or directory")

    return "\n".join(outputs)


_TXTCMDS_MAP: dict[str, Path] = {}


def _build_txtcmds_map() -> None:
    if not TXTCMDS_DIR.is_dir():
        return
    for p in TXTCMDS_DIR.rglob("*"):
        if p.is_file():
            _TXTCMDS_MAP[p.name] = p
            rel = p.relative_to(TXTCMDS_DIR).as_posix()
            _TXTCMDS_MAP["/" + rel] = p


_build_txtcmds_map()


def _handle_txtcmd(base_cmd: str) -> str | None:
    path = _TXTCMDS_MAP.get(base_cmd)
    if path is None:
        return None
    content = _read_file(path)
    if content is not None:
        return content.rstrip("\n")
    return None


_CPUINFO = _read_file(HONEYFS_DIR / "proc" / "cpuinfo") or ""
_MEMINFO = _read_file(HONEYFS_DIR / "proc" / "meminfo") or ""
_PASSWD = _read_file(HONEYFS_DIR / "etc" / "passwd") or ""
_SHADOW = _read_file(HONEYFS_DIR / "etc" / "shadow") or ""
_UPTIME_SECONDS = "3852041.18 7662292.32"


def _uname_response(args: list[str]) -> str:
    if not args:
        return KERNEL_NAME

    opts = {
        "name": False, "node": False, "release": False,
        "version": False, "machine": False, "os": False,
    }
    flag_map = {
        "a": "__ALL__", "s": "name", "n": "node", "r": "release",
        "v": "version", "m": "machine", "p": "machine", "i": "machine",
        "o": "os",
    }

    for a in args:
        a = a.strip()
        if a.startswith("--"):
            long_map = {
                "all": "__ALL__", "kernel-name": "name", "nodename": "node",
                "kernel-release": "release", "kernel-version": "version",
                "machine": "machine", "processor": "machine",
                "hardware-platform": "machine", "operating-system": "os",
            }
            key = long_map.get(a[2:])
            if key == "__ALL__":
                for k in opts:
                    opts[k] = True
            elif key:
                opts[key] = True
        elif a.startswith("-"):
            for ch in a[1:]:
                key = flag_map.get(ch)
                if key == "__ALL__":
                    for k in opts:
                        opts[k] = True
                elif key:
                    opts[key] = True

    parts: list[str] = []
    if opts["name"]:
        parts.append(KERNEL_NAME)
    if opts["node"]:
        parts.append(HOSTNAME)
    if opts["release"]:
        parts.append(KERNEL_VERSION)
    if opts["version"]:
        parts.append(KERNEL_BUILD)
    if opts["machine"]:
        parts.append(HARDWARE)
    if opts["os"]:
        parts.append(OPERATING_SYSTEM)
    if not parts:
        parts.append(KERNEL_NAME)
    return " ".join(parts)


def _ifconfig_response() -> str:
    return (
        "eth0      Link encap:Ethernet  HWaddr aa:bb:cc:dd:ee:ff\n"
        "          inet addr:192.168.1.10  Bcast:192.168.1.255  Mask:255.255.255.0\n"
        "          inet6 addr: fe80::a00:27ff:fed4:1001/64 Scope:Link\n"
        "          UP BROADCAST RUNNING MULTICAST  MTU:1500  Metric:1\n"
        "          RX packets:143240 errors:0 dropped:0 overruns:0 frame:0\n"
        "          TX packets:98750 errors:0 dropped:0 overruns:0 carrier:0\n"
        "          collisions:0 txqueuelen:1000\n"
        "          RX bytes:98453621 (98.4 MB)  TX bytes:32548790 (32.5 MB)\n"
        "\n"
        "\n"
        "lo        Link encap:Local Loopback\n"
        "          inet addr:127.0.0.1  Mask:255.0.0.0\n"
        "          inet6 addr: ::1/128 Scope:Host\n"
        "          UP LOOPBACK RUNNING  MTU:65536  Metric:1\n"
        "          RX packets:67850 errors:0 dropped:0 overruns:0 frame:0\n"
        "          TX packets:67850 errors:0 dropped:0 overruns:0 carrier:0\n"
        "          collisions:0 txqueuelen:0\n"
        "          RX bytes:45320120 (45.3 MB)  TX bytes:45320120 (45.3 MB)"
    )


def _free_response(args: list[str]) -> str:
    fmt = "kb"
    for a in args:
        if "-m" in a:
            fmt = "mb"
        elif "-h" in a:
            fmt = "human"

    if fmt == "mb":
        return (
            "              total        used        free      shared  buff/cache   available\n"
            "Mem:           3959        1214         974          48        1770        2397\n"
            "Swap:          2048          68        1979"
        )
    elif fmt == "human":
        return (
            "              total        used        free      shared  buff/cache   available\n"
            "Mem:           3G          1G          974M         48M        1G          2G\n"
            "Swap:          2G          68M         1G"
        )
    else:
        return (
            "              total        used        free      shared  buff/cache   available\n"
            "Mem:        4054744     1214880      997740       49908     1842124     2455556\n"
            "Swap:       2097148       70632     2026516"
        )


def _w_response() -> str:
    return (
        " 14:32:07 up 44 days, 13:40,  1 user,  load average: 0.08, 0.02, 0.01\n"
        "USER     TTY      FROM              LOGIN@   IDLE   JCPU   PCPU WHAT\n"
        "root     pts/0    192.168.1.100     14:31    0.00s  0.03s  0.01s w"
    )


def _uptime_response() -> str:
    return " 14:32:07 up 44 days, 13:40,  1 user,  load average: 0.08, 0.02, 0.01"


def _whoami_response() -> str:
    return "root"


def _last_response() -> str:
    return (
        "root     pts/0        192.168.1.100    Wed Mar 12 14:31   still logged in\n"
        "root     pts/0        192.168.1.50     Tue Mar 11 09:22 - 11:45  (02:23)\n"
        "reboot   system boot  3.2.0-4-amd64    Mon Mar 10 00:52 - 14:32 (2+13:40)\n"
        "\n"
        "wtmp begins Mon Mar 10 00:52:15 2026"
    )


def _ping_response(args: list[str]) -> str:
    host = ""
    for a in args:
        if not a.startswith("-"):
            host = a
            break
    if not host:
        return (
            "Usage: ping [-LRUbdfnqrvVaA] [-c count] [-i interval] [-w deadline]\n"
            "            [-p pattern] [-s packetsize] [-t ttl] [-I interface or address]\n"
            "            [-M mtu discovery hint] [-S sndbuf]\n"
            "            [ -T timestamp option ] [ -Q tos ] [hop1 ...] destination"
        )
    ip = "93.184.216.34"
    return (
        f"PING {host} ({ip}) 56(84) bytes of data.\n"
        f"64 bytes from {host} ({ip}): icmp_seq=1 ttl=50 time=42.3 ms\n"
        f"64 bytes from {host} ({ip}): icmp_seq=2 ttl=50 time=43.1 ms\n"
        f"64 bytes from {host} ({ip}): icmp_seq=3 ttl=50 time=41.8 ms\n"
        f"\n"
        f"--- {host} ping statistics ---\n"
        f"3 packets transmitted, 3 received, 0% packet loss, time 907ms\n"
        f"rtt min/avg/max/mdev = 41.800/42.400/43.100/0.534 ms"
    )


def _ps_response(args: list[str]) -> str:
    if any("ef" in a or "aux" in a for a in args):
        return (
            "UID        PID  PPID  C STIME TTY          TIME CMD\n"
            "root         1     0  0 Mar10 ?        00:00:03 /sbin/init\n"
            "root         2     0  0 Mar10 ?        00:00:00 [kthreadd]\n"
            "root       318     1  0 Mar10 ?        00:00:01 /usr/sbin/sshd -D\n"
            "root       420   318  0 14:31 pts/0    00:00:00 -bash\n"
            "root       523   420  0 14:32 pts/0    00:00:00 ps -ef"
        )
    return (
        "  PID TTY          TIME CMD\n"
        "  420 pts/0    00:00:00 bash\n"
        "  523 pts/0    00:00:00 ps"
    )


def _ls_response(args: list[str]) -> str:
    has_la = any("-l" in a or "-a" in a for a in args)
    if has_la:
        return (
            "total 28\n"
            "drwxr-xr-x  2 root root 4096 Mar 10 00:52 .\n"
            "drwxr-xr-x 22 root root 4096 Mar 10 00:52 ..\n"
            "-rw-------  1 root root  570 Mar 10 00:52 .bash_history\n"
            "-rw-r--r--  1 root root 3106 Mar 10 00:52 .bashrc\n"
            "-rw-r--r--  1 root root  148 Mar 10 00:52 .profile"
        )
    return ".bash_history  .bashrc  .profile"


DYNAMIC_RESPONSES: dict[str, str | callable] = {
    "uname": lambda args: _uname_response(args),
    "hostname": HOSTNAME,
    "whoami": "root",
    "id": "uid=0(root) gid=0(root) groups=0(root)",
    "w": _w_response(),
    "uptime": _uptime_response(),
    "last": _last_response(),
    "lscpu": (
        "Architecture:          x86_64\n"
        "CPU op-mode(s):        32-bit, 64-bit\n"
        "Byte Order:            Little Endian\n"
        "CPU(s):                2\n"
        "On-line CPU(s) list:   0,1\n"
        "Thread(s) per core:    1\n"
        "Core(s) per socket:    2\n"
        "Socket(s):             1\n"
        "Vendor ID:             GenuineIntel\n"
        "CPU family:            6\n"
        "Model:                 23\n"
        "Model name:            Intel(R) Core(TM)2 Duo CPU     E8200  @ 2.66GHz\n"
        "Stepping:              6\n"
        "CPU MHz:               2133.304\n"
        "BogoMIPS:              4270.03\n"
        "L1d cache:             32K\n"
        "L1i cache:             32K\n"
        "L2 cache:              6144K"
    ),
    "nproc": "2",

    "ifconfig": _ifconfig_response(),
    "netstat": (
        "Active Internet connections (servers and established)\n"
        "Proto Recv-Q Send-Q Local Address           Foreign Address         State\n"
        "tcp        0      0 0.0.0.0:22              0.0.0.0:*               LISTEN\n"
        "tcp        0      0 192.168.1.10:22         192.168.1.100:51234     ESTABLISHED\n"
        "tcp6       0      0 :::22                   :::*                    LISTEN"
    ),
    "ping": lambda args: _ping_response(args),

    "ls": lambda args: _ls_response(args),
    "pwd": "/root",
    "cd": "",  
    "mkdir": "",  
    "which": lambda args: f"/usr/bin/{args[0]}" if args else "",
    "locate": lambda args: f"-bash: locate: command not found" if not args else "",
    "find": "",
    "ps": lambda args: _ps_response(args),
    "kill": "",
    "pkill": "",
    "nohup": "",
    "sleep": "",
    "crontab": lambda args: "no crontab for root" if any("-l" in a for a in args) else "",

    "rm": "",  
    "cp": "",
    "mv": "",
    "chmod": "",  
    "chown": "",
    "chattr": "",
    "lockr": "-bash: lockr: command not found",
    "touch": "",
    "tee": "",

    "adduser": "",
    "chpasswd": "",
    "passwd": "passwd: password updated successfully",
    "groups": "root",

    "wget": lambda args: _wget_response(args),
    "curl": lambda args: _curl_response(args),
    "tftp": "",
    "ftpget": "",
    "scp": "",

    "bash": "",
    "sh": "",
    "export": "",  
    "source": "",
    "eval": "",
    "echo": lambda args: " ".join(args).strip('"').strip("'"),
    "cat": None,  
    "head": None,
    "tail": None,
    "grep": "",
    "awk": "",
    "sed": "",
    "cut": "",
    "wc": "",
    "sort": "",
    "uniq": "",
    "tr": "",
    "base64": "",

    "apt": "Reading package lists... Done",
    "apt-get": "Reading package lists... Done",
    "yum": "",
    "pip": "",
    "service": "",
    "systemctl": "",
    "iptables": "",

    "free": lambda args: _free_response(args),
    "top": None,   
    "df": None,    
    "du": "",
    "tar": "",
    "unzip": "",
    "dd": "",
    "nc": "",
    "sudo": lambda args: get_cowrie_response(" ".join(args)) if args else "",
}


def _wget_response(args: list[str]) -> str:
    url = ""
    for a in args:
        if a.startswith("http://") or a.startswith("https://") or a.startswith("ftp://"):
            url = a
            break
        if not a.startswith("-"):
            url = a

    if not url:
        return (
            "wget: missing URL\n"
            "Usage: wget [OPTION]... [URL]..."
        )

    host = url.split("//")[-1].split("/")[0].split(":")[0] if "//" in url else url.split("/")[0]
    filename = url.rstrip("/").split("/")[-1] or "index.html"

    return (
        f"--2026-03-12 14:32:07--  {url}\n"
        f"Resolving {host}... 93.184.216.34\n"
        f"Connecting to {host}|93.184.216.34|:80... connected.\n"
        f"HTTP request sent, awaiting response... 200 OK\n"
        f"Length: 45321 (44K) [application/octet-stream]\n"
        f"Saving to: '{filename}'\n"
        f"\n"
        f"     0K .......... .......... .......... .......... ....  100%  128K=0.3s\n"
        f"\n"
        f"2026-03-12 14:32:08 (128 KB/s) - '{filename}' saved [45321/45321]"
    )


def _curl_response(args: list[str]) -> str:
    url = ""
    for a in args:
        if a.startswith("http://") or a.startswith("https://") or a.startswith("ftp://"):
            url = a
            break
        if not a.startswith("-"):
            url = a
    if not url:
        return (
            "curl: try 'curl --help' for more information"
        )
    return (
        f"  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current\n"
        f"                                 Dload  Upload   Total   Spent    Left  Speed\n"
        f"100 45321  100 45321    0     0   128k      0  0:00:00  0:00:00 --:--:--  128k"
    )


def _safe_split(cmd_str: str) -> tuple[str, list[str]]:
    cmd_str = cmd_str.strip()
    try:
        parts = shlex.split(cmd_str)
    except ValueError:
        parts = cmd_str.split()
    if not parts:
        return "", []
    base = os.path.basename(parts[0])
    return base, parts[1:]


def get_cowrie_response(command: str) -> str | None:
    command = command.strip()
    if not command:
        return None

    base_cmd, args = _safe_split(command)
    if not base_cmd:
        return None

    if base_cmd in ("cat", "head", "tail"):
        result = _handle_file_reader(base_cmd, args)
        if result is not None:
            return result
    txtcmd_result = _handle_txtcmd(base_cmd)
    if txtcmd_result is not None:
        return txtcmd_result

    handler = DYNAMIC_RESPONSES.get(base_cmd)
    if handler is None:
        full_cmd = command.split()[0] if command.split() else ""
        handler = DYNAMIC_RESPONSES.get(os.path.basename(full_cmd))

    if handler is not None:
        if callable(handler):
            return handler(args)
        return handler  

    return f"-bash: {base_cmd}: command not found"


def _strip_redirections(cmd: str) -> str:
    cmd = re.sub(r'\s*\d*>\s*&\d+', '', cmd)
    cmd = re.sub(r'\s*\d*>\s*>?\s*/\S+', '', cmd)
    cmd = re.sub(r'\s*\d*>\s*>?\s*\S+', '', cmd)
    return cmd.strip()


def _extract_subshell_commands(text: str) -> list[str]:
    cmds: list[str] = []
    i = 0
    while i < len(text):
        idx = text.find('$(', i)
        if idx == -1:
            break
        depth = 0
        start = idx + 2
        j = start
        while j < len(text):
            if text[j] == '(':
                depth += 1
            elif text[j] == ')':
                if depth == 0:
                    inner = text[start:j].strip()
                    if inner:
                        first = inner.split('|')[0].strip()
                        first = re.split(r'\s*(?:&&|;)\s*', first)[0].strip()
                        first = _strip_redirections(first)
                        if first:
                            cmds.append(first)
                    break
                depth -= 1
            j += 1
        i = j + 1
    return cmds


def handle_chained_command(full_command: str) -> str | None:
    full_command = full_command.strip()
    if not full_command:
        return None
    sub_commands = re.split(r'\s*(?:&&|;)\s*', full_command)

    outputs: list[str] = []
    for sub in sub_commands:
        sub = sub.strip()
        if not sub:
            continue

        if "|" in sub:
            sub = sub.split("|")[0].strip()

        sub = _strip_redirections(sub)
        if not sub:
            continue

        if re.match(r'^[A-Za-z_]\w*=', sub):
            subcmds = _extract_subshell_commands(sub)
            for sc in subcmds:
                result = get_cowrie_response(sc)
                if result and result.strip():
                    outputs.append(result)
            continue

        result = get_cowrie_response(sub)
        if result is not None and result != "":
            outputs.append(result)

    if not outputs:
        return ""
    return "\n".join(outputs)


import random
import re as _re


def _parse_login_failed_entries(df: "pd.DataFrame") -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    mask = (df["source_cowrie"] == 1) & (df["event_type"] == "ssh_login_failed")
    rows = df.loc[mask, "request_data"].dropna()

    seen: set[str] = set()
    for raw in rows:
        raw = str(raw)
        m = _re.search(r"\[([^/]+)/([^\]]+)\]", raw)
        if not m:
            continue
        username, password = m.group(1).strip(), m.group(2).strip()
        key = f"{username}/{password}"
        if key in seen:
            continue
        seen.add(key)

        instruction = f"su {username}"
        output = f"Password: \nsu: Authentication failure"
        entries.append({"instruction": instruction, "output": output})
    return entries


def _parse_file_download_entries(df: "pd.DataFrame") -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    mask = (df["source_cowrie"] == 1) & (df["event_type"] == "ssh_file_downloaded")
    rows_raw = df.loc[mask, "raw"].dropna()

    seen_urls: set[str] = set()
    for raw_str in rows_raw:
        try:
            event = json.loads(str(raw_str))
        except (json.JSONDecodeError, ValueError):
            continue

        url = event.get("url", "")
        destfile = event.get("destfile", "/tmp/malware")
        outfile = os.path.basename(destfile) if destfile else "malware"
        shasum = event.get("shasum", "")

        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        use_wget = len(seen_urls) % 2 == 0
        if use_wget:
            instruction = f"wget {url} -O /tmp/{outfile}"
            output = (
                f"--{HOSTNAME}--  {url}\n"
                f"Resolving {url.split('/')[2]}... connected.\n"
                f"HTTP request sent, awaiting response... 200 OK\n"
                f"Length: {random.randint(10000, 500000)} bytes\n"
                f"Saving to: '/tmp/{outfile}'\n\n"
                f"/tmp/{outfile}    100%[===================>]   saved"
            )
        else:
            instruction = f"curl -o /tmp/{outfile} {url}"
            output = (
                f"  % Total    % Received % Xferd  Average Speed\n"
                f"\n"
                f"100  {random.randint(10000, 500000)}  100  {random.randint(5000, 200000)}  "
                f"0     0  {random.randint(50000, 600000)}      0 --:--:-- --:--:-- --:--:-- {random.randint(50000, 600000)}"
            )
        entries.append({"instruction": instruction, "output": output})
    return entries


def main() -> None:
    print("=" * 60)
    print("Synthetic Cowrie Dataset Generator — Extended")
    print("=" * 60)
    print(f"\n[1/5] Reading CSV: {CSV_PATH.name}")
    try:
        df = pd.read_csv(
            CSV_PATH,
            usecols=["source_cowrie", "event_type", "request_data", "raw"],
            dtype={"source_cowrie": "Int64"},
            low_memory=False,
        )
    except Exception as e:
        print(f"ERROR: Failed to read CSV: {e}")
        sys.exit(1)

    print(f"\n[2/5] Generating command entries (no limit)...")
    mask = (df["source_cowrie"] == 1) & (
        df["event_type"].str.contains("command", case=False, na=False)
    )
    cowrie_cmds = df.loc[mask, "request_data"].dropna()
    cmd_counts = cowrie_cmds.value_counts()
    unique_cmds = cmd_counts.index.tolist()

    print(f"  -> Total Cowrie command rows: {len(cowrie_cmds):,}")
    print(f"  -> Unique commands to process: {len(unique_cmds):,}")

    cmd_entries: list[dict[str, str]] = []
    skipped = 0
    for i, cmd in enumerate(unique_cmds, 1):
        cmd_str = str(cmd).strip()
        if not cmd_str:
            skipped += 1
            continue
        try:
            response = handle_chained_command(cmd_str)
        except Exception as e:
            print(f"  WARN: '{cmd_str[:60]}': {e}")
            skipped += 1
            continue
        if response is None:
            skipped += 1
            continue
        cmd_entries.append({"instruction": cmd_str, "output": response})
        if i % 20 == 0 or i == len(unique_cmds):
            print(f"  -> Processed {i}/{len(unique_cmds)} commands...", end="\r")
    print(f"\n  -> Command entries: {len(cmd_entries):,}  (skipped {skipped})")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for entry in cmd_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"  -> Wrote {OUTPUT_PATH.name}: {len(cmd_entries):,} lines")
    print(f"\n[3/5] Generating file-download entries (ssh_file_downloaded)...")
    dl_entries = _parse_file_download_entries(df)
    print(f"  -> File-download entries: {len(dl_entries):,}")

    print(f"\n[4/5] Generating login-failed entries (ssh_login_failed)...")
    login_entries = _parse_login_failed_entries(df)
    print(f"  -> Login-failed entries: {len(login_entries):,}")
    print(f"\n[5/5] Merging all sources -> {COMBINED_OUTPUT_PATH.name}")
    tty_entries: list[dict[str, str]] = []
    if TTY_DATASET_PATH.exists():
        with open(TTY_DATASET_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        tty_entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        print(f"  -> TTY log entries loaded: {len(tty_entries):,}")
    else:
        print(f"  -> {TTY_DATASET_PATH.name} not found, skipping TTY data.")

    all_entries = cmd_entries + dl_entries + login_entries + tty_entries

    seen_instructions: set[str] = set()
    deduped: list[dict[str, str]] = []
    for entry in all_entries:
        key = entry.get("instruction", "").strip()
        if key not in seen_instructions:
            seen_instructions.add(key)
            deduped.append(entry)

    with open(COMBINED_OUTPUT_PATH, "w", encoding="utf-8") as f:
        for entry in deduped:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    combined_size = COMBINED_OUTPUT_PATH.stat().st_size

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  CSV commands:        {len(cmd_entries):,}")
    print(f"  File downloads:      {len(dl_entries):,}")
    print(f"  Login-failed:        {len(login_entries):,}")
    print(f"  TTY log entries:     {len(tty_entries):,}")
    print(f"  Total (raw):         {len(all_entries):,}")
    print(f"  After dedup:         {len(deduped):,}")
    print(f"  Output file:         {COMBINED_OUTPUT_PATH}")
    print(f"  Output size:         {combined_size:,} bytes")
    print("\n--- Sample entries ---")
    for entry in deduped[:5]:
        instr = entry['instruction'][:70]
        out = entry['output'][:80].replace('\n', '\\n')
        print(f'  CMD: {instr}')
        print(f'  OUT: {out}')
        print()


if __name__ == "__main__":
    main()
