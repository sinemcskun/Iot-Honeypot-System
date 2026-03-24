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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
COWRIE_DIR = BASE_DIR / "cowrie_responses"
HONEYFS_DIR = COWRIE_DIR / "honeyfs"
TXTCMDS_DIR = COWRIE_DIR / "txtcmds"
CSV_PATH = BASE_DIR / "Processed_Data.csv"
OUTPUT_PATH = BASE_DIR / "synthetic_finetune_dataset.jsonl"

# Cowrie default identity settings (matching uname.py defaults)
HOSTNAME = "svr04"
KERNEL_NAME = "Linux"
KERNEL_VERSION = "3.2.0-4-amd64"
KERNEL_BUILD = "#1 SMP Debian 3.2.68-1+deb7u1"
HARDWARE = "x86_64"
OPERATING_SYSTEM = "GNU/Linux"


# ---------------------------------------------------------------------------
# Helper: read a local file safely
# ---------------------------------------------------------------------------
def _read_file(path: Path) -> str | None:
    """Read a file with graceful encoding fallback. Returns None on failure."""
    for enc in ("utf-8", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, OSError):
            continue
    return None


# ---------------------------------------------------------------------------
# Virtual files: fake content for paths not physically in honeyfs but known
# to Cowrie (attackers frequently read these).
# ---------------------------------------------------------------------------
VIRTUAL_FILES: dict[str, str] = {
    "/proc/uptime": "3852041.18 7662292.32",
    "/proc/loadavg": "0.08 0.02 0.01 1/120 523",
    "/proc/stat": "cpu  12345 678 9012 3456789 0 0 0 0 0 0",
    "/proc/self/status": "Name:\tbash\nState:\tS (sleeping)\nPid:\t420\nUid:\t0\t0\t0\t0\nGid:\t0\t0\t0\t0",
    "/etc/hosts.deny": "",
}


# ---------------------------------------------------------------------------
# Tier 1 — File-reading commands: cat, head, tail
# ---------------------------------------------------------------------------
def _handle_file_reader(cmd_name: str, args: list[str]) -> str | None:
    """
    Handle `cat`, `head`, `tail` by looking up paths in honeyfs/.
    Falls back to VIRTUAL_FILES for known paths not in honeyfs.
    Returns a realistic response string, or None if unhandled.
    """
    # Rejoin args and strip redirections first (e.g., "2 > /dev/null")
    raw = " ".join(args)
    raw = re.sub(r'\s*\d+\s*>\s*&?\S*', '', raw)   # 2>/dev/null, 2>&1
    raw = re.sub(r'\s*>\s*>?\s*\S+', '', raw)       # > file, >> file
    raw = raw.strip()

    # Re-split after redirection removal
    try:
        cleaned_args = shlex.split(raw)
    except ValueError:
        cleaned_args = raw.split()

    # Strip flags (e.g. -n 10) and collect file paths
    file_args: list[str] = []
    skip_next = False
    for i, a in enumerate(cleaned_args):
        if skip_next:
            skip_next = False
            continue
        if a.startswith("-"):
            # Some flags take a value (e.g. head -n 5)
            if a in ("-n", "-c"):
                skip_next = True
            continue
        # Skip bare numeric args (leftover FD numbers from "2 > /dev/null")
        if a.isdigit():
            continue
        file_args.append(a)

    if not file_args:
        return None  # no file path -> can't simulate

    outputs: list[str] = []
    for fpath in file_args:
        # Normalise: remove leading / and map to honeyfs
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


# ---------------------------------------------------------------------------
# Tier 2 — Static txtcmds lookup
# ---------------------------------------------------------------------------
# Build a map of command basenames → txtcmds file paths.
# txtcmds directory structure mirrors /bin, /usr/bin, etc.
_TXTCMDS_MAP: dict[str, Path] = {}


def _build_txtcmds_map() -> None:
    """Walk txtcmds/ and map command basenames to their content files."""
    if not TXTCMDS_DIR.is_dir():
        return
    for p in TXTCMDS_DIR.rglob("*"):
        if p.is_file():
            _TXTCMDS_MAP[p.name] = p
            # Also register full Unix path (e.g. /usr/bin/lscpu)
            rel = p.relative_to(TXTCMDS_DIR).as_posix()
            _TXTCMDS_MAP["/" + rel] = p


_build_txtcmds_map()


def _handle_txtcmd(base_cmd: str) -> str | None:
    """Return static txtcmd output if the command exists in txtcmds/."""
    path = _TXTCMDS_MAP.get(base_cmd)
    if path is None:
        return None
    content = _read_file(path)
    if content is not None:
        return content.rstrip("\n")
    return None


# ---------------------------------------------------------------------------
# Tier 3 — Dynamic / fallback responses (hardcoded, mimicking Cowrie logic)
# ---------------------------------------------------------------------------
# These responses are derived from reading the actual Python command modules
# in cowrie_responses/commands/*.py and the honeyfs fake files.

# Pre-load commonly referenced honeyfs files for dynamic responses
_CPUINFO = _read_file(HONEYFS_DIR / "proc" / "cpuinfo") or ""
_MEMINFO = _read_file(HONEYFS_DIR / "proc" / "meminfo") or ""
_PASSWD = _read_file(HONEYFS_DIR / "etc" / "passwd") or ""
_SHADOW = _read_file(HONEYFS_DIR / "etc" / "shadow") or ""

# Uptime value (Cowrie returns seconds since "boot")
_UPTIME_SECONDS = "3852041.18 7662292.32"


def _uname_response(args: list[str]) -> str:
    """
    Simulate `uname` with the same flag-parsing logic as Cowrie's uname.py.
    Supports -a, -s, -n, -r, -v, -m, -p, -i, -o and combined flags.
    """
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
        # Positional args (like "2" in "uname -m 2") are silently ignored

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
    """Static ifconfig output matching Cowrie style."""
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
    """Simulate `free` output. Supports -m and -h flags."""
    # Based on honeyfs/proc/meminfo values
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
    """Simulate the `w` command."""
    return (
        " 14:32:07 up 44 days, 13:40,  1 user,  load average: 0.08, 0.02, 0.01\n"
        "USER     TTY      FROM              LOGIN@   IDLE   JCPU   PCPU WHAT\n"
        "root     pts/0    192.168.1.100     14:31    0.00s  0.03s  0.01s w"
    )


def _uptime_response() -> str:
    """Simulate `uptime` output."""
    return " 14:32:07 up 44 days, 13:40,  1 user,  load average: 0.08, 0.02, 0.01"


def _whoami_response() -> str:
    return "root"


def _last_response() -> str:
    """Simulate `last` output (Cowrie's last.py returns a simple table)."""
    return (
        "root     pts/0        192.168.1.100    Wed Mar 12 14:31   still logged in\n"
        "root     pts/0        192.168.1.50     Tue Mar 11 09:22 - 11:45  (02:23)\n"
        "reboot   system boot  3.2.0-4-amd64    Mon Mar 10 00:52 - 14:32 (2+13:40)\n"
        "\n"
        "wtmp begins Mon Mar 10 00:52:15 2026"
    )


def _ping_response(args: list[str]) -> str:
    """Simulate `ping` with 3 pings (Cowrie generates random ttl/ms)."""
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
    # Generate a deterministic fake IP from the hostname
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
    """Simulate `ps` output. Always shows a few fake processes."""
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
    """Simulate `ls`. Basic directory listing of honeyfs root."""
    # We just return a plausible listing; real Cowrie uses fs.pickle
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


# Master fallback dictionary: base command name → handler or static string.
# All handlers accept (args: list[str]) and return str.
DYNAMIC_RESPONSES: dict[str, str | callable] = {
    # --- System info ---
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

    # --- Network ---
    "ifconfig": _ifconfig_response(),
    "netstat": (
        "Active Internet connections (servers and established)\n"
        "Proto Recv-Q Send-Q Local Address           Foreign Address         State\n"
        "tcp        0      0 0.0.0.0:22              0.0.0.0:*               LISTEN\n"
        "tcp        0      0 192.168.1.10:22         192.168.1.100:51234     ESTABLISHED\n"
        "tcp6       0      0 :::22                   :::*                    LISTEN"
    ),
    "ping": lambda args: _ping_response(args),

    # --- File system ---
    "ls": lambda args: _ls_response(args),
    "pwd": "/root",
    "cd": "",  # silent success
    "mkdir": "",  # silent success
    "which": lambda args: f"/usr/bin/{args[0]}" if args else "",
    "locate": lambda args: f"-bash: locate: command not found" if not args else "",
    "find": "",

    # --- Process management ---
    "ps": lambda args: _ps_response(args),
    "kill": "",
    "pkill": "",
    "nohup": "",
    "sleep": "",
    "crontab": lambda args: "no crontab for root" if any("-l" in a for a in args) else "",

    # --- File manipulation (silent success / deceptive) ---
    "rm": "",  # Cowrie silently "deletes" files
    "cp": "",
    "mv": "",
    "chmod": "",  # silent success (Cowrie's chmod.py does nothing)
    "chown": "",
    "chattr": "",
    "lockr": "-bash: lockr: command not found",
    "touch": "",
    "tee": "",

    # --- User management ---
    "adduser": "",
    "chpasswd": "",
    "passwd": "passwd: password updated successfully",
    "groups": "root",

    # --- Downloaders (Cowrie fakes download activity) ---
    "wget": lambda args: _wget_response(args),
    "curl": lambda args: _curl_response(args),
    "tftp": "",
    "ftpget": "",
    "scp": "",

    # --- Shells / scripting ---
    "bash": "",
    "sh": "",
    "export": "",  # sets env vars silently
    "source": "",
    "eval": "",

    # --- Text / data processing ---
    "echo": lambda args: " ".join(args).strip('"').strip("'"),
    "cat": None,  # handled by Tier 1 (file reader)
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

    # --- Package managers (Cowrie fakes installations) ---
    "apt": "Reading package lists... Done",
    "apt-get": "Reading package lists... Done",
    "yum": "",
    "pip": "",

    # --- System services ---
    "service": "",
    "systemctl": "",
    "iptables": "",

    # --- Misc ---
    "free": lambda args: _free_response(args),
    "top": None,   # handled by txtcmds
    "df": None,    # handled by txtcmds
    "du": "",
    "tar": "",
    "unzip": "",
    "dd": "",
    "nc": "",
    "sudo": lambda args: get_cowrie_response(" ".join(args)) if args else "",
}


def _wget_response(args: list[str]) -> str:
    """Simulate Cowrie's wget response: show connection + save output."""
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

    # Extract hostname from URL
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
    """Simulate Cowrie's curl response."""
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


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------
def _safe_split(cmd_str: str) -> tuple[str, list[str]]:
    """
    Split a command string into (base_command, args).
    Uses shlex for robust parsing; falls back to simple split on failure.
    """
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
    """
    Core engine: determine what Cowrie would respond to the given command.

    Resolution order:
      1. File-readers (cat, head, tail) → honeyfs lookup
      2. Static txtcmds → verbatim file content
      3. Dynamic / fallback → hardcoded dictionary
    Returns None if the command cannot be simulated.
    """
    command = command.strip()
    if not command:
        return None

    base_cmd, args = _safe_split(command)
    if not base_cmd:
        return None

    # ---------- Tier 1: File readers ----------
    if base_cmd in ("cat", "head", "tail"):
        result = _handle_file_reader(base_cmd, args)
        if result is not None:
            return result
        # If no file paths parsed, fall through to other tiers

    # ---------- Tier 2: Static txtcmds ----------
    txtcmd_result = _handle_txtcmd(base_cmd)
    if txtcmd_result is not None:
        return txtcmd_result

    # ---------- Tier 3: Dynamic / fallback ----------
    handler = DYNAMIC_RESPONSES.get(base_cmd)
    if handler is None:
        # Check if the full path form is known (e.g. /usr/bin/wget)
        full_cmd = command.split()[0] if command.split() else ""
        handler = DYNAMIC_RESPONSES.get(os.path.basename(full_cmd))

    if handler is not None:
        if callable(handler):
            return handler(args)
        return handler  # static string

    # ---------- Unknown command ----------
    return f"-bash: {base_cmd}: command not found"


def _strip_redirections(cmd: str) -> str:
    """Remove shell redirections from a command string."""
    # 2>&1, 2>/dev/null, >/dev/null, >> file, etc.
    cmd = re.sub(r'\s*\d*>\s*&\d+', '', cmd)
    cmd = re.sub(r'\s*\d*>\s*>?\s*/\S+', '', cmd)
    cmd = re.sub(r'\s*\d*>\s*>?\s*\S+', '', cmd)
    return cmd.strip()


def _extract_subshell_commands(text: str) -> list[str]:
    """
    Extract commands from $(...) subshell captures in a string.
    Returns a flat list of commands found inside subshells.
    """
    cmds: list[str] = []
    # Find balanced $(...) — simple approach for single-level nesting
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
                        # Inner may itself contain pipes/chains — take first cmd
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
    """
    Handle chained commands separated by ; or &&.
    Pipes (|) are processed by keeping only the leftmost command.
    Redirections (>, >>, 2>, 2>&1) are stripped.
    Subshells ($(...)) are recursively processed.
    """
    full_command = full_command.strip()
    if not full_command:
        return None

    # Split on ; and &&  (but not inside quoted strings)
    # Simple heuristic: split on these delimiters outside quotes
    sub_commands = re.split(r'\s*(?:&&|;)\s*', full_command)

    outputs: list[str] = []
    for sub in sub_commands:
        sub = sub.strip()
        if not sub:
            continue

        # Handle pipes: only process the leftmost command
        if "|" in sub:
            sub = sub.split("|")[0].strip()

        # Strip output redirections
        sub = _strip_redirections(sub)
        if not sub:
            continue

        # Handle variable assignments (VAR=value or VAR=$(cmd))
        if re.match(r'^[A-Za-z_]\w*=', sub):
            # Extract any $() subshell commands inside the assignment
            subcmds = _extract_subshell_commands(sub)
            for sc in subcmds:
                result = get_cowrie_response(sc)
                if result and result.strip():
                    outputs.append(result)
            continue

        # Handle $() subshells embedded in a command (e.g., echo "$var")
        # Process the top-level command itself
        result = get_cowrie_response(sub)
        if result is not None and result != "":
            outputs.append(result)

    if not outputs:
        return ""
    return "\n".join(outputs)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("Synthetic Cowrie Dataset Generator (Plan B)")
    print("=" * 60)

    # --- Step 1: Extract commands from CSV ---
    print(f"\n[1/3] Reading CSV: {CSV_PATH.name}")
    try:
        df = pd.read_csv(
            CSV_PATH,
            usecols=["source_cowrie", "event_type", "request_data"],
            dtype={"source_cowrie": "Int64"},
            low_memory=False,
        )
    except Exception as e:
        print(f"ERROR: Failed to read CSV: {e}")
        sys.exit(1)

    # Filter for Cowrie command events
    mask = (df["source_cowrie"] == 1) & (
        df["event_type"].str.contains("command", case=False, na=False)
    )
    cowrie_cmds = df.loc[mask, "request_data"].dropna()

    # Get unique commands (preserving frequency ranking)
    cmd_counts = cowrie_cmds.value_counts()
    unique_cmds = cmd_counts.index.tolist()

    # Cap at 500 most frequent if there are more
    if len(unique_cmds) > 500:
        unique_cmds = unique_cmds[:500]

    print(f"  -> Total Cowrie command rows:  {len(cowrie_cmds):,}")
    print(f"  -> Unique commands extracted:  {len(unique_cmds)}")
    print(f"  -> Top 5: {unique_cmds[:5]}")

    # --- Step 2: Generate responses ---
    print(f"\n[2/3] Simulating Cowrie responses...")
    results: list[dict[str, str]] = []
    skipped = 0

    for i, cmd in enumerate(unique_cmds, 1):
        cmd_str = str(cmd).strip()
        if not cmd_str:
            skipped += 1
            continue

        try:
            response = handle_chained_command(cmd_str)
        except Exception as e:
            print(f"  WARN: Error processing command '{cmd_str[:60]}': {e}")
            skipped += 1
            continue

        if response is None:
            skipped += 1
            continue

        results.append({
            "instruction": cmd_str,
            "output": response,
        })

        if i % 10 == 0 or i == len(unique_cmds):
            print(f"  -> Processed {i}/{len(unique_cmds)} commands...", end="\r")

    print(f"\n  -> Generated: {len(results)} entries, Skipped: {skipped}")

    # --- Step 3: Write JSONL ---
    print(f"\n[3/3] Writing JSONL: {OUTPUT_PATH.name}")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    file_size = OUTPUT_PATH.stat().st_size
    print(f"  -> Wrote {len(results)} lines ({file_size:,} bytes)")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Input CSV:         {CSV_PATH.name}")
    print(f"  Cowrie commands:   {len(cowrie_cmds):,} rows -> {len(unique_cmds)} unique")
    print(f"  Dataset entries:   {len(results)}")
    print(f"  Output file:       {OUTPUT_PATH}")
    print(f"  Output size:       {file_size:,} bytes")

    # Show a few sample entries
    print("\n--- Sample entries ---")
    for entry in results[:5]:
        instr = entry["instruction"][:70]
        out = entry["output"][:80].replace("\n", "\\n")
        print(f'  CMD: {instr}')
        print(f'  OUT: {out}')
        print()


if __name__ == "__main__":
    main()
