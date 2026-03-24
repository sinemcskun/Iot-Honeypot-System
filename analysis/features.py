import math
import re
import numpy as np
import pandas as pd


class SessionFeatureExtractor:
    _SENSITIVE_PORTS = {21, 22, 23, 80, 443, 3389}
    _DEFAULT_CREDS = {
        ("root", "root"), ("admin", "admin"), ("root", "123456"),
        ("admin", "password"), ("test", "test"), ("root", "toor"),
        ("admin", "1234"), ("root", "admin"), ("user", "user"),
    }
    _RECON = re.compile(r"\b(?:uname|whoami|id|ifconfig|ip\s+a|cat\s+/etc/passwd|hostname|netstat)\b", re.I)
    _DL_EXEC = re.compile(r"(?:wget|curl|tftp)\b.*(?:chmod|\.\\/|sh\s|bash\s|python)", re.I)
    _REV_SHELL = re.compile(r"(?:/dev/tcp/|nc\b.*-e|bash\b.*-i\s*>&|mkfifo)", re.I)
    _BASE64 = re.compile(r"(?:base64\s+-d|echo\s+.*\|\s*base64)", re.I)
    _PERSIST = re.compile(r"(?:crontab|\.bashrc|systemctl\s+enable|rc\.local)", re.I)
    _EXFIL = re.compile(r"(?:\bscp\b|\bcurl\b.*POST|\btar\b.*\|)", re.I)
    _DESTRUCT = re.compile(r"\b(?:rm\s+-rf|mkfs|dd\s+if=|killall)\b", re.I)
    _INTERNAL_IP = re.compile(r"(?:10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)")
    _PORT_FWD = re.compile(r"ssh\s+.*-[LRD]\s|socat\b|proxychains\b", re.I)
    _DOWNLOADER = re.compile(r"\b(?:wget|curl|tftp|ftp|scp)\b", re.I)
    _SQL_INJ = re.compile(r"(?:union\s+select|sleep\(|or\s+1\s*=\s*1|drop\s+table)", re.I)
    _PATH_TRAV = re.compile(r"(?:\.\./|/etc/passwd|%2e%2e)", re.I)
    _CMD_INJ = re.compile(r"(?:;\s*\w|`[^`]+`|\$\(|&&\s*\w)")

    def extract_all(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        print("[Features] Extraction starting...")
        df = self._command_features(df)
        df = self._timing_features(df)
        df = self._network_features(df)
        df = self._credential_features(df)
        df = self._tunneling_features(df)
        print(f"[Features] Complete -- {len(df.columns)} columns.")
        return df

    def _command_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cmds = df["commands"].fillna("").astype(str)
        cmd_lists = cmds.str.split("|")

        df["cmd_count"] = cmd_lists.apply(lambda x: len([c for c in x if c]))
        df["cmd_unique_count"] = cmd_lists.apply(lambda x: len(set(c for c in x if c)))
        df["cmd_unique_ratio"] = np.where(df["cmd_count"] > 0, df["cmd_unique_count"] / df["cmd_count"], 0.0)
        df["cmd_entropy"] = cmd_lists.apply(self._list_entropy)
        df["cmd_avg_length"] = cmd_lists.apply(lambda x: np.mean([len(c) for c in x if c]) if any(c for c in x) else 0.0)
        df["cmd_max_length"] = cmd_lists.apply(lambda x: max((len(c) for c in x if c), default=0))

        for col, pat in [
            ("has_download_execute_chain", self._DL_EXEC),
            ("has_reverse_shell", self._REV_SHELL),
            ("has_base64_decode", self._BASE64),
            ("has_recon_commands", self._RECON),
            ("has_persistence", self._PERSIST),
            ("has_file_exfiltration", self._EXFIL),
            ("has_destructive", self._DESTRUCT),
            ("has_downloader", self._DOWNLOADER),
            ("has_sql_injection", self._SQL_INJ),
            ("has_path_traversal", self._PATH_TRAV),
            ("has_cmd_injection", self._CMD_INJ),
        ]:
            df[col] = cmds.apply(lambda s, p=pat: int(bool(p.search(s))))
        return df

    def _timing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        raw_ts = df["raw_timestamps"].fillna("").astype(str)
        delays = raw_ts.apply(self._compute_delays)

        df["avg_inter_cmd_delay"] = delays.apply(lambda d: np.mean(d) if len(d) > 0 else 0.0)
        df["std_inter_cmd_delay"] = delays.apply(lambda d: np.std(d) if len(d) > 0 else 0.0)
        df["min_inter_cmd_delay"] = delays.apply(lambda d: np.min(d) if len(d) > 0 else 0.0)
        df["max_inter_cmd_delay"] = delays.apply(lambda d: np.max(d) if len(d) > 0 else 0.0)
        df["timing_variance_ratio"] = np.where(
            df["avg_inter_cmd_delay"] > 0,
            df["std_inter_cmd_delay"] / df["avg_inter_cmd_delay"], 0.0,
        )
        df["burst_count"] = delays.apply(self._count_bursts)
        df["events_per_minute"] = np.where(
            df["duration_sec"] > 0,
            df["event_count"] / (df["duration_sec"] / 60.0),
            df["event_count"].astype(float),
        )
        return df

    def _network_features(self, df: pd.DataFrame) -> pd.DataFrame:
        port_lists = df["dest_ports"].fillna("").str.split("|")
        df["unique_dest_ports"] = port_lists.apply(lambda x: len(set(p for p in x if p)))
        df["port_range_span"] = port_lists.apply(self._port_span)
        df["has_sensitive_ports"] = port_lists.apply(
            lambda x: int(any(p.isdigit() and int(p) in self._SENSITIVE_PORTS for p in x))
        )
        df["protocol_count"] = df["protocols"].fillna("").str.split("|").apply(lambda x: len(set(p for p in x if p)))
        df["has_internal_ip_scan"] = df["commands"].fillna("").apply(lambda s: int(bool(self._INTERNAL_IP.search(s))))

        dns_lists = df["dns_queries"].fillna("").str.split("|")
        df["dns_query_count"] = dns_lists.apply(lambda x: len([q for q in x if q]))
        df["dns_avg_query_length"] = dns_lists.apply(
            lambda x: np.mean([len(q) for q in x if q]) if any(q for q in x) else 0.0
        )
        df["http_unique_uris"] = df["http_uris"].fillna("").str.split("|").apply(lambda x: len(set(u for u in x if u)))
        return df

    def _credential_features(self, df: pd.DataFrame) -> pd.DataFrame:
        user_lists = df["usernames"].fillna("").str.split("|")
        pwd_lists = df["passwords"].fillna("").str.split("|")

        et = df["event_types"].fillna("").str.lower()
        df["credential_attempts"] = et.apply(lambda s: s.count("login") + s.count("auth") + s.count("connect"))
        df["unique_usernames"] = user_lists.apply(lambda x: len(set(u for u in x if u)))
        df["unique_passwords"] = pwd_lists.apply(lambda x: len(set(p for p in x if p)))
        df["credential_retry_count"] = df["unique_passwords"]
        df["has_default_creds"] = df.apply(
            lambda row: self._check_default_creds(row.get("usernames", ""), row.get("passwords", "")), axis=1,
        )
        df["password_entropy_avg"] = pwd_lists.apply(
            lambda x: np.mean([self._entropy(p) for p in x if p]) if any(p for p in x) else 0.0
        )
        return df

    def _tunneling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cmds = df["commands"].fillna("").astype(str)
        df["has_port_forwarding"] = cmds.apply(lambda s: int(bool(self._PORT_FWD.search(s))))
        df["has_encoded_payloads"] = cmds.apply(
            lambda s: int(bool(self._BASE64.search(s)) or bool(re.search(r"(?:[0-9a-fA-F]{2}){10,}", s)))
        )
        cmd_lists = cmds.str.split("|")
        df["payload_entropy_avg"] = cmd_lists.apply(
            lambda x: np.mean([self._entropy(c) for c in x if c]) if any(c for c in x) else 0.0
        )
        df["payload_entropy_max"] = cmd_lists.apply(
            lambda x: max((self._entropy(c) for c in x if c), default=0.0)
        )
        df["avg_payload_length"] = cmd_lists.apply(
            lambda x: np.mean([len(c) for c in x if c]) if any(c for c in x) else 0.0
        )
        df["tls_on_non_standard_port"] = df.apply(
            lambda row: self._check_tls_ns(row.get("commands", ""), row.get("dest_ports", "")), axis=1,
        )
        dns_lists = df["dns_queries"].fillna("").str.split("|")
        df["dns_tunnel_indicator"] = dns_lists.apply(
            lambda x: int(any(len(q) > 50 or q.count(".") > 4 for q in x if q))
        )
        return df

    # -- Helper methods --

    @staticmethod
    def _entropy(text: str) -> float:
        if not text:
            return 0.0
        n = len(text)
        freq = {}
        for ch in text:
            freq[ch] = freq.get(ch, 0) + 1
        return -sum((c / n) * math.log2(c / n) for c in freq.values())

    @staticmethod
    def _list_entropy(items: list) -> float:
        items = [i for i in items if i]
        if not items:
            return 0.0
        n = len(items)
        freq = {}
        for i in items:
            freq[i] = freq.get(i, 0) + 1
        return -sum((c / n) * math.log2(c / n) for c in freq.values())

    @staticmethod
    def _compute_delays(ts_str: str) -> list:
        parts = [p.strip() for p in ts_str.split("|") if p.strip()]
        if len(parts) < 2:
            return []
        try:
            times = pd.to_datetime(parts, errors="coerce", utc=True).dropna()
            if len(times) < 2:
                return []
            return (np.diff(times.values).astype("timedelta64[ms]").astype(float) / 1000.0).tolist()
        except Exception:
            return []

    @staticmethod
    def _count_bursts(delays: list) -> int:
        if len(delays) < 2:
            return 0
        return sum(1 for i in range(len(delays) - 1) if delays[i] < 0.5 and delays[i + 1] > 5.0)

    @staticmethod
    def _port_span(port_list: list) -> int:
        nums = [int(p) for p in port_list if p.isdigit()]
        return max(nums) - min(nums) if len(nums) >= 2 else 0

    def _check_default_creds(self, u_str: str, p_str: str) -> int:
        for u in (u_str or "").split("|"):
            for p in (p_str or "").split("|"):
                if u and p and (u.lower(), p.lower()) in self._DEFAULT_CREDS:
                    return 1
        return 0

    @staticmethod
    def _check_tls_ns(cmds: str, ports: str) -> int:
        if not re.search(r"(?:ClientHello|TLS|SSL)", cmds or "", re.I):
            return 0
        nums = [int(p) for p in (ports or "").split("|") if p.isdigit()]
        return int(any(p != 443 for p in nums)) if nums else 0
