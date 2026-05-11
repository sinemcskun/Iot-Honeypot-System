import sqlite3
import numpy as np
import pandas as pd


class Loader:
    def __init__(self, db_path: str, table_name: str = "Preprocessed_Log"):
        self.db_path = db_path
        self.table_name = table_name

    def load(self) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query(f'SELECT * FROM "{self.table_name}"', conn)
        finally:
            conn.close()

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

        for col in ["dest_port", "src_port", "severity"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        text_cols = [
            "request_data", "dns_query", "http_uri", "http_user_agent",
            "http_method", "event_type", "protocol", "username", "password",
            "alert_type", "raw", "log_source", "src_ip", "dest_ip", "session_id",
        ]
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)

        print(f"[Loader] {len(df):,} events loaded ({self.db_path})")
        return df


class SessionBuilder:
    def __init__(self, gap_minutes: int = 30):
        self.gap_minutes = gap_minutes

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        print(f"[SessionBuilder] Building sessions from {len(df):,} events...")

        df = df.dropna(subset=["timestamp"])
        df = df.sort_values(["src_ip", "dest_ip", "timestamp"])

        df["_pair"] = df["src_ip"].fillna("") + "::" + df["dest_ip"].fillna("")
        df["_gap"] = df.groupby("_pair")["timestamp"].diff()
        df["_gap_sec"] = pd.to_timedelta(df["_gap"], errors="coerce").dt.total_seconds()
        df["_gap_sec"] = df["_gap_sec"].fillna(self.gap_minutes * 60 + 1)
        df["_new"] = (df["_gap_sec"] > self.gap_minutes * 60).astype(int)
        df["_snum"] = df.groupby("_pair")["_new"].cumsum()
        df["session_id"] = df["_pair"] + "::s" + df["_snum"].astype(str)

        agg = df.groupby("session_id", dropna=False).agg(
            src_ip=("src_ip", "first"),
            dest_ip=("dest_ip", "first"),
            log_source=("log_source", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else ""),
            start_time=("timestamp", "min"),
            end_time=("timestamp", "max"),
            event_count=("session_id", "count"),
            commands=("request_data", lambda x: "|".join(x.dropna().astype(str).unique())),
            event_types=("event_type", lambda x: "|".join(x.dropna().astype(str).unique())),
            usernames=("username", lambda x: "|".join(v for v in x.dropna().astype(str).unique() if v)),
            passwords=("password", lambda x: "|".join(v for v in x.dropna().astype(str).unique() if v)),
            dest_ports=("dest_port", lambda x: "|".join(x.dropna().astype(int).astype(str).unique())),
            protocols=("protocol", lambda x: "|".join(v for v in x.dropna().astype(str).unique() if v)),
            dns_queries=("dns_query", lambda x: "|".join(v for v in x.dropna().astype(str).unique() if v)),
            http_uris=("http_uri", lambda x: "|".join(v for v in x.dropna().astype(str).unique() if v)),
            http_user_agents=("http_user_agent", lambda x: "|".join(v for v in x.dropna().astype(str).unique() if v)),
            alert_types=("alert_type", lambda x: "|".join(v for v in x.dropna().astype(str).unique() if v)),
            raw_timestamps=("timestamp", lambda x: "|".join(x.astype(str))),
        ).reset_index()

        agg["duration_sec"] = (agg["end_time"] - agg["start_time"]).dt.total_seconds().fillna(0.0)

        print(f"[SessionBuilder] {len(agg):,} sessions created.")
        return agg
