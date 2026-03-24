import os
import json
import pandas as pd


class IntelExporter:
    def save_parquet(self, df: pd.DataFrame, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path, index=False, engine="pyarrow")
        print(f"[Exporter] Parquet: {len(df):,} rows -> {path}")

    def save_json(self, df: pd.DataFrame, path: str, max_sessions: int = 5000) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        subset = df.head(max_sessions)

        records = []
        for _, row in subset.iterrows():
            records.append({
                "session_id": str(row.get("session_id", "")),
                "src_ip": str(row.get("src_ip", "")),
                "dest_ip": str(row.get("dest_ip", "")),
                "start_time": str(row.get("start_time", "")),
                "end_time": str(row.get("end_time", "")),
                "duration_sec": float(row.get("duration_sec", 0)),
                "event_count": int(row.get("event_count", 0)),
                "behavior_labels": [
                    l for l in str(row.get("attack_labels", "")).split("|") if l
                ],
                "attack_label_count": int(row.get("attack_label_count", 0)),
                "automation": {
                    "likelihood": str(row.get("automation_likelihood", "mixed")),
                    "score": float(row.get("automation_score", 0.5)),
                },
                "features": {
                    "cmd_count": int(row.get("cmd_count", 0)),
                    "cmd_entropy": round(float(row.get("cmd_entropy", 0)), 2),
                    "events_per_minute": round(float(row.get("events_per_minute", 0)), 2),
                    "unique_dest_ports": int(row.get("unique_dest_ports", 0)),
                    "credential_attempts": int(row.get("credential_attempts", 0)),
                    "payload_entropy_avg": round(float(row.get("payload_entropy_avg", 0)), 2),
                },
            })

        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2, default=str)
        print(f"[Exporter] JSON: {len(records):,} sessions -> {path}")
