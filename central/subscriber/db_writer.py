import sqlite3
import json
from pathlib import Path

from central.database.init_db import load_config, get_connection

config = load_config()
DB_PATH = config["database"]["db_path"]

def _write_to_table(entry: dict, table_name: str):
    conn = get_connection(DB_PATH)
    cur = conn.cursor()

    raw_val = json.dumps(entry.get("raw")) if entry.get("raw") is not None else None

    if table_name == "LLM_Log":
        generation_model = entry.get("generation_model", "Phi-3-Mini-4k-Instruct")
        
        sql = f"""
        INSERT INTO {table_name} (
            version, log_source, timestamp, event_type,
            src_ip, src_port, dest_ip, dest_port,
            protocol, session_id, username, password,
            request_data, dns_query, http_method, http_uri, http_user_agent,
            alert_type, severity, raw, generation_model
        ) VALUES (?,?,?,?, ?,?,?,?, ?,?,?,?, ?,?,?,?, ?,?, ?,?, ?)
        """
        values = (
            entry.get("version"), entry.get("log_source"), entry.get("timestamp"), entry.get("event_type"),
            entry.get("src_ip"), entry.get("src_port"), entry.get("dest_ip"), entry.get("dest_port"),
            entry.get("protocol"), entry.get("session_id"), entry.get("username"), entry.get("password"),
            entry.get("request_data"), entry.get("dns_query"), entry.get("http_method"), entry.get("http_uri"),
            entry.get("http_user_agent"), entry.get("alert_type"), entry.get("severity"), 
            raw_val, generation_model
        )
    else:
        sql = f"""
        INSERT INTO {table_name} (
            version, log_source, timestamp, event_type,
            src_ip, src_port, dest_ip, dest_port,
            protocol, session_id, username, password,
            request_data, dns_query, http_method, http_uri, http_user_agent,
            alert_type, severity, raw
        ) VALUES (?,?,?,?, ?,?,?,?, ?,?,?,?, ?,?,?,?, ?,?, ?,?)
        """
        values = (
            entry.get("version"), entry.get("log_source"), entry.get("timestamp"), entry.get("event_type"),
            entry.get("src_ip"), entry.get("src_port"), entry.get("dest_ip"), entry.get("dest_port"),
            entry.get("protocol"), entry.get("session_id"), entry.get("username"), entry.get("password"),
            entry.get("request_data"), entry.get("dns_query"), entry.get("http_method"), entry.get("http_uri"),
            entry.get("http_user_agent"), entry.get("alert_type"), entry.get("severity"), raw_val
        )

    cur.execute(sql, values)
    conn.commit()
    conn.close()

def insert_preprocessed_log(entry: dict):
    _write_to_table(entry, "Preprocessed_Log")

def insert_llm_log(entry: dict):
    _write_to_table(entry, "LLM_Log")