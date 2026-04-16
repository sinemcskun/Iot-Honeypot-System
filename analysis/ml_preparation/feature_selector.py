SAFE_FEATURES = [
    "duration_sec",
    "event_count",
    "min_inter_cmd_delay",
    "max_inter_cmd_delay",
    "cmd_count",
    "cmd_unique_count",
    "cmd_entropy",
    "cmd_avg_length",
    "cmd_max_length",
    "port_range_span",
    "has_sensitive_ports",
    "protocol_count",
    "dns_query_count",
    "dns_avg_query_length",
    "http_unique_uris",
    "unique_usernames",
    "password_entropy_avg",
    "payload_entropy_avg",
    "payload_entropy_max",
    "avg_payload_length",
    "avg_cmd_per_event",
    "log_duration",
    "log_event_count",
    "entropy_ratio",
]

HEURISTIC_FEATURES = [
    "has_download_execute_chain",
    "has_reverse_shell",
    "has_base64_decode",
    "has_recon_commands",
    "has_persistence",
    "has_file_exfiltration",
    "has_destructive",
    "has_downloader",
    "has_sql_injection",
    "has_path_traversal",
    "has_cmd_injection",
    "has_internal_ip_scan",
    "has_port_forwarding",
    "has_encoded_payloads",
    "tls_on_non_standard_port",
    "dns_tunnel_indicator",
    "has_default_creds",
    "automation_score",
    "timing_variance_ratio",
    "events_per_minute",
    "cmd_unique_ratio",
    "burst_count",
    "unique_dest_ports",
    "unique_passwords",
    "credential_retry_count",
    "credential_attempts",
    "credential_attempts",
    "avg_inter_cmd_delay",
    "std_inter_cmd_delay",
]

MULTI_LABEL_COLS = [
    "label_bruteforce",
    "label_malware_dropper",
    "label_reconnaissance",
    "label_lateral_movement",
    "label_credential_spray",
    "label_tunneling",
    "label_data_exfiltration",
    "label_destructive",
    "label_port_scan",
    "label_service_interaction",
    "label_network_probe",
]

BINARY_LABELS = [
    "bot_label",
    "tunnel_label",
]

LABEL_COLS = MULTI_LABEL_COLS + BINARY_LABELS


def get_safe_features(df_columns) -> list:
    return [c for c in SAFE_FEATURES if c in df_columns]


def get_heuristic_features(df_columns) -> list:
    return [c for c in HEURISTIC_FEATURES if c in df_columns]


def get_label_cols(df_columns) -> list:
    return [c for c in LABEL_COLS if c in df_columns]
