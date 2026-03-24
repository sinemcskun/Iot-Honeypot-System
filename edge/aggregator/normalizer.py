from typing import Dict, Any, Optional

PREPROCESSED_VERSION = 1

def get_nested(data: Dict, path: str, default=None):
    if path in data:
        return data[path]

    keys = path.split('.')
    val = data
    for key in keys:
        if isinstance(val, dict):
            val = val.get(key)
        else:
            return default
        if val is None: 
            return default
    return val

def _base_preprocessed(log_source: str) -> Dict[str, Any]:
    '''
    Base structure for preprocessed logs.
    Includes metadata about the log source and version.
    log_source: "cowrie", "honeytrap", "suricata"
    
    '''
    return {
        "version": PREPROCESSED_VERSION,
        "log_source": log_source,

        "timestamp": None,
        "event_type": None,

        "src_ip": None,
        "src_port": None,
        "dest_ip": None,
        "dest_port": None,

        "protocol": None,
        "session_id": None,
        "username": None,
        "password": None,

        "request_data": None,
        "dns_query": None,
        "http_method": None,
        "http_uri": None,
        "http_user_agent": None,

        "alert_type": None,
        "severity": None,

        "raw": None,
    }

_COWRIE_EVENT_MAP = {
    "cowrie.login.failed": "ssh_login_failed",
    "cowrie.login.success": "ssh_login_success",
    "cowrie.client.fingerprint": "ssh_publickey_attempt",
    "cowrie.session.connect": "ssh_connect",
    "cowrie.session.closed": "ssh_disconnect",
    "cowrie.session.params":"ssh_session_parameters",
    "cowrie.command.input": "ssh_command",
    "cowrie.command.failed": "ssh_command_failed",
    "cowrie.session.file_upload": "ssh_file_uploaded",
    "cowrie.session.file_download": "ssh_file_downloaded",
    "cowrie.virustotal.scanfile": "malware_virustotal_scan",
    "cowrie.client.version": "ssh_client_version",
    "cowrie.client.kex":"ssh_key_exchange",
    "cowrie.client.size":"ssh_terminal_resize",
    "cowrie.direct-tcpip.request": "ssh_port_forward_request",
    "cowrie.direct-tcpip.data": "ssh_port_forward_data",
    "cowrie.log.closed": "ssh_tty_log_closed",
    "cowrie.client.var": "ssh_env_variable",
    "cowrie.direct-tcpip.ja4": "ssh_direct_tcpip_ja4_fingerprint",
    "cowrie.direct-tcpip.ja4h": "ssh_direct_tcpip_ja4_fingerprint_hashed"
}

def _map_cowrie_event_type(eventid: Optional[str]) -> str:
    if not eventid:
        return "unknown"
    
    if eventid not in _COWRIE_EVENT_MAP:
        return "unknown"
    
    if eventid in _COWRIE_EVENT_MAP:
        return _COWRIE_EVENT_MAP[eventid]
    
    if eventid.startswith("cowrie."):
        return eventid.replace("cowrie.", "")
    
    return eventid

def normalize_cowrie_event(raw: Dict[str, Any], honeypot_ip: Optional[str] = None) -> Dict[str, Any]:
    '''
    It transforms a raw Cowrie log event into a preprocessed log format.
    raw: Raw Cowrie log event as a dictionary.
    honeypot_ip: Optional IP address of the honeypot to set as dest_ip if not present in raw log.
    '''

    doc = _base_preprocessed(log_source="cowrie")

    doc["timestamp"] = raw.get("timestamp")
    doc["event_type"] = _map_cowrie_event_type(raw.get("eventid"))
    doc["src_ip"] = raw.get("src_ip") or raw.get("src_host") or raw.get("peer_ip")
    doc["src_port"] = raw.get("src_port") or raw.get("peer_port")
    dest_ip = raw.get("dst_ip") or raw.get("dest_ip") or honeypot_ip
    doc["dest_ip"] = dest_ip
    dest_port = raw.get("dst_port") or raw.get("dest_port") or 22
    doc["dest_port"] = dest_port
    protocol = raw.get("protocol") or "ssh"
    doc["protocol"] = protocol
    doc["session_id"] = raw.get("session")
    doc["username"] = raw.get("username")   
    doc["password"] = raw.get("password")

    request_data = (
        raw.get("input") or 
        raw.get("message") or 
        raw.get("command") or 
        None
    )
    doc["request_data"] = request_data

    doc["dns_query"] = None
    doc["http_method"] = None
    doc["http_uri"] = None
    doc["http_user_agent"] = None
    doc["alert_type"] = None
    doc["severity"] = None

    doc["raw"] = raw
    return doc

def normalize_honeytrap_event(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    category = raw.get("category")
    event_type = raw.get("type")

    if category == "heartbeat" or event_type == "info":
        return None

    doc = _base_preprocessed(log_source="honeytrap")
 
    doc["timestamp"] = (
        raw.get("date")         
        or raw.get("timestamp")
        or raw.get("@timestamp")
        or raw.get("time")
    )

    doc["event_type"] = (
        raw.get("type") 
        or raw.get("event") 
        or raw.get("category") 
        or "honeytrap_event"
    )

    doc["src_ip"] = raw.get("source-ip") or raw.get("src_ip") or raw.get("source_ip") or raw.get("src")
    doc["src_port"] = raw.get("source-port") or raw.get("src_port") or raw.get("sport")
    doc["dest_ip"] = raw.get("destination-ip") or raw.get("dst_ip") or raw.get("dest_ip") or raw.get("dst")
    doc["dest_port"] = raw.get("destination-port") or raw.get("dst_port") or raw.get("dport")
    if category == "http":
        doc["protocol"] = "HTTP"
        doc["http_method"] = raw.get("http.method")
        doc["http_uri"] = raw.get("http.url")

        ua_list = raw.get("http.header.user-agent") or []
        doc["http_user_agent"] = ua_list[0] if ua_list else None

        doc["session_id"] = raw.get("http.sessionid")
        doc["request_data"] = raw.get("payload") or raw.get("payload-hex")

    elif category == "ftp":
        doc["protocol"] = "FTP"
        cmd = raw.get("ftp.command") or ""
        doc["session_id"] = raw.get("ftp.sessionid")

        if cmd.startswith("USER "):
            doc["username"] = cmd.split(" ", 1)[1].strip() or None
        elif cmd.startswith("PASS "):
            doc["password"] = cmd.split(" ", 1)[1].strip() or None

        doc["request_data"] = cmd

    doc["raw"] = raw
    return doc


def normalize_suricata_event(raw: Dict[str, Any]) -> Dict[str, Any]:
    if raw.get("event_type") == "stats":
        return None

    doc = _base_preprocessed(log_source="suricata")
    doc["timestamp"] = raw.get("timestamp")
    doc["event_type"] = raw.get("event_type") or "suricata_event"

    doc["src_ip"] = raw.get("src_ip")
    doc["src_port"] = raw.get("src_port")
    doc["dest_ip"] = raw.get("dest_ip")
    doc["dest_port"] = raw.get("dest_port")
    doc["protocol"] = raw.get("proto")
    doc["session_id"] = str(raw.get("flow_id")) if raw.get("flow_id") is not None else None
    doc["username"] = raw.get("username")
    doc["password"] = raw.get("password")

    http = raw.get("http") or {}
    doc["http_method"] = http.get("method") or http.get("http_method")

    if http:
        host = http.get("host") or "" or http.get("hostname")
        url = http.get("url") or "" or http.get("uri")
        doc["http_uri"] = f"{host}{url}" if (host or url) else None
        doc["http_user_agent"] = (
            http.get("user_agent")
            or http.get("http_user_agent")
        )
    else:
        doc["http_uri"] = None
        doc["http_user_agent"] = None

    dns = raw.get("dns") or {}
    doc["dns_query"] = dns.get("rrname")
    alert = raw.get("alert") or {}
    doc["alert_type"] = alert.get("signature")
    doc["severity"] = alert.get("severity")
    doc["request_data"] = (
        raw.get("payload_printable")
        or raw.get("payload")
    )

    doc["raw"] = raw
    return doc
