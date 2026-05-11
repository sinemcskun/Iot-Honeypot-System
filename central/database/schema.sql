CREATE TABLE IF NOT EXISTS Preprocessed_Log (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    version INTEGER,
    log_source TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL,

    src_ip TEXT NOT NULL,
    src_port INTEGER,
    dest_ip TEXT,
    dest_port INTEGER,

    protocol TEXT,
    session_id TEXT,
    username TEXT,
    password TEXT,

    request_data TEXT,
    dns_query TEXT,
    http_method TEXT,
    http_uri TEXT,
    http_user_agent TEXT,

    alert_type TEXT,
    severity INTEGER,

    raw TEXT
);

CREATE TABLE IF NOT EXISTS LLM_Log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        version INTEGER,
        log_source TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        event_type TEXT NOT NULL,
        src_ip TEXT NOT NULL,
        src_port INTEGER,
        dest_ip TEXT,
        dest_port INTEGER,
        protocol TEXT,
        session_id TEXT,
        username TEXT,
        password TEXT,
        request_data TEXT,
        dns_query TEXT,
        http_method TEXT,
        http_uri TEXT,
        http_user_agent TEXT,
        alert_type TEXT,
        severity INTEGER,
        raw TEXT,
        generation_model TEXT 
    );
