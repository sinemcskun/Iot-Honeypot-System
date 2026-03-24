"""
Database initialization and connection utilities.
"""
import sqlite3
import yaml
from pathlib import Path
from typing import Optional


def load_config() -> dict:
    """
    Load configuration from central_config.yaml.
    
    Returns:
        dict: Configuration dictionary
    """
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "config" / "central_config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_connection(db_path: str, timeout: int = 60) -> sqlite3.Connection:
    """
    Get a SQLite database connection.
    
    Args:
        db_path: Path to the SQLite database file
        timeout: Connection timeout in seconds
        
    Returns:
        sqlite3.Connection: Database connection
    """
    # Ensure parent directory exists
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path, timeout=timeout)
    return conn


def init_database(db_path: Optional[str] = None, schema_path: Optional[str] = None) -> None:
    """
    Initialize the database with schema if it doesn't exist.
    
    Args:
        db_path: Path to database file (if None, uses config)
        schema_path: Path to schema SQL file (if None, uses default)
    """
    if db_path is None:
        config = load_config()
        db_path = config["database"]["db_path"]
    
    if schema_path is None:
        project_root = Path(__file__).resolve().parents[1]
        schema_path = project_root / "schema.sql"
    
    if not Path(schema_path).exists():
        raise FileNotFoundError(f"Schema file not found at: {schema_path}")
    
    conn = get_connection(db_path)
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_sql = f.read()
        conn.executescript(schema_sql)
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    # Initialize database when run directly
    init_database()
    print("Database initialized successfully.")

