import json
import logging
import ssl
from pathlib import Path

import paho.mqtt.client as mqtt
import yaml

from .db_writer import insert_preprocessed_log
from .db_writer import insert_llm_log

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def load_config():
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "config" / "central_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

config = load_config()

MQTT_BROKER_HOST = config["mqtt"]["broker_host"]
MQTT_BROKER_PORT = config["mqtt"]["broker_port"]
MQTT_TOPIC       = config["mqtt"]["topic"]

MQTT_USERNAME    = config["mqtt"].get("username")
MQTT_PASSWORD    = config["mqtt"].get("password")
MQTT_USE_TLS     = config["mqtt"].get("tls", False)

REQUIRED_FIELDS = ["version", "log_source", "timestamp", "event_type", "src_ip"]


def is_valid_entry(entry: dict) -> bool:
    for field in REQUIRED_FIELDS:
        if field not in entry or entry[field] is None:
            logging.warning(f"Missing required field: {field}")
            return False
    return True


def on_connect(client, userdata, flags, rc):
    logging.info(f"on_connect called with rc={rc}")
    if rc == 0:
        logging.info("Connected to MQTT broker successfully")
        client.subscribe(MQTT_TOPIC)
        logging.info(f"Subscribed to topic: {MQTT_TOPIC}")
    else:
        logging.error(f"Failed to connect to MQTT broker. rc={rc}")


def on_message(client, userdata, msg):
    try:
        payload_str = msg.payload.decode("utf-8")
        logging.info(f"Received MQTT message on topic {msg.topic}: {payload_str}")

        data = json.loads(payload_str)
        if not is_valid_entry(data):
            logging.error("Invalid entry, skipping insert.")
            return

        insert_preprocessed_log(data)
        logging.info("Log inserted into SQLite successfully.")

    except json.JSONDecodeError:
        logging.exception("Failed to decode JSON payload.")
    except Exception:
        logging.exception("Error while processing MQTT message.")


def create_mqtt_client() -> mqtt.Client:
    logging.info(
        f"Creating MQTT client to {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT} "
        f"(topic={MQTT_TOPIC}, tls={MQTT_USE_TLS}, user={'set' if MQTT_USERNAME else 'none'})"
    )

    client = mqtt.Client()

    client.on_connect = on_connect
    client.on_message = on_message

    if MQTT_USERNAME:
        client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

    if MQTT_USE_TLS:
        client.tls_set(cert_reqs=ssl.CERT_REQUIRED)

    client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, keepalive=60)

    logging.info("MQTT client created and connected. Ready to start loop.")
    return client
