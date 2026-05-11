from __future__ import annotations

import json
from typing import Any, Dict, Optional

import paho.mqtt.client as mqtt


class MQTTPublisher:
    def __init__(
        self,
        host: str,
        port: int,
        topic: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_tls: bool = False,
        client_id: Optional[str] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.topic = topic

        self.client = mqtt.Client(client_id=client_id or "iot-honeypot-edge")
        if username is not None:
            self.client.username_pw_set(username, password=password)

        if use_tls:
            self.client.tls_set()

        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect

    def on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            print("[MQTT] Connected successfully.")
        else:
            print(f"[MQTT] Connection failed with code {rc}")

    def on_disconnect(self, client, userdata, rc, properties=None):  
        if rc != 0:
            print("[MQTT] Unexpected disconnection.")

    def connect(self) -> None:
        self.client.connect(self.host, self.port, keepalive=60)
        self.client.loop_start()

    def publish(self, msg: Dict[str, Any]) -> None:
        try:
            payload = json.dumps(msg, ensure_ascii=False)
            self.client.publish(self.topic, payload)
        except Exception as e:
            print(f"[MQTT] Publish Error: {e}")
