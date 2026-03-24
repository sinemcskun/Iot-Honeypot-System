import logging
from .mqtt_subscriber import create_mqtt_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def main():
    logging.info("Starting central subscriber...")
    client = create_mqtt_client()

    logging.info("Starting MQTT loop. Waiting for messages...")

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        logging.info("Subscriber stopped by user. Exiting cleanly...")
        try:
            client.disconnect()
        except Exception as e:
            logging.warning(f"Error during disconnect: {e}")


if __name__ == "__main__":
    main()
