import argparse, json, os, time, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from edge.publisher.mqtt_publisher import MQTTPublisher
from edge.config_loader import load_config

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    
    fifo_path = cfg["pipeline"]["fifo_path"]
    backup_path = cfg["pipeline"]["backup_file"]
    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
    
    mq = cfg["mqtt"]
    pub = MQTTPublisher(mq["host"], mq["port"], mq["topic"], mq["username"], mq["password"], mq["tls"])
    pub.connect()
    
    print(f"[Publisher] Listening on {fifo_path}")
    while not os.path.exists(fifo_path): time.sleep(1)
    
    while True:
        try:
            with open(fifo_path, "r") as pipe, open(backup_path, "a") as backup:
                for line in pipe:
                    line = line.strip()
                    if not line: continue
                    backup.write(line + "\n")
                    backup.flush()
                    try:
                        pub.publish(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"[Publisher] JSON decode error: {e}")
                    except Exception as e:
                        print(f"[Publisher] Publish error: {e}")
        except Exception as e:
            print(f"[Publisher] Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()