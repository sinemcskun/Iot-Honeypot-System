import argparse
import json
import sys
import threading
import os
import stat
from queue import Queue, Empty
from typing import Callable, Dict, Any
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from edge.aggregator.cowrie_reader import CowrieReader
from edge.aggregator.honeytrap_reader import HoneytrapReader
from edge.aggregator.suricata_reader import SuricataReader
from edge.aggregator.normalizer import normalize_cowrie_event, normalize_honeytrap_event, normalize_suricata_event
from edge.config_loader import load_config

def ensure_fifo(path: str) -> None:
    if os.path.exists(path):
        if stat.S_ISFIFO(os.stat(path).st_mode):
            return
        else:
            try:
                os.remove(path)
            except OSError:
                pass
    os.mkfifo(path)
    print(f"[Aggregator] FIFO created at: {path}")

def reader_worker(name, reader, norm_fn, q):
    try:
        for raw in reader.read_logs():
            try:
                processed = norm_fn(raw)
                if processed: q.put(processed)
            except Exception as e:
                print(f"[{name}] Error: {e}")
    except Exception as e:
        print(f"[{name}] Crash: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    
    cfg = load_config(args.config)

    cowrie_path = cfg["logs"]["cowrie"]
    honeytrap_path = cfg["logs"]["honeytrap"]
    suricata_path = cfg["logs"]["suricata"]
    fifo_path = cfg["pipeline"]["fifo_path"]

    ensure_fifo(fifo_path)
    queue = Queue(maxsize=10000)


    threads = [
        threading.Thread(target=reader_worker, args=("cowrie", CowrieReader(cowrie_path), normalize_cowrie_event, queue), daemon=True),
        threading.Thread(target=reader_worker, args=("honeytrap", HoneytrapReader(honeytrap_path), normalize_honeytrap_event, queue), daemon=True),
        threading.Thread(target=reader_worker, args=("suricata", SuricataReader(suricata_path), normalize_suricata_event, queue), daemon=True),
    ]
    for t in threads: t.start()

    print("[Aggregator] Started. Writing to FIFO...")
    try:
        with open(fifo_path, "w", encoding="utf-8") as pipe:
            while True:
                try:
                    item = queue.get(timeout=1.0)
                    pipe.write(json.dumps(item, ensure_ascii=False) + "\n")
                    pipe.flush()
                except Empty:
                    continue
                except BrokenPipeError:
                    print("[Aggregator] Pipe broken, waiting for reader...")
                    continue 
    except KeyboardInterrupt:
        pass
    finally:
        if os.path.exists(fifo_path): os.remove(fifo_path)

if __name__ == "__main__":
    main()