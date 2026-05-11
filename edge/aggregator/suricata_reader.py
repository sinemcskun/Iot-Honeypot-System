import json
import time
from typing import Generator, Dict

class SuricataReader:
    def __init__(self, filepath: str, start_from_end: bool = True, poll_interval: float = 0.5):
        self.filepath = filepath
        self.start_from_end = start_from_end
        self.poll_interval = poll_interval

    def read_logs(self) -> Generator[Dict, None, None]:
        with open(self.filepath, 'r', encoding="utf-8") as file:
            if self.start_from_end:
                file.seek(0, 2)

            while True:
                line = file.readline()
                if not line:
                    time.sleep(self.poll_interval)
                    continue

                line = line.strip()
                if not line:
                    continue

                try:
                    event = json.loads(line)

                except json.JSONDecodeError:

                    continue

                if isinstance(event, dict):
                    yield event
                else:
                    continue