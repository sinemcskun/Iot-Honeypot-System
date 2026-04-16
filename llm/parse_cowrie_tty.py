import os
import re
import json
import struct
import sys

TTY_LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cowrie_tty_logs")
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fine_tune_dataset.jsonl")

RECORD_HEADER_SIZE = 24  # 6 x 4 bytes
OP_OPEN = 1
OP_CLOSE = 2
OP_DATA = 3
DIR_OUTPUT = 2   
DIR_INPUT = 3    

ANSI_ESCAPE_RE = re.compile(
    r'('
    r'\x1b'               
    r'('
    r'\[[0-?]*[ -/]*[@-~]'   
    r'|'
    r'\].*?(?:\x07|\x1b\\)'  
    r'|'
    r'[()][AB012]'            
    r'|'
    r'[ -/]*[0-~]'           
    r')'
    r'|'
    r'[\x00-\x06\x0e-\x1a\x1c-\x1f]'  
    r'|'                                
    r'\x7f'                             
    r'|'
    r'\x9b[0-?]*[ -/]*[@-~]'           
    r')',
    re.DOTALL
)


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub('', text)


def resolve_backspaces(text: str) -> str:
    result = []
    for ch in text:
        if ch in ('\x08', '\x7f'):
            if result:
                result.pop()
        else:
            result.append(ch)
    return ''.join(result)


def clean_input(raw: str) -> str:
    text = strip_ansi(raw)
    text = resolve_backspaces(text)
    text = text.strip('\r\n')
    return text.strip()


def clean_output(raw: str) -> str:
    text = strip_ansi(raw)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    lines = [line.rstrip() for line in text.split('\n')]
    text = '\n'.join(lines).strip()
    return text


def parse_tty_log(filepath: str):
    try:
        with open(filepath, 'rb') as f:
            data = f.read()
    except (IOError, OSError) as e:
        print(f"  [ERROR] Cannot read file: {e}", file=sys.stderr)
        return

    filesize = len(data)
    if filesize < RECORD_HEADER_SIZE:
        print(f"  [WARN] File too small ({filesize} bytes), skipping.", file=sys.stderr)
        return

    offset = 0

    while offset + RECORD_HEADER_SIZE <= filesize:
        try:
            op, tty, size, direction, sec, usec = struct.unpack_from(
                '<IIIIII', data, offset
            )
        except struct.error as e:
            print(f"  [ERROR] Failed to unpack header at offset {offset}: {e}",
                  file=sys.stderr)
            break

        if op not in (OP_OPEN, OP_CLOSE, OP_DATA):
            print(f"  [WARN] Unknown op={op} at offset {offset}, stopping parse.",
                  file=sys.stderr)
            break

        if size > filesize - offset - RECORD_HEADER_SIZE:
            print(f"  [WARN] Record size ({size}) exceeds remaining data at "
                  f"offset {offset}, stopping parse.", file=sys.stderr)
            break

        payload_start = offset + RECORD_HEADER_SIZE
        payload_end = payload_start + size

        if op == OP_DATA and size > 0 and direction in (DIR_INPUT, DIR_OUTPUT):
            try:
                payload = data[payload_start:payload_end].decode(
                    'utf-8', errors='replace'
                )
                yield (direction, payload)
            except Exception as e:
                print(f"  [WARN] Failed to decode payload at offset {offset}: {e}",
                      file=sys.stderr)

        offset = payload_end


def extract_pairs(filepath: str):
    current_input_parts = []
    current_output_parts = []
    state = 'IDLE'  

    for direction, payload in parse_tty_log(filepath):
        if direction == DIR_INPUT:
            if state == 'COLLECTING_OUTPUT' and current_input_parts:
                cmd = clean_input(''.join(current_input_parts))
                resp = clean_output(''.join(current_output_parts))
                if cmd and resp:
                    yield (cmd, resp)
                current_input_parts = []
                current_output_parts = []

            current_input_parts.append(payload)
            state = 'COLLECTING_INPUT'

        elif direction == DIR_OUTPUT:
            if state == 'IDLE' and not current_input_parts:
                continue
            current_output_parts.append(payload)
            state = 'COLLECTING_OUTPUT'
    if current_input_parts:
        cmd = clean_input(''.join(current_input_parts))
        resp = clean_output(''.join(current_output_parts))
        if cmd and resp:
            yield (cmd, resp)


def main():
    if not os.path.isdir(TTY_LOGS_DIR):
        print(f"[FATAL] Directory not found: {TTY_LOGS_DIR}", file=sys.stderr)
        sys.exit(1)

    log_files = sorted([
        f for f in os.listdir(TTY_LOGS_DIR)
        if os.path.isfile(os.path.join(TTY_LOGS_DIR, f)) and f != '.gitignore'
    ])

    print(f"Found {len(log_files)} TTY log files in: {TTY_LOGS_DIR}")

    total_pairs = 0
    files_parsed = 0
    files_skipped = 0
    files_with_pairs = 0

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for i, filename in enumerate(log_files, 1):
            filepath = os.path.join(TTY_LOGS_DIR, filename)

            pair_count = 0
            try:
                for cmd, resp in extract_pairs(filepath):
                    record = {
                        "instruction": cmd,
                        "output": resp
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    pair_count += 1
            except Exception as e:
                print(f"  [{i}/{len(log_files)}] {filename} -> ERROR: {e}",
                      file=sys.stderr)
                files_skipped += 1
                continue

            files_parsed += 1
            total_pairs += pair_count
            if pair_count > 0:
                files_with_pairs += 1

            if i % 50 == 0 or i == len(log_files):
                print(f"  Progress: {i}/{len(log_files)} files processed, "
                      f"{total_pairs} pairs so far...")

    print(f"\n{'='*60}")
    print(f"  DONE!")
    print(f"  Files processed : {files_parsed}")
    print(f"  Files skipped   : {files_skipped}")
    print(f"  Files with pairs: {files_with_pairs}")
    print(f"  Total pairs     : {total_pairs}")
    print(f"  Output file     : {OUTPUT_FILE}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
