#!/bin/bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORIGINAL_CONFIG="$BASE_DIR/config/edge_config.yaml"
TEMP_CONFIG="/tmp/manual_capture_config.yaml"
TEMP_FIFO="/tmp/manual_capture_pipe"
OUTPUT_FILE="$BASE_DIR/manual_logs.json"

if [ -d "$BASE_DIR/venv" ]; then
    source "$BASE_DIR/venv/bin/activate"
else
    echo "[HATA] Sanal ortam bulunamadı."
    exit 1
fi

echo "=============================================="
echo "   Manuel Log Yakalama (İzole Mod)            "
echo "=============================================="

rm -f "$TEMP_FIFO"
rm -f "$OUTPUT_FILE"
rm -f "$TEMP_CONFIG"

cp "$ORIGINAL_CONFIG" "$TEMP_CONFIG"

sed -i "s|fifo_path:.*|fifo_path: $TEMP_FIFO|g" "$TEMP_CONFIG"

echo "[INFO] Geçici config oluşturuldu: $TEMP_CONFIG"
echo "[INFO] Hedef dosya: $OUTPUT_FILE"
cleanup() {
    echo ""
    echo "[INFO] İşlem durduruluyor..."
    kill "${AGGREGATOR_PID:-0}" 2>/dev/null || true
    kill "${RECORDER_PID:-0}"  2>/dev/null || true
    rm -f "$TEMP_FIFO"
    rm -f "$TEMP_CONFIG"
    echo "[INFO] Temizlik yapıldı. Çıktı dosyası: $OUTPUT_FILE"
    exit 0
}
trap cleanup SIGINT SIGTERM

mkfifo "$TEMP_FIFO"

cat "$TEMP_FIFO" > "$OUTPUT_FILE" &
RECORDER_PID=$!
echo "[INFO] Aggregator başlatılıyor (Config: $TEMP_CONFIG)..."

python3 "$BASE_DIR/edge/aggregator/aggregator_main.py" --config "$TEMP_CONFIG" &
AGGREGATOR_PID=$!

echo "[INFO] Loglar toplanıyor... Bitirmek için CTRL+C basın."

wait