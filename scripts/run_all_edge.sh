#!/bin/bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="$BASE_DIR/config/edge_config.yaml"

if [ -d "$BASE_DIR/venv" ]; then
    source "$BASE_DIR/venv/bin/activate"
else
    echo "[HATA] Sanal ortam (venv) bulunamadı: $BASE_DIR/venv"
    echo "Lütfen önce 'python3 -m venv venv' ile ortam kurun."
    exit 1
fi

echo "=========================================="
echo "   IoT Honeypot Edge Pipeline (MANUEL)    "
echo "=========================================="
echo "[INFO] Config: $CONFIG_FILE"

cleanup() {
    echo ""
    echo "[INFO] Sistem kapatılıyor..."
    kill "${AGGREGATOR_PID:-0}" 2>/dev/null || true
    kill "${PUBLISHER_PID:-0}" 2>/dev/null || true
    echo "[INFO] Tüm süreçler durduruldu."
    exit 0
}
trap cleanup SIGINT SIGTERM

echo "[INFO] Aggregator başlatılıyor..."
python3 "$BASE_DIR/edge/aggregator/aggregator_main.py" --config "$CONFIG_FILE" &
AGGREGATOR_PID=$!
echo "[INFO] Aggregator PID: $AGGREGATOR_PID"
sleep 2

echo "[INFO] Publisher başlatılıyor..."
python3 "$BASE_DIR/edge/publisher/publisher_main.py" --config "$CONFIG_FILE" &
PUBLISHER_PID=$!
echo "[INFO] Publisher PID: $PUBLISHER_PID"
echo "------------------------------------------"
echo "[INFO] Sistem çalışıyor."
echo "[INFO] Loglar 'edge_config.yaml' içindeki ayarlarını kullanıyor."
echo "[INFO] Çıkmak için CTRL+C yapabilirsiniz."

wait