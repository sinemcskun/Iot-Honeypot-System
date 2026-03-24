#!/bin/bash
set -euo pipefail

# --- DİZİN AYARLARI ---
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORIGINAL_CONFIG="$BASE_DIR/config/edge_config.yaml"
TEMP_CONFIG="/tmp/manual_capture_config.yaml"
TEMP_FIFO="/tmp/manual_capture_pipe"
OUTPUT_FILE="$BASE_DIR/manual_logs.json"

# --- SANAL ORTAM ---
if [ -d "$BASE_DIR/venv" ]; then
    source "$BASE_DIR/venv/bin/activate"
else
    echo "[HATA] Sanal ortam bulunamadı."
    exit 1
fi

echo "=============================================="
echo "   Manuel Log Yakalama (İzole Mod)            "
echo "=============================================="

# --- TEMİZLİK ---
rm -f "$TEMP_FIFO"
rm -f "$OUTPUT_FILE"
rm -f "$TEMP_CONFIG"

# --- GEÇİCİ CONFIG OLUŞTURMA ---
# Ana config dosyasını kopyalayıp FIFO yolunu değiştiriyoruz.
# Böylece ana sistem (servisler) çalışıyorsa onlarla çakışmıyoruz.
cp "$ORIGINAL_CONFIG" "$TEMP_CONFIG"

# Linux sed komutu ile yaml içindeki fifo yolunu değiştiriyoruz
# (Basit string değiştirme yapıyoruz)
sed -i "s|fifo_path:.*|fifo_path: $TEMP_FIFO|g" "$TEMP_CONFIG"

echo "[INFO] Geçici config oluşturuldu: $TEMP_CONFIG"
echo "[INFO] Hedef dosya: $OUTPUT_FILE"

# --- KAPANIŞ YÖNETİMİ ---
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

# --- 1. ADIM: KAYITÇI (Okuyucu) ---
# Aggregator yazmadan önce dinlemeye başlamalıyız
# FIFO henüz yoksa Aggregator oluşturacak, o yüzden önce mkfifo yapmıyoruz,
# Aggregator kodundaki ensure_fifo bunu halledecek. 
# Ancak 'cat' komutu FIFO yoksa hata verebilir, bu yüzden Python kodu önce başlatıp 
# FIFO oluşunca okumayı başlatmak daha güvenli ama senkronizasyon zor.
# En temizi: FIFO'yu script oluştursun.
mkfifo "$TEMP_FIFO"

cat "$TEMP_FIFO" > "$OUTPUT_FILE" &
RECORDER_PID=$!

# --- 2. ADIM: AGGREGATOR ---
echo "[INFO] Aggregator başlatılıyor (Config: $TEMP_CONFIG)..."

# Python'a değiştirilmiş geçici config dosyasını veriyoruz
python3 "$BASE_DIR/edge/aggregator/aggregator_main.py" --config "$TEMP_CONFIG" &
AGGREGATOR_PID=$!

echo "[INFO] Loglar toplanıyor... Bitirmek için CTRL+C basın."

wait