#!/usr/bin/env bash
# Aumenta un dataset limpio simulando el canal telefónico.
#
#   bash scripts/telephony_augment.sh data/clean data/raw/synthetic
#
# Para qué: casi todo modelo anti-spoofing preentrenado viene de audio
# 16 kHz limpio. El reto es 8 kHz telefónico, que recorta la banda a
# 300-3400 Hz y mete el codec G.711. Entrenar sin simular eso es entrenar
# para un dominio que no vas a ver en el scoring.
set -euo pipefail

SRC="${1:?uso: telephony_augment.sh <carpeta_entrada> <carpeta_salida>}"
DST="${2:?uso: telephony_augment.sh <carpeta_entrada> <carpeta_salida>}"
mkdir -p "$DST"

find "$SRC" -type f \( -name '*.wav' -o -name '*.flac' -o -name '*.mp3' \) | while read -r f; do
  base="$(basename "${f%.*}")"
  out="$DST/${base}_tel.wav"
  # Banda telefónica -> ida y vuelta por G.711 mu-law -> 8 kHz.
  ffmpeg -hide_banner -loglevel error -y -i "$f" \
    -af "highpass=f=300,lowpass=f=3400,volume=0.9" \
    -ar 8000 -ac 1 -c:a pcm_mulaw -f wav - 2>/dev/null \
  | ffmpeg -hide_banner -loglevel error -y -i - -ar 8000 -ac 1 -c:a pcm_s16le "$out"
  echo "  $out"
done

echo "Listo. Archivos en $DST"
