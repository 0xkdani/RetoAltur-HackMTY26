"""Primer contacto con el dataset: qué hay, cómo está balanceado, qué formato trae.

    python scripts/ingest.py [--root data/raw]

Correr esto ANTES de entrenar nada. Si el balance de clases está muy
cargado o las duraciones son dispares, eso cambia las decisiones de modelo.
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import soundfile as sf

from scripts.dataset import DEFAULT_AUDIO_DIR, DEFAULT_MANIFEST, DEFAULT_TURNS_DIR, discover


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--limit", type=int, default=0, help="0 = todos")
    args = parser.parse_args()

    items = discover(args.manifest, args.audio_dir, DEFAULT_TURNS_DIR)
    if not items:
        print(f"No encontré audio etiquetado via {args.manifest}.")
        print("Esperaba data/raw/{human,synthetic}/*.wav o data/labels.csv")
        return 1

    if args.limit:
        items = items[:args.limit]

    print(f"Archivos encontrados: {len(items)}")
    labels = collections.Counter("sintético" if i.label else "humano" for i in items)
    for name, count in labels.items():
        print(f"  {name:>10}: {count:5d}  ({count / len(items):.1%})")

    speakers = {i.speaker for i in items if i.speaker}
    engines = {i.engine for i in items if i.engine}
    print(f"Hablantes distintos: {len(speakers) or 'desconocido'}")
    print(f"Motores distintos:   {len(engines) or 'desconocido'}")
    splits = {}
    for item in items:
        if item.split:
            splits[item.split] = splits.get(item.split, 0) + 1
    if splits:
        detalle = " | ".join(f"{k}: {v}" for k, v in sorted(splits.items()))
        print(f"Splits del manifiesto: {detalle}  (speaker-disjoint segun el reto)")
    elif not speakers and not engines:
        print("  AVISO: sin split ni speaker/engine, la particion sera aleatoria")
        print("         y la metrica saldra inflada por fuga entre train y test.")

    durations, rates, channels, fallidos = [], collections.Counter(), collections.Counter(), []
    for item in items:
        try:
            info = sf.info(item.path)
            durations.append(info.duration)
            rates[info.samplerate] += 1
            channels[info.channels] += 1
        except Exception as exc:  # noqa: BLE001
            fallidos.append((item.path, str(exc)))

    if durations:
        arr = np.asarray(durations)
        print(f"\nDuración (s): media {arr.mean():.1f} | mediana {np.median(arr):.1f} "
              f"| min {arr.min():.1f} | max {arr.max():.1f} | total {arr.sum() / 3600:.2f} h")
    print(f"Sample rates: {dict(rates)}")
    print(f"Canales:      {dict(channels)}")
    if channels and set(channels) != {2}:
        print("  AVISO: hay audio no estéreo; sin canal de agente las features de turno se apagan.")
    if fallidos:
        print(f"\nNo se pudieron leer {len(fallidos)} archivos. Primeros 5:")
        for path, err in fallidos[:5]:
            print(f"  {os.path.basename(path)}: {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
