"""Extrae el vector de features de todo el dataset a un CSV.

    python scripts/extract_features.py --out data/features_vad.csv
    python scripts/extract_features.py --turns --out data/features_ref.csv

`--turns` usa los segmentos de referencia que trae el dataset en
turns/<id>.json en vez de correr nuestro VAD. Sirve para medir el techo:
si con turnos de referencia el EER baja mucho, el cuello de botella es la
segmentación y conviene invertir ahí (silero) antes que en más features.

Para el modelo que se sirve hay que usar la versión SIN --turns: en el
scoring solo llega el WAV, así que entrenar con segmentación de referencia
crearía un mismatch entre train y producción.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from app.audio import load_call_from_path
from app.detector import analyze_call
from scripts.dataset import (DEFAULT_AUDIO_DIR, DEFAULT_MANIFEST, DEFAULT_TURNS_DIR,
                             Item, discover)


def process_one(args: tuple[Item, bool]) -> dict | None:
    item, use_turns = args
    try:
        call = load_call_from_path(item.path)
        turns = item.load_turns() if use_turns else None
        if use_turns and not turns[0] and not turns[1]:
            turns = None  # sin JSON utilizable, caemos al VAD
        analysis = analyze_call(call, turns=turns)
        row = dict(analysis["features"])
        row.update({
            "call_id": item.call_id,
            "path": item.path,
            "label": item.label,
            "split": item.split,
            "speaker": item.speaker,
            "engine": item.engine,
            "group": item.group,
            "duration_s": round(call.duration, 3),
            "has_agent": int(analysis["has_agent_channel"]),
            "n_caller_segs": analysis["caller_segments"],
            "n_agent_segs": analysis["agent_segments"],
        })
        return row
    except Exception as exc:  # noqa: BLE001
        print(f"  [error] {os.path.basename(item.path)}: {type(exc).__name__}: {exc}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--turns-dir", default=DEFAULT_TURNS_DIR)
    parser.add_argument("--turns", action="store_true",
                        help="usar segmentos de referencia en vez de nuestro VAD")
    parser.add_argument("--out", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    args = parser.parse_args()

    out = args.out or ("data/features_ref.csv" if args.turns else "data/features_vad.csv")

    items = discover(args.manifest, args.audio_dir, args.turns_dir)
    if not items:
        print(f"No hay audio etiquetado. Revisa {args.manifest} y {args.audio_dir}.")
        return 1
    missing = [i for i in items if not os.path.exists(i.path)]
    if missing:
        print(f"AVISO: {len(missing)} archivos del manifiesto no están en disco "
              f"(p.ej. {os.path.basename(missing[0].path)}); se omiten.")
        items = [i for i in items if os.path.exists(i.path)]
    if args.limit:
        items = items[:args.limit]

    source = "turnos de referencia" if args.turns else "nuestro VAD"
    print(f"Extrayendo features de {len(items)} llamadas usando {source} "
          f"({args.workers} procesos)...")

    started = time.perf_counter()
    rows = []
    payload = [(item, args.turns) for item in items]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_one, p) for p in payload]
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                rows.append(result)
            if done % 50 == 0 or done == len(items):
                print(f"  {done}/{len(items)}")

    if not rows:
        print("No se pudo procesar ningún archivo.")
        return 1

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out, index=False)
    elapsed = time.perf_counter() - started
    print(f"\nGuardado: {out}  ({len(frame)} filas, {frame.shape[1]} columnas)")
    print(f"Tiempo: {elapsed:.1f}s  ->  {elapsed / len(rows) * 1000:.0f} ms por llamada")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
