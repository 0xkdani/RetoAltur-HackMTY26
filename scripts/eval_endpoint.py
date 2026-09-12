"""Evalúa el endpoint HTTP tal como lo hará el scorer del reto.

    python scripts/eval_endpoint.py --split val

Manda WAVs reales en base64 por HTTP y mide acierto y latencia extremo a
extremo. Es la única prueba que cubre todo el camino: serialización,
decodificación, features, modelo y respuesta. Todo lo demás puede estar
bien y esto fallar.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts.dataset import DEFAULT_AUDIO_DIR, DEFAULT_MANIFEST, DEFAULT_TURNS_DIR, discover


def call_endpoint(url: str, wav_path: str, timeout: float) -> tuple[dict, float]:
    with open(wav_path, "rb") as fh:
        payload = json.dumps({"audio": base64.b64encode(fh.read()).decode()}).encode()
    request = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read())
    return body, (time.perf_counter() - started) * 1000.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/detect")
    parser.add_argument("--split", default="val", help="train | val | all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    items = discover(DEFAULT_MANIFEST, DEFAULT_AUDIO_DIR, DEFAULT_TURNS_DIR)
    items = [i for i in items if os.path.exists(i.path)]
    if args.split != "all":
        items = [i for i in items if i.split == args.split]
    if args.limit:
        items = items[:args.limit]
    if not items:
        print(f"No hay llamadas en el split '{args.split}'.")
        return 1

    print(f"Evaluando {len(items)} llamadas del split '{args.split}' contra {args.url}\n")
    correct = 0
    latencies, errors, confidences = [], [], []
    wrong: list[tuple[str, int, dict]] = []

    for done, item in enumerate(items, 1):
        try:
            body, elapsed = call_endpoint(args.url, item.path, args.timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            errors.append((item.call_id, f"{type(exc).__name__}: {exc}"))
            continue
        predicted = int(bool(body.get("is_synthetic")))
        confidences.append(float(body.get("confidence", 0.5)))
        latencies.append(elapsed)
        if predicted == item.label:
            correct += 1
        else:
            wrong.append((item.call_id, item.label, body))
        if done % 25 == 0 or done == len(items):
            print(f"  {done}/{len(items)}  acierto parcial {correct}/{done - len(errors)}")

    evaluated = len(latencies)
    if not evaluated:
        print("\nNinguna llamada se pudo evaluar. ¿Está levantado el servicio?")
        for call_id, message in errors[:3]:
            print(f"  {call_id}: {message}")
        return 1

    arr = np.asarray(latencies)
    print(f"\n{'=' * 60}")
    print(f"Acierto:   {correct}/{evaluated}  ({correct / evaluated:.1%})")
    print(f"Latencia:  media {arr.mean():.0f} ms | mediana {np.median(arr):.0f} ms "
          f"| p95 {np.percentile(arr, 95):.0f} ms | max {arr.max():.0f} ms")
    print(f"Confianza: media {np.mean(confidences):.3f}")
    if errors:
        print(f"Errores de red/servidor: {len(errors)}")
        for call_id, message in errors[:3]:
            print(f"  {call_id}: {message}")
    if wrong:
        print(f"\nFalladas ({len(wrong)}):")
        for call_id, truth, body in wrong[:10]:
            label = "sintética" if truth else "humana"
            print(f"  {call_id}  era {label}, dijo is_synthetic={body.get('is_synthetic')} "
                  f"conf={body.get('confidence')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
