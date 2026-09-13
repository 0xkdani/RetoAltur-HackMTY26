"""Demostración en vivo: manda llamadas reales al servicio y muestra el veredicto.

    python scripts/demo.py --url https://retoaltur-hackmty26.onrender.com

Pensado para enseñarlo en pantalla: imprime una línea por llamada conforme
van respondiendo, con el veredicto, la confianza y cuánto tardó.

Es el mismo camino que recorre el jurado: WAV real -> base64 -> HTTP -> veredicto.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import random
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="https://retoaltur-hackmty26.onrender.com")
    p.add_argument("--n", type=int, default=10)
    args = p.parse_args()
    base = args.url.rstrip("/")

    with open("data/manifest.csv", encoding="utf-8-sig") as f:
        todas = [r for r in csv.DictReader(f)
                 if r["split"] == "val" and os.path.exists(f"data/audio/{r['anon_id']}.wav")]
    random.seed(42)
    muestra = random.sample(todas, min(args.n, len(todas)))

    print()
    print(f"  Probando el detector contra {base}")
    print(f"  {len(muestra)} llamadas reales del dataset")
    print()
    print(f"  {'#':>4}  {'llamada':<20} {'peso':>8}   {'veredicto':<9} {'seguro':>7} {'tiempo':>8}")
    print(f"  {'-'*4}  {'-'*20} {'-'*8}   {'-'*9} {'-'*7} {'-'*8}")

    aciertos, tiempos = 0, []
    for i, fila in enumerate(muestra, 1):
        ruta = f"data/audio/{fila['anon_id']}.wav"
        with open(ruta, "rb") as f:
            audio = f.read()
        cuerpo = json.dumps({"audio": base64.b64encode(audio).decode()}).encode()

        req = urllib.request.Request(base + "/detect", data=cuerpo,
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=300) as r:
            resp = json.loads(r.read())
        ms = (time.perf_counter() - t0) * 1000
        tiempos.append(ms)

        dijo = "MAQUINA" if resp["is_synthetic"] else "PERSONA"
        era_maquina = fila["label"] == "synthetic"
        bien = resp["is_synthetic"] == era_maquina
        aciertos += bien

        print(f"  {i:>3}.  {fila['anon_id']:<20} {len(audio)/1e6:>6.1f} MB   "
              f"{dijo:<9} {resp['confidence']*100:>6.0f}% {ms:>7.0f} ms   "
              f"{'OK' if bien else 'FALLA'}", flush=True)

    print()
    media = sum(tiempos) / len(tiempos)
    print(f"  {aciertos} de {len(muestra)} correctas  ·  {media:.0f} ms de promedio")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
