"""Simulacro del día del judging, contra el servicio real.

    python scripts/test_judging.py --url https://tu-servicio.onrender.com

Corre las mismas cosas que va a hacer el jurado, más las que NO va a hacer
pero que podrían tumbarnos: audio corrupto, payloads raros, mono en vez de
estéreo. Un 500 durante la calificación es cero puntos, así que preferimos
encontrarlo aquí.

Pruebas:
  1. Arranque en frío   — cuánto tarda la primera petición
  2. Robustez           — basura de entrada: ¿aguanta sin tronar?
  3. Llamadas reales    — acierto y latencia sobre el dataset
  4. Estado final       — lo que el servicio registró de todo esto
"""

from __future__ import annotations

import argparse
import base64
import sys as _sys
_sys.stdout.reconfigure(encoding="utf-8")
import csv
import io
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import soundfile as sf

RESULTADOS: list[tuple[str, bool, str]] = []


def check(nombre: str, ok: bool, detalle: str = "") -> None:
    RESULTADOS.append((nombre, ok, detalle))
    marca = "OK   " if ok else "FALLA"
    print(f"  [{marca}] {nombre}" + (f"  -- {detalle}" if detalle else ""))


def pedir(url: str, cuerpo: bytes, timeout: float = 180) -> tuple[int, dict | str, float]:
    """Manda una petición y devuelve (código, respuesta, milisegundos)."""
    req = urllib.request.Request(url, data=cuerpo,
                                 headers={"Content-Type": "application/json"})
    inicio = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            ms = (time.perf_counter() - inicio) * 1000
            try:
                return r.status, json.loads(r.read()), ms
            except json.JSONDecodeError:
                return r.status, "(no es JSON)", ms
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "ignore")[:120], (time.perf_counter() - inicio) * 1000
    except Exception as e:  # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}", (time.perf_counter() - inicio) * 1000


def envolver(audio: bytes) -> bytes:
    return json.dumps({"audio": base64.b64encode(audio).decode()}).encode()


def wav_falso(canales: int = 2, segundos: float = 3.0, sr: int = 8000) -> bytes:
    buf = io.BytesIO()
    datos = np.random.default_rng(0).standard_normal(
        (int(sr * segundos), canales)).astype(np.float32) * 0.05
    sf.write(buf, datos if canales > 1 else datos[:, 0], sr,
             format="WAV", subtype="PCM_16")
    return buf.getvalue()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True, help="URL base del servicio")
    p.add_argument("--llamadas", type=int, default=30)
    args = p.parse_args()
    base = args.url.rstrip("/")

    print(f"\nSimulacro de judging contra {base}")
    print("=" * 64)

    # ---------------------------------------------------------------
    print("\n1. ARRANQUE EN FRÍO")
    print("-" * 64)
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=180) as r:
            salud = json.loads(r.read())
        frio = (time.perf_counter() - t) * 1000
    except Exception as e:  # noqa: BLE001
        check("el servicio responde", False, str(e))
        return 1

    check("el servicio responde", salud.get("status") == "ok")
    check("el modelo está cargado", salud.get("model_trained") is True)
    if frio > 10000:
        print(f"  [aviso] la primera petición tardó {frio/1000:.0f}s: estaba dormido.")
        print("          Despiértalo unos minutos antes de que lleguen los jueces.")
    else:
        print(f"  [info ] primera petición: {frio:.0f} ms (estaba despierto)")

    # ---------------------------------------------------------------
    print("\n2. ROBUSTEZ — lo que NO debe tumbarnos")
    print("-" * 64)
    casos = [
        ("base64 inválido", json.dumps({"audio": "esto no es base64 ###"}).encode()),
        ("audio vacío", json.dumps({"audio": ""}).encode()),
        ("JSON sin el campo audio", json.dumps({"otra_cosa": 123}).encode()),
        ("bytes que no son WAV", envolver(b"no soy un wav, soy texto")),
        ("WAV mono en vez de estéreo", envolver(wav_falso(canales=1))),
        ("WAV estéreo de puro ruido", envolver(wav_falso(canales=2))),
        ("WAV muy corto (0.2 s)", envolver(wav_falso(canales=2, segundos=0.2))),
    ]
    for nombre, cuerpo in casos:
        codigo, resp, _ = pedir(f"{base}/detect", cuerpo, timeout=120)
        # Lo importante: NUNCA un 500. Un 400 con mensaje claro está bien;
        # un 200 con veredicto neutro también.
        acepta = codigo in (200, 400, 422)
        detalle = f"HTTP {codigo}"
        if codigo == 200 and isinstance(resp, dict):
            detalle += f" -> is_synthetic={resp.get('is_synthetic')}"
        check(nombre, acepta, detalle)

    # ---------------------------------------------------------------
    print(f"\n3. LLAMADAS REALES — {args.llamadas} al azar del dataset")
    print("-" * 64)
    manifest = "data/manifest.csv"
    if not os.path.exists(manifest):
        print("  (sin dataset local, se omite esta parte)")
    else:
        with open(manifest, encoding="utf-8-sig") as f:
            todas = [r for r in csv.DictReader(f)
                     if os.path.exists(f"data/audio/{r['anon_id']}.wav")]
        random.seed(13)
        muestra = random.sample(todas, min(args.llamadas, len(todas)))

        aciertos, tiempos, fallidas = 0, [], []
        for i, fila in enumerate(muestra, 1):
            with open(f"data/audio/{fila['anon_id']}.wav", "rb") as f:
                audio = f.read()
            codigo, resp, ms = pedir(f"{base}/detect", envolver(audio))
            if codigo != 200 or not isinstance(resp, dict):
                fallidas.append((fila["anon_id"], f"HTTP {codigo}"))
                continue
            tiempos.append(ms)
            bien = bool(resp.get("is_synthetic")) == (fila["label"] == "synthetic")
            aciertos += bien
            if not bien:
                fallidas.append((fila["anon_id"], f"era {fila['label']}"))
            if i % 10 == 0:
                print(f"    {i}/{len(muestra)}...")

        total = len(tiempos)
        if total:
            arr = np.array(tiempos)
            check("todas las peticiones respondieron", len(fallidas) == 0 or
                  all("HTTP" not in f[1] for f in fallidas),
                  f"{total}/{len(muestra)} con respuesta")
            check("acierto por encima del 90%", aciertos / total >= 0.90,
                  f"{aciertos}/{total} ({aciertos/total:.0%})")
            print(f"  [info ] latencia: media {arr.mean():.0f} ms | "
                  f"mediana {np.median(arr):.0f} ms | p95 {np.percentile(arr,95):.0f} ms")
            if fallidas:
                print(f"  [info ] no acertadas: {', '.join(f[0] for f in fallidas[:5])}")

    # ---------------------------------------------------------------
    print("\n4. LO QUE EL SERVICIO REGISTRÓ")
    print("-" * 64)
    try:
        with urllib.request.urlopen(f"{base}/stats", timeout=60) as r:
            st = json.loads(r.read())
        print(f"  consultas totales : {st.get('consultas')}")
        print(f"  dijimos sintético : {st.get('dijimos_sintetico')}")
        print(f"  dijimos humano    : {st.get('dijimos_humano')}")
        check("cero errores internos", st.get("errores", 0) == 0,
              f"{st.get('errores')} errores")
    except Exception as e:  # noqa: BLE001
        check("/stats responde", False, str(e))

    # ---------------------------------------------------------------
    print("\n" + "=" * 64)
    pasaron = sum(1 for _, ok, _ in RESULTADOS if ok)
    print(f"  {pasaron}/{len(RESULTADOS)} verificaciones pasaron")
    if pasaron == len(RESULTADOS):
        print("  Listo para el jurado.")
    else:
        print("  Revisa las que fallaron antes de presentar:")
        for nombre, ok, detalle in RESULTADOS:
            if not ok:
                print(f"    - {nombre}: {detalle}")
    print("=" * 64)
    return 0 if pasaron == len(RESULTADOS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
