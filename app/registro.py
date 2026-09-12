"""Registro de cada consulta que llega al endpoint.

Para qué sirve durante el judging: saber si los jueces ya empezaron a
mandar llamadas, cuántas llevan, qué contestamos y cuánto tardamos. Sin
esto uno está a ciegas mirando una terminal vacía.

Qué NO se guarda, a propósito: **el audio**. Son voces de personas reales
que prestaron su voz para el reto, y los términos de Altur piden no
intentar identificar a nadie. Guardamos solo los datos de la consulta
(cuándo, cuánto pesó, qué respondimos), nunca el contenido.

El registro nunca debe tumbar el servicio: si escribir falla, se ignora.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone

RUTA = os.environ.get("VOICEGUARD_LOG", "logs/consultas.jsonl")

# Varias peticiones pueden llegar a la vez; el candado evita que dos
# escrituras se encimen y dejen una línea rota.
_candado = threading.Lock()


def anotar(evento: dict) -> None:
    """Escribe una línea en el registro. Si algo falla, no pasa nada."""
    try:
        ahora = datetime.now().astimezone()
        fila = {
            "hora": ahora.strftime("%Y-%m-%d %H:%M:%S"),   # hora local (México)
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **evento,
        }
        with _candado:
            os.makedirs(os.path.dirname(RUTA) or ".", exist_ok=True)
            with open(RUTA, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(fila, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001
        pass  # un fallo al registrar jamás debe costar una consulta


def leer_todo() -> list[dict]:
    """Devuelve todas las consultas registradas. Ignora líneas corruptas."""
    if not os.path.exists(RUTA):
        return []
    filas = []
    try:
        with open(RUTA, encoding="utf-8") as fh:
            for linea in fh:
                linea = linea.strip()
                if not linea:
                    continue
                try:
                    filas.append(json.loads(linea))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return filas


def resumen() -> dict:
    """Resumen en vivo para el endpoint /stats."""
    filas = leer_todo()
    if not filas:
        return {"consultas": 0, "mensaje": "todavía no llega ninguna consulta"}

    latencias = sorted(f["latencia_ms"] for f in filas if "latencia_ms" in f)
    sinteticas = sum(1 for f in filas if f.get("is_synthetic") is True)
    humanas = sum(1 for f in filas if f.get("is_synthetic") is False)
    errores = [f for f in filas if f.get("error")]

    def percentil(datos: list, p: float) -> float:
        if not datos:
            return 0.0
        i = min(len(datos) - 1, int(len(datos) * p))
        return round(datos[i], 1)

    return {
        "consultas": len(filas),
        "dijimos_sintetico": sinteticas,
        "dijimos_humano": humanas,
        "errores": len(errores),
        "latencia_ms": {
            "media": round(sum(latencias) / len(latencias), 1) if latencias else 0,
            "mediana": percentil(latencias, 0.50),
            "p95": percentil(latencias, 0.95),
            "maxima": round(latencias[-1], 1) if latencias else 0,
        },
        "primera": filas[0].get("hora"),
        "ultima": filas[-1].get("hora"),
        "ultimo_error": errores[-1].get("error") if errores else None,
    }
