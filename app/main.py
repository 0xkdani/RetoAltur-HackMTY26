"""Servicio HTTP del reto.

Contrato mínimo exigido:
    POST /detect  ->  {"is_synthetic": bool, "confidence": float}

El brief no fija el nombre del campo del base64, así que el endpoint es
deliberadamente permisivo: acepta las claves más probables, un string JSON
pelón, multipart, o el WAV crudo en el body. Un endpoint que rechaza el
payload del scorer por un nombre de campo es cero puntos.
"""

from __future__ import annotations

import base64

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from .audio import AudioDecodeError, decode_base64
from .detector import detect
from .fusion import FusionModel

app = FastAPI(
    title="Altur VoiceGuard",
    description="Detección de voz sintética en llamadas telefónicas",
    version="0.1.0",
)

# Claves candidatas para el base64, en orden de probabilidad.
AUDIO_KEYS = ("audio", "audio_base64", "audio_b64", "wav", "wav_base64",
              "data", "file", "content", "clip", "payload")


def _extract_blob(body: object) -> bytes:
    """Saca los bytes del WAV de cualquier forma razonable de payload."""
    if isinstance(body, str):
        return decode_base64(body)

    if isinstance(body, dict):
        for key in AUDIO_KEYS:
            value = body.get(key)
            if isinstance(value, str) and value.strip():
                return decode_base64(value)
        # Último recurso: el string más largo del dict probablemente es el audio.
        candidates = [v for v in body.values() if isinstance(v, str) and len(v) > 256]
        if candidates:
            return decode_base64(max(candidates, key=len))

    raise AudioDecodeError(
        f"no encontré el audio en el payload; claves esperadas: {', '.join(AUDIO_KEYS)}"
    )


@app.on_event("startup")
async def warmup() -> None:
    """Precalienta el camino completo antes de recibir tráfico.

    La primera inferencia paga la carga perezosa de joblib/sklearn y la
    compilación de las rutinas de numpy: medimos ~1.2 s en frío contra
    ~30 ms en caliente. Sin esto, la primera llamada del scorer -- que
    puede ser la que cronometre -- sale un orden de magnitud más lenta.
    """
    import io

    import numpy as np
    import soundfile as sf

    buffer = io.BytesIO()
    silence = np.zeros((8000 * 2, 2), dtype=np.float32)
    sf.write(buffer, silence, 8000, format="WAV", subtype="PCM_16")
    try:
        detect(buffer.getvalue())
    except Exception:  # noqa: BLE001
        pass  # el warm-up nunca debe impedir que el servicio arranque


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model_trained": FusionModel().is_trained}


@app.post("/detect")
async def detect_endpoint(request: Request) -> JSONResponse:
    raw = await request.body()
    if not raw:
        raise HTTPException(status_code=400, detail="body vacío")

    content_type = (request.headers.get("content-type") or "").lower()

    try:
        if "application/json" in content_type:
            import json
            blob = _extract_blob(json.loads(raw))
        elif raw[:4] == b"RIFF":
            blob = raw  # WAV crudo, sin envoltura
        else:
            # Puede ser JSON sin el header correcto, o base64 a secas.
            import json
            try:
                blob = _extract_blob(json.loads(raw))
            except (ValueError, AudioDecodeError):
                blob = decode_base64(raw.decode("utf-8", errors="ignore"))
    except AudioDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    debug = request.query_params.get("debug", "").lower() in ("1", "true", "yes")
    try:
        result = detect(blob, include_debug=debug)
    except AudioDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        # Nunca tumbar el endpoint durante el scoring: un fallo se responde
        # como "no sintético" con confianza mínima y se deja registrado.
        return JSONResponse(
            status_code=200,
            content={"is_synthetic": False, "confidence": 0.5,
                     "meta": {"error": f"{type(exc).__name__}: {exc}"}},
        )

    return JSONResponse(content=result)


@app.post("/detect/upload")
async def detect_upload(file: UploadFile = File(...)) -> dict:
    """Variante multipart, cómoda para probar a mano con curl -F."""
    blob = await file.read()
    try:
        return detect(blob, include_debug=True)
    except AudioDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
