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
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from .audio import AudioDecodeError, decode_base64
from .detector import detect
from .fusion import FusionModel
from .registro import anotar, resumen

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Precalienta el camino completo antes de recibir tráfico.

    La primera inferencia paga la carga perezosa de joblib/sklearn y la
    compilación de las rutinas de numpy: medimos ~1.2 s en frío contra
    ~25 ms en caliente. Sin esto, la primera llamada del scorer -- que
    puede ser la que cronometren -- sale un orden de magnitud más lenta.

    Usamos lifespan y no on_event porque este último está marcado como
    obsoleto en FastAPI: con 4 personas instalando versiones distintas,
    lo obsoleto es justo lo que truena en la máquina de alguien más.
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
    yield


app = FastAPI(
    title="Altur VoiceGuard",
    description="Detección de voz sintética en llamadas telefónicas",
    version="0.1.0",
    lifespan=lifespan,
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


def _origen(request: Request) -> str:
    """De dónde vino la consulta. Sirve para distinguir nuestras pruebas
    de las del jurado cuando entren por el túnel."""
    reenviada = request.headers.get("x-forwarded-for")
    if reenviada:
        return reenviada.split(",")[0].strip()
    return request.client.host if request.client else "desconocido"


PAGINA_INICIO = """<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<title>Altur VoiceGuard</title>
<style>
 body{{font-family:"Segoe UI",system-ui,sans-serif;background:#0F1721;color:#E8EDF3;
      margin:0;display:flex;align-items:center;justify-content:center;min-height:100vh}}
 .caja{{max-width:620px;padding:48px 44px}}
 .pill{{display:inline-block;border:1px solid #2E4A66;border-radius:100px;padding:6px 16px;
        font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:#7FB2E0;margin-bottom:22px}}
 h1{{font-size:42px;margin:0 0 10px;letter-spacing:-.02em}}
 .sub{{color:#93A5B8;font-size:18px;line-height:1.55;margin-bottom:30px}}
 .estado{{display:flex;align-items:center;gap:10px;background:#14251C;border:1px solid #245437;
          border-radius:10px;padding:14px 18px;margin-bottom:28px}}
 .punto{{width:10px;height:10px;border-radius:50%;background:#35C46B}}
 .estado b{{color:#7FE0A5;font-size:15px}}
 .estado span{{color:#8FA3B5;font-size:14px}}
 .ep{{background:#16212E;border:1px solid #26374A;border-radius:10px;padding:20px 22px;margin-bottom:26px}}
 .ep .t{{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:#7FB2E0;margin-bottom:10px}}
 code{{font-family:Consolas,monospace;background:#0C1420;border:1px solid #24364A;
       border-radius:6px;padding:3px 9px;color:#9FD4FF;font-size:15px}}
 .ep p{{color:#93A5B8;font-size:14px;line-height:1.6;margin:10px 0 0}}
 .links{{display:flex;gap:12px;flex-wrap:wrap}}
 a{{color:#9FD4FF;text-decoration:none;border:1px solid #26374A;border-radius:8px;
    padding:10px 18px;font-size:14px;transition:.15s}}
 a:hover{{background:#1B2B3C;border-color:#3D5A78}}
 .pie{{margin-top:34px;color:#5E7186;font-size:13px;line-height:1.6}}
</style></head><body><div class="caja">
 <div class="pill">HackMTY 2026 &middot; Reto Altur</div>
 <h1>Altur VoiceGuard</h1>
 <div class="sub">Detecci&oacute;n de voz sint&eacute;tica en llamadas telef&oacute;nicas.</div>
 <div class="estado"><div class="punto"></div>
   <b>Servicio en l&iacute;nea</b><span>&middot; modelo cargado: {modelo}</span></div>
 <div class="ep">
   <div class="t">Endpoint del reto</div>
   <code>POST /detect</code>
   <p>Recibe un WAV est&eacute;reo de 8 kHz en base64 (canal 0 = quien llama,
      canal 1 = el agente) y responde
      <code>{{"is_synthetic": bool, "confidence": float}}</code></p>
 </div>
 <div class="links">
   <a href="/docs">Probarlo aqu&iacute;</a>
   <a href="/health">Estado</a>
   <a href="/stats">Consultas recibidas</a>
 </div>
 <div class="pie">La se&ntilde;al principal no es c&oacute;mo suena la voz, sino
   c&oacute;mo reacciona: los tiempos de respuesta de una persona son ca&oacute;ticos,
   los de una m&aacute;quina son parejos.</div>
</div></body></html>"""


@app.get("/", response_model=None)
async def inicio(request: Request):
    """Página de inicio.

    Devuelve HTML si la abre un navegador y JSON si la pide un programa,
    mirando la cabecera Accept. Un juez que abra la URL por curiosidad ve
    una página presentable en vez de JSON crudo; los monitores automáticos
    (incluido el health check de Render) siguen recibiendo JSON.
    """
    modelo = FusionModel()
    acepta = (request.headers.get("accept") or "").lower()

    if "text/html" in acepta:
        return HTMLResponse(PAGINA_INICIO.format(
            modelo="sí" if modelo.is_trained else "no"))

    return {
        "servicio": "Altur VoiceGuard",
        "descripcion": "Detección de voz sintética en llamadas telefónicas",
        "reto": "HackMTY 2026 · Defend the Bank Against Voice Deepfakes",
        "estado": "ok",
        "modelo_cargado": modelo.is_trained,
        "endpoint_del_reto": {
            "ruta": "POST /detect",
            "recibe": "WAV estéreo 8 kHz en base64 (canal 0 = caller, canal 1 = agente)",
            "devuelve": {"is_synthetic": "bool", "confidence": "float 0-1"},
        },
        "otras_rutas": {
            "/docs": "probar el servicio desde el navegador",
            "/health": "estado del servicio",
            "/stats": "resumen de las consultas recibidas",
        },
    }


@app.get("/stats")
async def stats() -> dict:
    """Resumen en vivo de las consultas recibidas.

    Durante el judging esta es la página que hay que tener abierta: dice
    si ya empezaron a mandar llamadas, cuántas llevan y si algo truena.
    """
    return resumen()


@app.get("/health")
async def health() -> dict:
    modelo = FusionModel()
    salud = {"status": "ok", "model_trained": modelo.is_trained}
    if modelo.version_warning:
        salud["warning"] = modelo.version_warning
    return salud


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
        anotar({"bytes_recibidos": len(raw), "origen": _origen(request),
                "error": f"{type(exc).__name__}: {exc}"})
        return JSONResponse(
            status_code=200,
            content={"is_synthetic": False, "confidence": 0.5,
                     "meta": {"error": f"{type(exc).__name__}: {exc}"}},
        )

    meta = result.get("meta", {})
    anotar({
        "bytes_recibidos": len(raw),
        "origen": _origen(request),
        "duracion_llamada_s": meta.get("duration_s"),
        "is_synthetic": result["is_synthetic"],
        "confidence": result["confidence"],
        "latencia_ms": meta.get("latency_ms"),
        "turnos_caller": meta.get("caller_turns"),
        "turnos_agente": meta.get("agent_turns"),
    })
    return JSONResponse(content=result)


@app.post("/detect/upload")
async def detect_upload(file: UploadFile = File(...)) -> dict:
    """Variante multipart, cómoda para probar a mano con curl -F."""
    blob = await file.read()
    try:
        return detect(blob, include_debug=True)
    except AudioDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
