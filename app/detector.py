"""Orquestación: de bytes WAV a veredicto.

Cascada pensada para latencia (es criterio de evaluación):
  1. VAD + timing + acústica barata  -> milisegundos, siempre corre
  2. [pendiente] modelo acústico pesado -> solo si el score queda en zona gris
  3. [pendiente] señal semántica (Whisper) -> solo si sigue gris

Hoy están implementados el paso 1 y el andamiaje de la cascada.
"""

from __future__ import annotations

import time

from .audio import Call, load_call
from .features import acoustic, timing
from .fusion import FusionModel, Verdict
from .vad import get_vad

_vad = get_vad()
_model = FusionModel()


def analyze_call(call: Call, turns: tuple[list, list] | None = None) -> dict:
    """Extrae todas las features de una llamada ya decodificada.

    `turns` permite inyectar segmentos precalculados (los que trae el dataset
    en turns/<id>.json). Sirve para medir cuánto pierde nuestro VAD contra
    una segmentación de referencia. OJO: en producción el scorer solo manda
    el WAV, así que el modelo que se sirve tiene que entrenarse con los
    segmentos de NUESTRO VAD, no con los de referencia.
    """
    if turns is not None:
        caller_segs, agent_segs = turns
    else:
        caller_segs = _vad.segments(call.caller, call.sr)
        agent_segs = _vad.segments(call.agent, call.sr) if call.has_agent else []

    features: dict[str, float] = {}
    features.update(timing.extract(caller_segs, agent_segs, call.duration))
    features.update(acoustic.extract(call.caller, call.sr, caller_segs))
    return {
        "features": features,
        "caller_segments": len(caller_segs),
        "agent_segments": len(agent_segs),
        "has_agent_channel": call.has_agent,
    }


def detect(blob: bytes, include_debug: bool = False) -> dict:
    """Punto de entrada único del servicio."""
    started = time.perf_counter()
    call = load_call(blob)
    analysis = analyze_call(call)
    verdict: Verdict = _model.predict(analysis["features"])
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    response = {
        "is_synthetic": bool(verdict.is_synthetic),
        "confidence": float(verdict.confidence),
    }
    # Campos extra: el scorer los ignora, a nosotros nos sirven para depurar
    # y para explicarle al juez en qué se basó la decisión.
    response["meta"] = {
        "latency_ms": round(elapsed_ms, 2),
        "duration_s": round(call.duration, 2),
        "sample_rate": call.sr,
        "caller_turns": analysis["caller_segments"],
        "agent_turns": analysis["agent_segments"],
        "has_agent_channel": analysis["has_agent_channel"],
        "model_trained": verdict.trained,
        "reason": verdict.reason,
    }
    if include_debug:
        response["features"] = analysis["features"]
    return response
