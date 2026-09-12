"""Señal B: comportamiento conversacional.

La idea de fondo: un humano y un pipeline TTS no *reaccionan* igual.
El humano responde con latencias caóticas (a veces 150 ms, a veces 2 s),
se le encima al agente, se calla a medias cuando lo interrumpen, mete
pausas irregulares. Un bot responde con una latencia más alta y mucho
más *consistente*, porque siempre paga el mismo costo de ASR + LLM + TTS.

Por eso varias features miden dispersión (std, IQR, coef. de variación),
no solo promedios: la consistencia es la delación.
"""

from __future__ import annotations

import numpy as np

from ..vad import Segment, overlap_duration, total_speech

# Orden fijo: el vector de features tiene que ser idéntico en train y en serving.
TIMING_FEATURES: tuple[str, ...] = (
    "resp_latency_mean",
    "resp_latency_median",
    "resp_latency_std",
    "resp_latency_cv",
    "resp_latency_min",
    "resp_latency_p90",
    "resp_latency_iqr",
    "resp_fast_ratio",
    "resp_count",
    "bargein_rate",
    "bargein_reaction_mean",
    "yield_time_mean",
    "yield_time_std",
    "overlap_ratio",
    "caller_speech_ratio",
    "caller_turn_count",
    "caller_turn_dur_mean",
    "caller_turn_dur_std",
    "caller_pause_mean",
    "caller_pause_std",
    "caller_pause_count",
    "silence_probe_count",
    "turn_dur_cv",
)

MAX_RESPONSE_LATENCY = 8.0  # más que esto no es una respuesta, es otra cosa


def _stats(values: list[float]) -> dict[str, float]:
    """Estadísticas robustas a listas vacías o de un solo elemento."""
    if not values:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "cv": 0.0,
                "min": 0.0, "p90": 0.0, "iqr": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    return {
        "mean": mean,
        "median": float(np.median(arr)),
        "std": std,
        # El coeficiente de variación es la feature clave: normaliza la
        # dispersión por la magnitud, así no confunde "bot lento" con "bot
        # consistente". Un bot tiende a CV bajo.
        "cv": float(std / mean) if mean > 1e-6 else 0.0,
        "min": float(arr.min()),
        "p90": float(np.percentile(arr, 90)),
        "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
    }


def _response_latencies(caller: list[Segment], agent: list[Segment]) -> list[float]:
    """Para cada turno del agente, cuánto tardó el caller en arrancar."""
    latencies = []
    for _, agent_end in agent:
        onset = next((s for s, _ in caller if s >= agent_end), None)
        if onset is not None and onset - agent_end <= MAX_RESPONSE_LATENCY:
            latencies.append(onset - agent_end)
    return latencies


def _bargeins(caller: list[Segment], agent: list[Segment]) -> tuple[int, list[float]]:
    """Turnos del caller que arrancan mientras el agente sigue hablando.

    Devuelve cuántos hubo y, de cada uno, en qué punto del turno del agente
    ocurrió (reaccionar rápido a lo que el agente está diciendo es muy humano).
    """
    count, reactions = 0, []
    for c_start, _ in caller:
        for a_start, a_end in agent:
            if a_start < c_start < a_end:
                count += 1
                reactions.append(c_start - a_start)
                break
    return count, reactions


def _yield_times(caller: list[Segment], agent: list[Segment]) -> list[float]:
    """Cuánto tarda el caller en callarse cuando el agente se le encima.

    Es el "recovery" del brief: al humano lo atropellan y se apaga de forma
    desordenada; el bot suele cortar parejo o no cortar nunca.
    """
    times = []
    for c_start, c_end in caller:
        interrupt = next((a_s for a_s, _ in agent if c_start < a_s < c_end), None)
        if interrupt is not None:
            times.append(c_end - interrupt)
    return times


def _internal_pauses(caller: list[Segment]) -> list[float]:
    """Huecos entre turnos consecutivos del caller que no son cambio de turno."""
    return [caller[i + 1][0] - caller[i][1] for i in range(len(caller) - 1)]


def _silence_probes(caller: list[Segment], agent: list[Segment]) -> int:
    """Veces que el caller habla sin que el agente haya dicho nada antes.

    Son los "¿bueno?", "¿hola?", "¿me escucha?" cuando el agente se queda
    callado. Los humanos los sueltan casi siempre; los bots, casi nunca.
    """
    if not agent:
        return 0
    probes = 0
    for c_start, _ in caller:
        prev_end = max((e for _, e in agent if e <= c_start), default=None)
        if prev_end is not None and c_start - prev_end > 3.0:
            probes += 1
    return probes


def extract(caller: list[Segment], agent: list[Segment], duration: float) -> dict[str, float]:
    """Vector de features conversacionales. Siempre devuelve TIMING_FEATURES completo."""
    latencies = _response_latencies(caller, agent)
    lat = _stats(latencies)

    bargein_count, bargein_reactions = _bargeins(caller, agent)
    yields = _yield_times(caller, agent)
    pauses = _internal_pauses(caller)
    turn_durs = [e - s for s, e in caller]

    caller_speech = total_speech(caller)
    overlap = overlap_duration(caller, agent) if agent else 0.0
    denom = max(caller_speech, 1e-6)
    turn_stats = _stats(turn_durs)

    return {
        "resp_latency_mean": lat["mean"],
        "resp_latency_median": lat["median"],
        "resp_latency_std": lat["std"],
        "resp_latency_cv": lat["cv"],
        "resp_latency_min": lat["min"],
        "resp_latency_p90": lat["p90"],
        "resp_latency_iqr": lat["iqr"],
        # Arrancar antes de 350 ms casi siempre es solapamiento humano real:
        # nadie corre un ASR+LLM+TTS en ese tiempo.
        "resp_fast_ratio": (float(np.mean([l < 0.35 for l in latencies]))
                            if latencies else 0.0),
        "resp_count": float(len(latencies)),
        "bargein_rate": bargein_count / max(len(caller), 1),
        "bargein_reaction_mean": _stats(bargein_reactions)["mean"],
        "yield_time_mean": _stats(yields)["mean"],
        "yield_time_std": _stats(yields)["std"],
        "overlap_ratio": overlap / denom,
        "caller_speech_ratio": caller_speech / max(duration, 1e-6),
        "caller_turn_count": float(len(caller)),
        "caller_turn_dur_mean": turn_stats["mean"],
        "caller_turn_dur_std": turn_stats["std"],
        "caller_pause_mean": _stats(pauses)["mean"],
        "caller_pause_std": _stats(pauses)["std"],
        "caller_pause_count": float(len(pauses)),
        "silence_probe_count": float(_silence_probes(caller, agent)),
        "turn_dur_cv": turn_stats["cv"],
    }
