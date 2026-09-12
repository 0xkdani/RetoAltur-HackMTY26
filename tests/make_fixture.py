"""Genera llamadas sintéticas de prueba con estructura de turnos controlada.

No sirve para entrenar -- el audio es ruido con forma de voz. Sirve para
verificar que el pipeline corre end-to-end y que las features de timing
distinguen los dos patrones de conversación que nos importan.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

SR = 8000
RNG = np.random.default_rng(7)


def _speech_like(duration: float, f0: float = 120.0, jitter: float = 0.02) -> np.ndarray:
    """Tono con armónicos y envolvente silábica: suficiente para activar el VAD."""
    n = int(duration * SR)
    t = np.arange(n) / SR
    freq = f0 * (1 + jitter * np.sin(2 * np.pi * 1.3 * t) + jitter * RNG.standard_normal(n) * 0.1)
    phase = np.cumsum(2 * np.pi * freq / SR)
    signal = sum(np.sin(k * phase) / k for k in (1, 2, 3, 4))
    envelope = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 4.0 * t))  # ritmo silábico ~4 Hz
    return (signal * envelope * 0.25).astype(np.float32)


def build_call(turns: list[tuple[str, float, float]], total: float) -> np.ndarray:
    """turns: lista de (canal, inicio_s, duracion_s) con canal en {caller, agent}."""
    n = int(total * SR)
    stereo = RNG.standard_normal((n, 2)).astype(np.float32) * 0.002  # piso de ruido
    for who, start, duration in turns:
        col = 0 if who == "caller" else 1
        f0 = 115.0 if who == "caller" else 165.0
        chunk = _speech_like(duration, f0=f0)
        a = int(start * SR)
        b = min(a + len(chunk), n)
        stereo[a:b, col] += chunk[:b - a]
    return np.clip(stereo, -1.0, 1.0)


def human_like() -> np.ndarray:
    """Latencias irregulares, un barge-in, una pausa larga y un '¿bueno?'."""
    turns = [
        ("agent", 0.5, 2.5),
        ("caller", 3.15, 1.4),   # 150 ms: se le encimó casi
        ("agent", 5.0, 2.0),
        ("caller", 8.1, 0.6),    # 1.1 s: se quedó pensando
        ("caller", 9.0, 1.8),    # pausa interna y retoma
        ("agent", 11.5, 3.0),
        ("caller", 12.8, 0.9),   # barge-in en medio del agente
        ("agent", 16.0, 1.5),
        ("caller", 21.0, 0.5),   # "¿bueno?" tras silencio largo
    ]
    return build_call(turns, 24.0)


def bot_like() -> np.ndarray:
    """Latencia casi constante (~900 ms), sin barge-in, turnos parejos."""
    turns = []
    t = 0.5
    for _ in range(5):
        turns.append(("agent", t, 2.2))
        t += 2.2 + 0.9              # siempre la misma latencia
        turns.append(("caller", t, 1.6))
        t += 1.6 + 0.4
    return build_call(turns, t + 1.0)


if __name__ == "__main__":
    sf.write("tests/fixture_human.wav", human_like(), SR, subtype="PCM_16")
    sf.write("tests/fixture_bot.wav", bot_like(), SR, subtype="PCM_16")
    print("Escritos tests/fixture_human.wav y tests/fixture_bot.wav")
