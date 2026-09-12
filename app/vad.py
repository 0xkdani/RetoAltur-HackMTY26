"""Detección de actividad de voz.

Por defecto usa un VAD de energía con umbral adaptativo e histéresis: no
necesita descargar nada, corre en microsegundos y a 8 kHz telefónico se
comporta razonablemente. Si más adelante quieren precisión, `SileroVAD`
tiene el mismo contrato -- cámbienlo en un solo lugar.

Todo devuelve segmentos como lista de (inicio_s, fin_s).
"""

from __future__ import annotations

import numpy as np

Segment = tuple[float, float]


def _frame_energies(x: np.ndarray, sr: int, frame_ms: float, hop_ms: float) -> np.ndarray:
    """Energía RMS por frame, en dB."""
    frame = max(1, int(sr * frame_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))
    if x.size < frame:
        return np.zeros(0, dtype=np.float32)
    n_frames = 1 + (x.size - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx]
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    return (20.0 * np.log10(rms + 1e-12)).astype(np.float32)


def _merge(segments: list[Segment], min_speech: float, max_gap: float) -> list[Segment]:
    """Pega segmentos separados por huecos cortos y tira los muy breves."""
    if not segments:
        return []
    merged = [list(segments[0])]
    for start, end in segments[1:]:
        if start - merged[-1][1] <= max_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged if e - s >= min_speech]


class EnergyVAD:
    """VAD de energía con piso de ruido estimado por percentil.

    El umbral no es fijo: se calcula sobre cada canal, así que aguanta
    llamadas con niveles muy distintos (es justo lo que pasa en telefonía).
    """

    def __init__(
        self,
        frame_ms: float = 30.0,
        hop_ms: float = 10.0,
        onset_db: float = 9.0,    # cuánto sobre el piso de ruido para abrir un turno
        offset_db: float = 5.0,   # histéresis: cerrar cuesta menos que abrir
        min_speech: float = 0.12,
        max_gap: float = 0.25,    # huecos < esto se consideran pausa interna, no fin de turno
    ) -> None:
        self.frame_ms = frame_ms
        self.hop_ms = hop_ms
        self.onset_db = onset_db
        self.offset_db = offset_db
        self.min_speech = min_speech
        self.max_gap = max_gap

    def segments(self, x: np.ndarray, sr: int) -> list[Segment]:
        db = _frame_energies(x, sr, self.frame_ms, self.hop_ms)
        if db.size == 0:
            return []

        noise_floor = float(np.percentile(db, 10))
        peak = float(np.percentile(db, 95))
        # Canal prácticamente plano => o es puro silencio o puro ruido; no inventamos turnos.
        if peak - noise_floor < 6.0:
            return []

        hi = noise_floor + self.onset_db
        lo = noise_floor + self.offset_db

        raw: list[Segment] = []
        active = False
        start_idx = 0
        for i, level in enumerate(db):
            if not active and level >= hi:
                active, start_idx = True, i
            elif active and level < lo:
                active = False
                raw.append((start_idx * self.hop_ms / 1000.0, i * self.hop_ms / 1000.0))
        if active:
            raw.append((start_idx * self.hop_ms / 1000.0, len(db) * self.hop_ms / 1000.0))

        return _merge(raw, self.min_speech, self.max_gap)


class SileroVAD:
    """Envoltorio opcional de silero-vad (ONNX). Mismo contrato que EnergyVAD.

    Requiere descargar el modelo una vez:
        python scripts/fetch_silero.py
    Si el modelo no está, `available()` devuelve False y el servicio sigue
    con EnergyVAD sin romperse.
    """

    MODEL_PATH = "models/silero_vad.onnx"

    def __init__(self, threshold: float = 0.5, min_speech: float = 0.12, max_gap: float = 0.25):
        self.threshold = threshold
        self.min_speech = min_speech
        self.max_gap = max_gap
        self._session = None

    def available(self) -> bool:
        import os
        return os.path.exists(self.MODEL_PATH)

    def _load(self):
        if self._session is None:
            import onnxruntime as ort
            self._session = ort.InferenceSession(
                self.MODEL_PATH, providers=["CPUExecutionProvider"]
            )
        return self._session

    def segments(self, x: np.ndarray, sr: int) -> list[Segment]:
        if not self.available():
            raise RuntimeError(f"falta {self.MODEL_PATH}; corre scripts/fetch_silero.py")
        sess = self._load()
        # silero acepta 8k y 16k nativamente; a 8 kHz el frame es de 256 muestras.
        window = 256 if sr == 8000 else 512
        state = np.zeros((2, 1, 128), dtype=np.float32)
        probs, times = [], []
        for start in range(0, len(x) - window + 1, window):
            chunk = x[start:start + window].astype(np.float32)[None, :]
            out, state = sess.run(None, {
                "input": chunk,
                "state": state,
                "sr": np.array(sr, dtype=np.int64),
            })
            probs.append(float(out[0][0]))
            times.append(start / sr)

        raw: list[Segment] = []
        active, seg_start = False, 0.0
        for t, p in zip(times, probs):
            if not active and p >= self.threshold:
                active, seg_start = True, t
            elif active and p < self.threshold:
                active = False
                raw.append((seg_start, t))
        if active and times:
            raw.append((seg_start, times[-1] + window / sr))
        return _merge(raw, self.min_speech, self.max_gap)


def get_vad(prefer_silero: bool = True):
    """Devuelve el mejor VAD disponible sin obligar a nadie a descargar nada."""
    if prefer_silero:
        silero = SileroVAD()
        if silero.available():
            return silero
    return EnergyVAD()


def total_speech(segments: list[Segment]) -> float:
    return float(sum(e - s for s, e in segments))


def overlap_duration(a: list[Segment], b: list[Segment]) -> float:
    """Segundos en que ambos lados hablan al mismo tiempo."""
    total = 0.0
    j = 0
    for a_start, a_end in a:
        while j < len(b) and b[j][1] < a_start:
            j += 1
        k = j
        while k < len(b) and b[k][0] < a_end:
            total += max(0.0, min(a_end, b[k][1]) - max(a_start, b[k][0]))
            k += 1
    return total
