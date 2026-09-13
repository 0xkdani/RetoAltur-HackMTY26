"""Señal A: artefactos acústicos, baseline barato.

Esto NO es un detector de última generación -- es el punto de partida para
tener una métrica contra la cual comparar, y corre en numpy puro sin GPU.
El plan es sustituirlo por AASIST o wav2vec2 fine-tuneado, manteniendo el
mismo contrato: `extract(...) -> dict`.

Ojo con el ancho de banda: a 8 kHz Nyquist es 4 kHz, así que los artefactos
de vocoder que viven arriba de eso ya no existen aquí. Lo que sí sobrevive:
suavidad espectral anormal, ruido de fondo demasiado limpio y prosodia con
poca variación.
"""

from __future__ import annotations

import numpy as np

from ..vad import Segment

ACOUSTIC_FEATURES: tuple[str, ...] = (
    "spec_flatness_mean",
    "spec_flatness_std",
    "spec_centroid_mean",
    "spec_centroid_std",
    "spec_rolloff_mean",
    "spec_bandwidth_mean",
    "spec_flux_mean",
    "spec_flux_std",
    "hf_ratio_mean",
    "hf_ratio_std",
    "zcr_mean",
    "zcr_std",
    "energy_std_db",
    "noise_floor_db",
    "snr_est_db",
    "f0_mean",
    "f0_std",
    "f0_voiced_ratio",
    "modulation_4hz",
)


def _stft_mag(x: np.ndarray, n_fft: int = 256, hop: int = 128) -> np.ndarray:
    """Magnitud STFT con ventana Hann. (frames, bins)

    sliding_window_view devuelve una vista sobre el mismo buffer, sin copiar
    nada. El fancy indexing que había antes materializaba una matriz de
    n_frames x n_fft, que en una llamada larga son varios millones de
    valores copiados solo para tirarlos. Resultado idéntico.
    """
    if x.size < n_fft:
        return np.zeros((0, n_fft // 2 + 1), dtype=np.float64)
    window = np.hanning(n_fft)
    vistas = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop]
    return np.abs(np.fft.rfft(vistas * window, axis=1))


def _voiced_frames(x: np.ndarray, segments: list[Segment], sr: int) -> np.ndarray:
    """Concatena solo lo que el VAD marcó como voz; el silencio ensucia las stats."""
    if not segments:
        return x
    pieces = [x[int(s * sr):int(e * sr)] for s, e in segments]
    pieces = [p for p in pieces if p.size > 0]
    return np.concatenate(pieces) if pieces else x


def _f0_track(x: np.ndarray, sr: int, frame: int = 512, hop: int = 256) -> tuple[np.ndarray, float]:
    """F0 por autocorrelación. Tosco pero suficiente para medir variabilidad prosódica."""
    fmin, fmax = 60.0, 400.0
    lag_min, lag_max = int(sr / fmax), int(sr / fmin)
    if x.size < frame or lag_max >= frame:
        return np.zeros(0), 0.0

    f0s, voiced = [], 0
    n_frames = 1 + (x.size - frame) // hop
    for i in range(n_frames):
        seg = x[i * hop:i * hop + frame].astype(np.float64)
        seg = seg - seg.mean()
        norm = np.dot(seg, seg)
        if norm < 1e-8:
            continue
        corr = np.correlate(seg, seg, mode="full")[frame - 1:]
        window = corr[lag_min:lag_max]
        if window.size == 0:
            continue
        peak = int(np.argmax(window)) + lag_min
        # Correlación normalizada baja = frame no sonoro (fricativa, ruido).
        if corr[peak] / norm > 0.3:
            f0s.append(sr / peak)
            voiced += 1
    ratio = voiced / max(n_frames, 1)
    return np.asarray(f0s), ratio


def extract(x: np.ndarray, sr: int, segments: list[Segment]) -> dict[str, float]:
    """Vector acústico del canal del caller. Siempre devuelve ACOUSTIC_FEATURES completo."""
    zeros = {name: 0.0 for name in ACOUSTIC_FEATURES}
    voiced = _voiced_frames(x, segments, sr)
    if voiced.size < 512:
        return zeros

    mag = _stft_mag(voiced)
    if mag.shape[0] == 0:
        return zeros

    power = mag ** 2 + 1e-12
    freqs = np.fft.rfftfreq(256, d=1.0 / sr)
    total = power.sum(axis=1)

    # Planitud espectral: media geométrica / media aritmética. La voz sintética
    # suele salir más "lisa" porque el vocoder no reproduce bien el ruido fino.
    flatness = np.exp(np.mean(np.log(power), axis=1)) / (power.mean(axis=1))
    centroid = (power * freqs).sum(axis=1) / total

    cumulative = np.cumsum(power, axis=1) / total[:, None]
    rolloff = freqs[np.argmax(cumulative >= 0.85, axis=1)]
    bandwidth = np.sqrt((power * (freqs[None, :] - centroid[:, None]) ** 2).sum(axis=1) / total)

    flux = np.sqrt(((np.diff(mag, axis=0)) ** 2).sum(axis=1)) if mag.shape[0] > 1 else np.zeros(1)

    # Energía en la banda alta útil que deja pasar el canal telefónico.
    hf_band = freqs >= 3000.0
    hf_ratio = power[:, hf_band].sum(axis=1) / total

    zcr_frames = np.mean(np.abs(np.diff(np.sign(voiced))) > 0)

    frame_rms = np.sqrt(np.mean(_stft_mag(voiced, 256, 128) ** 2, axis=1) + 1e-12)
    energy_db = 20 * np.log10(frame_rms + 1e-12)

    full_db = 20 * np.log10(np.abs(x) + 1e-12)
    noise_floor = float(np.percentile(full_db, 5))
    speech_level = float(np.percentile(full_db, 95))

    f0s, voiced_ratio = _f0_track(voiced, sr)

    # Modulación ~4 Hz: el ritmo silábico del habla natural. Se mide como la
    # energía de la envolvente alrededor de esa frecuencia.
    env = frame_rms - frame_rms.mean()
    if env.size > 16:
        env_spec = np.abs(np.fft.rfft(env))
        env_freqs = np.fft.rfftfreq(env.size, d=128.0 / sr)
        band = (env_freqs >= 2.0) & (env_freqs <= 6.0)
        mod = float(env_spec[band].sum() / (env_spec.sum() + 1e-12)) if band.any() else 0.0
    else:
        mod = 0.0

    return {
        "spec_flatness_mean": float(flatness.mean()),
        "spec_flatness_std": float(flatness.std()),
        "spec_centroid_mean": float(centroid.mean()),
        "spec_centroid_std": float(centroid.std()),
        "spec_rolloff_mean": float(rolloff.mean()),
        "spec_bandwidth_mean": float(bandwidth.mean()),
        "spec_flux_mean": float(flux.mean()),
        "spec_flux_std": float(flux.std()),
        "hf_ratio_mean": float(hf_ratio.mean()),
        "hf_ratio_std": float(hf_ratio.std()),
        "zcr_mean": float(zcr_frames),
        "zcr_std": 0.0,
        "energy_std_db": float(energy_db.std()),
        "noise_floor_db": noise_floor,
        "snr_est_db": speech_level - noise_floor,
        "f0_mean": float(f0s.mean()) if f0s.size else 0.0,
        "f0_std": float(f0s.std()) if f0s.size else 0.0,
        "f0_voiced_ratio": float(voiced_ratio),
        "modulation_4hz": mod,
    }
