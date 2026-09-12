"""Carga y normalización del audio que entra por el endpoint.

El reto entrega WAV estéreo a 8 kHz: canal 0 = caller (el que hay que
clasificar), canal 1 = agente. Aquí no asumimos que siempre llegue así:
si llega mono, tratamos todo como caller y el agente queda vacío.
"""

from __future__ import annotations

import base64
import binascii
import io
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np
import soundfile as sf


class AudioDecodeError(ValueError):
    """El payload no se pudo convertir a audio."""


@dataclass
class Call:
    """Una llamada ya separada en sus dos lados."""

    caller: np.ndarray  # mono float32
    agent: np.ndarray   # mono float32, puede venir vacío
    sr: int

    @property
    def duration(self) -> float:
        return len(self.caller) / self.sr if self.sr else 0.0

    @property
    def has_agent(self) -> bool:
        # Un canal de agente todo en silencio no sirve para features de turno.
        return self.agent.size > 0 and float(np.max(np.abs(self.agent))) > 1e-4


def decode_base64(payload: str) -> bytes:
    """Acepta base64 pelón o con prefijo data: URI."""
    if not isinstance(payload, str) or not payload.strip():
        raise AudioDecodeError("payload de audio vacío")
    raw = payload.strip()
    if raw.startswith("data:"):
        _, _, raw = raw.partition(",")
    raw = "".join(raw.split())  # quita saltos de línea del base64 con wrap
    try:
        return base64.b64decode(raw, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise AudioDecodeError(f"base64 inválido: {exc}") from exc


def _decode_with_ffmpeg(blob: bytes) -> tuple[np.ndarray, int]:
    """Salida de emergencia para WAV con codecs que soundfile no abre (G.711, etc.)."""
    if shutil.which("ffmpeg") is None:
        raise AudioDecodeError("soundfile no pudo abrir el audio y no hay ffmpeg disponible")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-i", "pipe:0", "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"]
    proc = subprocess.run(cmd, input=blob, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        raise AudioDecodeError(f"ffmpeg no pudo decodificar: {proc.stderr[:200]!r}")
    # ffmpeg aplana los canales; recuperamos la forma con los metadatos del probe.
    info = subprocess.run(
        ["ffprobe", "-hide_banner", "-loglevel", "error", "-show_entries",
         "stream=channels,sample_rate", "-of", "csv=p=0", "pipe:0"],
        input=blob, capture_output=True, text=True,
    )
    try:
        sr_str, ch_str = info.stdout.strip().splitlines()[0].split(",")[:2]
        sr, channels = int(sr_str), int(ch_str)
    except (ValueError, IndexError):
        sr, channels = 8000, 2
    data = np.frombuffer(proc.stdout, dtype=np.float32)
    if channels > 1:
        data = data.reshape(-1, channels)
    return data, sr


def load_call(blob: bytes) -> Call:
    """Bytes de un WAV -> Call con los dos canales separados."""
    try:
        data, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=False)
    except Exception:
        data, sr = _decode_with_ffmpeg(blob)

    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        caller, agent = data, np.zeros(0, dtype=np.float32)
    else:
        caller = np.ascontiguousarray(data[:, 0])
        agent = (np.ascontiguousarray(data[:, 1]) if data.shape[1] > 1
                 else np.zeros(0, dtype=np.float32))

    if caller.size == 0:
        raise AudioDecodeError("el canal del caller vino vacío")
    return Call(caller=caller, agent=agent, sr=int(sr))


def load_call_from_path(path: str) -> Call:
    with open(path, "rb") as fh:
        return load_call(fh.read())
