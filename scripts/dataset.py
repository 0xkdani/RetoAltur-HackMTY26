"""Descubrimiento de las llamadas del dataset y sus etiquetas.

Formato principal: el manifiesto de Altur.

    data/manifest.csv    anon_id,label,split,duration_s
    data/audio/<anon_id>.wav
    data/turns/<anon_id>.json    segmentos de voz por canal, precalculados

El `split` que trae el manifiesto es speaker-disjoint según el README del
reto, así que lo respetamos tal cual: es una mejor estimación de
generalización que cualquier split que armemos nosotros.

También soporta carpetas human/ y synthetic/ por si alguien organiza datos
propios para aumentar el set.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")

SYNTHETIC_DIRS = {"synthetic", "spoof", "fake", "tts", "bot", "sintetico", "sintetica"}
HUMAN_DIRS = {"human", "bonafide", "real", "genuine", "humano"}

DEFAULT_MANIFEST = "data/manifest.csv"
DEFAULT_AUDIO_DIR = "data/audio"
DEFAULT_TURNS_DIR = "data/turns"


@dataclass
class Item:
    path: str
    label: int                  # 1 = sintético, 0 = humano
    call_id: str = ""
    split: str = ""             # train | val | "" si no viene
    speaker: str = ""
    engine: str = ""
    turns_path: str = ""        # JSON de turnos precalculados, si existe
    extra: dict = field(default_factory=dict)

    @property
    def group(self) -> str:
        """Clave de agrupación para evitar fuga entre folds."""
        return self.speaker or self.engine or self.call_id or os.path.basename(self.path)

    def load_turns(self) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Lee el JSON de turnos y devuelve (caller, agent). Vacío si no hay archivo."""
        return load_turns_json(self.turns_path)


def load_turns_json(path: str) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Parsea turns/<id>.json -> (segmentos canal 0, segmentos canal 1), ordenados."""
    if not path or not os.path.exists(path):
        return [], []
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    caller, agent = [], []
    for turn in payload.get("turns", []):
        try:
            start, end = float(turn["start"]), float(turn["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        (caller if int(turn.get("channel", 0)) == 0 else agent).append((start, end))
    caller.sort()
    agent.sort()
    return caller, agent


def _parse_label(value: str) -> int | None:
    v = str(value).strip().lower()
    if v in SYNTHETIC_DIRS or v in {"1", "true", "yes", "si", "sí"}:
        return 1
    if v in HUMAN_DIRS or v in {"0", "false", "no"}:
        return 0
    return None


def from_manifest(
    manifest: str = DEFAULT_MANIFEST,
    audio_dir: str = DEFAULT_AUDIO_DIR,
    turns_dir: str = DEFAULT_TURNS_DIR,
) -> list[Item]:
    items: list[Item] = []
    with open(manifest, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            keys = {k.lower().strip(): (v or "").strip() for k, v in row.items() if k}
            call_id = (keys.get("anon_id") or keys.get("id") or
                       keys.get("call_id") or keys.get("path") or "")
            if not call_id:
                continue
            label = _parse_label(keys.get("label") or keys.get("class") or
                                 keys.get("is_synthetic") or "")
            if label is None:
                continue

            stem = os.path.splitext(os.path.basename(call_id))[0]
            path = call_id if call_id.lower().endswith(AUDIO_EXTS) else os.path.join(
                audio_dir, f"{stem}.wav")
            turns_path = os.path.join(turns_dir, f"{stem}.json") if turns_dir else ""

            items.append(Item(
                path=path,
                label=label,
                call_id=stem,
                split=(keys.get("split") or "").lower(),
                speaker=keys.get("speaker") or keys.get("speaker_id") or "",
                engine=keys.get("engine") or keys.get("tts") or keys.get("system") or "",
                turns_path=turns_path if os.path.exists(turns_path) else "",
            ))
    return items


def from_folders(root: str) -> list[Item]:
    """Etiqueta por el nombre de la carpeta contenedora (para datos propios)."""
    items: list[Item] = []
    for dirpath, _, filenames in os.walk(root):
        parts = {p.lower() for p in dirpath.replace("\\", "/").split("/")}
        if parts & SYNTHETIC_DIRS:
            label = 1
        elif parts & HUMAN_DIRS:
            label = 0
        else:
            continue
        engine = os.path.basename(dirpath).lower()
        for name in filenames:
            if name.lower().endswith(AUDIO_EXTS):
                items.append(Item(
                    path=os.path.join(dirpath, name),
                    label=label,
                    call_id=os.path.splitext(name)[0],
                    speaker=name.split("_")[0] if "_" in name else "",
                    engine=engine if label == 1 else "",
                ))
    return items


def discover(
    manifest: str = DEFAULT_MANIFEST,
    audio_dir: str = DEFAULT_AUDIO_DIR,
    turns_dir: str = DEFAULT_TURNS_DIR,
    fallback_root: str = "data/raw",
) -> list[Item]:
    """Prefiere el manifiesto; si no está, deduce por carpetas."""
    if os.path.exists(manifest):
        items = from_manifest(manifest, audio_dir, turns_dir)
        if items:
            return items
    return from_folders(fallback_root)
