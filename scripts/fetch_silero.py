"""Descarga el modelo silero-vad ONNX a models/silero_vad.onnx (opcional).

    python scripts/fetch_silero.py

Sin esto el servicio usa el VAD de energía, que funciona. Silero da bordes
de turno más limpios, y como todas las features de timing dependen de esos
bordes, suele mover la aguja. Vale la pena si hay red.
"""

from __future__ import annotations

import os
import urllib.request

URL = "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx"
DEST = "models/silero_vad.onnx"


def main() -> int:
    os.makedirs("models", exist_ok=True)
    if os.path.exists(DEST):
        print(f"{DEST} ya existe ({os.path.getsize(DEST) / 1024:.0f} KB)")
        return 0
    print(f"Descargando {URL}")
    try:
        urllib.request.urlretrieve(URL, DEST)
    except Exception as exc:  # noqa: BLE001
        print(f"Falló la descarga: {exc}")
        print("No pasa nada: el servicio sigue con EnergyVAD.")
        return 1
    print(f"Guardado en {DEST} ({os.path.getsize(DEST) / 1024:.0f} KB)")
    print("AVISO: verifica que la firma de entrada del ONNX coincida con app/vad.py::SileroVAD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
