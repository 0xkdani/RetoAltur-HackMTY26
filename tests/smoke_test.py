"""Prueba de humo: el pipeline completo corre y las features discriminan.

    python tests/smoke_test.py            # solo el pipeline, sin servidor
    python tests/smoke_test.py --http     # además pega al endpoint local
"""

from __future__ import annotations

import argparse
import base64
import time
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import soundfile as sf

from app.detector import detect
from tests.make_fixture import SR, bot_like, human_like

CHECKS: list[tuple[str, bool]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    CHECKS.append((name, condition))
    mark = "OK  " if condition else "FALLA"
    print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail else ""))


def to_wav_bytes(audio) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, SR, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", action="store_true")
    parser.add_argument("--url", default="http://127.0.0.1:8000/detect")
    args = parser.parse_args()

    human_wav = to_wav_bytes(human_like())
    bot_wav = to_wav_bytes(bot_like())

    print("\n1. El pipeline corre end-to-end")
    # Primera llamada del proceso: paga la carga de joblib/sklearn y numpy.
    # El servicio se ahorra esto con el warm-up de arranque (app/main.py),
    # asi que la latencia representativa es la de la segunda en adelante.
    cold_started = time.perf_counter()
    detect(human_wav)
    cold_ms = (time.perf_counter() - cold_started) * 1000.0

    human = detect(human_wav, include_debug=True)
    bot = detect(bot_wav, include_debug=True)
    check("responde con el contrato del reto",
          {"is_synthetic", "confidence"} <= set(human))
    check("is_synthetic es bool", isinstance(human["is_synthetic"], bool))
    check("confidence en [0,1]", 0.0 <= human["confidence"] <= 1.0)

    print("\n2. El VAD encuentra los turnos que pusimos")
    hm, bm = human["meta"], bot["meta"]
    check("detecta turnos del caller", hm["caller_turns"] > 0, f"{hm['caller_turns']} turnos")
    check("detecta turnos del agente", hm["agent_turns"] > 0, f"{hm['agent_turns']} turnos")
    check("reconoce el canal del agente", hm["has_agent_channel"])

    print("\n3. Las features de timing separan los dos patrones")
    hf, bf = human["features"], bot["features"]
    check("el bot tiene latencia más consistente (CV menor)",
          bf["resp_latency_cv"] < hf["resp_latency_cv"],
          f"bot {bf['resp_latency_cv']:.3f} vs humano {hf['resp_latency_cv']:.3f}")
    check("el humano responde más rápido en su mejor caso",
          hf["resp_latency_min"] < bf["resp_latency_min"],
          f"humano {hf['resp_latency_min']:.2f}s vs bot {bf['resp_latency_min']:.2f}s")
    check("solo el humano hace barge-in",
          hf["bargein_rate"] > 0 and bf["bargein_rate"] == 0,
          f"humano {hf['bargein_rate']:.2f} vs bot {bf['bargein_rate']:.2f}")

    print("\n4. Latencia del servicio")
    print(f"  [info ] arranque en frio: {cold_ms:.0f} ms (el servicio lo absorbe en el warm-up)")
    check("en caliente responde en menos de 300 ms",
          hm["latency_ms"] < 300, f"{hm['latency_ms']:.0f} ms")

    if args.http:
        print("\n5. El endpoint HTTP responde")
        try:
            import urllib.request, json
            payload = json.dumps({"audio": base64.b64encode(human_wav).decode()}).encode()
            request = urllib.request.Request(
                args.url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=30) as response:
                body = json.loads(response.read())
            check("endpoint devuelve is_synthetic", "is_synthetic" in body, str(body)[:120])
        except Exception as exc:  # noqa: BLE001
            check("endpoint alcanzable", False, f"{type(exc).__name__}: {exc}")

    passed = sum(1 for _, ok in CHECKS if ok)
    print(f"\n{passed}/{len(CHECKS)} verificaciones pasaron")
    if not bot["meta"]["model_trained"]:
        print("Nota: aún no hay modelo entrenado, el veredicto es el neutro 0.5 a propósito.")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
