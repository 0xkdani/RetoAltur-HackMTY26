"""Diagnóstico de robustez: ¿el modelo aprende la voz o aprende el canal?

    python scripts/diagnose.py --features data/features_vad.csv

Un EER de 0.00 en validación es sospechoso hasta que se demuestre lo
contrario. Los dos modos de fallo que buscamos:

1. Artefacto de grabación. Si las llamadas sintéticas se capturaron en un
   setup con menos ruido de fondo, features como `noise_floor_db` separan
   perfecto sin decir nada sobre la voz. Contra un set oculto grabado en
   otras condiciones, eso se cae.
2. Atajo trivial. Si una sola feature da AUC ~1.0, el resto del sistema
   está de adorno y somos frágiles a que esa feature cambie.

La prueba: entrenar sin los grupos sospechosos y ver cuánto se pierde.
Si el sistema aguanta sin ellos, la señal es real.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from app.features.acoustic import ACOUSTIC_FEATURES
from app.features.timing import TIMING_FEATURES
from app.fusion import FEATURE_ORDER
from scripts.train_fusion import build_model, equal_error_rate, matrix

# Features que describen el canal de transmisión, no a quien habla.
CHANNEL_FEATURES = ("noise_floor_db", "snr_est_db", "hf_ratio_mean", "hf_ratio_std",
                    "spec_centroid_mean", "spec_rolloff_mean", "spec_bandwidth_mean")

# Features puramente conductuales: no tocan la forma de onda de la voz,
# solo cuándo se habla. Son las más difíciles de falsificar para un atacante.
BEHAVIOUR_ONLY = tuple(TIMING_FEATURES)


def evaluate(train: pd.DataFrame, val: pd.DataFrame, columns: list[str]) -> tuple[float, float]:
    if not columns:
        return float("nan"), float("nan")
    model = build_model()
    model.fit(matrix(train, columns), train["label"].to_numpy(int))
    scores = model.predict_proba(matrix(val, columns))[:, 1]
    y = val["label"].to_numpy(int)
    auc = roc_auc_score(y, scores)
    eer, _ = equal_error_rate(y, scores)
    return float(auc), eer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/features_vad.csv")
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    frame = pd.read_csv(args.features)
    train = frame[frame["split"] == "train"].reset_index(drop=True)
    val = frame[frame["split"] == "val"].reset_index(drop=True)
    available = [c for c in FEATURE_ORDER if c in frame.columns]
    y_train = train["label"].to_numpy(int)

    print("=" * 74)
    print("1. PODER DISCRIMINATIVO DE CADA FEATURE POR SEPARADO (AUC en train)")
    print("   AUC > 0.90 en solitario = revisar si es señal real o artefacto")
    print("=" * 74)
    singles = []
    for column in available:
        values = np.nan_to_num(train[column].to_numpy(float), nan=0.0,
                               posinf=0.0, neginf=0.0)
        if np.std(values) < 1e-12:
            continue
        auc = roc_auc_score(y_train, values)
        singles.append((column, max(auc, 1 - auc)))  # dirección da igual aquí
    singles.sort(key=lambda kv: kv[1], reverse=True)
    for column, auc in singles[:args.top]:
        family = "timing " if column in TIMING_FEATURES else "acústica"
        flag = "  <-- sospechosa de canal" if column in CHANNEL_FEATURES else ""
        print(f"  {auc:.3f}  [{family}] {column}{flag}")

    print()
    print("=" * 74)
    print("2. ABLACIÓN: ¿qué queda si le quitamos grupos de features?")
    print("   (entrenado en train, medido en val, hablantes no vistos)")
    print("=" * 74)
    ablations = {
        "todo": available,
        "sin features de canal": [c for c in available if c not in CHANNEL_FEATURES],
        "solo conductual (timing)": [c for c in BEHAVIOUR_ONLY if c in frame.columns],
        "solo acústica": [c for c in ACOUSTIC_FEATURES if c in frame.columns],
        "sin la feature #1": [c for c in available if c != singles[0][0]],
        "sin el top-5 individual": [c for c in available
                                    if c not in {s[0] for s in singles[:5]}],
    }
    for name, columns in ablations.items():
        auc, eer = evaluate(train, val, columns)
        print(f"  {name:<26} ({len(columns):2d} feats)  AUC {auc:.3f} | EER {eer:.3f}")

    print()
    print("=" * 74)
    print("3. CONTROL: ¿se puede separar con features que NO son de voz?")
    print("   Si la duración sola separa, el dataset tiene un sesgo de construcción")
    print("=" * 74)
    for column in ("duration_s", "noise_floor_db", "snr_est_db"):
        if column not in frame.columns:
            continue
        values = np.nan_to_num(train[column].to_numpy(float), nan=0.0)
        if np.std(values) < 1e-12:
            continue
        auc = roc_auc_score(y_train, values)
        verdict = "OK" if abs(auc - 0.5) < 0.25 else "ALERTA"
        print(f"  [{verdict:6}] {column:<18} AUC {max(auc, 1 - auc):.3f}")

    print()
    print("=" * 74)
    print("4. ESTABILIDAD DEL RESULTADO EN VAL")
    print("=" * 74)
    print(f"  val tiene {len(val)} llamadas ({int(val['label'].sum())} sintéticas, "
          f"{len(val) - int(val['label'].sum())} humanas).")
    print("  Con esa n, un EER de 0.00 significa 'ningún error en 71 intentos',")
    print("  no 'EER verdadero = 0'. El intervalo de confianza real es amplio:")
    print("  con 0 errores en 71, el 95% superior ronda 4-5% de error.")
    print("  Conclusión honesta para el jurado: el sistema separa bien este set,")
    print("  y el número a reportar es el de la CV, no el de val.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
