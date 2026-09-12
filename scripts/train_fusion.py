"""Entrena y calibra el clasificador de fusión.

    python scripts/train_fusion.py --features data/features_vad.csv

Protocolo de evaluación:

* Si el CSV trae la columna `split`, se respeta el split del manifiesto de
  Altur, que es speaker-disjoint. Entrena en `train`, reporta en `val`.
  Es la estimación más cercana al set oculto del judging.
* Además corre validación cruzada por grupo dentro de `train`, porque `val`
  son pocas llamadas y un EER medido ahí tiene bastante varianza.

Reporta EER (estándar de anti-spoofing) y Brier score (calidad de la
calibración, que el reto premia), y compara timing-solo vs acústica-sola
vs fusión para saber qué señal carga el peso.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.features.acoustic import ACOUSTIC_FEATURES
from app.features.timing import TIMING_FEATURES
from app.fusion import FEATURE_ORDER


def equal_error_rate(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2), float(thresholds[idx])


def build_model() -> Pipeline:
    # Escalado + L2. Con ~280 llamadas y ~40 features, la regularización es
    # lo que evita que esto memorice el set.
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")),
    ])


def matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return np.nan_to_num(frame[columns].to_numpy(dtype=float),
                         nan=0.0, posinf=0.0, neginf=0.0)


def oof_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray | None,
               n_splits: int = 5) -> np.ndarray:
    out = np.zeros(len(y), dtype=float)
    unique_groups = len(np.unique(groups)) if groups is not None else 0
    if groups is not None and unique_groups >= n_splits:
        folds = GroupKFold(n_splits=n_splits).split(X, y, groups)
    else:
        folds = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                random_state=42).split(X, y)
    for train_idx, test_idx in folds:
        model = build_model()
        model.fit(X[train_idx], y[train_idx])
        out[test_idx] = model.predict_proba(X[test_idx])[:, 1]
    return out


def report(name: str, y: np.ndarray, scores: np.ndarray, threshold: float | None = None) -> dict:
    auc = roc_auc_score(y, scores)
    eer, eer_threshold = equal_error_rate(y, scores)
    used = eer_threshold if threshold is None else threshold
    accuracy = float(((scores >= used).astype(int) == y).mean())
    brier = brier_score_loss(y, scores)
    print(f"  {name:<22} AUC {auc:.3f} | EER {eer:.3f} | acc {accuracy:.3f} | Brier {brier:.3f}")
    return {"name": name, "auc": float(auc), "eer": eer, "threshold": float(eer_threshold),
            "accuracy": accuracy, "brier": float(brier), "n": int(len(y))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/features_vad.csv")
    parser.add_argument("--out", default="models/fusion.joblib")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--train-only", action="store_true",
                        help="entrenar el modelo final SOLO con el split train, "
                             "para poder medir el endpoint contra val sin contaminar")
    args = parser.parse_args()

    if not os.path.exists(args.features):
        print(f"No existe {args.features}. Corre scripts/extract_features.py primero.")
        return 1

    frame = pd.read_csv(args.features)
    y_all = frame["label"].to_numpy(dtype=int)
    print(f"Llamadas: {len(frame)} | sintéticas: {y_all.sum()} | humanas: {len(y_all) - y_all.sum()}")
    if len(np.unique(y_all)) < 2:
        print("El dataset tiene una sola clase; no se puede entrenar.")
        return 1

    available = [c for c in FEATURE_ORDER if c in frame.columns]
    missing = [c for c in FEATURE_ORDER if c not in frame.columns]
    if missing:
        print(f"AVISO: faltan {len(missing)} features en el CSV; reextrae si cambiaste el código.")

    subsets = {
        "solo timing": [c for c in TIMING_FEATURES if c in frame.columns],
        "solo acústica": [c for c in ACOUSTIC_FEATURES if c in frame.columns],
        "fusión": available,
    }

    has_split = "split" in frame.columns and frame["split"].isin(["train", "val"]).any()
    if has_split:
        train_frame = frame[frame["split"] == "train"].reset_index(drop=True)
        val_frame = frame[frame["split"] == "val"].reset_index(drop=True)
        print(f"Split del manifiesto (speaker-disjoint): train {len(train_frame)} | val {len(val_frame)}")
    else:
        train_frame, val_frame = frame, None
        print("Sin columna split: se evalúa solo por validación cruzada.")

    y_train = train_frame["label"].to_numpy(dtype=int)
    groups = train_frame["group"].to_numpy() if "group" in train_frame else None

    print(f"\nValidación cruzada dentro de train ({args.folds} folds, out-of-fold):")
    cv_results = {}
    for name, columns in subsets.items():
        if columns:
            cv_results[name] = report(name, y_train,
                                      oof_scores(matrix(train_frame, columns), y_train,
                                                 groups, args.folds))

    val_results = {}
    if val_frame is not None and len(val_frame) and len(np.unique(val_frame["label"])) > 1:
        y_val = val_frame["label"].to_numpy(dtype=int)
        print(f"\nSet de validación, hablantes no vistos ({len(val_frame)} llamadas):")
        for name, columns in subsets.items():
            if not columns:
                continue
            model = build_model()
            model.fit(matrix(train_frame, columns), y_train)
            scores = model.predict_proba(matrix(val_frame, columns))[:, 1]
            # Umbral tomado de la CV en train: fijarlo con val sería hacer
            # trampa contra nosotros mismos al estimar el set oculto.
            cv_threshold = cv_results.get(name, {}).get("threshold", 0.5)
            val_results[name] = report(name, y_val, scores, threshold=cv_threshold)

    # Modelo final. Por defecto usa todo el dataset (train + val): más datos,
    # mejor modelo para el judging. Con --train-only se entrena solo con train,
    # que es lo que hay que usar para medir el endpoint contra val de forma
    # honesta -- si el modelo vio val, ese número no vale nada.
    fit_frame = train_frame if args.train_only else frame
    y_fit = fit_frame["label"].to_numpy(dtype=int)
    final = build_model()
    final.fit(matrix(fit_frame, available), y_fit)
    if args.train_only:
        print("--train-only: modelo entrenado solo con el split train.")
    threshold = cv_results.get("fusión", {}).get("threshold", 0.5)

    coefficients = final.named_steps["clf"].coef_[0]
    ranked = sorted(zip(available, coefficients), key=lambda kv: abs(kv[1]), reverse=True)
    print("\nFeatures con más peso (+ empuja a sintético, - a humano):")
    for name, weight in ranked[:15]:
        print(f"  {weight:+7.3f}  {name}")

    if args.no_save:
        print("\n--no-save: no se escribió el modelo.")
        return 0

    import joblib
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump({
        "model": final,
        "feature_order": tuple(available),
        "threshold": float(threshold),
        "metrics": {"cv": cv_results, "val": val_results},
        "n_train": int(len(fit_frame)),
        "train_only": bool(args.train_only),
        "source": args.features,
    }, args.out)
    print(f"\nModelo guardado en {args.out} (umbral {threshold:.3f})")
    print("El servicio lo recarga solo; no hace falta reiniciar uvicorn.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
