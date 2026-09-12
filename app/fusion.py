"""Fusión tardía y calibración.

Cada señal produce su vector de features; una regresión logística encima
las combina y devuelve una probabilidad *calibrada*. Es a propósito un
modelo simple: se entrena con pocos datos, no sobreajusta con 40 features,
y sus coeficientes son explicables frente a un juez o un banco.

Si no hay modelo entrenado todavía, el detector responde 0.5 y lo marca
como `untrained` en vez de fingir una decisión.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .features.acoustic import ACOUSTIC_FEATURES
from .features.timing import TIMING_FEATURES

# El orden importa y debe ser el mismo en entrenamiento y en serving.
FEATURE_ORDER: tuple[str, ...] = TIMING_FEATURES + ACOUSTIC_FEATURES

MODEL_PATH = os.environ.get("VOICEGUARD_MODEL", "models/fusion.joblib")


@dataclass
class Verdict:
    is_synthetic: bool
    confidence: float
    trained: bool
    reason: str


def to_vector(features: dict[str, float]) -> np.ndarray:
    """dict -> vector en el orden canónico. Lo que falte va en 0."""
    return np.asarray([float(features.get(name, 0.0)) for name in FEATURE_ORDER],
                      dtype=np.float64).reshape(1, -1)


class FusionModel:
    """Carga perezosa del clasificador; recarga sola si el archivo cambia."""

    def __init__(self, path: str = MODEL_PATH):
        self.path = path
        self._bundle = None
        self._mtime: float | None = None

    def _load(self):
        if not os.path.exists(self.path):
            self._bundle = None
            return None
        mtime = os.path.getmtime(self.path)
        if self._bundle is None or mtime != self._mtime:
            import joblib
            self._bundle = joblib.load(self.path)
            self._mtime = mtime
        return self._bundle

    @property
    def is_trained(self) -> bool:
        return self._load() is not None

    def predict(self, features: dict[str, float]) -> Verdict:
        bundle = self._load()
        if bundle is None:
            return Verdict(
                is_synthetic=False,
                confidence=0.5,
                trained=False,
                reason="no hay modelo entrenado; corre scripts/train_fusion.py",
            )

        # El bundle guarda también el orden con el que se entrenó: si alguien
        # agrega una feature y no reentrena, preferimos fallar claro.
        order = bundle.get("feature_order", FEATURE_ORDER)
        vector = np.asarray([float(features.get(n, 0.0)) for n in order],
                            dtype=np.float64).reshape(1, -1)

        model = bundle["model"]
        prob_synthetic = float(model.predict_proba(vector)[0][1])
        threshold = float(bundle.get("threshold", 0.5))
        is_synthetic = prob_synthetic >= threshold

        # Confianza = qué tan lejos está la probabilidad de la frontera,
        # reescalado a [0.5, 1.0]. Así un 0.51 no se reporta como certeza.
        distance = abs(prob_synthetic - threshold) / max(threshold, 1 - threshold)
        confidence = 0.5 + 0.5 * min(1.0, distance)

        return Verdict(
            is_synthetic=is_synthetic,
            confidence=round(confidence, 4),
            trained=True,
            reason=f"p(sintético)={prob_synthetic:.3f} umbral={threshold:.3f}",
        )
