from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from app.config import settings


def build_text(
    title: Optional[str] = None,
    category: Optional[str] = None,
    brand: Optional[str] = None,
    description: Optional[str] = None,
    details: Optional[str] = None,
    raw_text: Optional[str] = None,
) -> str:
    if raw_text and raw_text.strip():
        return raw_text.strip()

    parts = []
    if title:
        parts.append(f"Title: {title}")
    if category:
        parts.append(f"Category: {category}")
    if brand:
        parts.append(f"Brand: {brand}")
    if description:
        parts.append(f"Description: {description}")
    if details:
        parts.append(f"Details: {details}")
    return "\n".join(parts).strip()


class PriceModel:
    def __init__(self, model_path: str | Path = settings.model_path):
        self.model_path = Path(model_path)
        self.pipeline = None

    def load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {self.model_path}. Run: python -m app.train"
            )
        self.pipeline = joblib.load(self.model_path)
        return self

    def predict(self, text: str) -> float:
        if self.pipeline is None:
            self.load()
        # Our model is trained on log1p(price)
        pred_log = float(self.pipeline.predict([text])[0])
        pred = float(np.expm1(pred_log))
        # Guardrails
        if pred < 0:
            pred = 0.0
        return round(pred, 2)
