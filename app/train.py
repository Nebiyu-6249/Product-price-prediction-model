from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "human_out.csv"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "price_model.joblib"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, header=None, names=["text", "price"])
    df["text"] = df["text"].fillna("").astype(str)
    df["price"] = df["price"].astype(float)

    X = df["text"]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Predict log(price) to stabilize training
    y_train_log = np.log1p(y_train)

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=6000, ngram_range=(1, 2))),
            ("reg", Ridge(alpha=1.0)),
        ]
    )

    pipeline.fit(X_train, y_train_log)

    # Evaluate
    pred_log = pipeline.predict(X_test)
    pred = np.expm1(pred_log)
    mae = mean_absolute_error(y_test, pred)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print(f"Saved model to: {MODEL_PATH}")
    print(f"Holdout MAE: ${mae:.2f} (smaller is better)")


if __name__ == "__main__":
    main()
