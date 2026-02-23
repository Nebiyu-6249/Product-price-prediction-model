# 🏷️ Product Price Prediction Model
**Traditional ML (TF‑IDF + Ridge) · FastAPI UI + JSON API · Docker-ready**

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-informational" />
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-API-success" />
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-ML-blue" />
  <img alt="Docker" src="https://img.shields.io/badge/Docker-ready-2496ED" />
</p>

A simple, deployable **product price prediction** project.

- Trains a regression model on a small labeled dataset (`data/human_out.csv`)
- Uses a **TF‑IDF + Ridge regression** pipeline
- Ships with a **FastAPI** web UI + JSON API
- Includes a **pre-trained model artifact** for out-of-the-box predictions (`models/price_model.joblib`)

> ✅ No notebooks  
> ✅ No external model APIs required

---

## ✨ What’s inside

- **Model**: `TfidfVectorizer(ngram_range=(1,2), max_features=6000)` → `Ridge(alpha=1.0)`
- **Target transform**: trains on `log1p(price)` and predicts via `expm1()` (stabilizes training)
- **Guardrails**: prediction is floored at `0.0` and rounded to 2 decimals
- **App**:
  - UI: `GET /` + form submit to `POST /predict`
  - API: `POST /api/predict`
  - Health: `GET /healthz`

---

## 📦 Project layout

```text
app/
  main.py            # FastAPI app (UI + API routes)
  train.py           # training script (writes models/price_model.joblib)
  model.py           # model loader + prediction helpers
  schemas.py         # Pydantic request/response models
  config.py          # settings (model path)
data/
  human_out.csv       # training data (text, price) — no header
models/
  price_model.joblib  # pre-trained model artifact
templates/
  index.html          # Tailwind UI form
  result.html         # prediction result page
```

---

## 🚀 Quickstart (Local)

### Requirements
- Python **3.11+** recommended

### Install & run

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run the API + UI
uvicorn app.main:app --reload --port 8000
```

Open: http://localhost:8000

---

## 🧠 Train / Retrain

```bash
python -m app.train
```

This will:
- read `data/human_out.csv`
- split into train/test (80/20)
- train the pipeline on `log1p(price)`
- print a holdout **MAE**
- save the model to `models/price_model.joblib`

---

## 🔌 API

### `POST /api/predict`

You can send **structured fields** (recommended)…

```json
{
  "title": "Wireless Noise Cancelling Headphones",
  "category": "Electronics",
  "brand": "Acme",
  "description": "Over-ear, bluetooth 5.3, 40h battery",
  "details": "Includes carrying case"
}
```

…or a single raw text block via `text`:

```json
{
  "text": "Title: Wireless Noise Cancelling Headphones\nCategory: Electronics\nBrand: Acme\nDescription: Over-ear, bluetooth 5.3"
}
```

Response:

```json
{
  "predicted_price_usd": 129.5
}
```

---

## 🖥️ Web UI

- `GET /` shows a simple Tailwind form
- `POST /predict` renders a results page with the predicted price

---

## ⚙️ Configuration

Settings are loaded via `pydantic-settings` (supports `.env`).

| Variable | Default | Description |
|---|---:|---|
| `MODEL_PATH` | `models/price_model.joblib` | Path to the joblib model artifact |

Notes:
- If the model file is missing, the app will raise an error instructing you to run `python -m app.train`.

---

## 🐳 Docker

```bash
docker build -t product-price-prediction-model .
docker run --rm -p 8000:8000 product-price-prediction-model
```

---

## 📊 Dataset notes

The included dataset `data/human_out.csv` is intentionally small (**100 rows**) and formatted as:

- column 1: free-form product text (title/category/brand/description…)
- column 2: numeric price (USD)

Predictions are therefore **approximate** and best used as a demo baseline.

---

## 🛠️ Troubleshooting

- **`Model artifact not found`**
  - Run: `python -m app.train` (creates `models/price_model.joblib`)

- **Import / dependency issues**
  - Recreate your venv and reinstall: `pip install -r requirements.txt`

- **Docker build fails with compilation errors**
  - The Dockerfile installs `build-essential` for common compiled deps.

---

## 🧭 Ideas to improve accuracy

- Add more training data (largest driver of performance)
- Add numeric features (ratings, review count, shipping weight, etc.)
- Try models like LightGBM / XGBoost
- Add basic text normalization / deduping
- Track experiments with proper evaluation and cross-validation

---

