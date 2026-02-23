# Product Price Prediction Model (Traditional ML + Deployable API)

A simple, deployable **product price prediction** project.

- Trains a regression model on a small labeled dataset (`data/human_out.csv`)
- Uses a TF‑IDF + Ridge regression pipeline
- Ships with a FastAPI web UI + JSON API
- Includes a pre-trained model artifact for out-of-the-box predictions

> ✅ No notebooks  
> ✅ No external model APIs required

---

## Quickstart (Local)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run the API
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000

---

## Train / Retrain

```bash
python -m app.train
```

This will:
- read `data/human_out.csv`
- train the model
- save to `models/price_model.joblib`

---

## API

### POST `/api/predict`
```json
{
  "title": "Wireless Noise Cancelling Headphones",
  "category": "Electronics",
  "brand": "Acme",
  "description": "Over-ear, bluetooth 5.3, 40h battery",
  "details": "Includes carrying case"
}
```

Response:
```json
{
  "predicted_price_usd": 129.5
}
```

---

## Docker

```bash
docker build -t product-price-prediction-model .
docker run --rm -p 8000:8000 product-price-prediction-model
```

---

## Notes

This is a small demo dataset, so treat predictions as approximate.  
If you want higher accuracy:
- increase training data size
- add numerical features (ratings, review count, etc.)
- try LightGBM / XGBoost
