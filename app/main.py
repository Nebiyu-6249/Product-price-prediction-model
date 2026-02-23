from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.model import PriceModel, build_text
from app.schemas import PredictRequest, PredictResponse

load_dotenv(override=False)

app = FastAPI(title="Product Price Prediction Model", version="1.0.0")
templates = Jinja2Templates(directory="templates")

model = PriceModel().load()


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
def predict_ui(
    request: Request,
    title: str = Form(default=""),
    category: str = Form(default=""),
    brand: str = Form(default=""),
    description: str = Form(default=""),
    details: str = Form(default=""),
):
    text = build_text(
        title=title or None,
        category=category or None,
        brand=brand or None,
        description=description or None,
        details=details or None,
    )
    price = model.predict(text) if text else 0.0
    return templates.TemplateResponse(
        "result.html",
        {"request": request, "predicted_price": price, "text": text},
    )


@app.post("/api/predict", response_model=PredictResponse)
def api_predict(payload: PredictRequest):
    text = build_text(
        title=payload.title,
        category=payload.category,
        brand=payload.brand,
        description=payload.description,
        details=payload.details,
        raw_text=payload.text,
    )
    price = model.predict(text)
    return PredictResponse(predicted_price_usd=price)
