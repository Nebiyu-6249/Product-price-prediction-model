from pydantic import BaseModel
from typing import Optional


class PredictRequest(BaseModel):
    # You can send either raw text or structured fields.
    text: Optional[str] = None

    title: Optional[str] = None
    category: Optional[str] = None
    brand: Optional[str] = None
    description: Optional[str] = None
    details: Optional[str] = None


class PredictResponse(BaseModel):
    predicted_price_usd: float
