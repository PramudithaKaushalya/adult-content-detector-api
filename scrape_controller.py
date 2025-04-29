from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from model_service import predict_from_model
from util import extract_sinhala_text

router = APIRouter()

class ScrapeRequest(BaseModel):
    url: str

@router.post("/web_scrape")
def scrape_page(request: ScrapeRequest):
    try:
        response = requests.get(request.url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        all_text = soup.get_text(separator="\n", strip=True)
        sinhala_text = extract_sinhala_text(all_text)

        return predict_from_model(sinhala_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scraping URL: {str(e)}")

