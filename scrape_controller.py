from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import re

router = APIRouter()

class ScrapeRequest(BaseModel):
    url: str

def extract_sinhala_text(text: str) -> str:
    # Match only Sinhala characters and spaces
    sinhala_only = re.findall(r'[\u0D80-\u0DFF\s]+', text)
    # Join, collapse extra spaces, and return
    return " ".join("".join(sinhala_only).split())

@router.post("/scrape")
def scrape_page(request: ScrapeRequest):
    try:
        response = requests.get(request.url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        all_text = soup.get_text(separator="\n", strip=True)
        sinhala_text = extract_sinhala_text(all_text)

        return {"text": sinhala_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scraping URL: {str(e)}")

