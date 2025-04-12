from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import re

router = APIRouter()

class TextRequest(BaseModel):
    text: str

def extract_sinhala_text(text: str) -> str:
    # Match only Sinhala characters and spaces
    sinhala_only = re.findall(r'[\u0D80-\u0DFF\s]+', text)
    # Join, collapse extra spaces, and return
    return " ".join("".join(sinhala_only).split())

@router.post("/text")
def scrape_page(request: TextRequest):
    try:
        sinhala_text = extract_sinhala_text(request.text)

        return {"text": sinhala_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scraping URL: {str(e)}")

