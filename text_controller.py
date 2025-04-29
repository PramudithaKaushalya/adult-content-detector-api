from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from model_service import predict_from_model
from util import extract_sinhala_text

router = APIRouter()

class TextRequest(BaseModel):
    text: str

@router.post("/text")
def text_classify(request: TextRequest):
    print(f"Received request to classify text: ", request.text)
    try:
        sinhala_text = extract_sinhala_text(request.text)
        if sinhala_text == "":
            return "Not found sinhala text here"
        else:
            return predict_from_model(sinhala_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error classify text: {str(e)}")

