from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from model_service import (train_and_save_model, evaluate_model, upload_and_append, append_record, fully_evaluate_model, download_cm, download_roc)

router = APIRouter()

@router.get("/train_model")
def train_model():
    try:
        return train_and_save_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")


@router.get("/evaluate_model")
def evaluate_trained_model():
    try:
        return evaluate_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating model: {str(e)}")


@router.post("/upload_data")
async def upload_data(file: UploadFile = File(...)):
    # Validate file type
    if not file.filename.endswith((".csv", ".xlsx")):
        raise HTTPException(status_code=400, detail="Only .csv or .xlsx files are allowed.")

    return upload_and_append(file)


class TextUploadRequest(BaseModel):
    text: str
    label: int

@router.post("/append_text")
def append_text(request: TextUploadRequest):
    append_record(request)

@router.get("/evaluate")
def fully_evaluating_model():
    try:
        print("Received request to fully evaluate the model.")
        return fully_evaluate_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating model: {str(e)}")

@router.get("/download/confusion")
def downloading_cm():
    try:
        print("Received request to download confusion chart.")
        return download_cm
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating model: {str(e)}")

@router.get("/download/auc")
def downloading_roc():
    try:
        print("Received request to download auc chart.")
        return download_roc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating model: {str(e)}")