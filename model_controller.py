from fastapi import APIRouter, HTTPException, UploadFile, File
from model_service import train_and_save_model, evaluate_model, upload_and_append

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

    try:
        return upload_and_append(file)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
