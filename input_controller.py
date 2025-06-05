from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from model_service import predict_from_model
from util import extract_sinhala_text
from bs4 import BeautifulSoup
from pydantic import BaseModel
from util import extract_sinhala_text
import shutil
import whisper
import ffmpeg
import os
import requests

router = APIRouter()

# classify normal text
class TextRequest(BaseModel):
    text: str

@router.post("/text")
def text_classify(text_request: TextRequest):
    print(f"Received request to classify text: ", text_request.text)
    try:
        sinhala_text = extract_sinhala_text(text_request.text)
        if sinhala_text == "":
            return "Not found sinhala text here"
        else:
            return predict_from_model(sinhala_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error classify text: {str(e)}")

# classify scraped text from website
class ScrapeRequest(BaseModel):
    url: str

@router.post("/web_scrape")
def scrape_page(web_request: ScrapeRequest):
    try:
        response = requests.get(web_request.url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        all_text = soup.get_text(separator="\n", strip=True)
        sinhala_text = extract_sinhala_text(all_text)

        return predict_from_model(sinhala_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scraping URL: {str(e)}")

# classify scraped text from audio/ video
class VideoRequest(BaseModel):
    video: UploadFile = File(...)

@router.post("/video_transcribe")
async def transcribe_video(request: VideoRequest):
    try:
        print("#######################1")
        # video = request.video
        # # Save video temporarily
        # video_path = f"temp/{video.filename}"
        # audio_path = video_path.replace(".mp4", ".mp3")
        #
        # print("#######################2")
        # with open(video_path, "wb") as buffer:
        #     shutil.copyfileobj(video.file, buffer)
        #
        # # Extract audio from video
        # ffmpeg.input(video_path).output(audio_path).run(overwrite_output=True)
        #
        # print("#######################3")
        # # Load whisper model (tiny, base, medium, large)
        # model = whisper.load_model("base")
        # result = model.transcribe(audio_path)
        #
        # print("#######################4")
        # # Cleanup
        # os.remove(video_path)
        # os.remove(audio_path)
        #
        # print("#######################5")
        # sinhala_text = extract_sinhala_text(result["text"])
        # print("\n\n\nText: "+sinhala_text+"\n\n\n")
        return {"text": "sinhala_text"}

    except Exception as e:
        return {"error": str(e)}