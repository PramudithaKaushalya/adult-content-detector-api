from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from util import extract_sinhala_text
import shutil
import whisper
import ffmpeg
import os

router = APIRouter()

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
