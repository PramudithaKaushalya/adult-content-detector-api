#Fast API install

pip install fastapi[all]  # includes `fastapi`, `pydantic`, `uvicorn`, and `starlette`

#Add video transcribe libs

pip install fastapi uvicorn python-multipart ffmpeg-python openai-whisper

#run fast API application

uvicorn main:app --reload
