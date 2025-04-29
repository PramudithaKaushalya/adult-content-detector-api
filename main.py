from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from scrape_controller import router as scrape_router
from text_controller import router as text_router
from video_controller import router as video_router
from model_controller import router as model_router

app = FastAPI()

app.add_middleware( # type: ignore[arg-type]
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(scrape_router, prefix="/api")
app.include_router(text_router, prefix="/api")
app.include_router(video_router, prefix="/api")
app.include_router(model_router, prefix="/api")