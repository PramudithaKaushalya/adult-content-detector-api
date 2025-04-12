from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from scrape_controller import router as scrape_router
from text_controller import router as text_router

app = FastAPI()

# Enable CORS (adjust origin as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(scrape_router, prefix="/api")
app.include_router(text_router, prefix="/api")
