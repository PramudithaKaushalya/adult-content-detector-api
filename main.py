from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from input_controller import router as input_router
from admin_controller import router as admin_router

app = FastAPI()

app.add_middleware( # type: ignore[arg-type]
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(input_router, prefix="/api")
app.include_router(admin_router, prefix="/api")