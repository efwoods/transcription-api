from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

import uvicorn
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

from contextlib import asynccontextmanager

# Configurations & Metrics
from core.config import settings
from core.monitoring import metrics
from core.logging import logger
from model.whisper_model import model

# API Routes
from api.routes import router

app = FastAPI(title="Real-Time Whisper Transcription Service", root_path="/transcription-api")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/transcribe", tags=["Transcribe"])

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize Whisper model
    model.init()
    
    yield  # Application runs here

    # Shutdown: (optional cleanup)
    # e.g., release resources or shutdown thread pools

@app.get("/")
async def root():
    metrics.health_requests.inc()
    return RedirectResponse(url="/health")
    
@app.get("/health")
async def health():
    metrics.health_requests.inc()
    return {"status": "healthy"}

@app.get("/metrics")
def metrics_endpoint():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=settings.FASTAPI_PORT)