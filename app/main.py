from fastapi import FastAPI, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import sys

from app.config import settings
from app.models.schemas import (
    VoiceDetectionRequest,
    VoiceDetectionResponse,
    ErrorResponse
)
from app.core.auth import verify_api_key
from app.core.exceptions import AudioProcessingError

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)

# Initialize FastAPI app
app = FastAPI(
    title="AI Voice Detection API (Hugging Face Powered)",
    description="Detect AI-generated vs Human voices using state-of-the-art deep learning",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/demo", tags=["UI"])
async def serve_demo():
    """
    Serve the demo UI (legacy endpoint)
    """
    return FileResponse("app/static/index.html")

# Global detector instance
detector = None

@app.on_event("startup")
async def startup_event():
    """
    Load model on startup based on configuration
    """
    global detector
    logger.info("Starting AI Voice Detection API...")
    
    from app.models.hf_detector import HuggingFaceDetector
    detector = HuggingFaceDetector()
    logger.info("✅ Hugging Face detector loaded")
    
    logger.info(f"API ready! Environment: {settings.ENVIRONMENT}")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Cleanup on shutdown
    """
    logger.info("Shutting down API...")

@app.get("/", tags=["UI"])
async def root():
    """
    Serve the main HTML interface
    """
    return FileResponse("index.html")

@app.get("/api/info", tags=["Health"])
async def api_info():
    """
    API information endpoint
    """
    return {
        "service": "AI Voice Detection API",
        "status": "running",
        "version": "2.0.0",
        "model": settings.HF_MODEL_NAME,
        "supported_languages": settings.SUPPORTED_LANGUAGES,
        "powered_by": "Hugging Face Transformers"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "model_type": settings.MODEL_TYPE,
        "device": "cuda" if settings.USE_GPU else "cpu"
    }

@app.post(
    "/api/voice-detection",
    response_model=VoiceDetectionResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    },
    tags=["Detection"]
)
async def voice_detection(
    request: VoiceDetectionRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Detect if voice is AI-generated or Human using deep learning
    
    - **language**: One of Tamil, English, Hindi, Malayalam, Telugu
    - **audioFormat**: Must be mp3
    - **audioBase64**: Base64 encoded MP3 audio
    
    Returns classification with confidence score
    """
    try:
        # Normalize and validate language
        lang = request.language.strip().title()
        if lang not in settings.SUPPORTED_LANGUAGES:
            logger.warning(f"Unsupported language '{request.language}', proceeding with default thresholds")
        logger.info(f"Processing request for language: {lang}")
        
        # Perform detection
        result = detector.detect(request.audioBase64, lang)
        
        # Build response
        response = VoiceDetectionResponse(
            status="success",
            language=lang,
            classification=result["classification"],
            confidenceScore=result["confidence"],
            explanation=result["explanation"]
        )
        
        logger.info(f"Detection complete: {result['classification']} ({result['confidence']})")
        return response
    
    except AudioProcessingError as e:
        logger.error(f"Audio processing error: {str(e)}")
        raise
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# Custom exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            status="error",
            message=exc.detail
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            status="error",
            message="Internal server error"
        ).dict()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=(settings.ENVIRONMENT == "development")
    )