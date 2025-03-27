from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
from contextlib import asynccontextmanager
import logging
import traceback

from backend.constants import (
    APP_TITLE, 
    APP_DESCRIPTION, 
    APP_VERSION, 
    API_V1_STR,
    CORS_ORIGINS,
    CORS_METHODS,
    CORS_HEADERS,
    CORS_CREDENTIALS
)
from backend.api.routes import fighters, predictions
from backend.api.database import get_db_connection, check_database_connection
from backend.ml.model_loader import load_model, get_loaded_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    try:
        logger.info("Loading model on application startup...")
        load_model()
        model = get_loaded_model()
        if model:
            logger.info("Model loaded successfully!")
        else:
            logger.warning("Model loading failed or no model was loaded!")
    except Exception as e:
        logger.error(f"Error loading model on startup: {str(e)}")
        logger.error(traceback.format_exc())
    
    # Check database connection on startup
    try:
        db_connected = check_database_connection()
        if db_connected:
            logger.info("Database connection successful!")
        else:
            logger.warning("Database connection check failed!")
    except Exception as e:
        logger.error(f"Error checking database connection: {str(e)}")
        logger.error(traceback.format_exc())
    
    yield
    
    # Clean up resources if needed
    logger.info("Shutting down application...")

# Create FastAPI app
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan
)

# Add CORS middleware with configured origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,  # Use configured origins from constants
    allow_credentials=CORS_CREDENTIALS,
    allow_methods=CORS_METHODS,
    allow_headers=CORS_HEADERS,
)

# Include routers
app.include_router(fighters.router)
app.include_router(predictions.router)

@app.get("/")
def read_root():
    return {"message": "UFC Fighter Prediction API is running!"}

@app.get("/health")
def health_check():
    # Do basic checks
    model = get_loaded_model()
    db_connected = check_database_connection()
    
    return {
        "status": "healthy" if model and db_connected else "degraded",
        "model_loaded": bool(model),
        "database_connected": db_connected
    }

# Add exception handlers for better error responses
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions and return a consistent error response."""
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handler for HTTP exceptions to ensure consistent logging."""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

# Run the API with uvicorn
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=True)