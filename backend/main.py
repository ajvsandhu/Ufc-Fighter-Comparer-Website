from fastapi import FastAPI, Depends, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
from contextlib import asynccontextmanager
import logging
import traceback
import json
import time

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
from backend.utils import sanitize_json
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
    """Application startup and shutdown events"""
    # Log startup
    logger.info("Starting UFC Fighter Prediction API")
    
    # Create FastAPI app
    logger.info(f"Initializing {APP_TITLE} v{APP_VERSION}")
    
    # Setup model and database
    await setup_dependencies()
    
    # Log startup complete
    logger.info("Application startup complete")
    
    yield
    
    # Cleanup on shutdown
    logger.info("Application shutting down...")

async def setup_dependencies():
    """Setup required dependencies like model and database"""
    # Load model with multiple attempts
    model_loaded = False
    try:
        for attempt in range(3):  # Try up to 3 times
            logger.info(f"Loading model attempt {attempt+1}/3...")
            if load_model():
                model = get_loaded_model()
                if model:
                    logger.info("Model loaded successfully!")
                    model_loaded = True
                    break
                else:
                    logger.warning("Model loading returned True but no model was loaded. Retrying...")
            else:
                logger.warning("Model loading failed. Retrying...")
        
        if not model_loaded:
            logger.error("All model loading attempts failed.")
    except Exception as e:
        logger.error(f"Unexpected error loading model on startup: {str(e)}")
        logger.error(traceback.format_exc())
    
    # Check database connection with retry
    db_connected = False
    try:
        for attempt in range(3):  # Try up to 3 times
            logger.info(f"Checking database connection attempt {attempt+1}/3...")
            if check_database_connection():
                logger.info("Database connection successful!")
                db_connected = True
                break
            else:
                logger.warning("Database connection check failed. Retrying...")
                time.sleep(2)  # Wait a bit before retrying
        
        if not db_connected:
            logger.error("All database connection attempts failed.")
    except Exception as e:
        logger.error(f"Error checking database connection: {str(e)}")
        logger.error(traceback.format_exc())
    
    # Log dependency status
    logger.info(f"Dependencies initialized - Model loaded: {model_loaded}, Database connected: {db_connected}")
    return model_loaded, db_connected

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

# Log CORS configuration
logger.info(f"CORS configured with origins: {CORS_ORIGINS}")

# Import route modules directly to avoid circular imports
from backend.api.routes.fighters import router as fighters_router
from backend.api.routes.predictions import router as predictions_router

# Include routers
app.include_router(fighters_router)
app.include_router(predictions_router)

@app.get("/")
def read_root():
    """Root endpoint - API health check"""
    response_data = {"message": "UFC Fighter Prediction API is running!"}
    return sanitize_json(response_data)

@app.get("/health")
def health_check():
    """Health check endpoint - returns status of model and database"""
    # Do basic checks
    model = get_loaded_model()
    db_connected = check_database_connection()
    
    response_data = {
        "status": "healthy" if model and db_connected else "degraded",
        "model_loaded": bool(model),
        "database_connected": db_connected
    }
    return sanitize_json(response_data)

# Add exception handlers for better error responses
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions and return a consistent error response."""
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    
    # Add request path to help with debugging
    path = request.url.path
    method = request.method
    
    response_data = {
        "detail": f"Internal server error: {str(exc)}",
        "path": path,
        "method": method,
        "type": type(exc).__name__
    }
    
    # Sanitize the response data
    sanitized_data = sanitize_json(response_data)
    
    return JSONResponse(
        status_code=500,
        content=sanitized_data
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handler for HTTP exceptions to ensure consistent logging."""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    
    # Add request path to help with debugging
    path = request.url.path
    method = request.method
    
    response_data = {
        "detail": exc.detail,
        "path": path,
        "method": method
    }
    
    # Sanitize the response data
    sanitized_data = sanitize_json(response_data)
    
    return JSONResponse(
        status_code=exc.status_code,
        content=sanitized_data
    )

# Run the API with uvicorn when script is executed directly
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=True)