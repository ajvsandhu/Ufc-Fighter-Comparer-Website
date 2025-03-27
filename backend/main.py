from fastapi import FastAPI, Depends, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.base import BaseHTTPMiddleware
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

# Custom middleware to sanitize response JSON
class SanitizeJSONMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Process the request (normal flow)
        response = await call_next(request)
        
        # Only process JSON responses
        if response.headers.get("content-type") == "application/json":
            try:
                # Read the response body
                body = [chunk async for chunk in response.body_iterator]
                
                # Recreate the response with the sanitized JSON
                if body:
                    json_body = json.loads(b"".join(body))
                    
                    # Function to recursively sanitize JSON values
                    def sanitize_json(obj):
                        if isinstance(obj, dict):
                            # Clean dictionary values
                            return {k: sanitize_json(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            # Clean list items
                            return [sanitize_json(item) for item in obj]
                        elif obj is None:
                            # Convert None to empty string for string context
                            return ""
                        else:
                            # Keep other values as-is
                            return obj
                    
                    # Sanitize the data
                    sanitized_data = sanitize_json(json_body)
                    
                    # Create a new JSON response with sanitized data
                    return JSONResponse(
                        content=sanitized_data,
                        status_code=response.status_code,
                        headers=dict(response.headers)
                    )
            except Exception as e:
                # Log error but continue with original response
                logger.error(f"Error in SanitizeJSONMiddleware: {str(e)}")
                logger.error(traceback.format_exc())
                
                # We need to return a new response with the original body
                # since we've consumed the body_iterator
                return Response(
                    content=b"".join(body) if 'body' in locals() else b"",
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type
                )
        
        return response

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup with multiple attempts
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
    
    # Check database connection on startup with retry
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
    
    # Log startup status
    logger.info(f"Application startup complete. Model loaded: {model_loaded}, Database connected: {db_connected}")
    
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

# Log CORS configuration
logger.info(f"CORS configured with origins: {CORS_ORIGINS}")

# Include routers
app.include_router(fighters.router)
app.include_router(predictions.router)

# Add the sanitize JSON middleware
app.add_middleware(SanitizeJSONMiddleware)

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
    
    # Add request path to help with debugging
    path = request.url.path
    method = request.method
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"Internal server error: {str(exc)}",
            "path": path,
            "method": method,
            "type": type(exc).__name__
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handler for HTTP exceptions to ensure consistent logging."""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    
    # Add request path to help with debugging
    path = request.url.path
    method = request.method
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "path": path,
            "method": method
        }
    )

# Run the API with uvicorn
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=True)