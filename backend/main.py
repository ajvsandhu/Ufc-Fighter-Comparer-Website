import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes.fighters_order_fix import router as fighters_router
from backend.api.routes.predictions import router as predictions_router
from backend.api.database import get_db_connection
from backend.constants import (
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    CORS_ORIGINS,
    CORS_METHODS,
    CORS_HEADERS,
    CORS_CREDENTIALS,
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    SERVER_HOST,
    SERVER_PORT
)
from typing import List, Dict

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION
)

# Enable CORS so the frontend can communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=CORS_CREDENTIALS,
    allow_methods=CORS_METHODS,
    allow_headers=CORS_HEADERS,
)

# Register API routes
app.include_router(fighters_router)
app.include_router(predictions_router)

@app.get("/")
def home():
    logger.info("Home endpoint accessed")
    return {"message": "Welcome to the UFC Fighter Comparison API!"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)