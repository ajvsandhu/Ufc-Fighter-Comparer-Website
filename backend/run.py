"""
Run script for the UFC Fighter Prediction API.
This is the main entry point for the application.
"""

import logging
import os
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting UFC Fighter Prediction API...")
    
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    # Run the FastAPI application
    uvicorn.run(
        "backend.main:app", 
        host=host, 
        port=port, 
        reload=True if os.environ.get("DEBUG", "False").lower() == "true" else False
    )
    
    logger.info("Application stopped.") 