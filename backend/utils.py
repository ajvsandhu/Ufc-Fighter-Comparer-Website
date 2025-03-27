"""
Utility functions used across the application.
"""
import logging

logger = logging.getLogger(__name__)

def sanitize_json(obj):
    """Recursively sanitize JSON values to prevent client-side errors."""
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