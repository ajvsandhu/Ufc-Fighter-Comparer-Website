"""
Utility functions used across the application.
"""
import logging

logger = logging.getLogger(__name__)

def sanitize_json(obj):
    """
    Recursively sanitize JSON values to prevent client-side errors.
    Ensures all values that might be used with string methods are valid strings.
    """
    # None -> empty string
    if obj is None:
        return ""
    
    # Handle dictionaries recursively
    elif isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            # Don't allow None keys
            if k is None:
                k = "unknown_key"
            result[k] = sanitize_json(v)
        return result
    
    # Handle lists recursively
    elif isinstance(obj, list):
        return [sanitize_json(item) for item in obj]
    
    # Handle specific types that frontend might try to use string methods on
    elif isinstance(obj, (int, float)):
        # Convert numbers to strings only for specific fields that frontend might 
        # try to use string methods on (like replace, split, etc.)
        numeric_fields_to_stringify = [
            "Record", "Height", "Weight", "Reach", "STANCE", "DOB", 
            "Str. Acc.", "Str. Def", "TD Acc.", "TD Def."
        ]
        
        # Check if we're inside a call stack with a known key
        parent_key = getattr(sanitize_json, 'current_key', None)
        if parent_key in numeric_fields_to_stringify:
            return str(obj)
        return obj
    
    # Convert bool to their string representation
    elif isinstance(obj, bool):
        return str(obj).lower()
    
    # Ensure strings are properly sanitized
    elif isinstance(obj, str):
        # Return empty string for "null", "undefined" or "None" strings
        if obj.lower() in ("null", "undefined", "none"):
            return ""
        return obj
    
    # Return string representation of other types
    else:
        try:
            return str(obj)
        except:
            return ""

def set_parent_key(key):
    """Set the current parent key for context in nested sanitization."""
    sanitize_json.current_key = key 