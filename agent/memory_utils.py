"""
Utility functions for memory operations in SmartAgent.
"""
import json
from typing import Any, Dict, Optional, Union

def parse_response(text: str) -> Union[Dict[str, Any], str]:
    """
    Parse a text response from LLM that may contain JSON.
    
    Args:
        text: The text response from the LLM
        
    Returns:
        Parsed JSON object or the original string if parsing fails
    """
    # Try to find JSON content within the response
    try:
        # Look for content between curly braces
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx >= 0 and end_idx > start_idx:
            json_str = text[start_idx:end_idx+1]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    # If no JSON found or parsing failed, return the original text
    return text

def safe_serialize(obj: Any) -> Any:
    """
    Convert an object to a JSON-serializable format.
    
    Args:
        obj: The object to serialize
        
    Returns:
        JSON-serializable representation of the object
    """
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [safe_serialize(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): safe_serialize(v) for k, v in obj.items()}
    else:
        # For other types, convert to string representation
        return str(obj)

def create_structured_memory(raw_response: str) -> Dict[str, Any]:
    """
    Process a raw LLM response into structured memory.
    
    Args:
        raw_response: Raw text response from the LLM
        
    Returns:
        Dictionary with parsed and original response
    """
    parsed = parse_response(raw_response)
    
    memory = {
        "raw_llm_response": raw_response,
        "parsed_response": parsed if isinstance(parsed, dict) else {},
        "is_structured": isinstance(parsed, dict)
    }
    
    return memory
