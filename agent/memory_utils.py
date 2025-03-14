"""Utility functions for working with memory in the agent."""

import json
import re
from typing import Dict, Any

def parse_response(text: str) -> Any:
    """
    Parse a string response, attempting to extract JSON.
    
    Args:
        text: The text string to parse
        
    Returns:
        Parsed JSON object or the original text
    """
    try:
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx >= 0 and end_idx > start_idx:
            json_str = text[start_idx:end_idx+1]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    # If we couldn't parse JSON, return the original text
    return text

def create_structured_memory(raw_response: str) -> Dict[str, Any]:
    """
    Process a raw LLM response into a structured memory format.
    
    Args:
        raw_response: The raw text from the LLM
        
    Returns:
        A dictionary with structured memory
    """
    parsed = parse_response(raw_response)
    memory = {
        "raw_llm_response": raw_response,
        "parsed_response": parsed if isinstance(parsed, dict) else {},
        "is_structured": isinstance(parsed, dict)
    }
    
    # If the response is structured, add specific fields from it
    if memory["is_structured"]:
        if "result" in parsed:
            memory["result"] = parsed["result"]
        if "subtasks" in parsed:
            memory["subtasks"] = parsed["subtasks"]
            
    return memory

def safe_serialize(obj: Any) -> Any:
    """
    Convert complex objects to JSON-serializable types.
    
    Args:
        obj: Any Python object
        
    Returns:
        A serializable version of the object
    """
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [safe_serialize(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): safe_serialize(v) for k, v in obj.items()}
    else:
        return str(obj)
