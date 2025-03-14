"""
Utility functions for the SmartAgent.
"""
import time
import json
import re  # Ensure re is imported
from typing import Any, Callable, Dict, Optional, TypeVar, cast

# Import constants after defining the functions to avoid circular imports
from .constants import MAX_RETRIES, RETRY_DELAY

# Type variable for generic function
T = TypeVar('T')

def handle_retryable_error(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """
    Retry a function if it raises an exception.
    
    Args:
        func: The function to call
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        The result of the function
    """
    retries = 0
    while retries < MAX_RETRIES:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            retries += 1
            if retries >= MAX_RETRIES:
                raise
            print(f"Error: {e}. Retrying in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)
    
    # This should never be reached, but it satisfies the type checker
    raise RuntimeError("Unexpected error in handle_retryable_error")

def parse_constraint(constraint_str: str) -> Dict[str, Any]:
    """
    Parse a constraint string into a structured format.
    
    Args:
        constraint_str: A string representing a constraint, e.g., "format: json"
        
    Returns:
        A dictionary with the parsed constraint
    """
    # Default result
    result = {"type": "generic", "value": constraint_str}
    
    # Try to parse format constraints
    format_match = re.match(r"format:\s*(\w+)", constraint_str, re.IGNORECASE)
    if format_match:
        return {"type": "format", "format": format_match.group(1).lower()}
    
    # Try to parse max_length constraints
    length_match = re.match(r"max_length:\s*(\d+)", constraint_str, re.IGNORECASE)
    if length_match:
        return {"type": "max_length", "length": int(length_match.group(1))}
    
    # Try to parse contains constraints
    contains_match = re.match(r"contains:\s*(.+)", constraint_str, re.IGNORECASE)
    if contains_match:
        return {"type": "contains", "value": contains_match.group(1).strip()}
    
    return result

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract the first JSON object from a text string with enhanced robustness.
    
    Args:
        text: The text potentially containing a JSON object
        
    Returns:
        The parsed JSON object or None if no valid JSON was found
    """
    # First, look for standard JSON patterns
    json_pattern = r"\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{(?:[^{}])*\}))*\}))*\}"
    match = re.search(json_pattern, text)
    
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Try looking for JSON in markdown code blocks
    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    matches = re.findall(code_block_pattern, text)
    
    for potential_json in matches:
        try:
            return json.loads(potential_json)
        except json.JSONDecodeError:
            continue
    
    # Try to find any object with result or subtasks keys
    result_pattern = r"\{\s*\"result\"[\s\S]*?\}"
    match = re.search(result_pattern, text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
            
    subtasks_pattern = r"\{\s*\"subtasks\"[\s\S]*?\}"
    match = re.search(subtasks_pattern, text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    
    return None

# Node-specific retryable error handler - placed down here to avoid circular imports
def handle_node_retryable_error(node, attempt: int, exception: Exception) -> bool:
    """
    Handle retryable errors that occur during LLM API calls.
    
    Args:
        node: The node being processed
        attempt: The current attempt number (0-indexed)
        exception: The exception that was raised
        
    Returns:
        True if max retries exceeded, False otherwise
    """
    if attempt < MAX_RETRIES - 1:
        time.sleep(RETRY_DELAY)
        return False  # Continue retrying
    else:
        node.status = "failed"
        node.error_message = f"Error after {MAX_RETRIES} attempts: {str(exception)}"
        return True  # Max retries exceeded
