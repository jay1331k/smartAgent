"""Utility functions for the SmartAgent."""

import time
import re
import json
from typing import Dict, Any, Tuple, Optional, Union, Callable
from .constants import MAX_RETRIES, RETRY_DELAY

def handle_retryable_error(func: Callable, *args, **kwargs):
    """
    Retry a function on failure.
    
    Args:
        func: The function to retry
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Result of the function if successful
        
    Raises:
        The last exception if all retries fail
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
    raise RuntimeError("Unexpected error in handle_retryable_error")

def handle_node_retryable_error(node, attempt: int, exception: Exception) -> bool:
    """
    Handle retryable errors for node execution.
    
    Args:
        node: The node being executed
        attempt: The current attempt number (0-indexed)
        exception: The exception that occurred
        
    Returns:
        True if retries are exhausted, False if should retry
    """
    if attempt < MAX_RETRIES - 1:
        time.sleep(RETRY_DELAY)
        return False
    else:
        node.status = "failed"
        node.error_message = f"Error after {MAX_RETRIES} attempts: {str(exception)}"
        return True

def parse_constraint(constraint_str: str) -> Tuple[str, str]:
    """
    Parse a constraint string into type and value.
    
    Args:
        constraint_str: String containing the constraint
        
    Returns:
        Tuple of (constraint_type, constraint_value)
    """
    parts = constraint_str.split(':', 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return "generic", constraint_str.strip()

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON object from text.
    
    Args:
        text: Text that might contain JSON
        
    Returns:
        Extracted JSON object as dict or None if no valid JSON found
    """
    # Try to find JSON in code blocks
    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    matches = re.findall(code_block_pattern, text)
    for potential_json in matches:
        try:
            return json.loads(potential_json)
        except json.JSONDecodeError:
            continue
    
    # Try to find JSON with regex
    json_pattern = r"\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{(?:[^{}])*\}))*\}))*\}"
    match = re.search(json_pattern, text)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Look for specific patterns
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
