import streamlit as st
import google.generativeai as genai
import re
import json
import time
from typing import Optional, Dict, List, Any, Union, Tuple

# --- Constants ---
MAX_RETRIES = 3
RETRY_DELAY = 2
MAX_DEPTH = 5
GLOBAL_CONTEXT_SUMMARY_INTERVAL = 5
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_OVERRIDDEN = "overridden"  # Added from agent/constants.py

def initialize_gemini_api():
    api_key = st.secrets.get("GOOGLE_API_KEY", None)
    if api_key:
        genai.configure(api_key=api_key)
        return True
    else:
        st.error("Google API Key not found in secrets. Please configure it.")
        api_key = st.text_input("Enter Google API Key:", type="password")
        if api_key:
            genai.configure(api_key=api_key)
            return True
    return False

def get_model():
    models = [m for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    if not models:
        st.error("No suitable models found. Check your API key.")
        return None
    
    selected_model = st.session_state.get('selected_model', "gemini-2.0-pro-exp-02-05")
    model = genai.GenerativeModel(selected_model)
    return model

def handle_retryable_error(func, *args, **kwargs):
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

def parse_constraint(constraint_str):
    result = {"type": "generic", "value": constraint_str}
    format_match = re.match(r"format:\s*(\w+)", constraint_str, re.IGNORECASE)
    if format_match:
        return {"type": "format", "format": format_match.group(1).lower()}
    length_match = re.match(r"max_length:\s*(\d+)", constraint_str, re.IGNORECASE)
    if length_match:
        return {"type": "max_length", "length": int(length_match.group(1))}
    contains_match = re.match(r"contains:\s*(.+)", constraint_str, re.IGNORECASE)
    if contains_match:
        return {"type": "contains", "value": contains_match.group(1).strip()}
    return result

def extract_json_from_text(text):
    json_pattern = r"\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{(?:[^{}])*\}))*\}))*\}"
    match = re.search(json_pattern, text)
    if match:
        json_str = match.group(0)
        try: return json.loads(json_str)
        except json.JSONDecodeError: pass
    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    matches = re.findall(code_block_pattern, text)
    for potential_json in matches:
        try: return json.loads(potential_json)
        except json.JSONDecodeError: continue
    result_pattern = r"\{\s*\"result\"[\s\S]*?\}"
    match = re.search(result_pattern, text)
    if match:
        try: return json.loads(match.group(0))
        except json.JSONDecodeError: pass
    subtasks_pattern = r"\{\s*\"subtasks\"[\s\S]*?\}"
    match = re.search(subtasks_pattern, text)
    if match:
        try: return json.loads(match.group(0))
        except json.JSONDecodeError: pass
    return None

def handle_node_retryable_error(node, attempt, exception):
    if attempt < MAX_RETRIES - 1:
        time.sleep(RETRY_DELAY)
        return False
    else:
        node.status = "failed"
        node.error_message = f"Error after {MAX_RETRIES} attempts: {str(exception)}"
        return True

def parse_response(text: str):
    try:
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx >= 0 and end_idx > start_idx:
            json_str = text[start_idx:end_idx+1]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    return text

def safe_serialize(obj):
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [safe_serialize(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): safe_serialize(v) for k, v in obj.items()}
    else:
        return str(obj)

def create_structured_memory(raw_response):
    parsed = parse_response(raw_response)
    memory = {
        "raw_llm_response": raw_response,
        "parsed_response": parsed if isinstance(parsed, dict) else {},
        "is_structured": isinstance(parsed, dict)
    }
    return memory
