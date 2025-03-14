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
GLOBAL_CONTEXT_SUMMARY_INTERVAL = 10
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_OVERRIDDEN = "overridden"  # Added from agent/constants.py

def initialize_gemini_api():
    api_key = st.secrets.get("GOOGLE_API_KEY", None)
    if (api_key):
        genai.configure(api_key=api_key)
        return True
    else:
        st.error("Google API Key not found in secrets. Please configure it.")
        api_key = st.text_input("Enter Google API Key:", type="password")
        if (api_key):
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

def parse_constraint(constraint: str) -> Dict[str, str]:
    """Parse a constraint string into type and value."""
    if ":" in constraint:
        constraint_type, value = constraint.split(":", 1)
        return {"type": constraint_type.strip(), "value": value.strip()}
    return {"type": constraint, "value": ""}

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Extract JSON content from text."""
    json_pattern = r'```(?:json)?\s*([\s\S]*?)```'
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    if matches:
        # Try each match until we find valid JSON
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    # If no JSON in code blocks, try to parse the whole text
    try:
        # Clean up the text - remove markdown formatting if present
        clean_text = re.sub(r'^\s*```.*$', '', text, flags=re.MULTILINE)
        clean_text = re.sub(r'^\s*```\s*$', '', clean_text, flags=re.MULTILINE)
        
        return json.loads(clean_text)
    except json.JSONDecodeError:
        # Look for JSON-like structure without code blocks
        try:
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = text[start_idx:end_idx]
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass
    
    return {}

def handle_node_retryable_error(node: Any, attempt: int, error: Exception) -> bool:
    """Handle retryable errors during node execution."""
    if attempt < MAX_RETRIES - 1:
        node.error_message = f"Error (attempt {attempt + 1}): {str(error)}. Retrying..."
        time.sleep(RETRY_DELAY)
        return False
    else:
        node.status = STATUS_FAILED
        node.error_message = f"Max retries reached. Last error: {str(error)}"
        return True

def extract_code_blocks(text: str) -> List[Dict[str, str]]:
    """Extract code blocks from text with language and content."""
    code_blocks = []
    pattern = r'```(\w*)\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for lang, content in matches:
        if lang.strip():
            code_blocks.append({
                "language": lang.strip(),
                "content": content.strip()
            })
        else:
            code_blocks.append({
                "language": "text",
                "content": content.strip()
            })
            
    return code_blocks

def extract_code_with_filenames(text: str) -> Dict[str, Dict[str, str]]:
    """
    Advanced code extraction that looks for file paths in various formats.
    Returns a dictionary mapping filepath to {language, content}.
    """
    result = {}
    
    # Pattern 1: Code blocks with filepath comments (most common)
    pattern1 = r'```(\w*)\n(?:\/\/|#)\s*filepath:\s*(.*?)\n(.*?)```'
    matches1 = re.findall(pattern1, text, re.DOTALL)
    for lang, filepath, code in matches1:
        filepath = filepath.strip()
        result[filepath] = {
            "language": lang.strip() or "text",
            "content": code.strip()
        }
    
    # Pattern 2: Explicit filepath mentions followed by code blocks
    pattern2 = r'[Ff]ile(?:path)?:\s*[`"\']?(.*?)[`"\']?\n\s*```(\w*)\n(.*?)```'
    matches2 = re.findall(pattern2, text, re.DOTALL)
    for filepath, lang, code in matches2:
        filepath = filepath.strip()
        if filepath not in result:  # Don't override if already found
            result[filepath] = {
                "language": lang.strip() or "text",
                "content": code.strip()
            }
    
    # Pattern 3: Code blocks with likely filenames in headers or comments
    if not result:
        code_blocks = re.findall(r'```(\w+)\n(.*?)```', text, re.DOTALL)
        for lang, code in code_blocks:
            # Look for filename patterns in the first few lines
            first_lines = code.split("\n")[:3]
            for line in first_lines:
                file_match = re.search(r'(?:file|filename|path):\s*[`"\']?([\w\/\.-]+)[`"\']?', line, re.IGNORECASE)
                if file_match:
                    filepath = file_match.group(1).strip()
                    if filepath not in result:
                        result[filepath] = {
                            "language": lang.strip() or "text",
                            "content": code.strip()
                        }
                    break
    
    # If we still have no results, make best-effort guesses based on language
    if not result:
        # Map of language to typical file extension
        lang_to_ext = {
            "python": "py", 
            "py": "py",
            "javascript": "js",
            "js": "js",
            "typescript": "ts", 
            "ts": "ts",
            "html": "html",
            "css": "css", 
            "java": "java",
            "c": "c", 
            "cpp": "cpp", 
            "csharp": "cs",
            "cs": "cs",
            "go": "go",
            "rust": "rs",
            "ruby": "rb",
            "php": "php",
            "shell": "sh",
            "bash": "sh"
        }
        
        file_counter = {}  # Keep track of how many files per language
        code_blocks = re.findall(r'```(\w+)\n(.*?)```', text, re.DOTALL)
        for lang, code in code_blocks:
            if lang:
                lang_lower = lang.lower()
                ext = lang_to_ext.get(lang_lower, lang_lower)
                
                # Increment counter for this language
                if lang_lower not in file_counter:
                    file_counter[lang_lower] = 1
                else:
                    file_counter[lang_lower] += 1
                
                # Generate a filename
                count = file_counter[lang_lower]
                if lang_lower in ["python", "py"]:
                    filename = f"script_{count}.py" if count > 1 else "main.py"
                elif lang_lower in ["javascript", "js"]:
                    filename = f"script_{count}.js" if count > 1 else "main.js"
                elif lang_lower == "html":
                    filename = f"page_{count}.html" if count > 1 else "index.html"
                elif lang_lower == "css":
                    filename = f"style_{count}.css" if count > 1 else "styles.css"
                else:
                    filename = f"file_{count}.{ext}"
                
                # Add to results if we don't already have this filename
                if filename not in result:
                    result[filename] = {
                        "language": lang,
                        "content": code.strip()
                    }
    
    return result

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
