# SMARTAGENT/agent/utils.py
import time
import re
import streamlit as st
from agent import agent  # Import constants


def parse_constraint(constraint_string: str) -> tuple[str, str]:
    """Parses a constraint string into its type and value."""
    try:
        constraint_type, constraint_value = constraint_string.split(":", 1)
        return constraint_type.strip(), constraint_value.strip()
    except ValueError:
        return "unknown", constraint_string

def handle_retryable_error(node: "Node", attempt: int, error: Exception) -> bool:
    """Handles retryable LLM API errors (rate limits, timeouts)."""
    if "429" in str(error) or "rate limit" in str(error).lower() or "timeout" in str(error).lower():
        if attempt < agent.MAX_RETRIES - 1:
            delay = agent.RETRY_DELAY * (2 ** attempt)
            st.toast(f"{type(error).__name__}. Retrying in {delay} seconds (attempt {attempt + 2}/{agent.MAX_RETRIES})...")
            time.sleep(delay)
            return False
        else:
            node.status = "failed"
            node.error_message = f"LLM API Error: Max retries exceeded: {error}"
            st.error(node.error_message)
            return True
    else:
        node.status = "failed"
        node.error_message = f"LLM API Error: {error}"
        st.error(node.error_message)
        return True
