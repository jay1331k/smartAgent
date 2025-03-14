"""
Constants used across the SmartAgent modules.
"""

# Maximum number of retries for operations
MAX_RETRIES = 3

# Delay between retries (in seconds)
RETRY_DELAY = 2

# Agent Configuration
MAX_DEPTH = 5
GLOBAL_CONTEXT_SUMMARY_INTERVAL = 5
LLM_MODEL = "gemini-pro"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 2048

# Status constants
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_OVERRIDDEN = "overridden"
