"""
SmartAgent: An AI-powered agent for hierarchical task decomposition.
"""

from .constants import (
    MAX_RETRIES,
    RETRY_DELAY,
    MAX_DEPTH,
    GLOBAL_CONTEXT_SUMMARY_INTERVAL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    STATUS_PENDING,
    STATUS_RUNNING,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_OVERRIDDEN
)

__version__ = "0.1.0"
