"""Constants for the SmartAgent application."""

# Agent Settings
MAX_RETRIES = 3
RETRY_DELAY = 2
MAX_DEPTH = 5
GLOBAL_CONTEXT_SUMMARY_INTERVAL = 10

# Node Status Constants
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_OVERRIDDEN = "overridden"

# Default LLM Settings
LLM_MODEL = "gemini-2.0-pro-exp-02-05"
LLM_TEMPERATURE = 0.2
LLM_TOP_P = 0.95
LLM_TOP_K = 40
LLM_MAX_TOKENS = 8192

# File Management
CODE_LANGUAGES = {
    "py": "python",
    "js": "javascript",
    "ts": "typescript",
    "html": "html",
    "css": "css",
    "json": "json",
    "md": "markdown",
    "txt": "text",
    "sh": "bash",
    "bash": "bash",
    "c": "c",
    "cpp": "cpp",
    "cs": "csharp",
    "java": "java",
    "rb": "ruby",
    "php": "php",
    "go": "go",
    "rs": "rust",
    "swift": "swift",
    "kt": "kotlin",
    "yaml": "yaml",
    "yml": "yaml",
    "toml": "toml",
    "xml": "xml",
    "sql": "sql"
}

# Prompt Templates
CODE_PROMPT_TEMPLATE = """
When generating code:

1. Always include complete, working code in your response
2. Format code blocks with proper language identifiers: ```language
3. Always include filepath at the top of each code block as a comment:
   ```python
   # filepath: path/to/file.py
   # Rest of the code...
   ```
4. Your code should be production-ready with proper error handling
5. Include all necessary imports and dependencies
6. When implementing a full solution, split the code into logical files
"""

ROOT_NODE_PROMPT_TEMPLATE = """
Your task is to decompose this complex task into manageable subtasks.

Please follow these guidelines:
1. Break down the main task into 3-7 logical subtasks
2. Each subtask should be clearly defined and focused
3. Together, the subtasks should fully solve the original task
4. Return your response as JSON in this exact format:
```json
{
  "subtasks": [
    "First subtask description",
    "Second subtask description",
    "Third subtask description"
  ]
}
```

Keep your subtasks clear, specific, and actionable.
"""

# Add a template for code files
CODE_FILE_TEMPLATE = """
For code-related tasks, please provide complete implementation files.
Always include the filepath as a comment at the top of each code block.

Example:
```python
# filepath: src/main.py
def main():
    print("Hello, world!")

if __name__ == "__main__":
    main()
```

Make sure the code is properly organized into appropriate files.
"""
