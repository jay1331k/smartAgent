# Smart Agent IDE

A VSCode-like IDE with hierarchical task decomposition agent powered by Google Gemini AI.

## Features

- File explorer on the left side
- Agent interface for hierarchical task decomposition
- Code editor with syntax highlighting
- Integrated terminal
- Google Gemini AI integration

## Project Structure

```
e:\Projects\smartAgent\
├── app.py                    # Main application entry point
├── components\               # Component modules
│   ├── __init__.py           # Package initializer
│   ├── agent.py              # Agent class implementation
│   ├── attention_mechanism.py # Attention mechanism for task tracking
│   ├── editor.py             # Code editor component
│   ├── file_explorer.py      # File explorer component
│   ├── memory.py             # Memory components for agent
│   ├── node.py               # Node class for task tree
│   ├── terminal.py           # Terminal/CLI component
│   └── utils.py              # Utility functions
└── requirements.txt          # Python dependencies
```

## Setup and Installation

1. Make sure you have Python 3.8+ installed
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your Google API key:
   - Create a `.streamlit/secrets.toml` file in the project directory
   - Add your Google API key:
   ```toml
   GOOGLE_API_KEY = "your_google_api_key_here"
   ```

## Running the Application

Run the application using Streamlit:

```bash
streamlit run app.py
```

The application will open in your default web browser at http://localhost:8501.

## Usage

1. **Smart Agent**: Use the agent to break down complex tasks
2. **File Explorer**: Browse and open files from the left sidebar
3. **Code Editor**: Edit selected files in the center panel
4. **Terminal**: Run commands in the integrated terminal at the bottom

## Project Requirements

See the `requirements.txt` file for detailed dependencies.
