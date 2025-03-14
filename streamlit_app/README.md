# Cline Web IDE

A browser-based code editor with integrated AI assistance, powered by Streamlit and Cline.

## Features

- **File Explorer**: Browse, create, edit, and manage project files
- **Code Editor**: Syntax highlighting and auto-formatting for multiple languages
- **Terminal**: Execute shell commands directly within the browser
- **AI Assistant**: Get help from Cline AI to write and debug your code
- **Project Download**: Export your complete project as a ZIP file

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Quick Start

The easiest way to get started is with our quick start script:

```bash
python quick_start.py
```

This will:
1. Check and install dependencies
2. Launch the Cline Web IDE
3. Open it in your default browser

To also create sample projects:

```bash
python quick_start.py --sample
```

For other options:

```bash
python quick_start.py --help
```

### Quick Start for Windows Users

The easiest way to get started on Windows is to use the included batch script:

```bash
start_app.bat
```

This will:
1. Create a virtual environment if it doesn't exist
2. Install all required dependencies
3. Launch the application with authentication disabled
4. Open it in your default browser

### Manual Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cline-web-ide.git
cd cline-web-ide/streamlit_app
```

2. Set up the environment:
```bash
python run.py --setup
```

3. Launch the application:
```bash
python run.py
```

4. For development without authentication:
```bash
python run.py --no-auth
```

### Docker Installation

If you prefer using Docker:

1. Build and run with Docker Compose:
```bash
docker-compose up -d
```

2. Access the application at http://localhost:8501

3. To stop the application:
```bash
docker-compose down
```

## Usage

1. Access the web IDE at http://localhost:8501
2. Log in or continue with anonymous access if authentication is disabled
3. Enter your Google API key in the sidebar to enable AI assistance
4. Create or upload files using the file explorer
5. Edit your code in the editor tab
6. Run commands in the terminal tab
7. Get AI assistance in the Cline Assistant tab
8. Download your project when finished

## Deployment

For detailed deployment instructions, see [DEPLOYMENT.md](./DEPLOYMENT.md).

## Project Structure

```
streamlit_app/
├── app.py                # Main Streamlit application
├── cline_interface.py    # AI assistant integration
├── file_explorer.py      # File browser component
├── firebase_auth.py      # Authentication module
├── run.py                # Local development launcher
├── quick_start.py        # Quick start script 
├── server.py             # Production server launcher
├── terminal.py           # Terminal integration
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker configuration
└── docker-compose.yml    # Docker Compose configuration
```

## Security Notice

- The application runs commands with the same permissions as the user who started the server
- Ensure proper authentication is enabled in production environments
- For local use, you can disable authentication with the --no-auth flag
