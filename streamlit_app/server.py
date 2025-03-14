#!/usr/bin/env python
"""
Server configuration for deploying Cline Web IDE
This script allows deploying Cline Web IDE on a server with proper security settings
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path
import secrets

def generate_secret():
    """Generate a secure secret key for Streamlit."""
    return secrets.token_hex(32)

def create_config_file(port, workers, enable_auth, secret_key=None):
    """Create Streamlit config file."""
    config_dir = Path.home() / ".streamlit"
    config_dir.mkdir(exist_ok=True)
    
    config_path = config_dir / "config.toml"
    
    if secret_key is None:
        secret_key = generate_secret()
    
    config_content = f"""
[server]
port = {port}
enableCORS = false
enableXsrfProtection = true
enableWebsocketCompression = true
maxUploadSize = 50
maxMessageSize = 200

[browser]
serverAddress = "0.0.0.0"
gatherUsageStats = false

[runner]
magicEnabled = true
installTracker = false
fastReruns = true

[theme]
base = "dark"

[client]
toolbarMode = "minimal"
showSidebarNavigation = true

[logger]
level = "info"
messageFormat = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

[deploy]
stateStoreProvider = "filesystem"
stateStoreDirectory = ".streamlit/state"
"""

    with open(config_path, "w") as f:
        f.write(config_content)
    
    # Create .env file with the secret key
    env_path = Path.cwd() / ".env"
    
    # Read existing content if available
    env_content = ""
    if env_path.exists():
        with open(env_path, "r") as f:
            env_content = f.read()
    
    # Check if STREAMLIT_SECRET_KEY is already set
    if "STREAMLIT_SECRET_KEY" not in env_content:
        with open(env_path, "a") as f:
            f.write(f"\n# Auto-generated secret key for Streamlit\nSTREAMLIT_SECRET_KEY={secret_key}\n")

    # Set environment variable for authentication
    if not enable_auth:
        os.environ["DISABLE_AUTH"] = "true"

    print(f"✓ Created Streamlit config in {config_path}")
    return secret_key

def setup_server():
    """Set up the server environment."""
    print("Setting up server environment...")
    
    # Create directories
    data_dir = Path.cwd() / "data"
    data_dir.mkdir(exist_ok=True)
    
    temp_dir = Path.cwd() / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    # Install production dependencies
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gunicorn", "watchdog"])
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    print("✓ Server environment set up successfully!")

def run_production_server(port=8501, workers=4, enable_auth=True):
    """Run Streamlit with gunicorn for production."""
    print(f"🚀 Starting Cline Web IDE production server on port {port} with {workers} workers")
    
    # Create config file and get secret key
    secret_key = create_config_file(port, workers, enable_auth)
    
    # Set environment variables
    os.environ["STREAMLIT_SECRET_KEY"] = secret_key
    
    # Build gunicorn command
    cmd = [
        "gunicorn",
        "--worker-class", "gthread",
        "--workers", str(workers),
        "--threads", "4",
        "--bind", f"0.0.0.0:{port}",
        "--timeout", "120",
        "--preload",
        "streamlit.web.bootstrap:bootstrap_wrapper"
    ]
    
    # Add app entry point
    cmd.extend(["--", "app.py"])
    
    try:
        subprocess.call(cmd)
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Error running server: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Cline Web IDE in production mode")
    parser.add_argument("--setup", action="store_true", help="Set up server environment")
    parser.add_argument("--port", type=int, default=8501, help="Port to run the server on")
    parser.add_argument("--workers", type=int, default=4, help="Number of Gunicorn workers")
    parser.add_argument("--no-auth", action="store_true", help="Disable authentication (not recommended for production)")
    
    args = parser.parse_args()
    
    if args.setup:
        setup_server()
    
    run_production_server(port=args.port, workers=args.workers, enable_auth=not args.no_auth)
