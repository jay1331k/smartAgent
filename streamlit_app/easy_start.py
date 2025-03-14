"""
Easy start script for running Cline Web IDE
Automatically handles checking requirements and starting the app
"""
import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

print("Starting Cline Web IDE...")

# Check if required packages are installed
def check_requirements():
    try:
        import streamlit
        try:
            import google.generativeai
            return True
        except ImportError:
            return False
    except ImportError:
        return False

# Install requirements if needed
if not check_requirements():
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
    except Exception as e:
        print(f"❌ Error installing requirements: {str(e)}")
        print("Please try installing them manually: pip install -r requirements.txt")
        sys.exit(1)

# Create temp directory if needed
temp_dir = Path("temp")
if not temp_dir.exists():
    temp_dir.mkdir()
    print("✓ Created temp directory")

# Run Streamlit app
print("Launching Cline Web IDE...")
port = 8501

# Start Streamlit in a separate process
cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(port)]

try:
    process = subprocess.Popen(cmd)
    
    # Wait a moment for the server to start
    print(f"Starting server on port {port}...")
    time.sleep(2)
    
    # Open web browser
    url = f"http://localhost:{port}"
    print(f"Opening {url} in your browser...")
    webbrowser.open(url)
    
    print("\nCline Web IDE is now running!")
    print("Press Ctrl+C to stop the server when you're done")
    
    # Keep the script running until manually stopped
    process.wait()
    
except KeyboardInterrupt:
    print("\nShutting down Cline Web IDE...")
    sys.exit(0)
except Exception as e:
    print(f"\n❌ Error: {str(e)}")
    sys.exit(1)
