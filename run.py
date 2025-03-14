"""
Launcher for the Smart Agent IDE
"""

import os
import sys
import subprocess
import argparse
import webbrowser
import time
import platform

def main():
    parser = argparse.ArgumentParser(description="Run Smart Agent IDE")
    parser.add_argument("--port", type=int, default=8501, help="Port to run the app on")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    args = parser.parse_args()
    
    # Make sure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Check if required modules are installed
    try:
        import streamlit
        import google.generativeai
        from dotenv import load_dotenv
        print("✓ Dependencies found")
    except ImportError as e:
        print(f"Error: Missing dependency {e.name}")
        install = input("Would you like to install required dependencies? (y/n): ")
        if install.lower() == 'y':
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("Dependencies installed successfully!")
            # Try importing again after installation
            try:
                import streamlit
                import google.generativeai
                from dotenv import load_dotenv
                print("✓ All dependencies now available")
            except ImportError as e2:
                print(f"Error: Still missing dependency {e2.name}. Please install manually.")
                sys.exit(1)
        else:
            print("Please install the required dependencies and try again.")
            sys.exit(1)
    
    print(f"Starting Smart Agent IDE on port {args.port}...")
    
    # Start Streamlit in a subprocess
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py"]
    cmd.extend(["--server.port", str(args.port)])
    cmd.extend(["--server.headless", "true"])
    
    # Use the appropriate method to create a non-blocking subprocess
    if platform.system() == "Windows":
        process = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        process = subprocess.Popen(cmd)
    
    # Wait for the server to start
    print("Waiting for server to start...")
    time.sleep(3)
    
    # Open browser
    if not args.no_browser:
        url = f"http://localhost:{args.port}"
        print(f"Opening {url} in your browser...")
        webbrowser.open(url)
    
    print("Server is running. Press Ctrl+C in the terminal window to stop.")
    
    try:
        # Keep the script running to make it easy to Ctrl+C
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping server...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("Server didn't terminate gracefully, forcing shutdown...")
            process.kill()

if __name__ == "__main__":
    main()
