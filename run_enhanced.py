import subprocess
import os
import sys

def main():
    print("Starting Enhanced SmartAgent...")
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change to the script directory
    os.chdir(script_dir)
    
    # Run the Streamlit app
    cmd = [sys.executable, "-m", "streamlit", "run", "enhanced_app.py", "--server.port=8501"]
    
    try:
        process = subprocess.Popen(cmd)
        print("Enhanced SmartAgent is running at http://localhost:8501")
        print("Press Ctrl+C to stop")
        process.wait()
    except KeyboardInterrupt:
        print("\nStopping SmartAgent...")
        process.terminate()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
