#!/usr/bin/env python
"""
Script to install dependencies for Cline Web IDE
"""
import subprocess
import sys
import platform
import os

def install_requirements():
    print("Installing dependencies...")
    
    # Install all packages 
    packages = [
        "streamlit==1.29.0",
        "streamlit-ace==0.1.1",
        "pexpect==4.8.0", 
        "python-dotenv==1.0.0",
        "watchdog==3.0.0",
        "streamlit-tree-select==0.0.5",
        "zipfile36==0.1.3",
        "pyyaml==6.0.1",
        "google-generativeai==0.3.1"
    ]
    
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
    print("✓ All dependencies installed successfully!")

if __name__ == "__main__":
    install_requirements()
