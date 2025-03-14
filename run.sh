#!/bin/bash

echo "Setting up SmartAgent environment..."

# Ensure required packages are installed
pip install -r requirements.txt

# Set environment variable to enable better error reporting
export PYTHONPATH=$(pwd)
export PYTHONUNBUFFERED=1

# Run the application
streamlit run app.py "$@"
