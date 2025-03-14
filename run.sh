#!/bin/bash

echo "Starting SmartAgent IDE..."

# Ensure required packages are installed
pip install -r requirements.txt

# Set environment variable to enable better error reporting
export PYTHONPATH=$(pwd)
export PYTHONUNBUFFERED=1

# Run the application
streamlit run app.py "$@"
