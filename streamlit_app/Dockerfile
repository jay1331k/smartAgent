FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn watchdog

# Copy application code
COPY . .

# Configure environment
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Create necessary directories
RUN mkdir -p /app/temp
RUN mkdir -p /app/data
RUN mkdir -p ~/.streamlit

# Create Streamlit config
RUN echo '\
[server]\n\
port = 8501\n\
enableCORS = false\n\
enableXsrfProtection = true\n\
\n\
[browser]\n\
serverAddress = "0.0.0.0"\n\
gatherUsageStats = false\n\
\n\
[theme]\n\
base = "dark"\n\
' > ~/.streamlit/config.toml

# Expose port
EXPOSE 8501

# Command to run the application
CMD ["python", "server.py"]
