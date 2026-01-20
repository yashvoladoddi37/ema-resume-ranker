# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements-minimal.txt .
COPY requirements-ui.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-minimal.txt
RUN pip install --no-cache-dir -r requirements-ui.txt

# Copy the rest of the application code
COPY . .

# Expose the port Streamlit will run on
EXPOSE 8501

# Healthcheck to ensure container is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Command to run the Streamlit application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
