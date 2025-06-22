# Use a lightweight Python base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies
COPY ../requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy super_agent source code
COPY . /app

# Expose metrics endpoint (if needed by Prometheus)
EXPOSE 9000

# Start the FastAPI app with Uvicorn (you can customize host/port if needed)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
