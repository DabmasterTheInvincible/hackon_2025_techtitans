# Use a minimal Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install build dependencies (optional for some ML libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy shared requirements from root
COPY ../../../requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy sub-agent code (local dir)
COPY . /app

# Expose port (only if needed; usually not required for sub-agent)
# EXPOSE 8001

# Run the agent
CMD ["python", "agent.py"]
