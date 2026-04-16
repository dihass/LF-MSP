# Base image (stable + compatible)
FROM python:3.11-slim

# Prevent Python buffering
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system deps (needed for scipy, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (cache optimization)
COPY requirements.txt .

# Install Python deps
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy rest of project
COPY . .

# Expose port
EXPOSE 8000

# Run app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]