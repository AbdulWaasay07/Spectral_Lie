FROM python:3.10-slim

# Install system dependencies (ffmpeg for audio processing)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    gcc \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and install CPU-only PyTorch first (Significant space savings)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy requirements separately to leverage Docker cache
COPY part3_api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Part 1 and Part 2 packages
COPY part1_audio_features /part1_audio_features
COPY part2_detection /part2_detection

# Install Part 1 and Part 2 (they will use the already installed CPU torch)
RUN pip install --no-cache-dir -e /part1_audio_features
RUN pip install --no-cache-dir -e /part2_detection

# Copy Part 3 application code
COPY part3_api .

# Expose port
EXPOSE 8000

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PROMETHEUS_MULTIPROC_DIR=/tmp/metrics

# Create directory for prometheus multiprocess mode if needed
RUN mkdir -p /tmp/metrics

# Command to run the application (Gunicorn + Uvicorn)
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "app.main:app", "-b", "0.0.0.0:8000", "--timeout", "60"]
