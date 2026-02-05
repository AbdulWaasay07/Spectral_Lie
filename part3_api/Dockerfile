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

# Pre-download models to bake them into the image
# This prevents timeouts and OOM during first request on Free tier
RUN python download_models.py

# Expose port
EXPOSE 8000

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

# Command to run the application (Gunicorn + Uvicorn)
# Increased to 2 workers to handle validator concurrency; still safe for 512MB RAM if deep_features=False
CMD ["sh", "-c", "gunicorn -k uvicorn.workers.UvicornWorker -w 2 --preload app.main:app -b 0.0.0.0:${PORT:-8000} --timeout 300"]
