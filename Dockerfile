# ==========================================================
# Stage 1: Builder (Compile dependencies and build wheels)
# ==========================================================
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install build-essential only where it's needed to compile FAISS / C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Create a wheels directory to safely pass compiled assets to the next stage
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt

# ==========================================================
# Stage 2: Final Runtime (Lightweight execution layer)
# ==========================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Bring over the pre-compiled wheels from the builder stage
COPY --from=builder /app/wheels /app/wheels
COPY requirements.txt .

# Install the pre-compiled packages locally without needing build-essential
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --no-index --find-links=/app/wheels -r requirements.txt && \
    rm -rf /app/wheels

# Copy application code last to protect build cache layers
COPY . .

EXPOSE 8080

# Dynamic port binding designed for Google Cloud Run target environments
CMD ["sh", "-c", "streamlit run streamlit_app.py --server.port=${PORT:-8080} --server.address=0.0.0.0"]