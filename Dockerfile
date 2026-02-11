# ArXiv RAG System - AMD ROCm GPU
# Uses official ROCm PyTorch image for proper AMD GPU support
# PyTorch 2.7.1 + ROCm 6.4.4 (compatible with host ROCm 7.1)
FROM rocm/pytorch:rocm6.4.4_ubuntu22.04_py3.10_pytorch_release_2.7.1 AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HSA_OVERRIDE_GFX_VERSION=11.0.0

WORKDIR /app

# Copy and install Python dependencies (PyTorch already in base image)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/raw data/vector_store models/embedding models/llm && \
    chmod -R 755 data models

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - run API server
CMD ["python", "src/api/main.py"]
