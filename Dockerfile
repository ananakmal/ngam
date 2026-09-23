# Production Dockerfile for ngam Universal Neural Decision Engine Daemon
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    NGAM_HOST=0.0.0.0 \
    NGAM_PORT=8045

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy package definition and project files
COPY pyproject.toml README.md /app/
COPY ngam/ /app/ngam/

# Install ngam package and core dependencies
RUN pip install --upgrade pip && \
    pip install .

# Expose daemon port
EXPOSE 8045

# Health check against daemon /healthz endpoint
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8045/healthz || exit 1

# Launch in-memory HTTP daemon
CMD ["ngam-daemon", "--host", "0.0.0.0", "--port", "8045"]
