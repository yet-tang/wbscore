# Dockerfile for wbscore API
# Multi-stage build to keep image lean. Final image runs as non-root.
# Target: deployable on any Docker host (Hetzner CX22 recommended).
FROM python:3.11-slim AS builder

WORKDIR /build

# Build deps for tokenizers / pyarrow
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only torch (~200MB vs ~2GB CUDA build) — fits in slim image.
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch && \
    pip install --no-cache-dir -r requirements.txt

# ----- runtime stage -----
FROM python:3.11-slim

WORKDIR /app

# Runtime libs only (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# App code
COPY api_server.py recommend.py rule_checker.py wb_fetcher.py laya_engine.py wb_rules.json ./

# Model checkpoints (v4 is current best; v3 kept as fallback)
COPY checkpoints/ ./checkpoints/

# Training data + synthetic negatives (small, <2MB)
COPY data/ ./data/

# Frontend (single-page app)
COPY frontend/ ./frontend/

# Non-root user
RUN useradd -m -u 1000 app && chown -R app:app /app
USER app

EXPOSE 8080

# Warmup: hit /healthz on startup so the model loads before traffic.
# (uvicorn --workers > 1 means each worker loads its own copy of the model.)
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/v1/healthz || exit 1

# workers=2: 1 process per CPU core. Each loads the model (~400MB RAM).
# For 2 vCPU machines, this is optimal.
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2", "--log-level", "info"]