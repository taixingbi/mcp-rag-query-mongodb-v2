# ---- base image ----
FROM python:3.11-slim

# ---- system settings ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ---- install system deps (optional but safe) ----
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---- install python deps first (better layer caching) ----
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---- copy app code (.env is in .dockerignore; pass at run time via --env-file .env) ----
COPY . .

# ---- expose port ----
EXPOSE 8000

# ---- start MCP server ----
# Pass env via: docker run --env-file .env ...
# Required: MONGODB_URI, OLLAMA_BASE_URL (use http://host.docker.internal:11434 if Ollama runs on host)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
