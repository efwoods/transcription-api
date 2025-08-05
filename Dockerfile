# Stage 1: Build
FROM python:3.11-slim AS builder

WORKDIR /app

# Optional: reduce layer size
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git ffmpeg libsndfile1-dev build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

COPY . .

# Stage 2: Runtime image
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg libsndfile1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
COPY --from=builder /app /app

ENV PATH=/root/.local/bin:$PATH
ENV TRANSCRIPTION_MODEL=tiny

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
