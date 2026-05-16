# 🎙️ Async Audio Transcription Pipeline

A production-ready, event-driven audio transcription service built on **FastAPI**, **Celery**, **Redis**, and **faster-whisper**.

```
transcription_project/
├── config.py          # Centralised pydantic-settings configuration
├── tasks.py           # Celery app + lazy-loaded WhisperModel + inference task
├── app.py             # FastAPI gateway with upload & polling endpoints
└── requirements.txt   # Pinned Python dependencies
```

---

## Architecture Overview

```
Client
  │
  │  POST /v1/transcribe  (multipart form: user_id + audio file)
  ▼
┌─────────────┐        ┌───────────────────────────────┐
│  FastAPI    │──────▶│        Redis Broker             │
│  (app.py)   │        │  (task queue + result backend) │
└─────────────┘        └───────────────┬───────────────┘
  │                                    │
  │  GET /v1/tasks/{task_id}           │  dequeue
  │  (polls result backend)            ▼
  │                        ┌──────────────────────────┐
  └───────────────────────▶│   Celery Worker           │
                            │   (tasks.py)              │
                            │   └── WhisperModel (int8) │
                            └──────────────────────────┘
```

---

## Prerequisites

| Tool    | Minimum version |
|---------|----------------|
| Python  | 3.10+          |
| Docker  | 20.10+         |
| pip     | 23+            |

---

## Step 1 — Start Redis via Docker

```bash
# Pull and run Redis in the background, exposing the default port.
docker run -d \
  --name transcription-redis \
  -p 6379:6379 \
  --restart unless-stopped \
  redis:7.2-alpine

# Verify it is running.
docker ps | grep transcription-redis

# Quick connectivity check (expects "+PONG").
docker exec transcription-redis redis-cli ping
```

---

## Step 2 — Create & Activate the Python Virtual Environment

```bash
cd /home/zulqarnain/Desktop/volga-assenment/transcription_project

python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip inside the venv before installing dependencies.
pip install --upgrade pip

pip install -r requirements.txt
```

> **Note:** `faster-whisper` will download the selected Whisper model weights
> on first inference. With `WHISPER_MODEL_SIZE=base` this is ~145 MB and is
> cached in `~/.cache/huggingface/hub/`.

---

## Step 3 — (Optional) Environment Overrides

Create a `.env` file inside `transcription_project/` to override any default:

```ini
# .env  — all values are optional; defaults shown below
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB_BROKER=0
REDIS_DB_BACKEND=1

WHISPER_MODEL_SIZE=base   # tiny | base | small | medium | large-v3
DEVICE=cpu                # cpu | cuda
COMPUTE_TYPE=int8         # int8 | float16 | float32

UPLOADS_DIR=api_scratch/uploads
OUTPUTS_DIR=api_scratch/outputs
```

---

## Step 4 — Start the Celery Worker

Open a **dedicated terminal**, activate the venv, then run:

```bash
cd /home/zulqarnain/Desktop/volga-assenment/transcription_project

source .venv/bin/activate

celery -A tasks worker \
  --loglevel=info \
  --concurrency=1
```

> `--concurrency=1` is intentional — it prevents multiple worker processes
> from each loading a full Whisper model into RAM simultaneously.

Expected output:
```
[config] -> transport:redis://localhost:6379/0
[queues] -> celery
[tasks]
  . tasks.run_transcription_task
[2026-...] INFO/MainProcess  celery@hostname ready.
```

---

## Step 5 — Start the FastAPI Server

Open a **second terminal**, activate the venv, then run:

```bash
cd /home/zulqarnain/Desktop/volga-assenment/transcription_project

source .venv/bin/activate

uvicorn app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload
```

The interactive API docs will be available at:
- Swagger UI → http://localhost:8000/docs
- ReDoc      → http://localhost:8000/redoc

---

## Step 6 — Test with cURL

### 6.1 Submit a transcription job

```bash
curl -s -X POST http://localhost:8000/v1/transcribe \
  -F "user_id=zulqarnain" \
  -F "file=@/path/to/your/audio.mp3" | jq .
```

Expected HTTP 202 response:

```json
{
  "task_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "status": "QUEUED",
  "message": "File 'audio.mp3' has been queued for transcription. Poll GET /v1/tasks/3fa85f64-5717-4562-b3fc-2c963f66afa6 for progress."
}
```

### 6.2 Poll for status while processing

```bash
TASK_ID="3fa85f64-5717-4562-b3fc-2c963f66afa6"

curl -s http://localhost:8000/v1/tasks/${TASK_ID} | jq .
```

Progress response (HTTP 200):

```json
{
  "task_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "status": "PROCESSING",
  "progress": 0.4321,
  "detail": "Transcription in progress (43.2%)."
}
```

### 6.3 Retrieve the final result

Once `status` is `"SUCCESS"`:

```json
{
  "task_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "status": "SUCCESS",
  "result": {
    "task_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "metadata": {
      "user_id": "zulqarnain",
      "original_filename": "audio.mp3"
    },
    "language": "en",
    "total_duration": 47.232,
    "full_text": "Hello, this is a transcription test ...",
    "segments": [
      {
        "start": 0.0,
        "end": 3.14,
        "text": "Hello, this is a transcription test.",
        "confidence": -0.312456
      }
    ]
  }
}
```

### 6.4 Validate file-type rejection

```bash
curl -s -X POST http://localhost:8000/v1/transcribe \
  -F "user_id=test" \
  -F "file=@/path/to/document.pdf" | jq .
```

Expected HTTP 400:

```json
{
  "detail": "Unsupported file type '.pdf'. Accepted extensions: ['.flac', '.m4a', '.mp3', '.ogg', '.wav']"
}
```

---

## Teardown

```bash
# Stop Redis container (data is ephemeral unless you mount a volume).
docker stop transcription-redis && docker rm transcription-redis

# Deactivate the venv.
deactivate
```

---

## Configuration Reference

| Variable              | Default                  | Description                                  |
|-----------------------|--------------------------|----------------------------------------------|
| `REDIS_HOST`          | `localhost`              | Redis server hostname                        |
| `REDIS_PORT`          | `6379`                   | Redis server port                            |
| `REDIS_DB_BROKER`     | `0`                      | Redis DB index used as Celery broker         |
| `REDIS_DB_BACKEND`    | `1`                      | Redis DB index used as result backend        |
| `WHISPER_MODEL_SIZE`  | `base`                   | faster-whisper model size                    |
| `DEVICE`              | `cpu`                    | Inference device (`cpu` / `cuda`)            |
| `COMPUTE_TYPE`        | `int8`                   | Quantisation type (`int8` / `float16`)       |
| `UPLOADS_DIR`         | `api_scratch/uploads`    | Temporary upload storage path                |
| `OUTPUTS_DIR`         | `api_scratch/outputs`    | JSON result output path                      |
| `UPLOAD_CHUNK_SIZE`   | `65536`                  | File read chunk size in bytes (64 KB)        |