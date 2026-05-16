"""
app.py
------
FastAPI REST gateway for the asynchronous transcription pipeline.

Run with:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import uuid
from pathlib import Path
from typing import Any

from celery.result import AsyncResult
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from config import settings
from tasks import celery_app, run_transcription_task

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #
ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
)

# --------------------------------------------------------------------------- #
# FastAPI application                                                           #
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="Audio Transcription Pipeline",
    description=(
        "Event-driven, asynchronous audio transcription service powered by "
        "faster-whisper and Celery."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# --------------------------------------------------------------------------- #
# Startup: ensure scratch directories exist                                     #
# --------------------------------------------------------------------------- #
@app.on_event("startup")
async def _create_scratch_dirs() -> None:
    Path(settings.uploads_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.outputs_dir).mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# POST /v1/transcribe                                                           #
# --------------------------------------------------------------------------- #
@app.post(
    "/v1/transcribe",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit an audio file for asynchronous transcription",
    response_description="Task queued confirmation with a task_id",
)
async def submit_transcription(
    user_id: str = Form(..., description="Caller-supplied user identifier"),
    file: UploadFile = File(..., description="Audio file to transcribe"),
) -> JSONResponse:
    """
    Accept an audio file upload, persist it to the local scratch directory
    using 64 KB chunked reads, and dispatch a Celery background task.

    Returns HTTP 202 with a `task_id` that the client can poll via
    `GET /v1/tasks/{task_id}`.
    """
    # ------------------------------------------------------------------ #
    # 1. Server-side extension validation                                  #
    # ------------------------------------------------------------------ #
    original_filename: str = file.filename or ""
    suffix: str = Path(original_filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{suffix}'. "
                f"Accepted extensions: {sorted(ALLOWED_EXTENSIONS)}"
            ),
        )

    # ------------------------------------------------------------------ #
    # 2. Generate a collision-free task / file ID                         #
    # ------------------------------------------------------------------ #
    task_id: str = str(uuid.uuid4())
    dest_filename: str = f"{task_id}{suffix}"
    dest_path: Path = Path(settings.uploads_dir) / dest_filename

    # ------------------------------------------------------------------ #
    # 3. Stream file binary in 64 KB chunks                               #
    # ------------------------------------------------------------------ #
    try:
        with open(dest_path, "wb") as fh:
            while True:
                chunk: bytes = await file.read(settings.upload_chunk_size)
                if not chunk:
                    break
                fh.write(chunk)
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to persist uploaded file: {exc}",
        ) from exc

    # ------------------------------------------------------------------ #
    # 4. Enqueue Celery task                                              #
    # ------------------------------------------------------------------ #
    metadata: dict[str, Any] = {
        "user_id": user_id,
        "original_filename": original_filename,
    }

    run_transcription_task.apply_async(
        args=[str(dest_path.resolve()), metadata],
        task_id=task_id,
    )

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "task_id": task_id,
            "status": "QUEUED",
            "message": (
                f"File '{original_filename}' has been queued for transcription. "
                f"Poll GET /v1/tasks/{task_id} for progress."
            ),
        },
    )


# --------------------------------------------------------------------------- #
# GET /v1/tasks/{task_id}                                                      #
# --------------------------------------------------------------------------- #
@app.get(
    "/v1/tasks/{task_id}",
    summary="Poll the status and result of a transcription task",
)
async def get_task_status(task_id: str) -> JSONResponse:
    """
    Return the current state of a submitted transcription task.

    Possible `status` values in the response:
    - **QUEUED**     – Task received but not yet picked up by a worker.
    - **STARTED**    – Worker has acknowledged the task.
    - **PROCESSING** – Inference is running; `progress` (0.0 – 1.0) is included.
    - **SUCCESS**    – Transcription complete; full result payload is included.
    - **FAILURE**    – Task failed after all retries.
    - **UNKNOWN**    – Task ID not recognised.
    """
    async_result: AsyncResult = AsyncResult(task_id, app=celery_app)
    state: str = async_result.state

    # ------------------------------------------------------------------ #
    # Map Celery states → user-friendly responses                         #
    # ------------------------------------------------------------------ #
    if state == "PENDING":
        return JSONResponse(
            content={
                "task_id": task_id,
                "status": "QUEUED",
                "detail": "Task is waiting in the queue.",
            }
        )

    if state == "STARTED":
        return JSONResponse(
            content={
                "task_id": task_id,
                "status": "STARTED",
                "detail": "Worker has picked up the task.",
            }
        )

    if state == "PROCESSING":
        task_meta: dict = async_result.info or {}
        progress: float = task_meta.get("progress", 0.0)
        return JSONResponse(
            content={
                "task_id": task_id,
                "status": "PROCESSING",
                "progress": progress,
                "detail": f"Transcription in progress ({progress * 100:.1f}%).",
            }
        )

    if state == "SUCCESS":
        result: dict = async_result.result or {}
        return JSONResponse(
            content={
                "task_id": task_id,
                "status": "SUCCESS",
                "result": result,
            }
        )

    if state == "FAILURE":
        error_info = str(async_result.info) if async_result.info else "Unknown error"
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "task_id": task_id,
                "status": "FAILURE",
                "detail": error_info,
            },
        )

    if state == "RETRY":
        return JSONResponse(
            content={
                "task_id": task_id,
                "status": "RETRYING",
                "detail": "Task encountered an error and is being retried.",
            }
        )

    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "task_id": task_id,
            "status": "UNKNOWN",
            "detail": f"Unrecognised task state: '{state}'.",
        },
    )
