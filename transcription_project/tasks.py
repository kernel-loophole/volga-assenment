"""
tasks.py
--------
Celery application initialisation and the core transcription background task.

Worker launch command (from project root):
    celery -A tasks worker --loglevel=info --concurrency=1
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

from celery import Celery
from celery.utils.log import get_task_logger

from config import settings

# --------------------------------------------------------------------------- #
# Logging                                                                      #
# --------------------------------------------------------------------------- #
logger = get_task_logger(__name__)

# --------------------------------------------------------------------------- #
# Celery application                                                            #
# --------------------------------------------------------------------------- #
celery_app = Celery(
    "transcription_pipeline",
    broker=settings.broker_url,
    backend=settings.result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # Keep results for 24 hours so the polling endpoint can read them.
    result_expires=86_400,
    # Prevent the worker from fetching more tasks than it can handle in
    # memory – critical when running with concurrency=1.
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)

# --------------------------------------------------------------------------- #
# Lazy-loaded Whisper model singleton                                           #
# --------------------------------------------------------------------------- #
_whisper_model = None


def _get_whisper_model():
    """
    Return the process-global WhisperModel instance, loading it on first call.
    The model is intentionally NOT loaded at import time – it is only
    materialised when the first task is executed inside a live worker process.
    """
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        logger.info(
            "Loading WhisperModel (size=%s, device=%s, compute_type=%s) …",
            settings.whisper_model_size,
            settings.device,
            settings.compute_type,
        )
        _whisper_model = WhisperModel(
            settings.whisper_model_size,
            device=settings.device,
            compute_type=settings.compute_type,
        )
        logger.info("WhisperModel loaded and cached in worker process.")
    return _whisper_model


# --------------------------------------------------------------------------- #
# Transcription task                                                            #
# --------------------------------------------------------------------------- #
@celery_app.task(
    bind=True,
    name="tasks.run_transcription_task",
    max_retries=3,
    default_retry_delay=10,
    track_started=True,
)
def run_transcription_task(
    self,
    file_path: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Background task that:
      1. Loads (or reuses) the Whisper model singleton.
      2. Streams transcription segments from faster-whisper.
      3. Writes a structured JSON result to the configured outputs directory.
      4. Cleans up the uploaded audio file regardless of success or failure.

    Parameters
    ----------
    file_path : str
        Absolute path to the uploaded audio file.
    metadata  : dict
        Arbitrary caller-supplied metadata (e.g. user_id, original filename).

    Returns
    -------
    dict
        The final transcription payload, mirrored to the Celery result backend.
    """
    task_id: str = self.request.id

    # Signal that processing has begun.
    self.update_state(
        state="PROCESSING",
        meta={"progress": 0.0, "task_id": task_id},
    )

    try:
        # ---------------------------------------------------------------- #
        # 1. Obtain model                                                   #
        # ---------------------------------------------------------------- #
        model = _get_whisper_model()

        # ---------------------------------------------------------------- #
        # 2. Run inference                                                  #
        # ---------------------------------------------------------------- #
        logger.info("[%s] Starting transcription of '%s'", task_id, file_path)

        self.update_state(
            state="PROCESSING",
            meta={"progress": 0.05, "task_id": task_id},
        )

        segments_generator, transcription_info = model.transcribe(
            file_path,
            beam_size=5,
            vad_filter=True,
        )

        detected_language: str = transcription_info.language
        total_duration: float = transcription_info.duration

        logger.info(
            "[%s] Detected language='%s', duration=%.2fs",
            task_id,
            detected_language,
            total_duration,
        )

        # ---------------------------------------------------------------- #
        # 3. Collect segments                                               #
        # ---------------------------------------------------------------- #
        segments_list: list[dict[str, Any]] = []
        processed_duration: float = 0.0

        for segment in segments_generator:
            segments_list.append(
                {
                    "start": round(segment.start, 3),
                    "end": round(segment.end, 3),
                    "text": segment.text.strip(),
                    "confidence": round(segment.avg_logprob, 6),
                }
            )

            # Update progress proportionally to audio time consumed.
            processed_duration = segment.end
            progress_pct = (
                min(processed_duration / total_duration, 0.99)
                if total_duration > 0
                else 0.99
            )
            self.update_state(
                state="PROCESSING",
                meta={"progress": round(progress_pct, 4), "task_id": task_id},
            )

        # ---------------------------------------------------------------- #
        # 4. Assemble final payload                                         #
        # ---------------------------------------------------------------- #
        full_text: str = " ".join(
            seg["text"] for seg in segments_list if seg["text"]
        )

        result_payload: dict[str, Any] = {
            "task_id": task_id,
            "metadata": metadata,
            "language": detected_language,
            "total_duration": round(total_duration, 3),
            "full_text": full_text,
            "segments": segments_list,
        }

        # ---------------------------------------------------------------- #
        # 5. Persist result to disk                                         #
        # ---------------------------------------------------------------- #
        outputs_dir = Path(settings.outputs_dir)
        outputs_dir.mkdir(parents=True, exist_ok=True)

        output_file = outputs_dir / f"{task_id}.json"
        with open(output_file, "w", encoding="utf-8") as fh:
            json.dump(result_payload, fh, ensure_ascii=False, indent=2)

        logger.info("[%s] Result written to '%s'", task_id, output_file)

        return result_payload

    except Exception as exc:
        logger.exception(
            "[%s] Transcription failed: %s – retrying (attempt %d/%d)",
            task_id,
            exc,
            self.request.retries + 1,
            self.max_retries,
        )
        raise self.retry(exc=exc)

    finally:
        # ---------------------------------------------------------------- #
        # 6. Always remove the uploaded source file to reclaim disk space.  #
        # ---------------------------------------------------------------- #
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(
                    "[%s] Deleted source file '%s'", task_id, file_path
                )
        except OSError as cleanup_err:
            logger.warning(
                "[%s] Could not delete source file '%s': %s",
                task_id,
                file_path,
                cleanup_err,
            )
