import os
import shutil
import threading
from pathlib import Path

from config import settings
from utils.logger import get_logger

logger = get_logger("video_to_text")

_MODEL = None
_MODEL_LOCK = threading.Lock()
_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v", ".webm"}


def _candidate_binary_from_env(raw_value: str) -> str:
    candidate = str(raw_value or "").strip().strip('"').strip("'")
    if not candidate:
        return ""

    candidate_path = Path(candidate)
    if candidate_path.is_file():
        return str(candidate_path)

    if candidate_path.is_dir():
        exe_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
        exe_path = candidate_path / exe_name
        if exe_path.exists():
            return str(exe_path)

    return ""


def _resolve_ffmpeg_binary() -> str:
    explicit_candidates = [
        settings.get_string("FFMPEG_PATH", ""),
        settings.get_string("FFMPEG_BINARY", ""),
        os.getenv("FFMPEG_PATH", ""),
        os.getenv("FFMPEG_BINARY", ""),
    ]

    for raw_candidate in explicit_candidates:
        binary = _candidate_binary_from_env(raw_candidate)
        if binary:
            return binary

    discovered = shutil.which("ffmpeg")
    return str(discovered or "").strip()


def _ensure_ffmpeg_on_path() -> str:
    ffmpeg_binary = _resolve_ffmpeg_binary()
    if not ffmpeg_binary:
        raise RuntimeError(
            "Video transcription requires ffmpeg. "
            "Add the ffmpeg 'bin' folder to PATH or set FFMPEG_PATH to the full ffmpeg.exe path, then restart the app."
        )

    ffmpeg_dir = str(Path(ffmpeg_binary).resolve().parent)
    current_path = os.getenv("PATH", "")
    path_entries = [entry for entry in current_path.split(os.pathsep) if entry]
    normalized_dir = os.path.normcase(os.path.normpath(ffmpeg_dir))
    normalized_entries = {
        os.path.normcase(os.path.normpath(entry))
        for entry in path_entries
    }
    if normalized_dir not in normalized_entries:
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + current_path if current_path else ffmpeg_dir
        logger.info(f"Prepended ffmpeg directory to PATH: {ffmpeg_dir}")

    return ffmpeg_binary


def _get_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        try:
            import whisper
        except Exception as exc:
            raise RuntimeError(
                "Whisper is not installed. Add 'openai-whisper' to the environment to enable video transcription."
            ) from exc

        model_name = str(settings.get_string("WHISPER_MODEL_NAME", "small") or "small").strip() or "small"
        logger.info(f"Loading Whisper model: {model_name}")
        _MODEL = whisper.load_model(model_name)
        return _MODEL


def is_video_file(path_or_name: str) -> bool:
    suffix = Path(str(path_or_name or "")).suffix.lower()
    return suffix in _VIDEO_EXTENSIONS


def transcribe_video(video_path: str) -> str:
    """
    Transcribe a video file with Whisper and return plain text.
    """
    safe_path = str(video_path or "").strip()
    if not safe_path:
        raise ValueError("video_path is required")
    if not os.path.exists(safe_path):
        raise FileNotFoundError(f"Video not found: {safe_path}")

    ffmpeg_binary = _ensure_ffmpeg_on_path()
    model = _get_model()
    try:
        result = model.transcribe(safe_path)
        return str((result or {}).get("text") or "").strip()
    except Exception as exc:
        logger.error(f"Error during video transcription for {safe_path} using ffmpeg '{ffmpeg_binary}': {exc}")
        raise


def transcribe_and_save(video_path: str, output_file: str = "transcript.txt") -> str:
    """
    Transcribe a video file and save the transcript to a text file.
    """
    transcript = transcribe_video(video_path)
    output_path = str(output_file or "transcript.txt").strip() or "transcript.txt"
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(transcript)
    logger.info(f"Video transcription saved to {output_path}")
    return output_path
