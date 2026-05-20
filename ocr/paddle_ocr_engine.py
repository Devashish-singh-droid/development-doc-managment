from utils.logger import get_logger

logger = get_logger("paddle_ocr_engine")
"""
paddle_ocr_engine.py

Reusable PaddleOCR engine
- Thread-safe singleton loader
- Image -> text extraction
- Confidence scoring
- Project-wide reusable
"""

import inspect
import logging
import os
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
from PIL import Image

from config import settings

if TYPE_CHECKING:
    from paddleocr import PaddleOCR

try:
    import cv2
except Exception:
    cv2 = None

# ============================
# SINGLETON OCR LOADER
# ============================

__lock = threading.Lock()
__paddle_ocr = None
__ocr_runtime = threading.local()


def _set_last_ocr_diagnostics(
    *,
    status: str,
    detail: str = "",
    min_confidence: float = 0.5,
    raw_candidates: int = 0,
    kept_lines: int = 0,
) -> None:
    __ocr_runtime.last_diagnostics = {
        "status": status,
        "detail": str(detail or "").strip(),
        "min_confidence": float(min_confidence),
        "raw_candidates": int(raw_candidates or 0),
        "kept_lines": int(kept_lines or 0),
    }


def get_last_ocr_diagnostics() -> Dict[str, Any]:
    diagnostics = getattr(__ocr_runtime, "last_diagnostics", None)
    if not isinstance(diagnostics, dict):
        return {}
    return dict(diagnostics)


def _configure_paddle_internal_logging(show_log: bool) -> None:
    if show_log:
        return
    for logger_name in ("ppocr", "paddleocr"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)


def enhance_for_ocr(pil_img: Image.Image) -> Image.Image:
    """Enhance image before OCR to improve text detection/recognition quality."""
    if cv2 is None:
        return pil_img

    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    denoised = cv2.fastNlMeansDenoising(gray, h=30)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)

    thresh = cv2.adaptiveThreshold(
        contrast,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10,
    )

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(thresh, -1, kernel)

    return Image.fromarray(sharp)


def get_paddle_ocr(
    lang: str = None,
    use_gpu: bool = None,
) -> "PaddleOCR":
    """
    Load PaddleOCR only once (thread-safe singleton).
    Uses environment variables for configuration if not explicitly provided.
    """
    global __paddle_ocr

    if lang is None:
        lang = settings.ocr_language

    if __paddle_ocr is None:
        with __lock:
            if __paddle_ocr is None:
                try:
                    from paddleocr import PaddleOCR
                except ModuleNotFoundError as exc:
                    raise ModuleNotFoundError(
                        "PaddleOCR runtime dependency is missing. Install 'paddlepaddle==3.0.0' in the active environment."
                    ) from exc

                os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
                show_log = settings.ocr_show_log
                _configure_paddle_internal_logging(show_log)
                init_signature = inspect.signature(PaddleOCR.__init__)
                supported_params = set(init_signature.parameters.keys())
                init_kwargs = {"lang": lang}

                # Prefer the PaddleOCR 3.x flags when available. We explicitly
                # disable extra document preprocessing models because plain OCR
                # on uploaded images/PDF pages does not need them.
                if "use_doc_orientation_classify" in supported_params:
                    init_kwargs["use_doc_orientation_classify"] = False
                if "use_doc_unwarping" in supported_params:
                    init_kwargs["use_doc_unwarping"] = False
                if "use_textline_orientation" in supported_params:
                    init_kwargs["use_textline_orientation"] = True
                elif "use_angle_cls" in supported_params:
                    init_kwargs["use_angle_cls"] = True
                if "show_log" in supported_params:
                    init_kwargs["show_log"] = show_log

                try:
                    __paddle_ocr = PaddleOCR(**init_kwargs)
                except Exception as exc:
                    logger.warning(f"PaddleOCR init failed with preferred settings: {exc}")
                    try:
                        fallback_kwargs = dict(init_kwargs)
                        fallback_kwargs.pop("show_log", None)
                        __paddle_ocr = PaddleOCR(**fallback_kwargs)
                    except Exception as exc2:
                        logger.warning(f"PaddleOCR init failed with compatibility fallback: {exc2}")
                        final_kwargs = {"lang": lang}
                        if "show_log" in supported_params:
                            final_kwargs["show_log"] = show_log
                        __paddle_ocr = PaddleOCR(**final_kwargs)

    return __paddle_ocr


# ============================
# OCR EXTRACTION FUNCTION
# ============================

def paddle_ocr_extract(
    image,
    min_confidence: float = 0.1,
) -> Optional[Dict[str, Any]]:
    """
    Run PaddleOCR on an image.

    Args:
        image: PIL.Image | OpenCV image | numpy array
        min_confidence: Filter weak OCR lines

    Returns:
        {
            "raw_text": str,
            "confidence": float,
            "engine": "PaddleOCR"
        }
    """

    ocr = get_paddle_ocr()

    if not isinstance(image, np.ndarray):
        image = np.array(image)

    ocr_input = image
    enhancement_used = False
    try:
        pil_input = Image.fromarray(image.astype(np.uint8))
        enhanced = enhance_for_ocr(pil_input)
        ocr_input = np.array(enhanced, dtype=np.uint8)
        enhancement_used = True
    except Exception as exc:
        logger.warning(f"Image enhancement failed before OCR; using original image. Error: {exc}")
        ocr_input = image

    _set_last_ocr_diagnostics(
        status="started",
        detail=f"enhancement_used={enhancement_used}",
        min_confidence=min_confidence,
        raw_candidates=0,
        kept_lines=0,
    )

    attempts = [("enhanced", ocr_input)] if enhancement_used else []
    attempts.append(("original", image))
    attempt_summaries: List[Dict[str, Any]] = []
    inference_errors: List[str] = []
    fallback_texts_all: List[str] = []

    for attempt_name, attempt_image in attempts:
        try:
            if hasattr(ocr, "predict"):
                result = ocr.predict(attempt_image)
            else:
                result = ocr.ocr(attempt_image, cls=True)
        except TypeError:
            try:
                result = ocr.ocr(attempt_image)
            except Exception as exc:
                msg = f"{attempt_name}: {exc}"
                inference_errors.append(msg)
                logger.warning(f"PaddleOCR inference failed ({attempt_name}): {exc}")
                continue
        except Exception as exc:
            msg = f"{attempt_name}: {exc}"
            inference_errors.append(msg)
            logger.warning(f"PaddleOCR inference failed ({attempt_name}): {exc}")
            continue

        lines, confidences, raw_candidates = _collect_ocr_lines(result, min_confidence=min_confidence)
        candidate_scores = _collect_candidate_scores(result)
        candidate_texts = _collect_candidate_texts(result)
        for candidate_text in candidate_texts:
            if candidate_text not in fallback_texts_all:
                fallback_texts_all.append(candidate_text)
        best_score = max(candidate_scores) if candidate_scores else 0.0
        attempt_summary = {
            "attempt": attempt_name,
            "raw_candidates": raw_candidates,
            "kept_lines": len(lines),
            "best_score": best_score,
        }
        attempt_summaries.append(attempt_summary)

        if lines:
            avg_confidence = (
                sum(confidences) / len(confidences) * 100
                if confidences
                else 0.0
            )
            _set_last_ocr_diagnostics(
                status="success",
                detail=(
                    f"attempt={attempt_name} extracted={len(lines)} "
                    f"raw_candidates={raw_candidates} best_score={best_score:.4f}"
                ),
                min_confidence=min_confidence,
                raw_candidates=raw_candidates,
                kept_lines=len(lines),
            )
            return {
                "raw_text": "\n".join(lines),
                "confidence": round(avg_confidence, 2),
                "engine": "PaddleOCR",
            }

    if inference_errors and not attempt_summaries:
        _set_last_ocr_diagnostics(
            status="inference_failed",
            detail=" | ".join(inference_errors)[:600],
            min_confidence=min_confidence,
            raw_candidates=0,
            kept_lines=0,
        )
        return None

    total_raw_candidates = sum(int(item.get("raw_candidates") or 0) for item in attempt_summaries)
    best_score_seen = max((float(item.get("best_score") or 0.0) for item in attempt_summaries), default=0.0)
    attempt_detail = "; ".join(
        f"{item['attempt']}(raw={item['raw_candidates']},kept={item['kept_lines']},best={float(item['best_score']):.4f})"
        for item in attempt_summaries
    )
    if fallback_texts_all:
        _set_last_ocr_diagnostics(
            status="low_confidence_fallback",
            detail=(
                f"Recovered {len(fallback_texts_all)} line(s) from OCR candidates despite low scores. "
                f"attempts=[{attempt_detail}] best_score_seen={best_score_seen:.4f}"
            ),
            min_confidence=min_confidence,
            raw_candidates=total_raw_candidates,
            kept_lines=len(fallback_texts_all),
        )
        return {
            "raw_text": "\n".join(fallback_texts_all),
            "confidence": round(best_score_seen * 100, 2),
            "engine": "PaddleOCR",
        }
    if total_raw_candidates > 0:
        _set_last_ocr_diagnostics(
            status="all_lines_below_confidence",
            detail=(
                f"Detected {total_raw_candidates} text line(s) across attempts, but all were below "
                f"min_confidence={min_confidence}. attempts=[{attempt_detail}] best_score_seen={best_score_seen:.4f}"
            ),
            min_confidence=min_confidence,
            raw_candidates=total_raw_candidates,
            kept_lines=0,
        )
    else:
        _set_last_ocr_diagnostics(
            status="no_text_detected",
            detail=f"No OCR text candidates detected. attempts=[{attempt_detail}]",
            min_confidence=min_confidence,
            raw_candidates=0,
            kept_lines=0,
        )
    return None


def _collect_candidate_scores(result: Any) -> List[float]:
    scores: List[float] = []
    if not isinstance(result, list):
        return scores

    for item in result:
        if isinstance(item, dict):
            rec_scores = item.get("rec_scores")
            if isinstance(rec_scores, list):
                for score in rec_scores:
                    try:
                        scores.append(float(score))
                    except Exception:
                        continue
            continue
        _walk_legacy_collect_scores(item, scores)
    return scores


def _collect_candidate_texts(result: Any) -> List[str]:
    texts: List[str] = []
    if not isinstance(result, list):
        return texts

    for item in result:
        if isinstance(item, dict):
            rec_texts = item.get("rec_texts")
            if isinstance(rec_texts, list):
                for text in rec_texts:
                    normalized = str(text or "").strip()
                    if normalized:
                        texts.append(normalized)
            continue
        _walk_legacy_collect_texts(item, texts)
    return texts


def _walk_legacy_collect_scores(payload: Any, scores: List[float]) -> None:
    if not isinstance(payload, list):
        return

    if len(payload) >= 2 and isinstance(payload[1], (list, tuple)):
        text_score = payload[1]
        if len(text_score) >= 2:
            if _looks_like_text_score_pair(text_score[0], text_score[1]):
                try:
                    scores.append(float(text_score[1]))
                except Exception:
                    pass
                return

    for item in payload:
        _walk_legacy_collect_scores(item, scores)


def _walk_legacy_collect_texts(payload: Any, texts: List[str]) -> None:
    if not isinstance(payload, list):
        return

    if len(payload) >= 2 and isinstance(payload[1], (list, tuple)):
        text_score = payload[1]
        if len(text_score) >= 2 and _looks_like_text_score_pair(text_score[0], text_score[1]):
            normalized_text = str(text_score[0] or "").strip()
            if normalized_text:
                texts.append(normalized_text)
            return

    for item in payload:
        _walk_legacy_collect_texts(item, texts)


def _collect_ocr_lines(
    result: Any,
    min_confidence: float = 0.5,
) -> tuple[List[str], List[float], int]:
    lines: List[str] = []
    confidences: List[float] = []
    raw_candidates = 0

    if not isinstance(result, list):
        return lines, confidences, raw_candidates

    # PaddleOCR 3.x returns OCRResult-style dicts with rec_texts + rec_scores.
    for item in result:
        if not isinstance(item, dict):
            continue
        rec_texts = item.get("rec_texts")
        rec_scores = item.get("rec_scores")
        if isinstance(rec_texts, list) and isinstance(rec_scores, list):
            for text, conf in zip(rec_texts, rec_scores):
                raw_candidates += 1
                _append_ocr_line(lines, confidences, text, conf, min_confidence)
            if lines:
                return lines, confidences, raw_candidates

    # PaddleOCR 2.x returns nested [box, [text, score]] lists.
    for item in result:
        raw_candidates += _walk_legacy_ocr_payload(item, lines, confidences, min_confidence)

    return lines, confidences, raw_candidates


def _walk_legacy_ocr_payload(
    payload: Any,
    lines: List[str],
    confidences: List[float],
    min_confidence: float,
) -> int:
    if not isinstance(payload, list):
        return 0

    discovered = 0

    if len(payload) >= 2 and isinstance(payload[1], (list, tuple)):
        text_score = payload[1]
        if len(text_score) >= 2:
            if _looks_like_text_score_pair(text_score[0], text_score[1]):
                discovered = 1
                _append_ocr_line(lines, confidences, text_score[0], text_score[1], min_confidence)
                return discovered

    for item in payload:
        discovered += _walk_legacy_ocr_payload(item, lines, confidences, min_confidence)
    return discovered


def _append_ocr_line(
    lines: List[str],
    confidences: List[float],
    text: Any,
    confidence: Any,
    min_confidence: float,
) -> None:
    try:
        normalized_text = str(text or "").strip()
        normalized_confidence = float(confidence)
    except Exception:
        return

    if not normalized_text or normalized_confidence < min_confidence:
        return

    lines.append(normalized_text)
    confidences.append(normalized_confidence)


def _looks_like_text_score_pair(text: Any, score: Any) -> bool:
    if not isinstance(text, str):
        return False
    normalized_text = text.strip()
    if not normalized_text:
        return False
    try:
        numeric_score = float(score)
    except Exception:
        return False
    # OCR recognition confidence is expected in [0, 1].
    return 0.0 <= numeric_score <= 1.0
