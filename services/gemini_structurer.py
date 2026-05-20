import json
import os
import re
import threading
import time

from config import settings
from pydantic import ValidationError

from services.gemini_client import (
    generate_json_text as gemini_generate_json_text,
    is_configured as gemini_is_configured,
    model_name as gemini_model_name,
)
from services.json_utils import safe_json_parse
from services.models import StructurerChunkResponse
from utils.logger import get_logger

logger = get_logger("gemini_structurer")

os.makedirs("output", exist_ok=True)

_chunk_size = max(1000, settings.get_int("LLM_STRUCTURER_CHUNK_SIZE", 4000))
_chunk_overlap = max(0, settings.get_int("LLM_STRUCTURER_CHUNK_OVERLAP", 500))
_llm_max_tokens = max(512, settings.get_int("LLM_STRUCTURER_MAX_TOKENS", 3000))
_max_concurrency = max(1, settings.get_int("LLM_STRUCTURER_MAX_CONCURRENCY", 2))
_chunk_retry_count = max(1, settings.get_int("LLM_STRUCTURER_RETRY_COUNT", 3))
_retry_backoff_seconds = max(0.0, settings.get_float("LLM_STRUCTURER_RETRY_BACKOFF_SECONDS", 1.0))
_structurer_semaphore = threading.BoundedSemaphore(_max_concurrency)


def _model_json_schema(model_cls) -> dict:
    if hasattr(model_cls, "model_json_schema"):
        return model_cls.model_json_schema()
    return model_cls.schema()


STRUCTURER_CHUNK_RESPONSE_SCHEMA = _model_json_schema(StructurerChunkResponse)

SYSTEM_PROMPT = """
You are an intelligent document analyzer.

You will receive OCR-extracted text from one document chunk.
This may be part of a larger document.
Extract structured information from THIS chunk only.

OBJECTIVE
1. Identify the document_type from the chunk if it is clear.
2. Extract only information explicitly present in this chunk.
3. Preserve document values exactly as written.
4. Return exactly one valid JSON object and nothing else.

DOCUMENT TYPE RULES
- document_type must be a short snake_case label.
- You may create a new label if needed.
- If the type is unclear, use "unknown".

FIELD EXTRACTION RULES
- Keys must be snake_case.
- Values must be exact text from the chunk.
- Keep number formatting exactly as shown.
- Keep date formatting exactly as shown.
- Do not invent, normalize, or infer missing values.

JSON OUTPUT RULES
- Always return these 4 top-level keys:
  "document_type", "metadata_entries", "items", "chunk_summary"
- Always return exactly one JSON object.
- Do not return markdown, code fences, comments, or prose.
- Do not use trailing commas.
- metadata_entries must be an array of objects with exactly:
  {"key": "...", "value": "..."}
- items must be an array of objects with exactly:
  {"fields": [{"key": "...", "value": "..."}]}
- If no metadata is present, return "metadata_entries": []
- If no item rows are present, return "items": []
- Omit empty keys and empty values.
- Omit any item whose fields array would be empty.
- chunk_summary must be a short plain-text summary string.

IMPORTANT OCR HANDLING
- OCR text may contain brackets, totals, or table fragments.
- Treat document text as plain text, not as JSON syntax.
- If a value contains characters like [ ] { } : , they must remain inside a quoted JSON string.
- Never output a closing bracket ] or brace } unless it is valid JSON syntax.

Return JSON matching this exact shape:
{
  "document_type": "unknown",
  "metadata_entries": [],
  "items": [],
  "chunk_summary": ""
}
"""

REPAIR_SYSTEM_PROMPT = """
You repair malformed JSON for a document structuring pipeline.

You will receive a broken JSON-like response for one document chunk.
Return exactly one valid JSON object with these 4 top-level keys:
"document_type", "metadata_entries", "items", "chunk_summary"

Required output shape:
{
  "document_type": "unknown",
  "metadata_entries": [],
  "items": [],
  "chunk_summary": ""
}

Rules:
- Output only valid JSON
- No markdown, no prose, no comments
- Use empty arrays instead of malformed or partial arrays
- Preserve all usable fields from the malformed input
- Do not invent missing values
- metadata_entries entries must be {"key": "...", "value": "..."}
- items entries must be {"fields": [{"key": "...", "value": "..."}]}
- If a field is unclear or empty, omit that field
"""


def split_document_into_chunks(text: str, chunk_size: int = None) -> list:
    chunks = []
    chunk_size = chunk_size or _chunk_size
    overlap = min(_chunk_overlap, max(0, chunk_size - 1))

    if len(text) <= chunk_size:
        return [text]

    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap

    return chunks


def normalize_document_type_label(value) -> str:
    if value is None:
        return "unknown"
    label = str(value).strip().lower()
    if not label:
        return "unknown"
    label = re.sub(r"[^a-z0-9]+", "_", label)
    label = re.sub(r"_+", "_", label).strip("_")
    return label or "unknown"


def _coerce_metadata_entries(raw) -> list:
    entries = []
    if isinstance(raw, dict):
        raw = [{"key": key, "value": value} for key, value in raw.items()]
    if not isinstance(raw, list):
        return entries

    for entry in raw:
        if not isinstance(entry, dict):
            continue
        if "key" in entry and "value" in entry:
            key = str(entry.get("key") or "").strip()
            value = str(entry.get("value") or "").strip()
            if key and value:
                entries.append({"key": key, "value": value})
            continue
        for key, value in entry.items():
            key = str(key or "").strip()
            value = str(value or "").strip()
            if key and value:
                entries.append({"key": key, "value": value})
    return entries


def _coerce_items(raw) -> list:
    items = []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return items

    for item in raw:
        if isinstance(item, dict) and isinstance(item.get("fields"), list):
            raw_fields = item.get("fields") or []
        elif isinstance(item, dict):
            raw_fields = [{"key": key, "value": value} for key, value in item.items()]
        else:
            raw_fields = []

        normalized_fields = []
        for field in raw_fields:
            if not isinstance(field, dict):
                continue
            if "key" in field and "value" in field:
                key = str(field.get("key") or "").strip()
                value = str(field.get("value") or "").strip()
                if key and value:
                    normalized_fields.append({"key": key, "value": value})
                continue
            for key, value in field.items():
                key = str(key or "").strip()
                value = str(value or "").strip()
                if key and value:
                    normalized_fields.append({"key": key, "value": value})

        if normalized_fields:
            items.append({"fields": normalized_fields})
    return items


def _validate_chunk_result(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Chunk response is not a JSON object")

    normalized_payload = dict(payload)
    if not normalized_payload.get("document_type") and payload.get("doc_type"):
        normalized_payload["document_type"] = payload.get("doc_type")
    if normalized_payload.get("chunk_summary") is None and payload.get("summary") is not None:
        normalized_payload["chunk_summary"] = payload.get("summary")
    if normalized_payload.get("chunk_summary") is None:
        normalized_payload["chunk_summary"] = ""
    metadata_source = normalized_payload.get("metadata_entries")
    if metadata_source in (None, "", []):
        metadata_source = payload.get("metadata") or payload.get("high_level_metadata")
    items_source = normalized_payload.get("items")
    if items_source in (None, "", []):
        items_source = payload.get("line_items") or payload.get("rows")
    normalized_payload["metadata_entries"] = _coerce_metadata_entries(metadata_source)
    normalized_payload["items"] = _coerce_items(items_source)

    try:
        if hasattr(StructurerChunkResponse, "model_validate"):
            validated = StructurerChunkResponse.model_validate(normalized_payload)
        else:
            validated = StructurerChunkResponse.parse_obj(normalized_payload)
    except ValidationError as exc:
        raise ValueError(f"Chunk response validation failed: {exc}") from exc

    if hasattr(validated, "model_dump"):
        validated_payload = validated.model_dump()
    else:
        validated_payload = validated.dict()

    metadata = {}
    for entry in validated_payload.get("metadata_entries", []) or []:
        key = str((entry or {}).get("key") or "").strip()
        value = str((entry or {}).get("value") or "").strip()
        if key and value:
            metadata[key] = value

    normalized_items = []
    for item in validated_payload.get("items", []) or []:
        fields = (item or {}).get("fields") or []
        item_dict = {}
        for field in fields:
            key = str((field or {}).get("key") or "").strip()
            value = str((field or {}).get("value") or "").strip()
            if key and value:
                item_dict[key] = value
        if item_dict:
            normalized_items.append(item_dict)

    return {
        "document_type": validated_payload.get("document_type"),
        "metadata": metadata,
        "items": normalized_items,
        "chunk_summary": str(validated_payload.get("chunk_summary") or "").strip(),
    }


def _repair_chunk_payload_with_gemini(
    malformed_content: str,
    chunk_num: int,
    total_chunks: int,
) -> dict:
    logger.warning(
        f"Chunk {chunk_num}/{total_chunks} returned malformed JSON. Attempting repair pass."
    )
    repaired_content = gemini_generate_json_text(
        user_prompt=(
            f"Repair this malformed structurer response for chunk {chunk_num}/{total_chunks}.\n\n"
            f"Malformed response:\n{malformed_content}"
        ),
        system_instruction=REPAIR_SYSTEM_PROMPT,
        temperature=0.0,
        max_output_tokens=_llm_max_tokens,
        response_schema=STRUCTURER_CHUNK_RESPONSE_SCHEMA,
    )
    repaired_payload = safe_json_parse(repaired_content)
    return _validate_chunk_result(repaired_payload)


def _call_gemini_structurer(
    chunk_text: str,
    chunk_num: int,
    total_chunks: int,
    ocr_confidence: float,
    attempt: int = 1,
) -> dict:
    if not gemini_is_configured():
        raise RuntimeError("Gemini structurer is not configured.")

    with _structurer_semaphore:
        retry_note = ""
        if attempt > 1:
            retry_note = (
                "\n\nIMPORTANT RETRY NOTE:\n"
                "The previous attempt did not return valid JSON. "
                "Return exactly one valid JSON object matching the schema. "
                "Always include the 4 top-level keys document_type, metadata_entries, items, and chunk_summary. "
                "Use [] for empty arrays. "
                "Do not add markdown, prose, stray brackets, or stray characters."
            )
        logger.info(
            f"Structuring with Gemini model {gemini_model_name()} "
            f"(chunk {chunk_num}/{total_chunks}, attempt {attempt}/{_chunk_retry_count}, "
            f"concurrency limit {_max_concurrency})"
        )
        content = gemini_generate_json_text(
            user_prompt=(
                f"OCR Confidence: {ocr_confidence}%\n\n"
                f"Chunk {chunk_num}/{total_chunks}:\n{chunk_text}{retry_note}"
            ),
            system_instruction=SYSTEM_PROMPT,
            temperature=0.0,
            max_output_tokens=_llm_max_tokens,
            response_schema=STRUCTURER_CHUNK_RESPONSE_SCHEMA,
        )
    try:
        parsed = safe_json_parse(content)
        return _validate_chunk_result(parsed)
    except Exception as exc:
        logger.warning(
            f"Chunk {chunk_num}/{total_chunks} parse/validation failed on primary response: {str(exc)[:200]}"
        )
        return _repair_chunk_payload_with_gemini(content, chunk_num, total_chunks)


def _parse_structurer_response(content: str) -> dict:
    try:
        parsed = safe_json_parse(content)
        if isinstance(parsed, dict) and "value" in parsed and isinstance(parsed["value"], dict):
            parsed = parsed["value"]
        if not isinstance(parsed, dict):
            raise ValueError("Structurer response is not a JSON object")
        return parsed
    except Exception as exc:
        raw = str(content or "").strip()
        snippet = raw[:2000] if raw else ""
        if snippet:
            logger.warning(f"[Structurer] Raw response preview: {snippet}")

        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = safe_json_parse(raw[start:end + 1])
                if isinstance(parsed, dict) and "value" in parsed and isinstance(parsed["value"], dict):
                    parsed = parsed["value"]
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        raise ValueError(f"Cannot parse structurer JSON response: {str(exc)[:200]}")


def process_chunk(chunk_text: str, chunk_num: int, total_chunks: int, ocr_confidence: float) -> dict:
    logger.info(f"Chunk {chunk_num}/{total_chunks} ({len(chunk_text)} chars)...")

    last_error = None
    for attempt in range(1, _chunk_retry_count + 1):
        try:
            result = _call_gemini_structurer(
                chunk_text,
                chunk_num,
                total_chunks,
                ocr_confidence,
                attempt=attempt,
            )
            result["document_type"] = normalize_document_type_label(result.get("document_type", "unknown"))

            doc_type = result.get("document_type", "unknown")
            items_count = len(result.get("items", []))
            metadata_count = len(result.get("metadata", {}))
            logger.info(f"({doc_type}, {metadata_count} fields, {items_count} items)")
            return result
        except Exception as exc:
            last_error = exc
            if attempt < _chunk_retry_count:
                logger.warning(
                    f"Chunk {chunk_num}/{total_chunks} attempt {attempt} failed: {str(exc)[:200]}. Retrying..."
                )
                if _retry_backoff_seconds > 0:
                    time.sleep(_retry_backoff_seconds * attempt)
                continue
            logger.error(f"Failed after {_chunk_retry_count} attempt(s) - {str(exc)[:200]}")

    return {
        "document_type": "unknown",
        "metadata": {},
        "items": [],
        "chunk_summary": (
            f"Chunk {chunk_num} could not be structured: "
            f"{str(last_error)[:120] if last_error else 'unknown error'}"
        ),
    }


def merge_chunk_results(chunk_results: list, document_type: str = None) -> dict:
    merged_metadata = {}
    all_items = []
    seen_items = set()
    detected_type = document_type

    for chunk_result in chunk_results:
        if not detected_type and chunk_result.get("document_type"):
            detected_type = chunk_result["document_type"]

        chunk_metadata = chunk_result.get("metadata", {})
        for key, value in chunk_metadata.items():
            if key not in merged_metadata and value:
                merged_metadata[key] = value

        for item in chunk_result.get("items", []) or []:
            item_str = json.dumps(item, sort_keys=True)
            if item_str not in seen_items:
                all_items.append(item)
                seen_items.add(item_str)

    result = {
        "document_type": normalize_document_type_label(detected_type or "unknown"),
        "high_level_metadata": merged_metadata,
        "items": all_items if all_items else None,
    }
    return {key: value for key, value in result.items() if value is not None}


def extract_metadata_with_gemini(all_text: str, ocr_confidence: float) -> dict:
    logger.info(f"\nDocument: {len(all_text)} chars")
    logger.info(f"Splitting into chunks ({_chunk_size} chars each with overlap {_chunk_overlap})...")

    chunks = split_document_into_chunks(all_text, chunk_size=_chunk_size)
    logger.info(f"Created {len(chunks)} chunk(s)\n")

    logger.info("Processing chunks with Gemini...")
    chunk_results = []
    for index, chunk in enumerate(chunks, 1):
        chunk_results.append(process_chunk(chunk, index, len(chunks), ocr_confidence))

    logger.info(f"\nMerging results from {len(chunk_results)} chunk(s)...")
    merged = merge_chunk_results(chunk_results)

    final_output = {
        "document_type": merged.get("document_type", "unknown"),
        "high_level_metadata": merged.get("high_level_metadata", {}),
        "items": merged.get("items", []),
        "confidence": {
            "ocr_engine": "PaddleOCR",
            "confidence_percent": ocr_confidence,
        },
        "content": {
            "full_text": all_text,
        },
    }

    if merged.get("items"):
        final_output["high_level_metadata"]["items"] = merged["items"]

    logger.info("\nExtraction complete!")
    logger.info(f"  Document Type: {final_output['document_type']}")
    logger.info(f"  Metadata Fields: {len(final_output['high_level_metadata'])}")
    if final_output["high_level_metadata"].get("items"):
        logger.info(f"  Total Items: {len(final_output['high_level_metadata']['items'])}")

    return final_output


def extract_metadata_with_llama(all_text: str, ocr_confidence: float) -> dict:
    return extract_metadata_with_gemini(all_text, ocr_confidence)

