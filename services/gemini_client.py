import json
from typing import Any, Iterator, Optional

from config import settings
from utils.logger import get_logger

logger = get_logger("gemini_client")

_client = None
_types = None
_configured_api_key = None


def _ensure_client() -> None:
    global _client, _types, _configured_api_key

    gemini_api_key = settings.gemini_api_key.strip()
    if not gemini_api_key:
        _client = None
        _types = None
        _configured_api_key = None
        return

    if _client is not None and _types is not None and _configured_api_key == gemini_api_key:
        return

    try:
        from google import genai
        from google.genai import types

        _client = genai.Client(api_key=gemini_api_key)
        _types = types
        _configured_api_key = gemini_api_key
    except Exception as exc:
        logger.warning(f"Gemini client initialization failed: {exc}")
        _client = None
        _types = None
        _configured_api_key = None


def is_configured() -> bool:
    _ensure_client()
    return _client is not None and _types is not None


def model_name() -> str:
    return settings.gemini_model_name.strip() or "gemini-2.5-flash"


def _extract_text_from_response(response: Any) -> str:
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        if hasattr(parsed, "model_dump"):
            parsed = parsed.model_dump()
        elif hasattr(parsed, "dict"):
            parsed = parsed.dict()
        return json.dumps(parsed, ensure_ascii=False)

    text = getattr(response, "text", None)
    if text:
        return text

    candidates = getattr(response, "candidates", None) or []
    parts = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                parts.append(part_text)
    if parts:
        return "\n".join(parts)

    return ""


def generate_json_text(
    user_prompt: str,
    system_instruction: str,
    temperature: float = 0.1,
    max_output_tokens: int = 2048,
    model: Optional[str] = None,
    response_schema: Any = None,
) -> str:
    if not is_configured():
        raise RuntimeError("Gemini client is not configured. Set GEMINI_API_KEY.")

    config = _types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
        response_schema=response_schema,
    )
    response = _client.models.generate_content(
        model=(model or model_name()),
        contents=user_prompt,
        config=config,
    )
    text = _extract_text_from_response(response)
    if text:
        return text

    raise RuntimeError("Gemini returned an empty response.")


def generate_text_stream(
    user_prompt: str,
    system_instruction: str,
    temperature: float = 0.1,
    max_output_tokens: int = 2048,
    model: Optional[str] = None,
) -> Iterator[str]:
    if not is_configured():
        raise RuntimeError("Gemini client is not configured. Set GEMINI_API_KEY.")

    config = _types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    stream = _client.models.generate_content_stream(
        model=(model or model_name()),
        contents=user_prompt,
        config=config,
    )
    for chunk in stream:
        text = _extract_text_from_response(chunk)
        if text:
            yield text
