import json
import re
from typing import Any, Dict

try:
    import json5
    HAS_JSON5 = True
except ImportError:
    HAS_JSON5 = False

def _strip_code_fences(text: str) -> str:
    raw = str(text or "").strip()
    if "```json" in raw:
        return raw.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in raw:
        return raw.split("```", 1)[1].split("```", 1)[0].strip()
    return raw


def _extract_balanced_json_object(text: str) -> str:
    raw = _strip_code_fences(text)
    start = raw.find("{")
    if start == -1:
        raise ValueError("No JSON opening brace found in response")

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(raw)):
        char = raw[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return raw[start:idx + 1]

    return raw[start:]


def _remove_trailing_commas(json_str: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", json_str)


def _remove_unmatched_closing_tokens(json_str: str) -> str:
    result = []
    brace_depth = 0
    bracket_depth = 0
    in_string = False
    escaped = False

    for char in json_str:
        if in_string:
            result.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            result.append(char)
            continue

        if char == "{":
            brace_depth += 1
            result.append(char)
            continue
        if char == "[":
            bracket_depth += 1
            result.append(char)
            continue
        if char == "}":
            if brace_depth <= 0:
                continue
            brace_depth -= 1
            result.append(char)
            continue
        if char == "]":
            if bracket_depth <= 0:
                continue
            bracket_depth -= 1
            result.append(char)
            continue

        result.append(char)

    return "".join(result)


def _extract_json_candidate(text: str) -> str:
    return _extract_balanced_json_object(text)


def _fix_incomplete_json(json_str: str) -> str:
    fixed = json_str.rstrip()
    if fixed.count('"') % 2 == 1:
        fixed += '"'

    open_braces = fixed.count("{")
    close_braces = fixed.count("}")
    open_brackets = fixed.count("[")
    close_brackets = fixed.count("]")

    while close_brackets < open_brackets:
        fixed += "]"
        close_brackets += 1

    while close_braces < open_braces:
        fixed += "}"
        close_braces += 1

    return fixed


def _attempt_json_load(candidate: str) -> Dict[str, Any]:
    decoder = json.JSONDecoder()
    parsed, _ = decoder.raw_decode(candidate)
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def safe_json_parse(text: str) -> Dict[str, Any]:
    json_str = _extract_json_candidate(text)
    compact = re.sub(r"\s+", " ", json_str.replace("\n", " ").replace("\r", " ").replace("\t", " ")).strip()
    no_trailing_commas = _remove_trailing_commas(json_str)
    no_unmatched_closers = _remove_unmatched_closing_tokens(no_trailing_commas)
    fixed = _fix_incomplete_json(no_unmatched_closers)
    candidate_variants = []
    for candidate in (
        json_str,
        compact,
        no_trailing_commas,
        no_unmatched_closers,
        _remove_trailing_commas(compact),
        fixed,
    ):
        candidate = str(candidate or "").strip()
        if candidate and candidate not in candidate_variants:
            candidate_variants.append(candidate)
    last_error: Exception | None = None

    for candidate in candidate_variants:
        try:
            return _attempt_json_load(candidate)
        except Exception as exc:
            last_error = exc

    if HAS_JSON5:
        for candidate in candidate_variants:
            try:
                parsed = json5.loads(candidate)
                return parsed if isinstance(parsed, dict) else {"value": parsed}
            except Exception as exc:
                last_error = exc

    raise ValueError(f"Cannot parse JSON response: {last_error}")
