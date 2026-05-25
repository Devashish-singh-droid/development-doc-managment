import json
import re
import time
import threading
from collections import OrderedDict
from typing import Any, Dict, List

from config import settings
from services.gemini_client import generate_json_text, is_configured, model_name
from services.json_utils import safe_json_parse
from utils.logger import get_logger

logger = get_logger("qa_query_understander")

QUERY_UNDERSTANDING_PROMPT = """
You analyze user questions for a document Q&A and retrieval system.

Return EXACTLY ONE valid minified JSON object and nothing else.
Use double quotes for every key and string value.
Do not add markdown, code fences, comments, explanations, trailing commas, or duplicate closing braces.
If a field is unknown, use an empty string, false, or [] instead of prose.

Return ONLY valid JSON with this schema:
{
  "topic": "short normalized topic",
  "intent": "short verb phrase for what user wants",
  "subject": "main person/company/object being asked about or empty string",
  "attribute": "specific requested field or concept such as father name, qualification, email, PO number, customer PO number, medicine, summary, expiry date, amount, or empty string",
  "document_type": "best inferred document type or empty string",
  "entity_type": "person/company/patient/document/medicine/general or empty string",
  "vendor": "vendor/company filter if clearly requested else empty string",
  "patient": "patient filter if clearly requested else empty string",
  "document_number": "document or invoice number if present else empty string",
  "year": "4 digit year if explicitly requested else empty string",
  "scenario_question": false,
  "simple_lookup": true,
  "needs_suggestion": false,
  "search_terms": ["best retrieval terms in priority order"],
  "must_match_terms": ["critical terms that should appear in relevant documents"]
}

Rules:
1. Extract meaning from the full question, not just keywords.
2. Be strict about the main subject and attribute.
3. For direct attribute lookups like email, phone, father name, qualification, PAN, GST, amount, date, expiry date, document number, PO number, customer PO number, medicine, medicine list, or prescription list, set simple_lookup to true.
4. For summaries, explanations, procedures, meanings, and how-to questions, set simple_lookup to false.
5. search_terms should help retrieval, mixing exact phrases and key entities.
6. must_match_terms should include only truly essential terms, especially the subject/entity.
7. Do not invent values not implied by the question.
8. Set scenario_question=true for hypothetical, policy, exception-handling, "what if", "in case", "how should", or scenario-based questions.
9. For document references like PO-001, INV-123, or DOC-45, copy that exact value into document_number.
10. For requests like "details about", "tell me about", "summary of", or "compare", set intent accordingly and keep the JSON compact.
"""

_CACHE_TTL_SECONDS = max(1, settings.get_int("QA_QUERY_ANALYSIS_CACHE_TTL_SECONDS", 21600))
_CACHE_MAX_SIZE = max(1, settings.get_int("QA_QUERY_ANALYSIS_CACHE_MAX_SIZE", 512))
_analysis_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_analysis_cache_lock = threading.Lock()

QUERY_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {"type": "string"},
        "intent": {"type": "string"},
        "subject": {"type": "string"},
        "attribute": {"type": "string"},
        "document_type": {"type": "string"},
        "entity_type": {"type": "string"},
        "vendor": {"type": "string"},
        "patient": {"type": "string"},
        "document_number": {"type": "string"},
        "year": {"type": "string"},
        "scenario_question": {"type": "boolean"},
        "simple_lookup": {"type": "boolean"},
        "needs_suggestion": {"type": "boolean"},
        "search_terms": {"type": "array", "items": {"type": "string"}},
        "must_match_terms": {"type": "array", "items": {"type": "string"}},
    },
}


_SCENARIO_PATTERNS = [
    r"\bwhat if\b",
    r"\bin case\b",
    r"\bscenario\b",
    r"\bsuppose\b",
    r"\bassume\b",
    r"\bassuming\b",
    r"\bwhat happens\b",
    r"\bhow should\b",
    r"\bhow do we handle\b",
    r"\bhow to handle\b",
    r"\bif\s+[a-z0-9]",
]

_SCENARIO_NOISE_TERMS = {
    "what", "when", "then", "will", "would", "should", "could", "scenario",
    "suppose", "assume", "assuming", "happen", "happens", "handle", "handled",
    "process", "procedure", "steps", "there", "case", "cases", "situation",
    "situations", "where", "does", "after", "before", "during", "under",
    "need", "needs", "required", "require", "policy",
}


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _tokenize_terms(value: str) -> List[str]:
    seen = set()
    terms = []
    normalized = _normalize_text(value)
    for doc_ref in re.findall(r"\b[a-z]{2,}(?:[-/][a-z0-9]+)+\b", normalized, flags=re.IGNORECASE):
        if doc_ref in seen:
            continue
        seen.add(doc_ref)
        terms.append(doc_ref)
    for token in re.findall(r"[a-z0-9]+", normalized):
        if len(token) < 3 or token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return terms


def _clean_list(values: Any) -> List[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []

    seen = set()
    cleaned = []
    for value in values:
        text = _clean_text(value)
        if not text:
            continue
        key = _normalize_text(text)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
    return cleaned


def _is_scenario_question_text(question: str) -> bool:
    q = _normalize_text(question)
    if not q:
        return False
    return any(re.search(pattern, q, flags=re.IGNORECASE) for pattern in _SCENARIO_PATTERNS)


def _scenario_terms(question: str) -> List[str]:
    terms = []
    seen = set()
    for token in _tokenize_terms(question):
        normalized = _normalize_text(token)
        if normalized in _SCENARIO_NOISE_TERMS:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        terms.append(token)
    return terms


def _cache_get(cache_key: str) -> Dict[str, Any] | None:
    now = time.time()
    with _analysis_cache_lock:
        cached = _analysis_cache.get(cache_key)
        if not cached:
            return None
        if cached.get("_expires_at", 0) <= now:
            _analysis_cache.pop(cache_key, None)
            return None
        _analysis_cache.move_to_end(cache_key)
        result = dict(cached)
        result.pop("_expires_at", None)
        return result


def _cache_set(cache_key: str, value: Dict[str, Any]) -> None:
    expires_at = time.time() + _CACHE_TTL_SECONDS
    with _analysis_cache_lock:
        entry = dict(value)
        entry["_expires_at"] = expires_at
        _analysis_cache[cache_key] = entry
        _analysis_cache.move_to_end(cache_key)
        while len(_analysis_cache) > _CACHE_MAX_SIZE:
            _analysis_cache.popitem(last=False)


def _heuristic_analysis(question: str) -> Dict[str, Any]:
    q = _normalize_text(question)
    topic = _clean_text(question)
    scenario_question = _is_scenario_question_text(question)
    out = {
        "topic": topic,
        "intent": "",
        "subject": "",
        "attribute": "",
        "document_type": "",
        "entity_type": "",
        "vendor": "",
        "patient": "",
        "document_number": "",
        "year": "",
        "scenario_question": scenario_question,
        "simple_lookup": False,
        "needs_suggestion": True,
        "search_terms": [],
        "must_match_terms": [],
    }

    subject_match = re.search(
        r"\b(?:of|for|about|regarding)\s+([a-z][a-z0-9\s.\-]{2,80})$",
        q,
        flags=re.IGNORECASE,
    )
    patient_match = re.search(
        r"\bpatient\s+([a-z][a-z0-9\s.\-]{2,80})$",
        q,
        flags=re.IGNORECASE,
    )
    if patient_match:
        out["patient"] = _clean_text(patient_match.group(1))
        out["subject"] = out["patient"]
        out["entity_type"] = "patient"
    elif subject_match:
        out["subject"] = _clean_text(subject_match.group(1))

    attribute_patterns = [
        ("customer po number", r"\bcustomer\s+(?:po|p\.?\s*o\.?|purchase\s+order)\s*(?:number|no|#)?\b"),
        ("po number", r"\b(?:po|p\.?\s*o\.?|purchase\s+order)\s+(?:number|no|#)\b"),
        ("invoice number", r"\binvoice\s+(?:number|no|#)\b"),
        ("father name", r"\bfather(?:'s)?\s+name\b"),
        ("qualification", r"\bqualification\b"),
        ("email", r"\b(?:email|mail id|email id|e-mail)\b"),
        ("phone", r"\b(?:phone|mobile|contact number|phone number|mobile number)\b"),
        ("medicine", r"\b(?:medicine|medicines|tablet|tablets|drug|drugs|prescribed medicine|prescribed medicines|prescription|prescriptions)\b"),
        ("gst", r"\bgst(?:in)?\b"),
        ("pan", r"\bpan(?: number)?\b"),
        ("amount", r"\b(?:amount|value|price|cost)\b"),
        ("date", r"\bdate\b"),
        ("expiry date", r"\bexpiry\s+date\b|\bexpir(?:y|ary)\s+date\b"),
        ("document number", r"\b(?:invoice|bill|document|doc)\s+(?:number|no|#)\b"),
        ("summary", r"\bsummary\b"),
    ]
    for label, pattern in attribute_patterns:
        if re.search(pattern, q, flags=re.IGNORECASE):
            out["attribute"] = label
            break

    if scenario_question:
        out["intent"] = "scenario_reasoning"
        out["simple_lookup"] = False
        out["needs_suggestion"] = True
    elif re.search(r"\bhow\b|\bprocedure\b|\bsteps\b|\bprocess\b|\bexplain\b|\bdetails?\b|\bdetail\b|\binformation\b|\binfo\b|\bdescribe\b|\btell me about\b", q):
        out["intent"] = "explain"
        out["simple_lookup"] = False
    elif out["attribute"]:
        out["intent"] = f"find {out['attribute']}"
        out["simple_lookup"] = out["attribute"] not in {"summary"}

    if re.search(r"\binvoice\b", q):
        out["document_type"] = "invoice"
    elif re.search(r"\bbill\b", q):
        out["document_type"] = "bill"
    elif re.search(r"\bmedical report\b", q):
        out["document_type"] = "medical_report"
    elif re.search(r"\bbid\b|\btender\b|\brfq\b", q):
        out["document_type"] = "bid"

    if out["subject"] and re.search(r"\b(?:po|purchase order|invoice|vendor|customer)\b", q):
        out["entity_type"] = out["entity_type"] or "company"

    year_match = re.search(r"\b(?:19|20)\d{2}\b", q)
    if year_match:
        out["year"] = year_match.group(0)

    doc_match = re.search(r"\b(?:po|inv|rfq|bid|doc|invoice)[-/ ]?\d{1,6}\b", q, flags=re.IGNORECASE)
    if not doc_match:
        doc_match = re.search(r"\b[a-z0-9]+(?:[/-][a-z0-9]+)+\b", q, flags=re.IGNORECASE)
    if doc_match:
        out["document_number"] = _clean_text(doc_match.group(0)).replace(" ", "-")
        if not out["subject"]:
            out["subject"] = out["document_number"]
        if not out["entity_type"]:
            out["entity_type"] = "document"

    search_terms = []
    if out["subject"]:
        search_terms.append(out["subject"])
    if out["attribute"]:
        search_terms.append(out["attribute"])
    if out["document_number"]:
        search_terms.append(out["document_number"])
    if scenario_question:
        search_terms.extend(_scenario_terms(question))
    else:
        search_terms.extend(_tokenize_terms(question))
    if out["attribute"] == "customer po number":
        search_terms.extend(["customer po number", "customer po", "po number", "purchase order number"])
    elif out["attribute"] == "po number":
        search_terms.extend(["po number", "purchase order number"])
    elif out["attribute"] == "invoice number":
        search_terms.extend(["invoice number", "document number"])

    must_match_terms = []
    if out["subject"]:
        must_match_terms.extend(_tokenize_terms(out["subject"]))
    if out["document_number"]:
        must_match_terms.append(out["document_number"])

    out["search_terms"] = _clean_list(search_terms)
    out["must_match_terms"] = _clean_list(must_match_terms)
    out["needs_suggestion"] = not out["simple_lookup"]
    return out


def _should_use_heuristic_analysis(question: str, analysis: Dict[str, Any]) -> bool:
    normalized_question = _normalize_text(question)
    if not normalized_question:
        return True

    token_count = len(re.findall(r"[a-z0-9]+", normalized_question))
    has_doc_number = bool(str(analysis.get("document_number") or "").strip())
    has_subject = bool(str(analysis.get("subject") or "").strip())
    has_attribute = bool(str(analysis.get("attribute") or "").strip())
    procurement_markers = [
        "po", "purchase order", "po number", "customer po", "customer po number", "invoice", "inv", "amount", "details", "detail",
        "summary", "compare", "comparison", "tell me about", "about", "list",
    ]
    if has_doc_number:
        return True
    if token_count <= 4 and any(marker in normalized_question for marker in procurement_markers):
        return True
    if token_count <= 6 and has_subject and has_attribute:
        return True
    return False


def _normalize_analysis(question: str, payload: Dict[str, Any] | None) -> Dict[str, Any]:
    fallback = _heuristic_analysis(question)
    payload = payload if isinstance(payload, dict) else {}

    normalized = {
        "topic": _clean_text(payload.get("topic") or fallback["topic"]),
        "intent": _clean_text(payload.get("intent") or fallback["intent"]),
        "subject": _clean_text(payload.get("subject") or fallback["subject"]),
        "attribute": _clean_text(payload.get("attribute") or fallback["attribute"]),
        "document_type": _clean_text(payload.get("document_type") or fallback["document_type"]),
        "entity_type": _clean_text(payload.get("entity_type") or fallback["entity_type"]),
        "vendor": _clean_text(payload.get("vendor") or fallback["vendor"]),
        "patient": _clean_text(payload.get("patient") or fallback["patient"]),
        "document_number": _clean_text(payload.get("document_number") or fallback["document_number"]),
        "year": _clean_text(payload.get("year") or fallback["year"]),
        "scenario_question": _coerce_bool(payload.get("scenario_question")) if "scenario_question" in payload else fallback["scenario_question"],
        "simple_lookup": _coerce_bool(payload.get("simple_lookup")) if "simple_lookup" in payload else fallback["simple_lookup"],
        "needs_suggestion": _coerce_bool(payload.get("needs_suggestion")) if "needs_suggestion" in payload else fallback["needs_suggestion"],
        "search_terms": _clean_list(payload.get("search_terms")) or fallback["search_terms"],
        "must_match_terms": _clean_list(payload.get("must_match_terms")) or fallback["must_match_terms"],
    }

    if normalized["scenario_question"]:
        normalized["simple_lookup"] = False
        normalized["needs_suggestion"] = True

    if normalized["patient"] and not normalized["subject"]:
        normalized["subject"] = normalized["patient"]
    if normalized["subject"] and not normalized["must_match_terms"]:
        normalized["must_match_terms"] = _tokenize_terms(normalized["subject"])
    if normalized["subject"] and normalized["subject"] not in normalized["search_terms"]:
        normalized["search_terms"] = [normalized["subject"]] + normalized["search_terms"]
    if normalized["attribute"] and normalized["attribute"] not in normalized["search_terms"]:
        normalized["search_terms"].append(normalized["attribute"])
    if normalized["document_number"] and normalized["document_number"] not in normalized["search_terms"]:
        normalized["search_terms"].insert(0, normalized["document_number"])
    if normalized["year"] and normalized["year"] not in normalized["search_terms"]:
        normalized["search_terms"].append(normalized["year"])

    if normalized["scenario_question"]:
        filtered_terms = []
        seen = set()
        for term in normalized["search_terms"]:
            key = _normalize_text(term)
            if key in _SCENARIO_NOISE_TERMS:
                continue
            if key in seen:
                continue
            seen.add(key)
            filtered_terms.append(term)
        normalized["search_terms"] = filtered_terms
        if not any([
            normalized["subject"],
            normalized["patient"],
            normalized["vendor"],
            normalized["document_number"],
        ]):
            normalized["must_match_terms"] = []

    normalized["search_terms"] = _clean_list(normalized["search_terms"])
    normalized["must_match_terms"] = _clean_list(normalized["must_match_terms"])
    return normalized


def analyze_query(question: str) -> Dict[str, Any]:
    normalized_question = _normalize_text(question)
    if not normalized_question:
        return _normalize_analysis(question, {})

    cached = _cache_get(normalized_question)
    if cached is not None:
        return cached

    heuristic = _normalize_analysis(question, {})
    if _should_use_heuristic_analysis(question, heuristic):
        _cache_set(normalized_question, heuristic)
        return heuristic

    if not is_configured():
        _cache_set(normalized_question, heuristic)
        return heuristic

    user_prompt = (
        f"Question: {question}\n\n"
        "Return only the JSON object."
    )
    
    # Retry logic with max 3 attempts
    max_retries = 3
    last_error = None
    
    for attempt in range(max_retries):
        try:
            if attempt == 0:
                logger.info(f"[QA] Query analysis using Gemini model: {model_name()}")
            else:
                logger.info(f"[QA] Query analysis retry {attempt}/{max_retries-1} using Gemini model: {model_name()}")
            
            content = generate_json_text(
                user_prompt=user_prompt,
                system_instruction=QUERY_UNDERSTANDING_PROMPT,
                temperature=0.0,
                max_output_tokens=320,
                response_schema=QUERY_ANALYSIS_SCHEMA,
            )
            parsed = safe_json_parse(content)
            analysis = _normalize_analysis(question, parsed)
            
            if attempt > 0:
                logger.info(f"[QA] Query analysis succeeded on retry {attempt}")
            
            _cache_set(normalized_question, analysis)
            return analysis
            
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                logger.warning(f"[QA] Query analysis attempt {attempt+1} failed: {str(e)[:150]}, retrying...")
                time.sleep(0.5 * (attempt + 1))  # Exponential backoff: 0.5s, 1s, 1.5s
            else:
                logger.warning(f"[QA] Query analysis failed after {max_retries} attempts: {str(e)[:200]}")
    
    # All retries failed, use fallback
    _cache_set(normalized_question, heuristic)
    return heuristic

