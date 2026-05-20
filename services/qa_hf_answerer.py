import json
import re
import time
import hashlib
import threading
from collections import OrderedDict
from difflib import SequenceMatcher
from typing import Iterator, List, Dict, Any, Tuple

from config import settings
from services.gemini_client import generate_json_text, generate_text_stream, is_configured, model_name
from services.json_utils import safe_json_parse
from services.qa_query_understander import analyze_query
from utils.logger import get_logger

logger = get_logger("qa_hf_answerer")

try:
    from langchain_core.documents import Document
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI
    _LANGCHAIN_AVAILABLE = True
except Exception as exc:
    Document = None
    JsonOutputParser = None
    PromptTemplate = None
    ChatPromptTemplate = None
    ChatGoogleGenerativeAI = None
    _LANGCHAIN_AVAILABLE = False
    _LANGCHAIN_IMPORT_ERROR = str(exc)

SYSTEM_PROMPT = """
You are a document Q&A assistant.

Rules:
1. Answer in the most appropriate length for the question.
2. Use ONLY facts from provided documents for the answer.
3. If not found, say "I wasn't able to find matching information in my knowledge sources for that request."
4. When the answer is found, include one genuinely useful suggestion that adds value beyond the direct answer.
5. The suggestion should be short, specific, and insightful, not generic.
6. The suggestion may use limited general knowledge to explain the answer, but must not contradict the documents.
7. For medical, legal, or financial content, keep the suggestion informational and cautious, not prescriptive.
8. Return valid JSON with answer, should_suggest, suggestion, topic, citations, and structured_answer.
9. Set should_suggest=true whenever you can provide a useful, concrete suggestion grounded in the answer or source context.
10. No guessing or unsupported claims in the answer.
11. The answer field must be complete, self-contained, and end cleanly. Never stop mid-sentence, mid-list, or mid-thought.
12. If the answer could be long, compress it into a shorter complete summary rather than truncating it.
13. If the question asks for multiple matching records, include all relevant matches visible in the provided sources, or clearly say how many are shown.
14. Output exactly one complete JSON object. Do not leave any field unfinished.
15. If conversation context is provided, use it to resolve references like "his", "her", "that resume", "same person", or "those details".
16. For resume, profile, or candidate-evaluation questions, infer a reasoned answer from the documented skills, experience, education, projects, and achievements when possible.
17. For situational questions such as hiring fit, suitability, or recommendation, do not say "Not found in documents" if the sources provide enough evidence to make a cautious evidence-based judgment.
18. If the evidence is partial, answer carefully using phrases like "Based on the available documents..." and explain the main reasons briefly.
19. Make the response user-friendly and easy to scan.
20. Adapt the structure to the question:
   - Use a direct short summary for simple factual questions.
   - Use bullets for lists or multi-point answers.
   - Use ordered step-style bullets for process or how-to questions.
   - Use clearly separated sections for comparisons or evaluations.
21. structured_answer.summary should be a concise plain-language summary.
22. structured_answer.highlights should contain short, readable bullet points only when they add value.
23. structured_answer.sections should contain clear headings with body text and/or bullets when the answer benefits from structure.
24. Keep structured_answer.closing short and helpful, or leave it empty.
25. When the answer contains category-wise numeric data, include graph_data with labels and values that match the answer exactly.
26. Include chart_options as a list from: ["bar", "pie", "line", "doughnut"] when graph_data is present, otherwise return an empty list.
27. If the user asks for a graph or chart, do not say you cannot plot, render, or generate visuals. Just provide the comparison data and let the system render the chart.
"""

STREAM_SYSTEM_PROMPT = """
You are a document Q&A assistant.

Rules:
1. Answer in plain text only, using ONLY facts from the provided documents.
2. If the answer is not supported by the documents, say "I wasn't able to find matching information in my knowledge sources for that request."
3. If conversation context is provided, use it to resolve references like "his", "her", "that resume", or "same person".
4. For resume, profile, or candidate-evaluation questions, make a cautious evidence-based judgment from the documented skills, experience, education, projects, and achievements when possible.
5. For situational questions such as hiring fit or suitability, do not say "Not found in documents" if the sources provide enough evidence for a careful conclusion.
6. If the evidence is partial, clearly say "Based on the available documents..." and explain briefly.
7. Make the answer user-friendly and easy to scan.
8. When the answer benefits from structure, stream it in this plain-text layout:
   - optional short title on its own line
   - one short summary paragraph
   - bullet points prefixed with "- "
   - section headings on their own line ending with ":" followed by bullets or short paragraphs
9. Keep line breaks meaningful so the UI can show headings, bullets, and sections while the text is still streaming.
10. End with a complete final sentence. Never stop mid-sentence, mid-list, or mid-thought.
11. If the answer could be long, compress it into a shorter complete answer rather than truncating it.
12. Do not return JSON, XML, markdown code fences, or artificial labels like "Answer:".
13. If the user asks for a graph or chart, never mention limitations like "I cannot plot" or "I cannot create charts". Just provide the comparison values or summary directly.
"""

SUGGESTION_SYSTEM_PROMPT = """
You generate one high-value follow-up insight for a document Q&A system.

Rules:
1. Read the question, answer, and source context carefully.
2. Produce multiple candidate suggestions and make them genuinely useful, grounded, and specific.
3. Think of angles such as meaning, implication, significance, likely use, what to verify next, notable pattern, anomaly, or the most relevant adjacent detail.
4. Prefer direct insight over vague advice. Do not frame it as "you may also want to know".
5. Do not repeat the answer in different words.
6. Do not give a generic suggestion.
7. You may use limited general knowledge when it helps explain the answer, but do not contradict the source context.
8. For medical, legal, or financial content, keep the suggestion informational and cautious, not prescriptive.
9. Usually set should_suggest=true when the answer is available and you can add a concrete, relevant next insight.
10. Return valid JSON only:
{{"should_suggest": true/false, "candidates": [{{"suggestion": "...", "angle": "...", "helpfulness": 1-10}}]}}
11. Only set should_suggest=false when the answer is missing or there is no reliable extra value to add.
"""

_CACHE_TTL_SECONDS = max(1, settings.qa_cache_ttl_seconds)  # 3 hours
_CACHE_MAX_SIZE = max(1, settings.qa_cache_max_size)
_QA_ANSWER_MAX_TOKENS = max(256, settings.get_int("QA_ANSWER_MAX_TOKENS", 2200))
_answer_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_answer_cache_lock = threading.Lock()

_LC_PROMPT_CACHE_TTL_SECONDS = max(1, settings.langchain_prompt_cache_ttl_seconds)
_LC_PROMPT_CACHE_MAX_SIZE = max(1, settings.langchain_prompt_cache_max_size)
_lc_prompt_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

_SOFT_NOT_FOUND_ANSWER = "I wasn't able to find matching information in my knowledge sources for that request."

ANSWER_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "should_suggest": {"type": "boolean"},
        "suggestion": {"type": "string"},
        "topic": {"type": "string"},
        "structured_answer": {
            "type": "object",
            "properties": {
                "style": {"type": "string"},
                "title": {"type": "string"},
                "summary": {"type": "string"},
                "highlights": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "heading": {"type": "string"},
                            "body": {"type": "string"},
                            "bullets": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    },
                },
                "closing": {"type": "string"},
            },
        },
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string"},
                    "snippet": {"type": "string"},
                },
            },
        },
        "graph_data": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "x_label": {"type": "string"},
                "y_label": {"type": "string"},
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "values": {
                    "type": "array",
                    "items": {"type": "number"},
                },
            },
        },
        "chart_options": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}

SUGGESTION_CANDIDATES_SCHEMA = {
    "type": "object",
    "properties": {
        "should_suggest": {"type": "boolean"},
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "suggestion": {"type": "string"},
                    "angle": {"type": "string"},
                    "helpfulness": {"type": "integer"},
                },
            },
        },
    },
}

DOC_PROMPT = (
    "SOURCE {index}: {document_type} (ID: {doc_id})\n"
    "Metadata:\n{metadata}\n"
    "Items: {items}\n"
    "Content:\n{content}\n"
    "---"
)

def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def is_langchain_available() -> bool:
    return _LANGCHAIN_AVAILABLE


def _langchain_enabled() -> bool:
    return settings.enable_langchain


def _build_lcel_chain(system_prompt: str, temperature: float, max_output_tokens: int):
    if not _LANGCHAIN_AVAILABLE:
        raise RuntimeError(f"LangChain is not available: {_LANGCHAIN_IMPORT_ERROR}")
    if not is_configured():
        raise RuntimeError("Gemini client is not configured. Set GEMINI_API_KEY.")

    model = ChatGoogleGenerativeAI(
        model=model_name(),
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        google_api_key=settings.gemini_api_key,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{user_prompt}"),
        ]
    )
    return prompt | model | JsonOutputParser()


def _invoke_lcel_json(system_prompt: str, user_prompt: str, temperature: float, max_output_tokens: int) -> Dict[str, Any]:
    chain = _build_lcel_chain(system_prompt, temperature, max_output_tokens)
    result = chain.invoke({"user_prompt": user_prompt})
    if isinstance(result, dict):
        return result
    return safe_json_parse(str(result or ""))

def _lc_tokenize(value: str) -> List[str]:
    normalized = _normalize_text(value)
    terms: List[str] = []
    seen = set()
    for doc_ref in re.findall(r"\b[a-z]{2,}(?:[-/][a-z0-9]+)+\b", normalized, flags=re.IGNORECASE):
        for variant in (doc_ref, re.sub(r"[-/]+", "", doc_ref)):
            if len(variant) < 3 or variant in seen:
                continue
            seen.add(variant)
            terms.append(variant)
    for token in re.findall(r"[a-z0-9]{3,}", normalized):
        if token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return terms


def _lc_source_text_for_scoring(source: Dict[str, Any]) -> str:
    parts = []
    metadata = source.get("metadata")
    snippet = source.get("snippet")
    full_text = source.get("full_text")
    items = source.get("items")

    if metadata:
        parts.append(str(metadata))
    if snippet:
        parts.append(str(snippet))
    if items and isinstance(items, list):
        parts.append(" ".join([str(it) for it in items[:4]]))
    if full_text:
        parts.append(str(full_text))
    return " ".join(parts)


def _lc_score_source(question_tokens: List[str], source: Dict[str, Any]) -> float:
    if not question_tokens:
        return 0.0

    source_text = _normalize_text(_lc_source_text_for_scoring(source))
    if not source_text:
        return 0.0

    score = 0.0
    for token in question_tokens:
        if token and token in source_text:
            score += 1.0
    return score


def _lc_select_sources(question: str, sources: List[Dict[str, Any]], max_sources: int) -> List[Dict[str, Any]]:
    if not sources:
        return []
    safe_max = max(1, min(int(max_sources or 1), len(sources)))
    tokens = _lc_tokenize(question)
    if not tokens:
        return sources[:safe_max]

    scored = []
    for idx, src in enumerate(sources):
        scored.append((
            _lc_score_source(tokens, src),
            idx,
            src,
        ))

    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    selected = [item[2] for item in scored[:safe_max]]
    return selected if selected else sources[:safe_max]


def _lc_cache_key(question: str, sources: List[Dict[str, Any]], max_sources: int, max_chars: int) -> str:
    fingerprint = {
        "q": _normalize_text(question),
        "max_sources": int(max_sources or 0),
        "max_chars": int(max_chars or 0),
        "sources": [],
    }
    for src in sources:
        fingerprint["sources"].append({
            "doc_id": src.get("doc_id", ""),
            "snippet": str(src.get("snippet", ""))[:200],
            "meta": str(src.get("metadata", ""))[:120],
        })
    encoded = repr(fingerprint).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _lc_cache_get(key: str) -> Dict[str, Any] | None:
    now = time.time()
    item = _lc_prompt_cache.get(key)
    if not item:
        return None
    if item.get("expires_at", 0) <= now:
        _lc_prompt_cache.pop(key, None)
        return None
    _lc_prompt_cache.move_to_end(key)
    return item


def _lc_cache_set(key: str, value: Dict[str, Any]) -> None:
    expires_at = time.time() + _LC_PROMPT_CACHE_TTL_SECONDS
    payload = dict(value)
    payload["expires_at"] = expires_at
    _lc_prompt_cache[key] = payload
    _lc_prompt_cache.move_to_end(key)
    while len(_lc_prompt_cache) > _LC_PROMPT_CACHE_MAX_SIZE:
        _lc_prompt_cache.popitem(last=False)


def _lc_build_documents(selected: List[Dict[str, Any]], max_chars: int) -> List[Document]:
    documents = []
    per_doc_chars = max(600, int(max_chars / max(1, len(selected)))) if max_chars else 1600
    for src in selected:
        doc_id = src.get("doc_id", "")
        doc_type = src.get("document_type", "unknown")
        metadata = src.get("metadata", "")
        snippet = src.get("snippet", "")
        items = src.get("items", [])
        full_text = src.get("full_text", "")

        content_parts = []
        if snippet:
            content_parts.append(str(snippet))
        if full_text:
            content_parts.append(str(full_text))
        content = "\n".join(part for part in content_parts if part)
        if per_doc_chars and len(content) > per_doc_chars:
            content = content[:per_doc_chars].rstrip() + "..."

        meta_items = ""
        if items and isinstance(items, list):
            meta_items = " | ".join([str(item) for item in items[:4]])

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "doc_id": doc_id,
                    "document_type": doc_type,
                    "metadata": str(metadata)[:2000] if metadata else "",
                    "items": meta_items,
                },
            )
        )
    return documents


def _lc_build_sources_block(
    question: str,
    sources: List[Dict[str, Any]],
    max_sources: int = 4,
    max_chars: int = 6000,
) -> Tuple[str, List[Dict[str, Any]]]:
    if not _LANGCHAIN_AVAILABLE:
        raise RuntimeError(f"LangChain is not available: {_LANGCHAIN_IMPORT_ERROR}")

    safe_sources = sources if isinstance(sources, list) else []
    if not safe_sources:
        return "", []

    safe_max_sources = max(1, min(int(max_sources or 1), len(safe_sources)))
    key = _lc_cache_key(question, safe_sources, safe_max_sources, max_chars)
    cached = _lc_cache_get(key)
    if cached is not None:
        return cached.get("block", ""), cached.get("selected_sources", [])

    selected = _lc_select_sources(question, safe_sources, safe_max_sources)
    docs = _lc_build_documents(selected, max_chars)

    template = PromptTemplate.from_template(DOC_PROMPT)
    lines = []
    for idx, doc in enumerate(docs, start=1):
        lines.append(
            template.format(
                index=idx,
                document_type=doc.metadata.get("document_type", "unknown"),
                doc_id=doc.metadata.get("doc_id", ""),
                metadata=doc.metadata.get("metadata", ""),
                items=doc.metadata.get("items", ""),
                content=doc.page_content,
            )
        )

    block = "\n".join(lines)
    _lc_cache_set(key, {"block": block, "selected_sources": selected})
    logger.info(f"[LangChain] Built prompt block using {len(selected)} source(s)")
    return block, selected


def _is_cacheable_answer(answer: str) -> bool:
    answer_norm = _normalize_text(answer)
    if not answer_norm:
        return False
    blocked_phrases = [
        "not found in documents",
        "wasn't able to find matching information in my knowledge sources",
        "based on documents found. see sources for details.",
        "llm not configured. showing related documents.",
    ]
    return not any(phrase in answer_norm for phrase in blocked_phrases)


def _is_no_data_answer(answer: str) -> bool:
    normalized = _normalize_text(answer)
    if not normalized:
        return True
    markers = [
        "not found in documents",
        "no relevant documents found",
        "wasn't able to find matching information in my knowledge sources",
        "was not able to find matching information in my knowledge sources",
    ]
    return any(marker in normalized for marker in markers)


def _preferred_chart_options_for_question(question: str, has_graph_data: bool) -> List[str]:
    if not has_graph_data:
        return []
    normalized = _normalize_text(question)
    preferred = []
    if any(token in normalized for token in ["doughnut", "donut", "donut chart"]):
        preferred.append("doughnut")
    elif any(token in normalized for token in ["pie chart", "pie graph", "pie"]):
        preferred.append("pie")
    elif any(token in normalized for token in ["line chart", "line chat", "line graph", "line plot", "trend line", "timeline", "over time", "trend"]):
        preferred.append("line")
    elif any(token in normalized for token in ["bar chart", "bar graph", "bar plot", "column chart", "column graph", "histogram"]):
        preferred.append("bar")

    defaults = ["bar", "line", "pie", "doughnut"]
    ordered = []
    seen = set()
    for option in preferred + defaults:
        if option in seen:
            continue
        ordered.append(option)
        seen.add(option)
    return ordered


def _sanitize_answer_text(question: str, answer: str) -> str:
    text = str(answer or "").replace("\r\n", "\n").strip()
    if not text:
        return _SOFT_NOT_FOUND_ANSWER

    if _normalize_text(question).find("graph") != -1 or _normalize_text(question).find("chart") != -1 or _normalize_text(question).find("plot") != -1:
        filtered_lines = []
        for raw_line in text.splitlines():
            clean_line = raw_line.strip()
            normalized_line = _normalize_text(clean_line)
            if not clean_line:
                if filtered_lines and filtered_lines[-1]:
                    filtered_lines.append("")
                continue
            if (
                ("cannot" in normalized_line or "can't" in normalized_line or "unable" in normalized_line)
                and any(token in normalized_line for token in ["plot", "graph", "chart", "visual"])
            ):
                continue
            if "generate visual chart" in normalized_line or "generate visual graph" in normalized_line:
                continue
            clean_line = re.sub(r"^\s*however,\s*", "", clean_line, flags=re.IGNORECASE)
            if re.match(r"^i can provide\b", clean_line, flags=re.IGNORECASE):
                clean_line = re.sub(
                    r"^i can provide\b[^.!\n]*",
                    "Here is the comparison data from the available documents.",
                    clean_line,
                    flags=re.IGNORECASE,
                ).strip()
            if clean_line:
                filtered_lines.append(clean_line)
        text = "\n".join(filtered_lines).strip()

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text or _SOFT_NOT_FOUND_ANSWER


def _query_analyses_match(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False

    left_subject = _normalize_text(left.get("subject"))
    right_subject = _normalize_text(right.get("subject"))
    if left_subject or right_subject:
        if not left_subject or not right_subject or left_subject != right_subject:
            return False

    left_attribute = _normalize_text(left.get("attribute"))
    right_attribute = _normalize_text(right.get("attribute"))
    if left_attribute or right_attribute:
        if not left_attribute or not right_attribute or left_attribute != right_attribute:
            return False

    left_doc_type = _normalize_text(left.get("document_type"))
    right_doc_type = _normalize_text(right.get("document_type"))
    if left_doc_type or right_doc_type:
        if not left_doc_type or not right_doc_type or left_doc_type != right_doc_type:
            return False

    left_doc_number = _normalize_text(left.get("document_number"))
    right_doc_number = _normalize_text(right.get("document_number"))
    if left_doc_number or right_doc_number:
        if not left_doc_number or not right_doc_number or left_doc_number != right_doc_number:
            return False

    left_intent = _normalize_text(left.get("intent"))
    right_intent = _normalize_text(right.get("intent"))
    if left_intent or right_intent:
        if not left_intent or not right_intent:
            return False
        if left_intent != right_intent and SequenceMatcher(None, left_intent, right_intent).ratio() < 0.82:
            return False

    return True


def _build_cache_key(question: str, sources: List[Dict[str, Any]], conversation_context: str = "") -> str:
    """Build a stable fingerprint for repeated QA requests on the same source context."""
    key_payload = {
        "question": _normalize_text(question),
        "conversation_context": _normalize_text(conversation_context)[:500],
        "sources": [],
    }

    for src in sources:
        source_fingerprint = {
            "doc_id": src.get("doc_id", ""),
            "document_type": src.get("document_type", ""),
            "metadata": str(src.get("metadata", ""))[:250],
            "snippet": str(src.get("snippet", ""))[:500],
            "full_text": str(src.get("full_text", ""))[:900],
            "items_count": len(src.get("items", [])) if isinstance(src.get("items"), list) else 0,
        }
        key_payload["sources"].append(source_fingerprint)

    encoded = json.dumps(key_payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_get(cache_key: str) -> Dict[str, Any] | None:
    now = time.time()
    with _answer_cache_lock:
        cached = _answer_cache.get(cache_key)
        if not cached:
            return None

        # Check if cached entry has expired
        if cached.get("_expires_at", 0) <= now:
            _answer_cache.pop(cache_key, None)
            return None

        # Move to end (most recently used)
        _answer_cache.move_to_end(cache_key)
        
        # Return copy of cached data without timestamp
        result = dict(cached)
        result.pop("_expires_at", None)
        return result


def _cache_set(cache_key: str, value: Dict[str, Any]) -> None:
    expires_at = time.time() + _CACHE_TTL_SECONDS
    with _answer_cache_lock:
        # Store value with expiration timestamp
        cache_entry = dict(value)
        cache_entry["_expires_at"] = expires_at
        _answer_cache[cache_key] = cache_entry
        _answer_cache.move_to_end(cache_key)

        # Remove oldest entries if cache is full
        while len(_answer_cache) > _CACHE_MAX_SIZE:
            _answer_cache.popitem(last=False)


def clear_cache() -> None:
    """Clear all cached Q&A answers. Called on user logout."""
    with _answer_cache_lock:
        _answer_cache.clear()
        logger.info("Q&A cache cleared on logout")


def _search_cache_by_topic(question: str) -> Dict[str, Any] | None:
    """Search cache by extracting question topic and matching cached topics semantically."""
    if not question:
        return None
    
    with _answer_cache_lock:
        if not _answer_cache:
            return None

        question_analysis = analyze_query(question)
        
        # Try word matching
        for entry in _answer_cache.values():
            topic = entry.get("topic", "")
            if not topic:
                continue
            if not _is_cacheable_answer(str(entry.get("answer") or "")):
                continue

            entry_analysis = entry.get("query_analysis")
            if not isinstance(entry_analysis, dict):
                entry_analysis = analyze_query(topic)

            if _query_analyses_match(question_analysis, entry_analysis):
                result = dict(entry)
                result.pop("_expires_at", None)
                logger.info(f"[Cache] Found semantic match: topic='{topic}', answer='{entry.get('answer', '')[:50]}'")
                return result
        
        return None

def _build_sources_block(sources: List[Dict[str, Any]]) -> str:
    """Build optimized source block with full text context for LLM (token-efficient)"""
    lines = []
    
    for idx, src in enumerate(sources, start=1):
        doc_id = src.get("doc_id", "")
        doc_type = src.get("document_type", "unknown")
        metadata = src.get("metadata", "")
        snippet = src.get("snippet", "")
        full_text = src.get("full_text", "")
        items = src.get("items", [])
        
        # Compact header (reduced verbosity)
        lines.append(f"SOURCE {idx}: {doc_type} (ID: {doc_id[:16]})")
        
        # Metadata - Send FULL metadata so LLM can see all fields including CMD, CEO, etc.
        if metadata:
            # Send up to 1000 chars of metadata (was 140, now much larger)
            lines.append(f"Metadata:\n{metadata[:10000]}")
        
        # Items summary (if present)
        if items and isinstance(items, list):
            lines.append(f"Items: {len(items)} total")
            for i, item in enumerate(items[:4], 1):
                if isinstance(item, dict):
                    item_str = " | ".join([f"{k}: {v}" for k, v in list(item.items())[:4]])
                    lines.append(f"  {i}. {item_str}")
        
        # Snippet or full text
        if snippet:
            lines.append(f"Content: {snippet[:400]}")
            # IMPORTANT: For Q&A, also send full text so LLM can search entire document
            if full_text:
                truncated = full_text[:1400]
                lines.append(f"Full Text: {truncated}")
        elif full_text:
            truncated = full_text[:1400]
            lines.append(f"Text: {truncated}")
        
        lines.append("---")
    
    return "\n".join(lines)


def _build_sources_block_with_langchain(
    question: str,
    sources: List[Dict[str, Any]],
    max_sources: int | None = None,
    max_chars: int | None = None,
) -> tuple[str, List[Dict[str, Any]]]:
    """Use LangChain for prompt assembly + chunk selection if enabled."""
    safe_sources = sources if isinstance(sources, list) else []
    if not safe_sources:
        return "", []

    use_langchain = settings.enable_langchain
    if use_langchain:
        try:
            block, selected = _lc_build_sources_block(
                question=question,
                sources=safe_sources,
                max_sources=max_sources or 4,
                max_chars=max_chars or 6000,
            )
            if block and selected:
                return block, selected
        except Exception as exc:
            logger.warning(f"[LangChain] Prompt build failed, falling back: {exc}")

    fallback = safe_sources[: max_sources or len(safe_sources)]
    return _build_sources_block(fallback), fallback


def _fallback_citations(sources: List[Dict[str, Any]], limit: int = 3) -> List[Dict[str, Any]]:
    fallback = []
    for src in sources[:limit]:
        doc_id = src.get("doc_id")
        snippet = src.get("snippet") or ""
        if doc_id:
            fallback.append({"doc_id": doc_id, "snippet": snippet})
    return fallback


def _clean_suggestion(value: str) -> str:
    suggestion = str(value or "").strip()
    suggestion = re.sub(r"^\s*(suggestion|helpful insight)\s*:\s*", "", suggestion, flags=re.IGNORECASE)
    suggestion = re.sub(r"\s+", " ", suggestion).strip()
    if _is_incomplete_suggestion(suggestion):
        return ""
    return suggestion


def _is_incomplete_suggestion(value: str) -> bool:
    suggestion = str(value or "").strip()
    if not suggestion:
        return False

    lowered = _normalize_text(suggestion)
    words = re.findall(r"[a-z']+", lowered)
    if not words:
        return True

    hanging_words = {
        "the", "a", "an", "and", "or", "but", "if", "to", "of", "for", "with",
        "by", "on", "in", "at", "from", "that", "this", "these", "those",
        "because", "which", "than", "then",
    }
    weak_prefixes = (
        "note that",
        "also",
        "because",
        "this means",
        "which means",
        "keep in mind",
    )

    if words[-1] in hanging_words:
        return True
    if len(words) <= 4 and any(lowered.startswith(prefix) for prefix in weak_prefixes):
        return True
    return False


def _fallback_suggestion_from_context(question: str, answer: str) -> str:
    question_norm = _normalize_text(question)
    answer_norm = _normalize_text(answer)
    if not answer_norm or _is_no_data_answer(answer):
        return ""

    procurement_markers = [
        "purchase order", "po-", "invoice", "amount", "payment terms", "delivery", "vendor",
    ]
    resume_markers = [
        "resume", "candidate", "profile", "skills", "experience", "job", "role fit",
    ]
    if any(marker in question_norm for marker in procurement_markers):
        return "You can also compare these records by amount, date, and payment terms for a faster review."
    if any(marker in question_norm for marker in resume_markers):
        return "You can next ask for a role-fit and skill-gap summary based on this profile."
    return "You can ask for a short point-wise comparison or a chart-focused summary as the next step."


def _clean_text_item(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^\s*(?:[-*•]+|\d+[.)])\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_procurement_question_text(question: str) -> str:
    normalized = _normalize_text(question)
    return re.sub(r"\bpo['’]s\b", "po", normalized, flags=re.IGNORECASE)


def _qa_requests_broad_record_listing(question: str) -> bool:
    normalized = _normalize_procurement_question_text(question)
    if not normalized:
        return False
    document_markers = [
        "po", "purchase order", "purchase orders", "invoice", "invoices", "document", "documents",
    ]
    broad_markers = [
        "all", "all the", "every", "list", "show", "what are all", "tell me about all",
    ]
    patterns = [
        r"\b(?:all|every)\s+(?:the\s+)?(?:po|purchase order|purchase orders|invoice|invoices|document|documents)\b",
        r"\b(?:list|show|tell me about)\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?(?:po|purchase order|purchase orders|invoice|invoices|document|documents)\b",
    ]
    if not any(marker in normalized for marker in document_markers):
        return False
    if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in patterns):
        return True
    return any(marker in normalized for marker in broad_markers) and any(
        marker in normalized for marker in document_markers
    )


def _clean_text_list(values: Any, max_items: int = 6) -> List[str]:
    items = values if isinstance(values, list) else []
    cleaned: List[str] = []
    seen = set()
    for value in items:
        text = _clean_text_item(value)
        if not text:
            continue
        normalized = _normalize_text(text)
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(text)
        if len(cleaned) >= max_items:
            break
    return cleaned


def _split_answer_sentences(answer: str, max_items: int = 5) -> List[str]:
    raw_parts = re.split(r"(?<=[.!?])\s+|\n+", str(answer or "").strip())
    parts = []
    for value in raw_parts:
        text = _clean_text_item(value)
        if text:
            parts.append(text)
        if len(parts) >= max_items:
            break
    return parts


def _guess_structured_style(question: str, answer: str, query_analysis: Dict[str, Any] | None = None) -> str:
    normalized_question = _normalize_text(question)
    intent = _normalize_text((query_analysis or {}).get("intent"))

    if any(token in normalized_question for token in ["compare", "comparison", "difference", "versus", " vs ", "better"]) or "compare" in intent:
        return "comparison"
    if any(token in normalized_question for token in ["how to", "steps", "process", "procedure", "workflow", "in case", "what should"]) or "procedure" in intent:
        return "steps"
    if any(token in normalized_question for token in ["list", "which", "what are", "show all", "top", "who all"]):
        return "bullets"
    if any(token in normalized_question for token in ["summary", "summarize", "overview", "brief"]):
        return "summary"
    if len(_split_answer_sentences(answer, max_items=4)) >= 3:
        return "bullets"
    return "paragraph"


def _build_default_structured_answer(
    question: str,
    answer: str,
    query_analysis: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    clean_answer = str(answer or "").strip() or _SOFT_NOT_FOUND_ANSWER
    sentences = _split_answer_sentences(clean_answer, max_items=5)
    style = _guess_structured_style(question, clean_answer, query_analysis=query_analysis)

    summary = clean_answer
    if style != "paragraph" and len(sentences) >= 2:
        summary = " ".join(sentences[:2]).strip()

    sections: List[Dict[str, Any]] = []
    highlights: List[str] = []
    if style == "steps":
        highlights = sentences[:4]
        sections = [{"heading": "Steps", "body": "", "bullets": highlights}]
    elif style == "bullets":
        highlights = sentences[:4]
        sections = [{"heading": "Key Points", "body": "", "bullets": highlights}]
    elif style == "comparison":
        sections = [{"heading": "Comparison", "body": clean_answer, "bullets": []}]
    elif style == "summary":
        sections = []
    else:
        sections = []

    return {
        "style": style,
        "title": "",
        "summary": summary,
        "highlights": highlights,
        "sections": sections,
        "closing": "",
    }


def _normalize_structured_answer(
    question: str,
    answer: str,
    structured_answer: Any,
    query_analysis: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    default_payload = _build_default_structured_answer(
        question,
        answer,
        query_analysis=query_analysis,
    )
    payload = structured_answer if isinstance(structured_answer, dict) else {}

    style = _clean_text_item(payload.get("style")) or default_payload["style"]
    title = _clean_text_item(payload.get("title"))
    summary = _clean_text_item(payload.get("summary")) or default_payload["summary"]
    highlights = _clean_text_list(payload.get("highlights"), max_items=5)
    closing = _clean_text_item(payload.get("closing"))

    sections: List[Dict[str, Any]] = []
    raw_sections = payload.get("sections") if isinstance(payload.get("sections"), list) else []
    for raw_section in raw_sections[:5]:
        if not isinstance(raw_section, dict):
            continue
        heading = _clean_text_item(raw_section.get("heading"))
        body = _clean_text_item(raw_section.get("body"))
        bullets = _clean_text_list(raw_section.get("bullets"), max_items=6)
        if heading or body or bullets:
            sections.append({
                "heading": heading,
                "body": body,
                "bullets": bullets,
            })

    if not sections and default_payload["sections"]:
        sections = default_payload["sections"]
    if not highlights and default_payload["highlights"] and style in {"bullets", "steps"}:
        highlights = default_payload["highlights"]
    if not summary:
        summary = default_payload["summary"]

    return {
        "style": style,
        "title": title,
        "summary": summary,
        "highlights": highlights,
        "sections": sections,
        "closing": closing,
    }


_ALLOWED_CHART_OPTIONS = ("bar", "pie", "line", "doughnut")


def _normalize_graph_data(graph_data: Any) -> Dict[str, Any]:
    payload = graph_data if isinstance(graph_data, dict) else {}
    labels = payload.get("labels")
    values = payload.get("values")

    if not isinstance(labels, list) or not isinstance(values, list):
        points = payload.get("points")
        if isinstance(points, list):
            labels = []
            values = []
            for point in points:
                if not isinstance(point, dict):
                    continue
                label = str(point.get("label") or "").strip()
                value = point.get("value")
                if not label:
                    continue
                try:
                    numeric_value = float(value)
                except Exception:
                    continue
                labels.append(label[:60])
                values.append(round(numeric_value, 4))

    if not isinstance(labels, list) or not isinstance(values, list):
        return {}

    normalized_labels: List[str] = []
    normalized_values: List[float] = []
    for label, value in zip(labels, values):
        safe_label = str(label or "").strip()
        if not safe_label:
            continue
        try:
            numeric_value = float(value)
        except Exception:
            continue
        normalized_labels.append(safe_label[:60])
        normalized_values.append(round(numeric_value, 4))
        if len(normalized_labels) >= 24:
            break

    if len(normalized_labels) < 2 or len(normalized_labels) != len(normalized_values):
        return {}

    return {
        "title": str(payload.get("title") or "").strip()[:120],
        "x_label": str(payload.get("x_label") or "").strip()[:80],
        "y_label": str(payload.get("y_label") or "").strip()[:80],
        "labels": normalized_labels,
        "values": normalized_values,
    }


def _normalize_chart_options(chart_options: Any, has_graph_data: bool) -> List[str]:
    options = chart_options if isinstance(chart_options, list) else []
    normalized: List[str] = []
    seen = set()
    for option in options:
        value = str(option or "").strip().lower()
        if value not in _ALLOWED_CHART_OPTIONS or value in seen:
            continue
        normalized.append(value)
        seen.add(value)
    if normalized:
        return normalized
    return ["bar", "pie"] if has_graph_data else []


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _score_suggestion_candidate(candidate: Dict[str, Any], answer: str, seed_suggestion: str = "") -> int:
    suggestion = _clean_suggestion(candidate.get("suggestion"))
    if not suggestion:
        return -10_000

    score = 0
    helpfulness = candidate.get("helpfulness")
    try:
        score += int(helpfulness) * 10
    except Exception:
        pass

    suggestion_norm = _normalize_text(suggestion)
    answer_norm = _normalize_text(answer)
    seed_norm = _normalize_text(seed_suggestion)

    if suggestion_norm and suggestion_norm == answer_norm:
        score -= 60
    if suggestion_norm and seed_norm and suggestion_norm == seed_norm:
        score += 5
    if 35 <= len(suggestion) <= 220:
        score += 8
    if str(candidate.get("angle") or "").strip():
        score += 3
    if suggestion.endswith("?"):
        score -= 5

    return score


def _pick_best_suggestion(candidates: List[Dict[str, Any]], answer: str, seed_suggestion: str = "") -> str:
    if not candidates:
        return _clean_suggestion(seed_suggestion)

    ranked = sorted(
        candidates,
        key=lambda candidate: _score_suggestion_candidate(candidate, answer, seed_suggestion=seed_suggestion),
        reverse=True,
    )
    best = ranked[0] if ranked else {}
    best_suggestion = _clean_suggestion(best.get("suggestion"))
    return best_suggestion or _clean_suggestion(seed_suggestion)


def _suggestion_prompt_analysis(question: str) -> Dict[str, Any]:
    analysis = analyze_query(question)
    prompt_analysis = {
        key: value
        for key, value in analysis.items()
        if key not in {"needs_suggestion", "simple_lookup"}
    }
    return prompt_analysis


def _build_mandatory_fallback_suggestion(
    question: str,
    answer: str,
    sources: List[Dict[str, Any]],
    query_analysis: Dict[str, Any] | None = None,
) -> str:
    normalized_question = _normalize_text(question)
    normalized_answer = _normalize_text(answer)
    analysis = query_analysis if isinstance(query_analysis, dict) else {}
    intent = _normalize_text(analysis.get("intent"))

    safe_sources = [src for src in (sources or []) if isinstance(src, dict)]
    first_source = safe_sources[0] if safe_sources else {}
    doc_type = str(first_source.get("document_type") or "").strip()
    doc_id = str(first_source.get("doc_id") or "").strip()
    source_label = doc_type or doc_id or "the related documents"

    if _is_no_data_answer(normalized_answer):
        return "Try asking with a specific name, document ID, invoice number, PO number, or date so I can narrow the search."

    if any(token in normalized_question for token in ["compare", "difference", "versus", " vs "]) or "compare" in intent:
        return "You can ask for a side-by-side comparison of dates, amounts, skills, status, or responsibilities for the same records."

    if any(token in normalized_question for token in ["how to", "steps", "process", "procedure", "workflow"]) or "procedure" in intent:
        return "You can ask for a step-by-step checklist, required documents, dependencies, or exception points from the same process."

    if any(token in normalized_question for token in ["summary", "summarize", "brief", "overview"]):
        return f"You can ask for key dates, owners, totals, risks, or a section-wise breakdown from {source_label}."

    if any(token in normalized_question for token in ["invoice", "amount", "po", "purchase order", "revenue", "cost", "total"]):
        return "You can ask me to extract totals, calculate differences, trace linked documents, or list pending amounts from the same records."

    if any(token in normalized_question for token in ["who is", "tell me about", "profile", "candidate", "resume", "employee"]):
        return "You can ask for skills, experience, education, contact details, project history, or a fitment comparison for the same profile."

    return f"You can ask a follow-up for key dates, totals, contacts, status, or a deeper section-wise breakdown from {source_label}."


def _build_answer_generation_inputs(
    question: str,
    sources: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    broad_record_listing = _qa_requests_broad_record_listing(question)
    source_budget = min(12 if broad_record_listing else 4, max(1, len(sources)))
    char_budget = 14000 if broad_record_listing else 6500
    sources_block, sources_for_llm = _build_sources_block_with_langchain(
        question=question,
        sources=sources,
        max_sources=source_budget,
        max_chars=char_budget,
    )
    if not sources_block:
        sources_for_llm = sources
        sources_block = _build_sources_block(sources)
    prompt_query_analysis = _suggestion_prompt_analysis(question)
    return sources_block, sources_for_llm, prompt_query_analysis


def _build_json_answer_prompt(
    question: str,
    sources_block: str,
    sources_for_llm: List[Dict[str, Any]],
    prompt_query_analysis: Dict[str, Any],
    conversation_context: str = "",
) -> str:
    return (
        f"Q: {question}\n\n"
        + (
            f"Conversation context:\n{conversation_context.strip()}\n\n"
            if str(conversation_context or "").strip()
            else ""
        )
        +
        f"Query understanding: {json.dumps(prompt_query_analysis, ensure_ascii=True)}\n\n"
        f"Based on these {len(sources_for_llm)} documents, answer naturally in the length the question requires.\n"
        "Give a complete final answer. If you need to be concise, shorten by summarizing, not by cutting off.\n"
        "Do not end the answer with an unfinished sentence or partial clause.\n"
        "If multiple records match, include the full set that is supported by the provided sources.\n"
        "For resume or candidate-evaluation questions, use the available document evidence to assess suitability instead of requiring an exact sentence match.\n"
        "Also return structured_answer with a user-friendly summary, optional highlights, and sections that fit the question type.\n"
        f"{sources_block}\n\n"
        "Respond ONLY with valid JSON: "
        "{\"answer\": \"complete answer here\", "
        "\"should_suggest\": true, "
        "\"suggestion\": \"one short useful suggestion here or empty string\", "
        "\"topic\": \"short normalized topic for this Q&A\", "
        "\"structured_answer\": {"
        "\"style\": \"paragraph|summary|bullets|steps|comparison\", "
        "\"title\": \"optional short title\", "
        "\"summary\": \"short user-friendly summary\", "
        "\"highlights\": [\"short point\"], "
        "\"sections\": [{\"heading\": \"section heading\", \"body\": \"plain-language explanation\", \"bullets\": [\"bullet\"]}], "
        "\"closing\": \"optional short wrap-up\""
        "}, "
        "\"citations\": [{\"doc_id\": \"...\", \"snippet\": \"...\"}]}"
    )


def _build_streaming_answer_prompt(
    question: str,
    sources_block: str,
    sources_for_llm: List[Dict[str, Any]],
    prompt_query_analysis: Dict[str, Any],
    conversation_context: str = "",
) -> str:
    return (
        f"Q: {question}\n\n"
        + (
            f"Conversation context:\n{conversation_context.strip()}\n\n"
            if str(conversation_context or "").strip()
            else ""
        )
        +
        f"Query understanding: {json.dumps(prompt_query_analysis, ensure_ascii=True)}\n\n"
        f"Based on these {len(sources_for_llm)} documents, answer naturally in the length the question requires.\n"
        "Stream a complete answer in plain text only.\n"
        "If the answer is detailed, structure it live using a short title, summary paragraph, bullets with '- ', and section headings ending with ':'.\n"
        "If multiple records match, include the full set that is supported by the provided sources.\n"
        "For resume or candidate-evaluation questions, use the available document evidence to assess suitability instead of requiring an exact sentence match.\n"
        f"{sources_block}\n"
    )


def _coalesce_stream_chunk(chunk_text: str, streamed_so_far: str) -> str:
    text = str(chunk_text or "")
    if not text:
        return ""
    if streamed_so_far and text.startswith(streamed_so_far):
        return text[len(streamed_so_far):]
    return text


def _generate_contextual_suggestion(
    question: str,
    answer: str,
    sources: List[Dict[str, Any]],
    seed_suggestion: str = "",
) -> str:
    answer_norm = _normalize_text(answer)
    if not answer or _is_no_data_answer(answer):
        return ""

    if not is_configured():
        return ""

    content = ""
    try:
        sources_block, _ = _build_sources_block_with_langchain(
            question=question,
            sources=sources[:3],
            max_sources=3,
            max_chars=3000,
        )
        query_analysis = _suggestion_prompt_analysis(question)
        user_prompt = (
            f"Question: {question}\n"
            f"Answer: {answer}\n\n"
            f"Query understanding: {json.dumps(query_analysis, ensure_ascii=True)}\n\n"
            f"Relevant source context:\n{sources_block}\n\n"
            f"Existing candidate suggestion: {seed_suggestion or '(none)'}\n\n"
            'Return ONLY valid JSON: {"should_suggest": true/false, "candidates": [{"suggestion": "...", "angle": "...", "helpfulness": 1-10}]}'
        )
        if _langchain_enabled():
            parsed = _invoke_lcel_json(
                system_prompt=SUGGESTION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.45,
                max_output_tokens=420,
            )
        else:
            content = generate_json_text(
                user_prompt=user_prompt,
                system_instruction=SUGGESTION_SYSTEM_PROMPT,
                temperature=0.45,
                max_output_tokens=420,
                response_schema=SUGGESTION_CANDIDATES_SCHEMA,
            )
            parsed = safe_json_parse(content)
        should_suggest = _coerce_bool(parsed.get("should_suggest"))
        candidates = parsed.get("candidates") if isinstance(parsed.get("candidates"), list) else []
        cleaned_candidates = []
        for candidate in candidates[:4]:
            if not isinstance(candidate, dict):
                continue
            suggestion = _clean_suggestion(candidate.get("suggestion"))
            angle = str(candidate.get("angle") or "").strip()
            helpfulness = candidate.get("helpfulness")
            if not suggestion:
                continue
            try:
                helpfulness = int(helpfulness)
            except Exception:
                helpfulness = 0
            cleaned_candidates.append({
                "suggestion": suggestion,
                "angle": angle,
                "helpfulness": helpfulness,
            })

        if cleaned_candidates:
            return _pick_best_suggestion(cleaned_candidates, answer, seed_suggestion=seed_suggestion)
        if should_suggest:
            return _clean_suggestion(seed_suggestion)
    except Exception as e:
        preview = content[:120] if content else "(empty)"
        logger.warning(f"[QA] Suggestion generation fallback: {str(e)[:200]} | preview={preview}")

    return _clean_suggestion(seed_suggestion)


def _ensure_suggestion(question: str, result: Dict[str, Any], sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    normalized = dict(result or {})
    answer = _sanitize_answer_text(question, str(normalized.get("answer") or "").strip() or _SOFT_NOT_FOUND_ANSWER)
    citations = normalized.get("citations") if isinstance(normalized.get("citations"), list) else []
    seed_suggestion = str(normalized.get("suggestion") or "").strip()
    suggestion_reviewed = _coerce_bool(normalized.get("_suggestion_reviewed"))
    query_analysis = normalized.get("query_analysis") if isinstance(normalized.get("query_analysis"), dict) else analyze_query(question)
    if suggestion_reviewed:
        suggestion = _clean_suggestion(seed_suggestion)
    else:
        suggestion = _generate_contextual_suggestion(question, answer, sources, seed_suggestion=seed_suggestion)
        normalized["_suggestion_reviewed"] = True
    if not suggestion:
        suggestion = _fallback_suggestion_from_context(question, answer)

    normalized["answer"] = answer
    normalized["citations"] = citations
    normalized["suggestion"] = suggestion
    normalized["query_analysis"] = query_analysis
    normalized["structured_answer"] = _normalize_structured_answer(
        question,
        answer,
        normalized.get("structured_answer"),
        query_analysis=query_analysis,
    )
    normalized_graph = _normalize_graph_data(normalized.get("graph_data"))
    normalized["graph_data"] = normalized_graph
    normalized["chart_options"] = _normalize_chart_options(
        normalized.get("chart_options"),
        has_graph_data=bool(normalized_graph),
    )
    if normalized_graph:
        preferred = _preferred_chart_options_for_question(question, has_graph_data=True)
        merged = []
        seen = set()
        for option in preferred + list(normalized.get("chart_options") or []):
            if option in _ALLOWED_CHART_OPTIONS and option not in seen:
                merged.append(option)
                seen.add(option)
        normalized["chart_options"] = merged
    return normalized


def _extract_json_string_field(payload: str, field_name: str) -> str:
    raw = str(payload or "")
    match = re.search(rf'"{re.escape(field_name)}"\s*:\s*"', raw)
    if not match:
        return ""

    start = match.end()
    chars = []
    escaped = False
    for ch in raw[start:]:
        if escaped:
            chars.append(ch)
            escaped = False
            continue
        if ch == "\\":
            chars.append(ch)
            escaped = True
            continue
        if ch == '"':
            break
        chars.append(ch)

    candidate = '"' + "".join(chars) + '"'
    try:
        return str(json.loads(candidate))
    except Exception:
        value = "".join(chars)
        value = value.replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t").replace('\\"', '"')
        return value.strip()


def _extract_json_bool_field(payload: str, field_name: str) -> bool | None:
    raw = str(payload or "")
    match = re.search(rf'"{re.escape(field_name)}"\s*:\s*(true|false)', raw, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip().lower() == "true"


def _recover_answer_payload(content: str, question: str, sources: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    raw = str(content or "").strip()
    if not raw:
        return None

    answer = _extract_json_string_field(raw, "answer")
    if not answer:
        plain_text = re.sub(r"^invalid json output:\s*", "", raw, flags=re.IGNORECASE).strip()
        if "For troubleshooting, visit:" in plain_text:
            plain_text = plain_text.split("For troubleshooting, visit:", 1)[0].strip()
        if plain_text and not plain_text.startswith("{"):
            answer = plain_text
    if not answer:
        return None

    suggestion = _extract_json_string_field(raw, "suggestion")
    topic = _extract_json_string_field(raw, "topic") or question[:60]
    should_suggest = _extract_json_bool_field(raw, "should_suggest")

    recovered = {
        "answer": answer,
        "suggestion": suggestion if (should_suggest or suggestion) else "",
        "topic": topic,
        "citations": _fallback_citations(sources),
    }
    return recovered


def answer_question(question: str, sources: List[Dict[str, Any]], conversation_context: str = "") -> Dict[str, Any]:
    """Answer question using sources with full context"""
    if not question:
        return _ensure_suggestion(question, {"answer": _SOFT_NOT_FOUND_ANSWER, "citations": []}, sources)

    cache_key = _build_cache_key(question, sources, conversation_context)
    cached_answer = _cache_get(cache_key)
    if cached_answer is not None:
        logger.info("[QA] Cache hit")
        cached_answer = _ensure_suggestion(question, cached_answer, sources)
        _cache_set(cache_key, cached_answer)
        return cached_answer
    
    if not is_configured():
        logger.warning("[QA] LLM client not configured (GEMINI_API_KEY missing)")
        fallback = _ensure_suggestion(question, {
            "answer": "LLM not configured. Showing related documents.",
            "citations": _fallback_citations(sources),
        }, sources)
        _cache_set(cache_key, fallback)
        return fallback
    
    # Check semantic cache (topic-based matching)
    cached_result = None if conversation_context.strip() else _search_cache_by_topic(question)
    if cached_result is not None:
        logger.info("[QA] Semantic cache hit")
        cached_result = _ensure_suggestion(question, cached_result, sources)
        return cached_result
    
    sources_block, sources_for_llm, prompt_query_analysis = _build_answer_generation_inputs(question, sources)
    user_prompt = _build_json_answer_prompt(
        question,
        sources_block,
        sources_for_llm,
        prompt_query_analysis,
        conversation_context=conversation_context,
    )
    
    logger.info(
        f"[QA] Sending question to Gemini (cache miss) using model {model_name()} | sources={len(sources_for_llm)} | question_preview='{question[:120]}'"
    )
    
    content = ""
    try:
        if _langchain_enabled():
            result = _invoke_lcel_json(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.1,
                max_output_tokens=_QA_ANSWER_MAX_TOKENS,
            )
            logger.info("[QA] LLM Response received (LCEL)")
        else:
            content = generate_json_text(
                user_prompt=user_prompt,
                system_instruction=SYSTEM_PROMPT,
                temperature=0.1,
                max_output_tokens=_QA_ANSWER_MAX_TOKENS,
                response_schema=ANSWER_RESPONSE_SCHEMA,
            )
            logger.info(f"[QA] LLM Response: {content[:150]}")
            result = safe_json_parse(content)
    except Exception as e:
        raw_content = content
        llm_output = getattr(e, "llm_output", "")
        if not raw_content and llm_output:
            raw_content = str(llm_output or "")
        if str(raw_content or "").strip():
            logger.warning("[QA] LLM returned malformed JSON; attempting recovery from raw text")
        else:
            logger.exception("[QA] LLM Error while answering question")
        recovered = _recover_answer_payload(raw_content, question, sources_for_llm or sources)
        if recovered is not None:
            logger.info("[QA] Recovered answer from malformed JSON response")
            recovered = _ensure_suggestion(question, recovered, sources_for_llm or sources)
            _cache_set(cache_key, recovered)
            return recovered
        logger.info(f"[QA] Falling back to snippets only")
        fallback = {
            "answer": "Based on documents found. See sources for details.",
            "topic": question[:60],
            "citations": _fallback_citations(sources_for_llm or sources),
        }
        fallback = _ensure_suggestion(question, fallback, sources_for_llm or sources)
        _cache_set(cache_key, fallback)
        return fallback
    
    answer = _sanitize_answer_text(question, str(result.get("answer") or "").strip() or _SOFT_NOT_FOUND_ANSWER)
    should_suggest = _coerce_bool(result.get("should_suggest"))
    raw_suggestion = result.get("suggestion")
    suggestion = _clean_suggestion(raw_suggestion) if (should_suggest or raw_suggestion) else ""
    topic = str(result.get("topic") or "").strip() or question[:60]  # Use question as fallback topic
    citations = result.get("citations") if isinstance(result.get("citations"), list) else []
    
    # Validate citations
    allowed_ids = {src.get("doc_id") for src in (sources_for_llm or sources) if src.get("doc_id")}
    filtered = []
    for cite in citations:
        if not isinstance(cite, dict):
            continue
        doc_id = cite.get("doc_id")
        if doc_id in allowed_ids:
            snippet = str(cite.get("snippet") or "").strip()
            filtered.append({"doc_id": doc_id, "snippet": snippet})
    
    if not filtered and (sources_for_llm or sources):
        filtered = _fallback_citations(sources_for_llm or sources)

    final_result = {
        "answer": answer,
        "suggestion": suggestion,
        "topic": topic,  # Store semantic topic for cache search
        "structured_answer": result.get("structured_answer"),
        "citations": filtered,
        "graph_data": result.get("graph_data"),
        "chart_options": result.get("chart_options"),
    }
    final_result = _ensure_suggestion(question, final_result, sources_for_llm or sources)
    _cache_set(cache_key, final_result)
    return final_result


def answer_question_stream(
    question: str,
    sources: List[Dict[str, Any]],
    conversation_context: str = "",
) -> Iterator[Dict[str, Any]]:
    """Yield streaming deltas followed by a final normalized answer payload."""
    if not question:
        final = _ensure_suggestion(question, {"answer": _SOFT_NOT_FOUND_ANSWER, "citations": []}, sources)
        yield {"type": "final", "data": final}
        return

    cache_key = _build_cache_key(question, sources, conversation_context)
    cached_answer = _cache_get(cache_key)
    if cached_answer is not None:
        logger.info("[QA] Cache hit (stream)")
        cached_answer = _ensure_suggestion(question, cached_answer, sources)
        _cache_set(cache_key, cached_answer)
        yield {"type": "final", "data": cached_answer}
        return

    if not is_configured():
        logger.warning("[QA] LLM client not configured (GEMINI_API_KEY missing)")
        fallback = _ensure_suggestion(question, {
            "answer": "LLM not configured. Showing related documents.",
            "citations": _fallback_citations(sources),
        }, sources)
        _cache_set(cache_key, fallback)
        yield {"type": "final", "data": fallback}
        return

    cached_result = None if conversation_context.strip() else _search_cache_by_topic(question)
    if cached_result is not None:
        logger.info("[QA] Semantic cache hit (stream)")
        cached_result = _ensure_suggestion(question, cached_result, sources)
        yield {"type": "final", "data": cached_result}
        return

    sources_block, sources_for_llm, prompt_query_analysis = _build_answer_generation_inputs(question, sources)
    user_prompt = _build_streaming_answer_prompt(
        question,
        sources_block,
        sources_for_llm,
        prompt_query_analysis,
        conversation_context=conversation_context,
    )

    logger.info(
        f"[QA] Streaming question to Gemini using model {model_name()} | sources={len(sources_for_llm)} | question_preview='{question[:120]}'"
    )

    streamed_answer = ""
    try:
        for chunk_text in generate_text_stream(
            user_prompt=user_prompt,
            system_instruction=STREAM_SYSTEM_PROMPT,
            temperature=0.1,
            max_output_tokens=_QA_ANSWER_MAX_TOKENS,
        ):
            delta = _coalesce_stream_chunk(chunk_text, streamed_answer)
            if not delta:
                continue
            streamed_answer += delta
            yield {"type": "delta", "delta": delta}
    except Exception:
        logger.exception("[QA] LLM Error while streaming answer")
        if not streamed_answer.strip():
            fallback = {
                "answer": "Based on documents found. See sources for details.",
                "topic": question[:60],
                "citations": _fallback_citations(sources_for_llm or sources),
            }
            fallback = _ensure_suggestion(question, fallback, sources_for_llm or sources)
            _cache_set(cache_key, fallback)
            yield {"type": "final", "data": fallback}
            return

    answer = _sanitize_answer_text(question, streamed_answer.strip() or _SOFT_NOT_FOUND_ANSWER)
    final_result = {
        "answer": answer,
        "suggestion": "",
        "topic": question[:60],
        "structured_answer": None,
        "citations": _fallback_citations(sources_for_llm or sources),
    }
    try:
        enriched_result = answer_question(
            question,
            sources,
            conversation_context=conversation_context,
        )
        if isinstance(enriched_result, dict):
            enriched_answer = _sanitize_answer_text(
                question,
                str(enriched_result.get("answer") or "").strip() or _SOFT_NOT_FOUND_ANSWER,
            )
            enriched_suggestion = _clean_suggestion(enriched_result.get("suggestion"))
            enriched_topic = str(enriched_result.get("topic") or "").strip()
            enriched_citations = enriched_result.get("citations") if isinstance(enriched_result.get("citations"), list) else []
            enriched_structured = enriched_result.get("structured_answer")
            enriched_graph_data = enriched_result.get("graph_data")
            enriched_chart_options = enriched_result.get("chart_options")
            if _is_no_data_answer(final_result.get("answer")) and not _is_no_data_answer(enriched_answer):
                final_result["answer"] = enriched_answer
            if isinstance(enriched_structured, dict):
                final_result["structured_answer"] = enriched_structured
            if enriched_graph_data:
                final_result["graph_data"] = enriched_graph_data
            if isinstance(enriched_chart_options, list):
                final_result["chart_options"] = enriched_chart_options
            if enriched_suggestion:
                final_result["suggestion"] = enriched_suggestion
                final_result["_suggestion_reviewed"] = True
            if enriched_topic:
                final_result["topic"] = enriched_topic
            if enriched_citations:
                final_result["citations"] = enriched_citations
    except Exception as exc:
        logger.warning(f"[QA] Suggestion enrichment fallback in stream mode: {str(exc)[:180]}")
    final_result = _ensure_suggestion(question, final_result, sources_for_llm or sources)
    logger.info(f"[QA] Stream suggestion generated: {'yes' if final_result.get('suggestion') else 'no'}")
    _cache_set(cache_key, final_result)
    yield {"type": "final", "data": final_result}

