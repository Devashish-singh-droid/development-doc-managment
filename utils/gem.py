"""
GEM (Goods and Services Exchange Model) PDF Processing Module
"""
from uuid import uuid4
from copy import copy

import requests
import tempfile
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.logger import get_logger
from typing import Dict, Any, List, TypedDict, Optional
import os
import fitz
import json
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from bs4 import BeautifulSoup
import re
import hashlib

logger = get_logger('gem')
load_dotenv()


# ---------------------------------------------------------------------------
# State Definition for LangGraph
# ---------------------------------------------------------------------------

class ChunkProcessingState(TypedDict):
    all_chunks: List[str]
    current_chunk_index: int
    accumulated_results: Dict[str, Dict]
    master_schema: Dict[str, Any]
    chunk_results: List[Dict[str, Dict]]
    is_complete: bool


# ---------------------------------------------------------------------------
# Text Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 180000, overlap: int = 500) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap if end < len(text) else end
    logger.info(f"[GEM] Text split into {len(chunks)} chunks (size: {chunk_size}, overlap: {overlap})")
    return chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_sheet(wb, target_name: str):
    target = target_name.strip().lower()
    for name in wb.sheetnames:
        if name.strip().lower() == target:
            return wb[name]
    return None

# ---------------------------------------------------------------------------
# OCR helper for scanned PDFs
# ---------------------------------------------------------------------------
def _ocr_pdf_bytes(pdf_bytes: bytes, url: str) -> str:
    """
    Render each PDF page to image using PyMuPDF, then OCR with shared PaddleOCR helper.
    Returns extracted text, or empty string if OCR fails.
    """
    def _extract_page_text(result: Any) -> str:
        """
        Normalize OCR output defensively inside GEM so linked-PDF extraction
        still works even if the shared OCR helper returns a slightly different shape.
        """
        if not result:
            return ""

        if isinstance(result, dict):
            raw_text = str(result.get("raw_text") or "").strip()
            if raw_text:
                return raw_text

            rec_texts = result.get("rec_texts")
            if isinstance(rec_texts, list):
                return "\n".join(str(item).strip() for item in rec_texts if str(item).strip()).strip()

            text = str(result.get("text") or "").strip()
            if text:
                return text

        if hasattr(result, "get"):
            try:
                raw_text = str(result.get("raw_text") or "").strip()
                if raw_text:
                    return raw_text
            except Exception:
                pass

        if isinstance(result, list):
            lines = []
            for item in result:
                if isinstance(item, dict):
                    text = str(item.get("text") or item.get("raw_text") or "").strip()
                    if text:
                        lines.append(text)
                        continue

                    rec_texts = item.get("rec_texts")
                    if isinstance(rec_texts, list):
                        lines.extend(str(text).strip() for text in rec_texts if str(text).strip())
                        continue

                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    candidate = item[1]
                    if isinstance(candidate, (list, tuple)) and candidate:
                        text = str(candidate[0] or "").strip()
                        if text:
                            lines.append(text)
                            continue

                text = str(item or "").strip()
                if text and text not in {"[]", "{}"}:
                    lines.append(text)

            return "\n".join(lines).strip()

        return ""

    try:
        from PIL import Image
        from ocr.paddle_ocr_engine import paddle_ocr_extract
    except Exception as exc:
        logger.warning(f"[GEM][OCR] Shared PaddleOCR engine unavailable: {exc}")
        return ""

    text_parts = []
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            page_count = doc.page_count
            logger.info(f"[GEM][OCR] Starting PaddleOCR on {page_count} pages | URL: {url}")

            for page_no in range(page_count):
                page = doc.load_page(page_no)
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better accuracy
                pix = page.get_pixmap(matrix=mat)

                mode = "RGBA" if pix.n == 4 else "RGB"
                img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                if mode == "RGBA":
                    img = img.convert("RGB")

                result = paddle_ocr_extract(img)
                page_text = _extract_page_text(result)
                if page_text:
                    text_parts.append(page_text)
                    logger.info(
                        f"[GEM][OCR] Page {page_no + 1}/{page_count} - "
                        f"{len(page_text)} chars extracted"
                    )
                else:
                    logger.warning(
                        f"[GEM][OCR] Page {page_no + 1}/{page_count} - "
                        f"no text detected | result_type={type(result).__name__}"
                    )

        result_text = "\n\n".join(text_parts).strip()
        logger.info(
            f"[GEM][OCR] PaddleOCR complete - {len(result_text)} chars "
            f"from {page_count} pages | URL: {url}"
        )
        return result_text

    except Exception as e:
        logger.error(f"[GEM][OCR] PaddleOCR failed: {e} | URL: {url}")
        return ""

# ---------------------------------------------------------------------------
# HTML scraping Ã¢â‚¬â€ find PDF links inside HTML pages
# ---------------------------------------------------------------------------

def _extract_pdf_links_from_html(html_content: bytes, base_url: str) -> List[str]:
    """
    Parse an HTML page and return all URLs that point to PDFs.
    Checks anchor/iframe/embed/object tags and raw URL regex.
    """
    from urllib.parse import urljoin

    soup     = BeautifulSoup(html_content, "html.parser")
    pdf_urls = []
    seen     = set()

    for tag in soup.find_all(["a", "iframe", "embed", "object"]):
        href = tag.get("href") or tag.get("src") or tag.get("data")
        if not href:
            continue
        if ".pdf" in href.lower() or "pdf" in href.lower():
            absolute = urljoin(base_url, href)
            if absolute not in seen:
                seen.add(absolute)
                pdf_urls.append(absolute)
                logger.info(f"[GEM][HTML] Found PDF link in tag: {absolute}")

    # Also scan raw text for PDF URLs
    raw_text = html_content.decode("utf-8", errors="ignore")
    pattern  = r'https?://[^\s\'"<>]+\.pdf[^\s\'"<>]*'
    for m in re.findall(pattern, raw_text, re.IGNORECASE):
        if m not in seen:
            seen.add(m)
            pdf_urls.append(m)
            logger.info(f"[GEM][HTML] Found PDF URL via regex: {m}")

    logger.info(f"[GEM][HTML] Total PDF links found: {len(pdf_urls)} | Base: {base_url}")
    return pdf_urls


# ---------------------------------------------------------------------------
# Core PDF downloader + extractor
# ---------------------------------------------------------------------------

def _download_and_extract_pdf(url: str, prefetched_bytes: Optional[bytes] = None) -> str:
    """
    Extract text from a direct PDF URL (or pre-downloaded bytes).
    Automatically falls back to OCR if text extraction yields 0 chars.
    """
    temp_path = None
    try:
        if prefetched_bytes is not None:
            pdf_bytes = prefetched_bytes
        else:
            logger.info(f"[GEM][PDF] Downloading: {url}")
            resp = requests.get(url, timeout=30, verify=False)
            resp.raise_for_status()
            pdf_bytes = resp.content
            logger.info(f"[GEM][PDF] Downloaded {len(pdf_bytes)} bytes | URL: {url}")

        # Write to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            temp_path = tmp.name

        # Normal text extraction
        full_text  = []
        page_count = 0
        try:
            with fitz.open(temp_path) as doc:
                page_count = doc.page_count
                for page_no in range(page_count):
                    page_text = doc.load_page(page_no).get_text("text")
                    if page_text and page_text.strip():
                        full_text.append(page_text.strip())
        except fitz.FileDataError as e:
            logger.error(f"[GEM][PDF] Invalid PDF: {e} | URL: {url}")
            return f"Error: Invalid PDF Ã¢â‚¬â€ {str(e)}"

        extracted  = "\n".join(full_text).strip()
        char_count = len(extracted)
        logger.info(f"[GEM][PDF] Extracted Ã¢â‚¬â€ Pages: {page_count}, Chars: {char_count} | URL: {url}")

        # Scanned PDF: 0 chars Ã¢â€ â€™ OCR
        if char_count == 0 and page_count > 0:
            logger.warning(
                f"[GEM][PDF] Zero chars from {page_count} pages Ã¢â‚¬â€ trying OCR | URL: {url}"
            )
            ocr_text = _ocr_pdf_bytes(pdf_bytes, url)
            if ocr_text:
                logger.info(f"[GEM][PDF] OCR succeeded Ã¢â‚¬â€ {len(ocr_text)} chars | URL: {url}")
                return ocr_text
            else:
                msg = f"Warning: zero chars even after OCR Ã¢â‚¬â€ scanned PDF or OCR not installed (pages: {page_count})"
                logger.warning(f"[GEM][PDF] {msg} | URL: {url}")
                return msg

        return extracted

    except Exception as e:
        msg = f"Error: {type(e).__name__}: {str(e)}"
        logger.error(f"[GEM][PDF] {msg} | URL: {url}")
        return msg
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"[GEM][PDF] Cleaned up: {temp_path}")


# ---------------------------------------------------------------------------
# Link extractor Ã¢â‚¬â€ handles HTML pages, direct PDFs, scanned PDFs
# ---------------------------------------------------------------------------

def extract_text_from_pdf_link(link: str) -> str:
    """
    Download a URL and extract text. Handles:
      1. Direct PDF             Ã¢â€ â€™ text extraction (+ OCR fallback)
      2. Scanned PDF (0 chars)  Ã¢â€ â€™ OCR via pytesseract
      3. HTML page              Ã¢â€ â€™ scrape for embedded PDF links, recurse once
      4. HTML with no PDFs      Ã¢â€ â€™ extract visible page text as last resort
    """
    logger.info(f"[GEM][LINK] Attempting: {link}")

    try:
        response     = requests.get(link, timeout=30, verify=False)
        http_status  = response.status_code
        content_type = response.headers.get("Content-Type", "unknown")
        content_len  = len(response.content)
        logger.info(
            f"[GEM][LINK] HTTP {http_status} | Content-Type: {content_type} | "
            f"Bytes: {content_len} | URL: {link}"
        )

        response.raise_for_status()

        # Ã¢â€â‚¬Ã¢â€â‚¬ HTML page Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        if "text/html" in content_type.lower():
            logger.info(f"[GEM][LINK] HTML detected Ã¢â‚¬â€ scanning for embedded PDF links | URL: {link}")
            pdf_links = _extract_pdf_links_from_html(response.content, link)

            if pdf_links:
                all_texts = []
                for pdf_url in pdf_links:
                    logger.info(f"[GEM][LINK] Processing embedded PDF: {pdf_url}")
                    result = _download_and_extract_pdf(pdf_url)
                    if result and not result.startswith("Error:") and not result.startswith("Warning:"):
                        all_texts.append(result)
                        logger.info(f"[GEM][LINK] Embedded PDF extracted: {len(result)} chars | {pdf_url}")
                    else:
                        logger.warning(f"[GEM][LINK] Embedded PDF failed: {result} | {pdf_url}")

                if all_texts:
                    combined = "\n\n".join(all_texts)
                    logger.info(f"[GEM][LINK] {len(all_texts)} embedded PDFs extracted, total {len(combined)} chars")
                    return combined
                else:
                    logger.warning(f"[GEM][LINK] PDF links found but all failed to extract | URL: {link}")

            # Fallback: visible text from the HTML page itself
            soup      = BeautifulSoup(response.content, "html.parser")
            html_text = soup.get_text(separator="\n", strip=True)
            if html_text and len(html_text) > 100:
                logger.info(f"[GEM][LINK] Extracted {len(html_text)} chars from HTML text | URL: {link}")
                return html_text

            return f"Skipped: HTML page with no useful PDF or text content"

        # Ã¢â€â‚¬Ã¢â€â‚¬ PDF response Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        is_pdf = "pdf" in content_type.lower() or ".pdf" in link.lower()
        if not is_pdf:
            msg = f"Skipped: unrecognized content-type '{content_type}'"
            logger.warning(f"[GEM][LINK] {msg} | URL: {link}")
            return msg

        return _download_and_extract_pdf(link, prefetched_bytes=response.content)

    except requests.exceptions.SSLError as e:
        msg = f"Error: SSL failed Ã¢â‚¬â€ {str(e)}"
        logger.error(f"[GEM][LINK] {msg} | URL: {link}")
        return msg
    except requests.exceptions.ConnectionError as e:
        msg = f"Error: Connection failed Ã¢â‚¬â€ {str(e)}"
        logger.error(f"[GEM][LINK] {msg} | URL: {link}")
        return msg
    except requests.exceptions.Timeout:
        msg = "Error: Timed out after 30s"
        logger.error(f"[GEM][LINK] {msg} | URL: {link}")
        return msg
    except requests.exceptions.HTTPError as e:
        msg = f"Error: HTTP {response.status_code} Ã¢â‚¬â€ {str(e)}"
        logger.error(f"[GEM][LINK] {msg} | URL: {link}")
        return msg
    except Exception as e:
        msg = f"Error: {type(e).__name__}: {str(e)}"
        logger.error(f"[GEM][LINK] {msg} | URL: {link}")
        return msg


# ---------------------------------------------------------------------------
# Main PDF extraction
# ---------------------------------------------------------------------------

def extract_pdf_text_and_links(file_path: str) -> Dict[str, Any]:
    full_text  = []
    all_links  = []
    page_count = 0

    with fitz.open(file_path) as doc:
        page_count = doc.page_count
        for page_no in range(page_count):
            page = doc.load_page(page_no)
            text = page.get_text("text")
            if text:
                full_text.append(text.strip())
            for link in page.get_links():
                uri = link.get("uri")
                if uri:
                    all_links.append(uri)

    os.makedirs("gem-testing", exist_ok=True)
    if all_links:
        with open("gem-testing/extracted_links.txt", "w") as f:
            for link in all_links:
                f.write(link + "\n")
        logger.info(f"[GEM] Saved {len(all_links)} links to gem-testing/extracted_links.txt")

    extracted_text = "\n".join(full_text)
    logger.info(f"[GEM] Main PDF: {page_count} pages, {len(extracted_text)} chars")
    return {"text": extracted_text, "links": all_links}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def gem_processing(file_path: str) -> Dict[str, Any]:
    logger.info(f"[GEM] Starting with file_path: {file_path}")

    try:
        result = extract_pdf_text_and_links(file_path)
        if result is None:
            raise ValueError("extract_pdf_text_and_links returned None")
        text  = result["text"]
        links = result.get("links", [])
    except Exception as e:
        logger.error(f"[GEM] PDF extraction failed: {e}")
        return {"status": "error", "message": f"PDF extraction failed: {str(e)}"}

    logger.info(f"[GEM] Main PDF: {len(text)} chars, {len(links)} links found")

    # Download and extract linked PDFs
    sub_pdf_texts = []
    link_summary  = []
    duplicate_links_skipped = 0
    seen_link_content_hashes = set()

    for idx, link in enumerate(links):
        logger.info(f"[GEM][LINK] Processing link {idx + 1}/{len(links)}: {link}")
        extracted  = extract_text_from_pdf_link(link)
        is_failure = (
            extracted.startswith("Error:")
            or extracted.startswith("Skipped:")
            or extracted.startswith("Warning:")
        )
        if is_failure:
            link_summary.append({"link": link, "status": "failed", "detail": extracted, "chars": 0})
            logger.warning(f"[GEM][LINK] Link {idx + 1} FAILED: {extracted}")
        else:
            char_count = len(extracted)
            content_hash = hashlib.sha256(extracted.encode("utf-8", errors="ignore")).hexdigest()
            is_duplicate_content = content_hash in seen_link_content_hashes
            if not is_duplicate_content:
                seen_link_content_hashes.add(content_hash)

            if is_duplicate_content:
                duplicate_links_skipped += 1
                link_summary.append({
                    "link": link,
                    "status": "duplicate_skipped",
                    "chars": char_count,
                })
                logger.info(
                    f"[GEM][LINK] Link {idx + 1} DUPLICATE CONTENT - skipped ({char_count} chars)"
                )
            else:
                link_summary.append({"link": link, "status": "success", "chars": char_count})
                sub_pdf_texts.append(extracted)
                logger.info(f"[GEM][LINK] Link {idx + 1} SUCCESS - {char_count} chars")

    os.makedirs("gem-testing", exist_ok=True)
    with open("gem-testing/link_summary.json", "w", encoding="utf-8") as f:
        json.dump(link_summary, f, indent=4)

    successful = sum(1 for s in link_summary if s["status"] == "success")
    failed = sum(1 for s in link_summary if s["status"] == "failed")
    logger.info(
        f"[GEM][LINK] Summary - Total: {len(links)}, Success: {successful}, "
        f"DuplicateSkipped: {duplicate_links_skipped}, Failed: {failed}"
    )

    try:
        cell_map = LLM_processing(text)

        if sub_pdf_texts:
            logger.info(f"[GEM] Running call2 LLM on {len(sub_pdf_texts)} linked PDF(s)")
            call2_results = call2_LLM_processing(sub_pdf_texts)

            PRIORITY_SHEETS = ["Asset Details", "Resource Cost"]

            #FORCE OVERRIDE (call2 always wins)
            for sheet in PRIORITY_SHEETS:
                if sheet in call2_results:
                    logger.info(f"[GEM] FORCING sub PDF data for '{sheet}' (discarding call1)")
                    cell_map[sheet] = call2_results[sheet]
                    
        # Normalize all sheet names by stripping whitespace
        normalized_cell_map = {}
        for key, value in cell_map.items():
            normalized_key = key.strip()
            if normalized_key in normalized_cell_map:
                logger.warning(
                    f"[GEM] Duplicate sheet key after normalization: '{normalized_key}' - "
                    f"keeping later value from '{key}'"
                )
            normalized_cell_map[normalized_key] = value
        
        output_path = write_to_excel(normalized_cell_map)

    except Exception as e:
        logger.error(f"[GEM] Processing failed: {e}")
        return {"status": "error", "message": f"Processing failed: {str(e)}"}

    return {
        "status":       "completed",
        "message":      "PDF processing completed.",
        "output_path":  output_path,
        "link_summary": link_summary,
    }


# ---------------------------------------------------------------------------
# LangGraph Chunk Processing
# ---------------------------------------------------------------------------

def process_single_chunk(state: ChunkProcessingState) -> ChunkProcessingState:
    if state["current_chunk_index"] >= len(state["all_chunks"]):
        state["is_complete"] = True
        return state

    current_chunk = state["all_chunks"][state["current_chunk_index"]]
    schema        = state["master_schema"]

    logger.info(f"[GEM] Processing chunk {state['current_chunk_index'] + 1}/{len(state['all_chunks'])}")

    previous_context = ""
    if state["accumulated_results"]:
        previous_context = "\n\nPrevious chunk results (for context):\n" + json.dumps(
            state["accumulated_results"], indent=2
        )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_template("""You are an expert tender document analyst filling an Excel workbook from extracted PDF text.

You are processing chunk {chunk_number} of a multi-chunk document.

Return ONLY a valid JSON object. No markdown. No explanation.

Structure:
{{
  "<exact_sheet_name>": {{ "<cell_address>": <value_or_null> }},
  ...
}}

EXCEPTION Asset Details must be a plain JSON array (NOT a dict, NOT wrapped in "rows" or any key):
"Asset Details": [
  {{"asset_type": "Laptop", "brand": null, "qty": 50}},
  {{"asset_type": "Desktop", "brand": "Dell", "qty": 20}}
]

GENERAL RULES
- Use EXACT sheet names and cell addresses from schema.
- Return null if value not found. Do NOT invent values.
- Do NOT duplicate values already present in previous chunks.

TENDER CHECKLIST:
Extract: tender number, bid date, customer/dept name, scope of work, duration, bid due date, EMD/PBG, delivery location, OEM/MAF requirements, compliance.

ASSET DETAILS (plain array):
- Normalize: "Laptops"Ã¢â€ â€™"Laptop", "Desktops"Ã¢â€ â€™"Desktop", "Printers"Ã¢â€ â€™"Printer"
- Total qty rule: If total stated Ã¢â€ â€™ use total. If only model-wise Ã¢â€ â€™ sum them.
- ONE entry per asset type. No duplicates.
- Brand: null when using total qty. Set only if one clear brand dominates.
- Types to capture: Desktop, Laptop, Printer, Scanner, Server, UPS, Network Switch, Mouse, Keyboard, Monitor, Webcam, Storage Device

RESOURCE COST:
Extract manpower roles, qualification, qty, monthly rate (only if explicitly stated).

COSTING:
Fill only explicitly stated values. No guessing.

Schema:
{schema}

Chunk {chunk_number}:
{text}

{previous_context}""")

    try:
        chain    = prompt | llm
        response = chain.invoke({
            "schema":           json.dumps(schema, indent=2),
            "text":             current_chunk,
            "chunk_number":     state["current_chunk_index"] + 1,
            "previous_context": previous_context
        })

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        chunk_result = json.loads(raw)
        state["chunk_results"].append(chunk_result)

        for sheet_name, cells in chunk_result.items():
            if sheet_name not in state["accumulated_results"]:
                state["accumulated_results"][sheet_name] = [] if isinstance(cells, list) else {}

            if isinstance(cells, list):
                if not isinstance(state["accumulated_results"][sheet_name], list):
                    state["accumulated_results"][sheet_name] = []
                state["accumulated_results"][sheet_name].extend(cells)
            elif isinstance(cells, dict):
                if isinstance(state["accumulated_results"][sheet_name], dict):
                    state["accumulated_results"][sheet_name].update(cells)

        logger.info(f"[GEM] Chunk {state['current_chunk_index'] + 1} processed successfully")

    except json.JSONDecodeError as e:
        logger.error(f"[GEM] JSON parse error chunk {state['current_chunk_index'] + 1}: {e}")
    except Exception as e:
        logger.error(f"[GEM] Error chunk {state['current_chunk_index'] + 1}: {e}")

    state["current_chunk_index"] += 1
    return state


def create_chunk_processing_workflow():
    workflow = StateGraph(ChunkProcessingState)
    workflow.add_node("process_chunk", process_single_chunk)
    workflow.set_entry_point("process_chunk")

    def should_continue(state: ChunkProcessingState) -> str:
        if state["is_complete"] or state["current_chunk_index"] >= len(state["all_chunks"]):
            return "end"
        return "process_chunk"

    workflow.add_conditional_edges(
        "process_chunk",
        should_continue,
        {"process_chunk": "process_chunk", "end": "__end__"}
    )

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# ---------------------------------------------------------------------------
# Schema extraction
# ---------------------------------------------------------------------------

def extract_master_schema_json() -> Dict:
    import openpyxl

    file_path = r"templates/template.xlsx"
    wb        = openpyxl.load_workbook(file_path, data_only=False)
    logger.info(f"Workbook sheets: {wb.sheetnames}")

    master_schema = {"workbook_name": "GEM Template", "sheets": {}}

    ws = _find_sheet(wb, "Tender Checklist")
    if ws:
        fields = []
        for row in range(2, 40):
            label = ws[f"A{row}"].value
            if label:
                fields.append({"label": str(label).strip(), "value_cell": f"B{row}", "value": None})
        master_schema["sheets"]["Tender Checklist"] = {
            "sheet_name": ws.title, "type": "key_value_form", "fields": fields
        }
        logger.info(f"[GEM] Schema extracted: '{ws.title}'")
    else:
        logger.warning("[GEM] 'Tender Checklist' sheet not found")

    ws = _find_sheet(wb, "Asset Details")
    if ws:
        master_schema["sheets"]["Asset Details"] = {
            "sheet_name": ws.title, "type": "flat_asset_table",
            "start_row": 3,
            "columns": {"sr_no": "A", "asset_type": "B", "brand": "C", "qty": "D"},
            "rows": []
        }
        logger.info(f"[GEM] Schema extracted: '{ws.title}'")
    else:
        logger.warning("[GEM] 'Asset Details' sheet not found")

    ws = _find_sheet(wb, "Resource Cost")
    if ws:
        resources = []
        for row in range(2, 20):
            sr = ws[f"A{row}"].value
            if sr:
                resources.append({
                    "row": row, "sr_no": sr,
                    "profile_cell": f"B{row}", "qualification_cell": f"C{row}",
                    "qty_cell": f"D{row}", "per_month_cell": f"E{row}",
                    "for_12_month_cell": f"F{row}",
                    "profile": None, "qualification": None,
                    "qty": None, "per_month": None, "for_12_month": None,
                })
        master_schema["sheets"]["Resource Cost"] = {
            "sheet_name": ws.title, "type": "resource_table",
            "resources": resources, "total_cell": "F4"
        }
        logger.info(f"[GEM] Schema extracted: '{ws.title}'")
    else:
        logger.warning("[GEM] 'Resource Cost' sheet not found")

    ws = _find_sheet(wb, "Costing")
    if ws:
        costing_fields = []
        for row in range(1, ws.max_row + 1):
            label = ws[f"C{row}"].value or ws[f"D{row}"].value
            if label:
                costing_fields.append({
                    "label": str(label).strip(), "row": row,
                    "input_cell": f"G{row}", "value": None
                })
        master_schema["sheets"]["Costing"] = {
            "sheet_name": ws.title, "type": "financial_model", "fields": costing_fields
        }
        logger.info(f"[GEM] Schema extracted: '{ws.title}'")
    else:
        logger.warning("[GEM] 'Costing' sheet not found")

    wb.close()

    os.makedirs("gem-testing", exist_ok=True)
    with open("gem-testing/master_schema.json", "w", encoding="utf-8") as f:
        json.dump(master_schema, f, indent=4)

    return master_schema


# ---------------------------------------------------------------------------
# LLM Processing
# ---------------------------------------------------------------------------

def LLM_processing(text: str) -> Dict:
    schema = extract_master_schema_json()
    chunks = chunk_text(text, chunk_size=180000, overlap=500)

    if not chunks:
        logger.error("[GEM] No chunks created from text")
        return {}
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0,
    )

    initial_state: ChunkProcessingState = {
        "all_chunks":          chunks,
        "current_chunk_index": 0,
        "accumulated_results": {},
        "master_schema":       schema,
        "chunk_results":       [],
        "is_complete":         False,
    }

    graph  = create_chunk_processing_workflow()
    config = {"configurable": {"thread_id": f"gem-session-{uuid4().hex[:8]}"}}

    logger.info(f"[GEM] Starting chunk processing Ã¢â‚¬â€ {len(chunks)} chunks")

    try:
        final_state = None
        for state in graph.stream(initial_state, config):
            if "__end__" in state:
                final_state = state["__end__"]
                break
            if "process_chunk" in state:
                final_state = state["process_chunk"]
        if final_state is None:
            final_state = initial_state
    except Exception as e:
        logger.error(f"[GEM] Workflow failed: {e}")
        final_state = initial_state

    cell_map = final_state.get("accumulated_results", {})

    os.makedirs("gem-testing", exist_ok=True)
    with open("gem-testing/final_cell_map.json", "w", encoding="utf-8") as f:
        json.dump(cell_map, f, indent=4)

    logger.info("[GEM] Chunk processing complete.")
    return cell_map


# ---------------------------------------------------------------------------
# Resource Cost sheet helper functions
# ---------------------------------------------------------------------------

def _find_resource_total_row(ws, start_row: int = 2) -> int:
    """Find the row containing 'Total' in Resource Cost sheet."""
    for row in range(start_row, ws.max_row + 50):
        e_val = ws[f"E{row}"].value
        f_val = ws[f"F{row}"].value
        if isinstance(e_val, str) and e_val.strip().lower() == "total":
            return row
        if isinstance(f_val, str) and "SUM(" in f_val.upper():
            return row
    return 0


def _copy_row_style(ws, source_row: int, target_row: int, min_col: int = 1, max_col: int = 6):
    """Copy formatting from one row to another."""
    for col in range(min_col, max_col + 1):
        src = ws.cell(row=source_row, column=col)
        dst = ws.cell(row=target_row, column=col)
        if src.has_style:
            dst.font = copy(src.font)
            dst.border = copy(src.border)
            dst.fill = copy(src.fill)
            dst.number_format = copy(src.number_format)
            dst.protection = copy(src.protection)
            dst.alignment = copy(src.alignment)


def _write_resource_cost_sheet(ws, cells: Any) -> int:
    """Write Resource Cost data with dynamic row insertion and formula preservation."""
    parsed_rows: Dict[int, Dict[str, Any]] = {}

    if isinstance(cells, list):
        for idx, item in enumerate(cells, start=1):
            row = idx + 1
            parsed_rows[row] = {
                "B": item.get("profile"),
                "C": item.get("qualification"),
                "D": item.get("qty"),
                "E": item.get("per_month"),
                "F": item.get("for_12_month"),
            }
    else:
        for cell_addr, value in (cells or {}).items():
            match = re.fullmatch(r"([A-Za-z]+)(\d+)", str(cell_addr).strip())
            if not match:
                continue
            col = match.group(1).upper()
            row = int(match.group(2))
            if row < 2 or col not in {"A", "B", "C", "D", "E", "F"}:
                continue
            parsed_rows.setdefault(row, {})[col] = value

    source_rows = [
        row_num
        for row_num in sorted(parsed_rows.keys())
        if any(parsed_rows[row_num].get(col) is not None for col in ("A", "B", "C", "D", "E", "F"))
    ]
    if not source_rows:
        return 0

    data_start_row = 2
    total_row = _find_resource_total_row(ws, start_row=data_start_row)
    if total_row <= 0:
        total_row = max(ws.max_row + 1, data_start_row + 1)
        ws[f"E{total_row}"] = "Total"

    existing_capacity = max(0, total_row - data_start_row)
    required_rows = len(source_rows)
    if required_rows > existing_capacity:
        extra_rows = required_rows - existing_capacity
        ws.insert_rows(total_row, amount=extra_rows)
        for new_row in range(total_row, total_row + extra_rows):
            _copy_row_style(ws, source_row=data_start_row, target_row=new_row, min_col=1, max_col=6)
        total_row += extra_rows

    for offset, src_row in enumerate(source_rows):
        target_row = data_start_row + offset
        row_data = parsed_rows[src_row]

        ws[f"A{target_row}"] = offset + 1
        ws[f"B{target_row}"] = row_data.get("B")
        ws[f"C{target_row}"] = row_data.get("C")
        ws[f"D{target_row}"] = row_data.get("D")
        ws[f"E{target_row}"] = row_data.get("E")
        ws[f"F{target_row}"] = f"=D{target_row}*E{target_row}*12"

    first_data_row = data_start_row
    last_data_row = data_start_row + required_rows - 1
    ws[f"E{total_row}"] = "Total"
    ws[f"F{total_row}"] = f"=SUM(F{first_data_row}:F{last_data_row})"

    return required_rows * 6


# ---------------------------------------------------------------------------
# Excel writer
# ---------------------------------------------------------------------------

def write_to_excel(cell_map: Dict) -> str:
    import shutil
    import openpyxl

    source_file = r"templates/template.xlsx"
    output_dir  = r"templates/outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"output_{uuid4().hex[:8]}.xlsx")

    shutil.copy(source_file, output_file)
    wb = openpyxl.load_workbook(output_file)

    for target_sheet_name, cells in cell_map.items():
        ws = _find_sheet(wb, target_sheet_name)
        if ws is None:
            logger.warning(f"[GEM] Sheet '{target_sheet_name}' not found Ã¢â‚¬â€ skipping")
            continue

        written = 0
        success = False

        # Add Status and Remarks headers for Asset Details sheet
        if target_sheet_name.strip().lower() == "asset details":
            if ws[f"F2"].value is None:
                ws[f"F2"] = "Status"
            if ws[f"G2"].value is None:
                ws[f"G2"] = "Remarks"
            logger.info(f"[GEM] Added Status and Remarks headers to '{target_sheet_name}'")

        # Unwrap {"rows": [...]} if LLM accidentally wrapped it
        if isinstance(cells, dict) and set(cells.keys()) == {"rows"} and isinstance(cells.get("rows"), list):
            logger.info(f"[GEM] Unwrapping 'rows' wrapper for '{target_sheet_name}'")
            cells = cells["rows"]

        # Resource Cost requires dynamic row insertion so total/formula rows remain intact.
        if (
            isinstance(cells, list)
            and target_sheet_name.strip().lower() == "resource cost"
        ):
            try:
                written = _write_resource_cost_sheet(ws, cells)
                success = True
                logger.info(
                    f"[GEM] Wrote '{target_sheet_name}' - dynamic resource table, {written} cells"
                )
            except Exception as e:
                logger.error(f"[GEM] Resource Cost write failed for '{target_sheet_name}': {e}")
                success = False

        # Attempt 1: Raw list-of-dicts write for asset-style tables
        if not success and isinstance(cells, list):
            try:
                start_row = 3

                for idx, row_data in enumerate(cells, start=1):
                    row_num = start_row + idx - 1

                    ws[f"A{row_num}"] = idx
                    ws[f"B{row_num}"] = row_data.get("asset_type")
                    ws[f"C{row_num}"] = row_data.get("brand")
                    ws[f"D{row_num}"] = row_data.get("details")
                    ws[f"E{row_num}"] = row_data.get("qty")
                    
                    # Add validation status and remarks columns
                    status = row_data.get("status")
                    remarks = row_data.get("remarks")
                    ws[f"F{row_num}"] = status if status else ""
                    ws[f"G{row_num}"] = remarks if remarks else ""

                written = len(cells) * 7  # 7 columns now (A-G)
                success = True

                logger.info(
                    f"[GEM] RAW WRITE '{target_sheet_name}' — "
                    f"{len(cells)} rows written ({written} cells)"
                )

            except Exception as e:
                logger.error(f"[GEM] Raw write failed for '{target_sheet_name}': {e}")
                success = False

        if (
            not success
            and isinstance(cells, dict)
            and target_sheet_name.strip().lower() == "resource cost"
        ):
            try:
                written = _write_resource_cost_sheet(ws, cells)
                success = True
                logger.info(
                    f"[GEM] Wrote '{target_sheet_name}' - dynamic resource table, {written} cells"
                )
            except Exception as e:
                logger.error(f"[GEM] Resource Cost write failed for '{target_sheet_name}': {e}")
                success = False

        # Attempt 2: Standard dict {cell: value}
        if not success and isinstance(cells, dict):
            try:
                for cell_addr, value in cells.items():
                    if value is None:
                        continue
                    ws[cell_addr] = value
                    written += 1
                success = True
                logger.info(f"[GEM] Wrote '{target_sheet_name}' Ã¢â‚¬â€ dict, {written} cells")
            except Exception as e:
                logger.error(f"[GEM] Dict write failed for '{target_sheet_name}': {e}")

        # Attempt 3: Flatten list-of-dicts only for actual cell maps
        if not success and isinstance(cells, list):
            try:
                flat = {}
                for item in cells:
                    if isinstance(item, dict):
                        flat.update(item)
                if not flat:
                    raise ValueError("list payload did not contain any dict values")
                cell_addr_pattern = re.compile(r"^[A-Z]+[0-9]+$", re.IGNORECASE)
                invalid_keys = [key for key in flat.keys() if not cell_addr_pattern.match(str(key))]
                if invalid_keys:
                    raise ValueError(
                        "list payload is row data, not cell map; invalid keys: "
                        + ", ".join(str(key) for key in invalid_keys[:5])
                    )
                for cell_addr, value in flat.items():
                    if value is None:
                        continue
                    ws[cell_addr] = value
                    written += 1
                success = True
                logger.info(f"[GEM] Wrote '{target_sheet_name}' Ã¢â‚¬â€ flattened, {written} cells")
            except Exception as e:
                logger.error(f"[GEM] All write attempts failed for '{target_sheet_name}': {e}")

        if not success:
            logger.error(
                f"[GEM] FAILED '{target_sheet_name}' | type={type(cells)} | "
                f"preview={str(cells)[:300]}"
            )

        logger.info(f"[GEM] Total written to '{ws.title}': {written} cells")

    wb.save(output_file)
    wb.close()
    logger.info(f"[GEM] Excel saved Ã¢â€ â€™ {output_file}")
    return output_file


from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from rapidfuzz import fuzz

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHUNK_SIZE      = 100_000
CHUNK_OVERLAP   = 3_000
FUZZY_THRESHOLD = 85


# ---------------------------------------------------------------------------
# Smart chunker
# ---------------------------------------------------------------------------

def _smart_split(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start  = 0

    while start < len(text):
        end = start + size

        if end >= len(text):
            chunks.append(text[start:])
            break

        search_start = max(start, end - 2000)
        segment      = text[search_start:end]
        blank_pos    = segment.rfind("\n\n")

        if blank_pos != -1:
            cut = search_start + blank_pos + 2
        else:
            newline_pos = segment.rfind("\n")
            cut = (search_start + newline_pos + 1) if newline_pos != -1 else end

        chunks.append(text[start:cut])
        start = max(start + 1, cut - overlap)

    return chunks


# ---------------------------------------------------------------------------
# Fuzzy dedup
# ---------------------------------------------------------------------------

def _fuzzy_dedup(assets: List[dict]) -> List[dict]:
    groups: dict[tuple, List[int]] = {}
    for i, item in enumerate(assets):
        key = (
            (item.get("asset_type") or "").strip().upper(),
            (item.get("brand")      or "").strip().upper(),
        )
        groups.setdefault(key, []).append(i)

    canonical: List[dict] = []

    for _, indices in groups.items():
        merged_indices: List[List[int]] = []

        for idx in indices:
            bare    = assets[idx].get("model") or ""
            matched = False

            for group in merged_indices:
                rep_bare = assets[group[0]].get("model") or ""
                if fuzz.token_sort_ratio(bare, rep_bare) >= FUZZY_THRESHOLD:
                    group.append(idx)
                    matched = True
                    break

            if not matched:
                merged_indices.append([idx])

        for group in merged_indices:
            base = dict(assets[group[0]])
            for idx in group[1:]:
                other  = assets[idx]
                eq, nq = base.get("qty"), other.get("qty")
                base["qty"] = (
                    None if (eq is None and nq is None)
                    else (eq or 0) + (nq or 0)
                )
                if len(other.get("details") or "") > len(base.get("details") or ""):
                    base["details"] = other["details"]
            canonical.append(base)

    for i, item in enumerate(canonical, start=1):
        item["sr_no"] = i

    return canonical


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """You are an expert tender document analyst specializing in IT hardware procurement tenders in India (GeM Portal, government PSUs, banks, central government undertakings).

Your task is to extract Asset Details and Resource Cost data from the provided tender document chunk and return it as strict JSON — no markdown, no code fences, no explanation, no extra wrapper keys.

---

## OUTPUT FORMAT (return EXACTLY this structure, nothing else):

{{
  "Asset Details": [
    {{
      "sr_no": <integer, 1-based sequential>,
      "asset_type": <string>,
      "brand": <string or null>,
      "model": <string or null>,
      "details": <string or null>,
      "qty": <integer or null>
    }}
  ],
  "Resource Cost": [
    {{
      "sr_no": <integer, 1-based sequential>,
      "profile": <string>,
      "qualification": <string or null>,
      "qty": <integer or null>,
      "per_month": <number or null>,
      "for_12_month": <number or null>
    }}
  ]
}}

---

## STRICT ASSET WHITELIST — EXTRACT ONLY THESE CATEGORIES:

You must ONLY extract assets that belong to the following IT and office hardware categories.
If an item does not clearly belong to one of these categories → SKIP IT entirely.

ALLOWED categories (extract these):
- Computers / Desktops / All-in-One PCs
- Laptops / Notebooks / Ultrabooks
- Servers (rack, tower, blade)
- Printers (laser, inkjet, dot matrix, thermal, multifunction/MFP)
- Scanners (flatbed, document, handheld)
- Photocopiers / Reprographic machines
- UPS (Uninterruptible Power Supply)
- Networking equipment: Switches (managed/unmanaged), Routers, Firewalls, Access Points, Load Balancers
- Storage devices: NAS, SAN, external HDD, tape libraries
- Monitors / Display screens
- Projectors / Interactive flat panels / Smart boards
- KVM switches
- Racks and cabinets (server/network racks)
- Thin clients / Zero clients
- Tablets / iPads (only if listed as IT assets in a BOQ)
- Biometric devices / Fingerprint scanners / Attendance systems
- CCTV cameras / IP cameras / DVR / NVR (only if listed in IT/security BOQ)
- Video conferencing systems / Webcams / Conference phones
- External storage: pen drives, hard disks (only if in BOQ with qty)
- POS terminals / Barcode scanners (only if in IT BOQ)
- Power Distribution Units (PDU)
- Patch panels / Structured cabling (only if in BOQ with qty)
- Any other clearly identifiable IT or computer hardware with a brand, model, and qty

STRICTLY FORBIDDEN — never extract these even if they appear in the document:
- Military / defence / weapons equipment of any kind
- Naval vessels, ships, boats, submarines, aircraft
- Aircraft engines, generators, flight deck equipment, radars, navigation systems
- Tanks, armoured vehicles, all-terrain vehicles
- Communication jamming systems, encryption military devices, battlefield systems
- Drones / UAVs (unless clearly a commercial IT asset with brand/model in BOQ)
- Weapons, ammunition, helmets, jackets, uniforms, clothing
- Medical equipment, surgical instruments
- Industrial machinery: cranes, compressors, pumps, dehumidifiers
- Marine equipment: RO plants, propulsion machinery, underwater systems
- Vehicles of any kind
- Civil / construction / engineering materials
- Furniture, fixtures, fittings
- Consumables: toner, ink, paper, cables, cleaning supplies
- Software, licenses, subscriptions
- AMC / CAMC service charges
- Manpower / staffing (those go in Resource Cost)

---

## EXHAUSTIVE EXTRACTION — COMPLETENESS RULES:

You MUST extract EVERY qualifying IT hardware item visible in this chunk. Do not stop early.
This is one segment of a larger document — extract everything present in this segment.

NOTE: This chunk may begin mid-table. If you see rows that clearly belong to a hardware
table even without a visible header, extract them using context clues (columns that look
like brand, model, qty, specs).

---

## CORE CONCEPT — ONE ROW PER UNIQUE VARIANT:

A "variant" = unique combination of asset_type + brand + model + all technical specs.

- Same asset_type + same brand + same model + same specs → SAME variant → SUM qty
- Any spec differs → DIFFERENT variant → separate row

---

## MODEL FIELD RULES:

The `model` field must contain the model / part number exactly as it appears in the document,
with ONLY label prefix noise removed.

- Strip ONLY these leading label words (case-insensitive):
  "Model", "Model No", "Model No.", "Model Number",
  "Part No", "Part No.", "Part Number", "SKU", "Code"
  followed by an optional colon or dash.
- Do NOT strip the brand name from the model string — brand is captured separately.
- Do NOT abbreviate, summarise, or invent a model number.
- Examples:
    "Model No. Z4665G"           → model: "Z4665G"
    "Model: HP EliteBook 840 G9" → model: "HP EliteBook 840 G9"
    "SKU: ABC-123-XYZ"           → model: "ABC-123-XYZ"
    Not stated                   → model: null
- If no model / part number is present → model: null

---

## TWO-PASS READING — CRITICAL:

LEVEL 1: Summary BOQ — item name + total qty (e.g. "Laptops — 92")
LEVEL 2: Detail sub-table — individual variants with brand, model, specs, per-variant qty

- Level 2 exists with per-variant qty → one row per variant, use sub-table qty
- Level 2 exists but no per-variant qty → one row per variant, qty = null
- Only Level 1 exists → one row, brand if mentioned, model = null, details = null, qty from Level 1
- NEVER skip sub-tables

---

## QUANTITY RULES:
- Use qty exactly as written
- "as required" / "as per actual" / "to be confirmed" → qty = null
- Range given → use lower bound
- Confirmed duplicate after normalization → SUM quantities

---

## BRAND RULES:
- Manufacturer name only (e.g. HP, Dell, Lenovo, Cisco, APC)
- Not stated or unknown → brand = null

---

## DETAILS FIELD RULES:
- Capture ALL useful identifying detail present for the asset.
- Include technical specs when present: CPU, RAM, Storage, Display, OS, speed, interface, print type, scan type, etc.
- If the row only provides a model / part number and no other specs, use that model / part number text in `details` instead of null.
- If the row provides brand + model in one combined string, keep the identifying model text in `details`.
- Prefer fuller identifying text over null whenever any model/spec text is present.
- Format examples:
  "CPU: Intel i5-1235U | RAM: 8GB DDR4 | Storage: 512GB SSD | Display: 15.6 inch | OS: Windows 11"
  "Model: ACER VERITON Z4665G"
  "Model: HP LaserJet Pro M403DN"
- Only include attributes actually present — skip blank columns
- No qty in details
- Use `details = null` only when absolutely no model text and no specs are present

---

## ASSET TYPE RULES:
- Category name only — never include brand/model words
- Wrong: "HP Laptop", "Canon LaserJet" → Right: "Laptop", "Printer"
- Preserve the tender's own terminology exactly if it is a valid category name

---

## RESOURCE COST RULES:

Extract named manpower roles with qty. Look in: Manpower sections, Staffing annexures,
Scope of Work ("bidder shall deploy N engineers"), Pre-Qualification criteria.

- qty = null if not an explicit integer
- qualification = null if not stated
- per_month / for_12_month = null unless explicit INR figure — NEVER calculate
- Nothing found → "Resource Cost": []

---

## ABSOLUTE RULES:
1. Return ONLY the JSON object — no preamble, no markdown, no code fences
2. qty and cost fields = JSON numbers, not strings
3. sr_no = 1-based sequential within this chunk
4. Do NOT fabricate — only extract what is explicitly present
5. Do NOT skip sub-tables
6. Do NOT merge variants with different specs
7. DO merge confirmed duplicates (same asset_type + brand + model + specs) — sum their qty
8. Zero assets + zero resources → {{"Asset Details": [], "Resource Cost": []}}

---

DOCUMENT CHUNK:
{text}"""

_PROMPT = ChatPromptTemplate.from_template(_PROMPT_TEMPLATE)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def call2_LLM_processing(sub_pdf_texts: List[str]) -> Dict:
    if not sub_pdf_texts:
        logger.info("[GEM] No sub PDF texts for call2")
        return {}

    combined_text = "\n\n--- SUB PDF ---\n\n".join(sub_pdf_texts)
    logger.info(f"[GEM][CALL2] Combined text: {len(combined_text)} chars")

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-pro-preview",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0,
    )

    chain = _PROMPT | llm

    chunks = _smart_split(combined_text)
    logger.info(f"[GEM][CALL2] Split into {len(chunks)} chunk(s)")

    # ------------------------------------------------------------------
    # Process one chunk
    # ------------------------------------------------------------------
    def process_chunk(idx: int, chunk: str):
        try:
            response = chain.invoke({"text": chunk})
            content = response.content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        text = item.get("text")
                        if text:
                            parts.append(str(text))
                    else:
                        text = getattr(item, "text", None)
                        if text:
                            parts.append(str(text))
                raw = "\n".join(part.strip() for part in parts if str(part).strip())
            else:
                raw = str(content or "").strip()

            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            chunk_result = json.loads(raw)
            assets    = chunk_result.get("Asset Details", [])
            resources = chunk_result.get("Resource Cost", [])
            assets    = assets    if isinstance(assets,    list) else []
            resources = resources if isinstance(resources, list) else []

            logger.info(f"[GEM][CALL2] Chunk {idx}: {len(assets)} assets, {len(resources)} resources")
            return idx, assets, resources

        except json.JSONDecodeError as e:
            logger.error(f"[GEM][CALL2] Chunk {idx} JSON parse failed: {e}")
            return idx, [], []
        except Exception as e:
            logger.error(f"[GEM][CALL2] Chunk {idx} LLM call failed: {e}")
            return idx, [], []

    # ------------------------------------------------------------------
    # Parallel execution
    # ------------------------------------------------------------------
    chunk_results: Dict[int, tuple] = {}

    with ThreadPoolExecutor(max_workers=min(len(chunks), 10)) as executor:
        futures = {
            executor.submit(process_chunk, idx, chunk): idx
            for idx, chunk in enumerate(chunks, start=1)
        }
        for future in as_completed(futures):
            idx, assets, resources = future.result()
            chunk_results[idx] = (assets, resources)

    # ------------------------------------------------------------------
    # Merge in document order
    # ------------------------------------------------------------------
    all_assets:    List[dict] = []
    all_resources: List[dict] = []

    for idx in sorted(chunk_results):
        assets, resources = chunk_results[idx]
        all_assets.extend(assets)
        all_resources.extend(resources)

    logger.info(
        f"[GEM][CALL2] Total before dedup — "
        f"assets: {len(all_assets)}, resources: {len(all_resources)}"
    )

    # ------------------------------------------------------------------
    # Fuzzy dedup assets
    # ------------------------------------------------------------------
    deduped_assets = _fuzzy_dedup(all_assets)

    # Simple exact dedup for resources by profile name
    seen_profiles: dict[str, bool] = {}
    deduped_resources: List[dict]  = []
    for item in all_resources:
        key = (item.get("profile") or "").strip().upper()
        if key not in seen_profiles:
            seen_profiles[key] = True
            item["sr_no"] = len(deduped_resources) + 1
            deduped_resources.append(item)

    logger.info(
        f"[GEM][CALL2] After dedup — "
        f"assets: {len(deduped_assets)}, resources: {len(deduped_resources)}"
    )

    # ------------------------------------------------------------------
    # Validate assets against combined text
    # ------------------------------------------------------------------
    logger.info(f"[GEM][CALL2] Running validation on {len(deduped_assets)} assets")
    validated_assets = validate_assets(deduped_assets, combined_text)

    result = {
        "Asset Details": validated_assets,
        "Resource Cost": deduped_resources,
    }

    os.makedirs("gem-testing", exist_ok=True)
    with open("gem-testing/call2_raw_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    logger.info(f"[GEM][CALL2] Done. Keys: {list(result.keys())}")
    return result

def validate_assets(assets: List[dict], combined_text: str) -> List[dict]:

    if not assets:
        logger.info("[GEM][VALIDATE] No assets to validate")
        return assets

    if not combined_text:
        logger.warning("[GEM][VALIDATE] No source text available — skipping validation")
        for item in assets:
            item["status"]  = "NOT_VALIDATED"
            item["remarks"] = "Source text not available for validation."
        return assets

    source_chunks = _smart_split(combined_text)
    logger.info(f"[GEM][VALIDATE] Validating {len(assets)} assets across {len(source_chunks)} chunk(s)")

    flash_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0,
    )
    validator_chain = _VALIDATOR_PROMPT | flash_llm

    def _validate_chunk(chunk_idx: int, chunk: str) -> tuple:
        try:
            response = validator_chain.invoke({
                "asset_json":  json.dumps(assets, ensure_ascii=False),
                "source_chunk": chunk,
            })
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            return chunk_idx, json.loads(raw)

        except Exception as e:
            logger.error(f"[GEM][VALIDATE] Chunk {chunk_idx} failed: {e}")
            return chunk_idx, []

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(_validate_chunk, idx, chunk): idx
            for idx, chunk in enumerate(source_chunks, start=1)
        }
        chunk_results: Dict[int, list] = {}
        for future in as_completed(futures):
            idx, result = future.result()
            chunk_results[idx] = result

    # Merge results across chunks per asset — VERIFIED wins, else QTY_MISMATCH, else OCR_ERROR, else NOT_FOUND
    per_asset: Dict[int, List[dict]] = {asset["sr_no"]: [] for asset in assets}
    for idx in sorted(chunk_results):
        for item in chunk_results[idx]:
            sr_no = item.get("sr_no")
            if sr_no in per_asset:
                per_asset[sr_no].append(item)

    for asset in assets:
        results = per_asset[asset["sr_no"]]

        verified_found = any(r.get("status") == "VERIFIED"      for r in results)
        mismatch_found = any(r.get("status") == "QTY_MISMATCH"  for r in results)
        ocr_found      = any(r.get("status") == "OCR_ERROR"     for r in results)

        if verified_found:
            asset["status"]  = "VERIFIED"
            asset["remarks"] = None

        elif mismatch_found:
            remarks = [r.get("remarks") for r in results if r.get("status") == "QTY_MISMATCH" and r.get("remarks")]
            asset["status"]  = "QTY_MISMATCH"
            asset["remarks"] = " | ".join(remarks) if remarks else "Qty mismatch detected. Please double check."

        elif ocr_found:
            remarks = [r.get("remarks") for r in results if r.get("status") == "OCR_ERROR" and r.get("remarks")]
            asset["status"]  = "OCR_ERROR"
            asset["remarks"] = " | ".join(remarks) if remarks else "OCR/parsing issue detected. Please double check."

        else:
            asset["status"]  = "NOT_FOUND"
            asset["remarks"] = "Asset not found in any source chunk."

    logger.info(
        f"[GEM][VALIDATE] Done — "
        f"VERIFIED={sum(1 for a in assets if a['status']=='VERIFIED')} "
        f"QTY_MISMATCH={sum(1 for a in assets if a['status']=='QTY_MISMATCH')} "
        f"NOT_FOUND={sum(1 for a in assets if a['status']=='NOT_FOUND')} "
        f"OCR_ERROR={sum(1 for a in assets if a['status']=='OCR_ERROR')}"
    )
    return assets

_VALIDATOR_PROMPT_TEMPLATE = """You are a strict auditor validating one extracted IT hardware asset against a chunk of the original tender document source text.

You will be given:
1. ONE extracted asset row (JSON) — this is what the extraction model produced
2. ONE chunk of the original source document text (you will receive multiple chunks)

Your job: find this asset in the source chunk and verify if the extracted qty is correct.

---

## CRITICAL CONTEXT — READ BEFORE VALIDATING:

The same asset (same brand + model) may appear MULTIPLE TIMES across different tables,
annexures, or sections in the source document — for example once in a summary BOQ table
and again in a detailed asset list. The extraction model has already summed all occurrences
into ONE row with a combined qty.

So when you search the source chunk:
- Find ALL occurrences of this asset in this chunk
- Sum their individual qtys
- Compare that sum to the extracted qty

If the chunk only contains some occurrences (others are in different chunks), you will
return NOT_FOUND_IN_CHUNK so the caller knows to check other chunks too.

---

## OUTPUT FORMAT (return EXACTLY this, nothing else):

{{
  "status": <string>,
  "remarks": <string or null>
}}

---

## STATUS VALUES — pick exactly one:

- "VERIFIED"           → found in this chunk, qty matches (or sums to) extracted qty
- "QTY_MISMATCH"       → found in this chunk, but qty does NOT match — explain discrepancy
- "NOT_FOUND_IN_CHUNK" → this asset is not mentioned in this chunk at all — caller will check other chunks
- "OCR_ERROR"          → asset found but surrounding text is garbled / columns are shifted / numbers are corrupted

---

## REMARKS RULES:

- "VERIFIED" → remarks = null

- "QTY_MISMATCH" → Write in a clear, client-friendly way. Structure it as:
    "We found [asset] in the source document but the quantity doesn't add up.
    Here's what we saw: [describe each occurrence in plain English — e.g. 'Row 12 in the BOQ table lists qty 5', 'Row 34 in the Annexure A lists qty 3'].
    That gives us a source total of X, but the extracted figure is Y.
    This discrepancy may be because [plain English reason — e.g. 'two different variants appear to have been grouped under one line item', 'the summary table and the detail table show different numbers'].
    👉 We recommend opening the original document and checking [specific table/section/row] to confirm the correct quantity before proceeding."

- "NOT_FOUND_IN_CHUNK" → remarks = null

- "OCR_ERROR" → Write in a clear, client-friendly way. Structure it as:
    "We found a reference to [asset] in the source document, but the surrounding data appears to be corrupted or misaligned — likely due to a PDF scanning or OCR issue.
    Specifically, near [describe location in plain English — e.g. 'Row 8 of the hardware table on what appears to be page 3'], the text reads '[short garbled snippet]' which suggests the columns may have shifted or a digit was misread.
    👉 We recommend opening the original document and visually inspecting this section to retrieve the correct quantity."

- "NOT_FOUND" → Write in a clear, client-friendly way. Structure it as:
    "We were unable to locate [asset_type] ([brand] if available) anywhere in the source document.
    This could mean it was missed during extraction, the item description in the document uses different terminology, or the OCR did not capture it correctly.
    👉 We recommend searching the original document manually for this item to confirm whether it is present and what quantity is listed."

## AMBIGUITY RULES — handle these generically:

- If multiple assets share a single line item with one qty → status = "QTY_MISMATCH", explain in plain English that the document groups multiple items under one entry making it impossible to confirm individual quantities, recommend the client verify each item separately.
- If a qty appears as a range (e.g. "1-2") → status = "QTY_MISMATCH", mention the range found and that the lower bound was used.
- If qty column exists but is blank → status = "QTY_MISMATCH", mention the column was found but empty.
- If same item appears in both a summary table and a detail table with different qtys → status = "QTY_MISMATCH", clearly distinguish which table showed which number and recommend trusting the detail table.
- If item is found but has no qty anywhere nearby → status = "QTY_MISMATCH", mention item was located but no quantity could be confirmed from the surrounding text.
- NEVER speculate about intent — only describe what is literally present in the document.
---

## HOW TO SEARCH (follow in order):

1. Search for the brand name (case-insensitive) anywhere in the chunk
2. Near each brand match, look for the model number — first strip label noise:
   remove "Model", "Model No.", "Model No", "Model Number", "Part No.", "SKU", "Code" prefixes
   then compare the bare model string (case-insensitive, ignore extra spaces)
3. Also search by asset_type if brand+model search yields nothing
4. For every match found, read the qty from that row — look for columns named
   "Qty", "Quantity", "Nos", "No.", "Count", or a standalone number in the same row
5. If multiple matches found in this chunk → sum all their qtys
6. Compare sum to extracted qty

---

## ABSOLUTE RULES:
1. Return ONLY the JSON object — no preamble, no markdown, no code fences
2. NEVER fabricate quantities — only use numbers explicitly written in the source chunk
3. NEVER mark as VERIFIED if you are not 100% certain the qty matches
4. If unsure between VERIFIED and QTY_MISMATCH → always choose QTY_MISMATCH with your reasoning in remarks
5. If asset brand/model is simply not in this chunk → NOT_FOUND_IN_CHUNK (not an error, just means check other chunks)
6. Return ONLY the JSON array — no preamble, no markdown, no code fences
7. You MUST return an entry for EVERY asset in the input JSON — no skipping
8. sr_no MUST match exactly the sr_no from the input JSON — never renumber, never change
9. If asset is not found in all the chunks → return {{"sr_no": <original_sr_no>, "status": "NOT_FOUND_IN_CHUNK", "remarks": null}}
10. Total items in output array MUST equal total items in input array

---

EXTRACTED ASSET:
{asset_json}

SOURCE CHUNK:
{source_chunk}"""

_VALIDATOR_PROMPT = ChatPromptTemplate.from_template(_VALIDATOR_PROMPT_TEMPLATE)
