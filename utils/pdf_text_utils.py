from dataclasses import dataclass
from contextlib import redirect_stdout
import io
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.parse import unquote, urlparse

import fitz
from PIL import Image


@dataclass
class PdfLinkTarget:
    target: str
    page_number: int
    kind: str
    is_remote: bool

# Utility to check if PDF has selectable text
def pdf_has_selectable_text(pdf_path):
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text = page.get_text().strip()
                if text:
                    return True
        return False
    except Exception:
        return False

# Utility to extract all selectable text from PDF
def extract_pdf_text(pdf_path):
    sections = []
    try:
        with fitz.open(pdf_path) as doc:
            for page_number, page in enumerate(doc, start=1):
                text = page.get_text()
                page_sections = []
                if text:
                    page_sections.append(text.strip())

                table_text = _extract_page_tables_text(page, page_number)
                if table_text:
                    page_sections.append(table_text)

                if page_sections:
                    sections.append("\n\n".join(page_sections))
        return "\n\n".join(sections).strip()
    except Exception:
        return "\n\n".join(sections).strip()


def _extract_page_tables_text(page, page_number: int) -> str:
    try:
        with redirect_stdout(io.StringIO()):
            table_finder = page.find_tables()
    except Exception:
        return ""

    table_sections = []
    for table_index, table in enumerate(getattr(table_finder, "tables", []) or [], start=1):
        try:
            rows = table.extract()
        except Exception:
            continue

        normalized_rows = [
            ["" if cell is None else str(cell).replace("\n", " ").strip() for cell in row]
            for row in (rows or [])
            if row
        ]
        normalized_rows = [row for row in normalized_rows if any(row)]
        if not normalized_rows:
            continue

        max_cols = max(len(row) for row in normalized_rows)
        padded_rows = [row + [""] * (max_cols - len(row)) for row in normalized_rows]
        header = padded_rows[0]
        body = padded_rows[1:]

        markdown_rows = [
            "| " + " | ".join(_escape_markdown_table_cell(cell) for cell in header) + " |",
            "| " + " | ".join("---" for _ in header) + " |",
        ]
        markdown_rows.extend(
            "| " + " | ".join(_escape_markdown_table_cell(cell) for cell in row) + " |"
            for row in body
        )
        table_sections.append(
            f"--- EXTRACTED TABLE {table_index} ON PAGE {page_number} ---\n"
            + "\n".join(markdown_rows)
        )

    return "\n\n".join(table_sections)


def _escape_markdown_table_cell(value: str) -> str:
    return str(value).replace("|", "\\|")


def extract_pdf_link_targets(
    pdf_path: str,
    allowed_extensions: Optional[Iterable[str]] = None,
) -> List[PdfLinkTarget]:
    discovered: List[PdfLinkTarget] = []
    seen = set()
    allowed_exts = {
        str(ext).strip().lower()
        for ext in (allowed_extensions or [])
        if str(ext).strip()
    }

    try:
        base_dir = Path(pdf_path).resolve().parent
        with fitz.open(pdf_path) as doc:
            for page_number, page in enumerate(doc, start=1):
                for raw_link in page.get_links() or []:
                    normalized_target = _normalize_pdf_link_target(raw_link, base_dir)
                    if not normalized_target:
                        continue
                    if not _is_supported_link_target(normalized_target, allowed_exts):
                        continue
                    key = normalized_target.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    discovered.append(
                        PdfLinkTarget(
                            target=normalized_target,
                            page_number=page_number,
                            kind=_describe_pdf_link(raw_link),
                            is_remote=_is_remote_target(normalized_target),
                        )
                    )
    except Exception:
        return discovered

    return discovered


def _normalize_pdf_link_target(raw_link: dict, base_dir: Path) -> str:
    uri_target = str(raw_link.get("uri") or "").strip()
    if uri_target:
        parsed = urlparse(uri_target)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            return uri_target
        if parsed.scheme == "file":
            local_path = unquote(parsed.path or "").lstrip("/")
            return str(Path(local_path).resolve()) if local_path else ""

    file_target = str(raw_link.get("file") or "").strip()
    if not file_target:
        return ""

    parsed_file = urlparse(file_target)
    if parsed_file.scheme in {"http", "https"} and parsed_file.netloc:
        return file_target

    if parsed_file.scheme == "file":
        file_target = unquote(parsed_file.path or "").lstrip("/")
    else:
        file_target = unquote(file_target.split("#", 1)[0].strip())

    if not file_target:
        return ""

    resolved_path = Path(file_target)
    if not resolved_path.is_absolute():
        resolved_path = (base_dir / resolved_path).resolve()
    else:
        resolved_path = resolved_path.resolve()
    return str(resolved_path)


def _is_supported_link_target(target: str, allowed_extensions: set) -> bool:
    if _is_remote_target(target):
        return True
    target_ext = Path(target).suffix.lower()
    if allowed_extensions:
        return target_ext in allowed_extensions
    return target_ext == ".pdf"


def _is_remote_target(target: str) -> bool:
    parsed = urlparse(str(target or "").strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _describe_pdf_link(raw_link: dict) -> str:
    if raw_link.get("uri"):
        return "uri"
    if raw_link.get("file"):
        return "file"
    return "unknown"
