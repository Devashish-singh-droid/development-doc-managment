import base64
import html
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, quote, unquote, urljoin, urlparse, urlencode, urlunparse

import requests
from bs4 import BeautifulSoup

from config import settings
from utils.logger import get_logger

logger = get_logger("remote_link_importer")

_REQUEST_TIMEOUT = (
    settings.get_int("REMOTE_IMPORT_CONNECT_TIMEOUT_SECONDS", 15),
    settings.get_int("REMOTE_IMPORT_READ_TIMEOUT_SECONDS", 180),
)
_MAX_FILES_PER_IMPORT = max(1, settings.get_int("REMOTE_IMPORT_MAX_FILES", 25))
_MAX_FILE_SIZE_BYTES = max(
    1,
    settings.get_int("REMOTE_IMPORT_MAX_FILE_SIZE_MB", 50) * 1024 * 1024,
)
_HTML_CONTENT_TYPES = ("text/html", "application/xhtml+xml")
_USER_AGENT = settings.get_string(
    "REMOTE_IMPORT_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AI-Doc-Management/1.0",
)

_CONTENT_TYPE_EXTENSIONS = {
    "application/pdf": ".pdf",
    "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.ms-word.document.macroenabled.12": ".docm",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.ms-excel.sheet.macroenabled.12": ".xlsm",
    "application/vnd.ms-excel.sheet.binary.macroenabled.12": ".xlsb",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/tiff": ".tiff",
}


class RemoteImportError(Exception):
    """Raised when a remote link cannot be resolved into importable files."""


@dataclass
class RemoteImportCandidate:
    filename: str
    download_url: str
    source_url: str
    drive_id: Optional[str] = None
    item_id: Optional[str] = None


@dataclass
class RemoteImportFile:
    filename: str
    temp_path: str
    source_url: str
    size_bytes: int


def import_documents_from_link(
    link: str,
    allowed_extensions: Iterable[str],
    graph_access_token: Optional[str] = None,
) -> Tuple[List[RemoteImportFile], List[str]]:
    """Resolve a public link into downloadable files and save them to temp storage."""
    allowed_exts = {ext.lower() for ext in allowed_extensions if str(ext).strip()}
    normalized_link = str(link or "").strip()
    parsed = urlparse(normalized_link)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RemoteImportError("Please provide a valid public HTTP or HTTPS link.")

    session = requests.Session()
    session.headers.update({"User-Agent": _USER_AGENT})
    try:
        candidates = _resolve_candidates(session, normalized_link, allowed_exts, graph_access_token)
        if not candidates:
            raise RemoteImportError("No supported documents were found at this link.")

        unique_candidates = _dedupe_candidates(candidates)
        truncated = len(unique_candidates) > _MAX_FILES_PER_IMPORT
        candidates = unique_candidates[:_MAX_FILES_PER_IMPORT]
        warnings: List[str] = []
        imported_files: List[RemoteImportFile] = []

        for candidate in candidates:
            try:
                imported_files.append(
                    _download_candidate(
                        session,
                        candidate,
                        allowed_exts,
                        access_token=graph_access_token,
                    )
                )
            except RemoteImportError as exc:
                warnings.append(f"{candidate.filename}: {exc}")
            except Exception as exc:
                logger.warning(f"Unexpected remote import failure for {candidate.download_url}: {exc}")
                warnings.append(f"{candidate.filename}: download failed")

        if not imported_files:
            failure_details = " ".join(warnings[:3]).strip()
            if failure_details:
                raise RemoteImportError(f"No supported documents could be downloaded. {failure_details}")
            raise RemoteImportError("No supported documents could be downloaded from this link.")

        if truncated:
            warnings.append(
                f"Only the first {_MAX_FILES_PER_IMPORT} supported files were imported from the link."
            )

        return imported_files, warnings
    finally:
        session.close()


def _resolve_candidates(
    session: requests.Session,
    url: str,
    allowed_extensions: set,
    graph_access_token: Optional[str] = None,
) -> List[RemoteImportCandidate]:
    if _looks_like_onedrive_or_sharepoint(url):
        onedrive_candidates = _resolve_onedrive_candidates(
            session,
            url,
            allowed_extensions,
            graph_access_token=graph_access_token,
        )
        if onedrive_candidates:
            return onedrive_candidates
        if graph_access_token:
            graph_path_candidate = _resolve_personal_drive_path_candidate(url, allowed_extensions)
            if graph_path_candidate:
                return [graph_path_candidate]

    html_candidates = _resolve_html_candidates(session, url, allowed_extensions)
    if html_candidates:
        return html_candidates

    return [
        RemoteImportCandidate(
            filename=_guess_filename_from_url(url) or "remote_document",
            download_url=url,
            source_url=url,
        )
    ]


def _resolve_onedrive_candidates(
    session: requests.Session,
    shared_url: str,
    allowed_extensions: set,
    graph_access_token: Optional[str] = None,
) -> List[RemoteImportCandidate]:
    graph_base = "https://graph.microsoft.com/v1.0"
    share_urls = []
    for candidate_url in [shared_url, _sharepoint_raw_file_url(shared_url)]:
        normalized_candidate = str(candidate_url or "").strip()
        if normalized_candidate and normalized_candidate not in share_urls:
            share_urls.append(normalized_candidate)

    item = None
    share_token = ""
    for share_url in share_urls:
        encoded = _encode_share_url(share_url)
        if not encoded:
            continue
        payload = _get_json(
            session,
            f"{graph_base}/shares/{encoded}/driveItem",
            access_token=graph_access_token,
        )
        if payload:
            item = payload
            share_token = encoded
            break

    if not item or not share_token:
        return []

    if item.get("file"):
        candidate = _graph_item_to_candidate(
            share_token,
            item,
            shared_url,
            drive_id=_extract_drive_id(item),
        )
        return [candidate] if candidate and _is_supported_filename(candidate.filename, allowed_extensions) else []

    if not item.get("folder"):
        return []

    candidates: List[RemoteImportCandidate] = []
    root_drive_id = _extract_drive_id(item)
    root_item_id = str(item.get("id") or "").strip() or None
    stack: List[Tuple[Optional[str], Optional[str], bool]] = [
        (root_drive_id, root_item_id, not bool(root_drive_id and root_item_id))
    ]
    while stack:
        drive_id, item_id, use_share_root = stack.pop()
        children = _list_graph_children(
            session,
            share_token=share_token if use_share_root else None,
            drive_id=drive_id,
            item_id=item_id,
            access_token=graph_access_token,
        )
        for child in children:
            child_id = str(child.get("id") or "").strip() or None
            child_drive_id = _extract_drive_id(child) or drive_id
            if child.get("folder"):
                if child_id:
                    if child_drive_id:
                        stack.append((child_drive_id, child_id, False))
                    elif use_share_root:
                        logger.info(
                            "Skipping nested folder traversal for shared folder '%s' because Graph did not return a drive id.",
                            child.get("name") or child_id,
                        )
                continue
            candidate = _graph_item_to_candidate(
                share_token,
                child,
                shared_url,
                drive_id=child_drive_id,
            )
            if candidate and _is_supported_filename(candidate.filename, allowed_extensions):
                candidates.append(candidate)
    return candidates


def _resolve_html_candidates(
    session: requests.Session,
    url: str,
    allowed_extensions: set,
) -> List[RemoteImportCandidate]:
    try:
        response = session.get(url, timeout=_REQUEST_TIMEOUT, allow_redirects=True)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RemoteImportError(f"Could not access the link: {exc}") from exc

    content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    if content_type and content_type not in _HTML_CONTENT_TYPES:
        filename = _filename_from_response(response) or _guess_filename_from_url(response.url) or "remote_document"
        return [
            RemoteImportCandidate(
                filename=filename,
                download_url=response.url,
                source_url=url,
            )
        ]

    soup = BeautifulSoup(response.text, "html.parser")
    candidates: List[RemoteImportCandidate] = []

    for anchor in soup.find_all("a", href=True):
        href = urljoin(response.url, anchor["href"])
        candidate = _candidate_from_url(href, source_url=url, fallback_name=anchor.get_text(" ", strip=True))
        if candidate and _is_supported_filename(candidate.filename, allowed_extensions):
            candidates.append(candidate)

    for match in re.findall(r'https?://[^\s"\'>]+', response.text):
        candidate = _candidate_from_url(match, source_url=url)
        if candidate and _is_supported_filename(candidate.filename, allowed_extensions):
            candidates.append(candidate)

    return candidates


def _resolve_personal_drive_path_candidate(
    url: str,
    allowed_extensions: set,
) -> Optional[RemoteImportCandidate]:
    relative_path = _extract_personal_drive_relative_path(url)
    if not relative_path:
        return None

    filename = _guess_filename_from_url(url) or Path(relative_path).name
    filename = _normalize_filename(filename)
    if not _is_supported_filename(filename, allowed_extensions):
        return None

    encoded_relative_path = "/".join(quote(part, safe="") for part in relative_path.split("/") if part)
    graph_download_url = (
        f"https://graph.microsoft.com/v1.0/me/drive/root:/{encoded_relative_path}:/content"
    )
    return RemoteImportCandidate(
        filename=filename,
        download_url=graph_download_url,
        source_url=url,
    )


def _download_candidate(
    session: requests.Session,
    candidate: RemoteImportCandidate,
    allowed_extensions: set,
    access_token: Optional[str] = None,
) -> RemoteImportFile:
    temp_path = None
    download_succeeded = False
    attempted_urls = []
    pending_urls = _build_download_attempt_urls(candidate.download_url)

    try:
        while pending_urls:
            current_url = pending_urls.pop(0)
            if current_url in attempted_urls:
                continue
            attempted_urls.append(current_url)

            try:
                headers = None
                if access_token and "graph.microsoft.com" in urlparse(current_url).netloc.lower():
                    headers = {"Authorization": f"Bearer {access_token}"}
                response = session.get(
                    current_url,
                    timeout=_REQUEST_TIMEOUT,
                    allow_redirects=True,
                    headers=headers,
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                if (
                    "graph.microsoft.com" in urlparse(current_url).netloc.lower()
                    and "/me/drive/root:" in current_url
                    and getattr(getattr(exc, "response", None), "status_code", None) == 404
                ):
                    raise RemoteImportError(
                        "the connected Microsoft account cannot find this file in its own OneDrive. "
                        "Reconnect using the account that owns or can access this document link."
                    ) from exc
                if pending_urls:
                    logger.info(f"Retrying alternate download URL for {candidate.filename}: {exc}")
                    continue
                raise RemoteImportError(f"download failed: {exc}") from exc

            content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            content_bytes = response.content or b""
            final_url = response.url or current_url

            if _looks_like_microsoft_login_redirect(final_url, content_bytes):
                raise RemoteImportError(
                    "this SharePoint/OneDrive link requires Microsoft sign-in for the server. "
                    "Use an anonymous 'Anyone with the link' download link or add Microsoft OAuth integration."
                )

            if not content_bytes:
                if pending_urls:
                    continue
                raise RemoteImportError("downloaded file is empty")

            if len(content_bytes) > _MAX_FILE_SIZE_BYTES:
                raise RemoteImportError(
                    f"file is too large ({round(len(content_bytes) / 1024 / 1024, 2)} MB)"
                )

            if _looks_like_html_payload(content_type, content_bytes):
                html_links = _extract_download_links_from_html(response.url, content_bytes)
                for html_link in html_links:
                    if html_link not in attempted_urls and html_link not in pending_urls:
                        pending_urls.append(html_link)
                if pending_urls:
                    continue
                raise RemoteImportError("link resolved to a preview page instead of a downloadable file")

            filename = _filename_from_response(response) or candidate.filename or _guess_filename_from_url(response.url)
            filename = _normalize_filename(filename)
            filename = _ensure_supported_extension(filename, content_type, allowed_extensions)

            if not _is_supported_filename(filename, allowed_extensions):
                if pending_urls:
                    continue
                raise RemoteImportError("file type is not supported")

            suffix = Path(filename).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                temp_path = tmp.name
                tmp.write(content_bytes)

            download_succeeded = True
            return RemoteImportFile(
                filename=filename,
                temp_path=temp_path,
                source_url=final_url or candidate.source_url,
                size_bytes=len(content_bytes),
            )

        raise RemoteImportError("unable to resolve a direct downloadable file from the link")
    finally:
        if temp_path and os.path.exists(temp_path) and not download_succeeded:
            os.remove(temp_path)


def _build_download_attempt_urls(url: str) -> List[str]:
    urls = [url]
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if any(domain in host for domain in ("1drv.ms", "onedrive.live.com", "sharepoint.com")):
        raw_file_url = _sharepoint_raw_file_url(url)
        if raw_file_url and raw_file_url not in urls:
            urls.insert(0, raw_file_url)
        forced = _with_query_updates(url, {"download": "1"})
        if forced not in urls:
            urls.append(forced)
        if raw_file_url:
            forced_raw_file_url = _with_query_replacement(raw_file_url, {"download": "1"})
            if forced_raw_file_url not in urls:
                urls.insert(1, forced_raw_file_url)
    return urls


def _with_query_updates(url: str, updates: dict) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query, keep_blank_values=True)
    for key, value in updates.items():
        query[key] = [value]
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))


def _with_query_replacement(url: str, replacement: dict) -> str:
    parsed = urlparse(url)
    return urlunparse(parsed._replace(query=urlencode(replacement, doseq=True)))


def _sharepoint_raw_file_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if "sharepoint.com" not in host:
        return ""

    match = re.match(r"^/:[a-z]:/r/(.+)$", parsed.path, flags=re.IGNORECASE)
    if not match:
        return ""

    raw_path = "/" + match.group(1).lstrip("/")
    return urlunparse(parsed._replace(path=raw_path, query=""))


def _extract_personal_drive_relative_path(url: str) -> str:
    raw_url = _sharepoint_raw_file_url(url) or url
    parsed = urlparse(raw_url)
    match = re.match(r"^/personal/[^/]+/(.+)$", parsed.path, flags=re.IGNORECASE)
    if not match:
        return ""
    return unquote(match.group(1)).strip("/")


def _looks_like_html_payload(content_type: str, content_bytes: bytes) -> bool:
    if content_type in _HTML_CONTENT_TYPES:
        return True

    sample = content_bytes[:512].lstrip()
    lowered = sample.lower()
    return (
        lowered.startswith(b"<!doctype html")
        or lowered.startswith(b"<html")
        or lowered.startswith(b"<head")
        or lowered.startswith(b"<body")
    )


def _looks_like_microsoft_login_redirect(final_url: str, content_bytes: bytes) -> bool:
    parsed = urlparse(str(final_url or ""))
    host = parsed.netloc.lower()
    if "login.microsoftonline.com" in host:
        return True

    sample = content_bytes[:600].decode("utf-8", errors="ignore").lower()
    return "sign in to your account" in sample and "microsoft" in sample


def _extract_download_links_from_html(base_url: str, content_bytes: bytes) -> List[str]:
    try:
        html_text = content_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return []

    candidates: List[str] = []
    soup = BeautifulSoup(html_text, "html.parser")

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        if not href:
            continue
        absolute = urljoin(base_url, href)
        if _looks_like_direct_download_link(absolute):
            candidates.append(absolute)

    for pattern in [
        r'"@microsoft\.graph\.downloadUrl"\s*:\s*"([^"]+)"',
        r'"downloadUrl"\s*:\s*"([^"]+)"',
        r'"sourceUrl"\s*:\s*"([^"]+)"',
        r'"url"\s*:\s*"(https?:\\?/\\?/[^"]+download[^"]*)"',
    ]:
        for match in re.findall(pattern, html_text, flags=re.IGNORECASE):
            normalized = _normalize_embedded_url(match)
            if normalized:
                candidates.append(normalized)

    for match in re.findall(r'https?://[^\s"\'>]+', html_text):
        normalized = _normalize_embedded_url(match)
        if normalized and _looks_like_direct_download_link(normalized):
            candidates.append(normalized)

    unique = []
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def _normalize_embedded_url(value: str) -> str:
    normalized = html.unescape(str(value or "").strip())
    normalized = normalized.replace("\\u0026", "&").replace("\\/", "/")
    if normalized.startswith("http://") or normalized.startswith("https://"):
        return normalized
    return ""


def _looks_like_direct_download_link(url: str) -> bool:
    lowered = url.lower()
    parsed = urlparse(lowered)
    path = parsed.path or ""
    query_values = " ".join(sum(parse_qs(parsed.query).values(), []))
    return (
        "download=1" in lowered
        or "/download" in lowered
        or "download.aspx" in lowered
        or "graph.microsoft.com" in lowered
        or any(path.endswith(ext) for ext in _CONTENT_TYPE_EXTENSIONS.values())
        or any(query_values.lower().endswith(ext) for ext in _CONTENT_TYPE_EXTENSIONS.values())
    )


def _get_json(
    session: requests.Session,
    url: str,
    access_token: Optional[str] = None,
) -> Optional[dict]:
    try:
        headers = {"Authorization": f"Bearer {access_token}"} if access_token else {}
        if "/shares/" in url:
            headers["Prefer"] = "redeemSharingLinkIfNecessary"
        response = session.get(
            url,
            timeout=_REQUEST_TIMEOUT,
            allow_redirects=True,
            headers=headers or None,
        )
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        logger.info(f"Remote JSON request failed for {url}: {exc}")
        return None


def _list_graph_children(
    session: requests.Session,
    share_token: Optional[str] = None,
    drive_id: Optional[str] = None,
    item_id: Optional[str] = None,
    access_token: Optional[str] = None,
) -> List[dict]:
    if drive_id and item_id:
        next_url = (
            "https://graph.microsoft.com/v1.0/drives/"
            f"{quote(str(drive_id), safe='')}/items/{quote(str(item_id), safe='')}/children"
        )
    elif share_token:
        next_url = f"https://graph.microsoft.com/v1.0/shares/{share_token}/driveItem/children"
    else:
        return []

    results: List[dict] = []
    while next_url:
        payload = _get_json(session, next_url, access_token=access_token)
        if not payload:
            break
        results.extend(payload.get("value", []))
        next_url = payload.get("@odata.nextLink")
    return results


def _graph_item_to_candidate(
    share_token: str,
    item: dict,
    source_url: str,
    drive_id: Optional[str] = None,
) -> Optional[RemoteImportCandidate]:
    filename = _normalize_filename(item.get("name") or "remote_document")
    item_id = str(item.get("id") or "").strip() or None
    resolved_drive_id = drive_id or _extract_drive_id(item)
    download_url = item.get("@microsoft.graph.downloadUrl")
    if not download_url and resolved_drive_id and item_id:
        download_url = _graph_drive_item_content_url(resolved_drive_id, item_id)
    if not download_url and item_id:
        download_url = f"https://graph.microsoft.com/v1.0/shares/{share_token}/driveItem/items/{item_id}/content"
    if not download_url:
        return None
    return RemoteImportCandidate(
        filename=filename,
        download_url=download_url,
        source_url=item.get("webUrl") or source_url,
        drive_id=resolved_drive_id,
        item_id=item_id,
    )


def _extract_drive_id(item: dict) -> Optional[str]:
    for reference in (
        item.get("parentReference"),
        item.get("remoteItem", {}).get("parentReference"),
        item.get("remoteItem", {}).get("shared", {}).get("parentReference"),
    ):
        if isinstance(reference, dict):
            drive_id = str(reference.get("driveId") or "").strip()
            if drive_id:
                return drive_id
    return None


def _graph_drive_item_content_url(drive_id: str, item_id: str) -> str:
    return (
        "https://graph.microsoft.com/v1.0/drives/"
        f"{quote(str(drive_id), safe='')}/items/{quote(str(item_id), safe='')}/content"
    )


def _dedupe_candidates(candidates: List[RemoteImportCandidate]) -> List[RemoteImportCandidate]:
    unique: List[RemoteImportCandidate] = []
    seen = set()
    for candidate in candidates:
        key = (candidate.filename.lower(), candidate.download_url)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _candidate_from_url(
    url: str,
    source_url: str,
    fallback_name: Optional[str] = None,
) -> Optional[RemoteImportCandidate]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None

    filename = _guess_filename_from_url(url) or _normalize_filename(fallback_name) or "remote_document"
    return RemoteImportCandidate(
        filename=filename,
        download_url=url,
        source_url=source_url,
    )


def _filename_from_response(response: requests.Response) -> Optional[str]:
    content_disposition = response.headers.get("Content-Disposition", "")
    match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', content_disposition, flags=re.IGNORECASE)
    if match:
        return _normalize_filename(unquote(match.group(1)))
    return _guess_filename_from_url(response.url)


def _guess_filename_from_url(url: str) -> Optional[str]:
    parsed = urlparse(url)
    path_name = Path(unquote(parsed.path)).name
    if path_name and "." in path_name:
        return _normalize_filename(path_name)

    query = parse_qs(parsed.query)
    for key in ("filename", "file", "name", "download"):
        for value in query.get(key, []):
            value = unquote(str(value or "")).strip()
            if value and "." in value:
                return _normalize_filename(Path(value).name)
    return None


def _normalize_filename(name: Optional[str]) -> str:
    raw = str(name or "").strip().strip('"').strip("'")
    raw = raw.replace("\\", "/")
    cleaned = Path(raw).name.strip()
    if not cleaned:
        return "remote_document"
    cleaned = re.sub(r"[<>:\"/\\\\|?*\x00-\x1F]", "_", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return cleaned or "remote_document"


def _ensure_supported_extension(filename: str, content_type: str, allowed_extensions: set) -> str:
    current_ext = Path(filename).suffix.lower()
    if current_ext in allowed_extensions:
        return filename

    mapped_ext = _CONTENT_TYPE_EXTENSIONS.get(content_type)
    if mapped_ext and mapped_ext in allowed_extensions:
        base_name = Path(filename).stem if current_ext else filename
        return f"{base_name}{mapped_ext}"
    return filename


def _is_supported_filename(filename: str, allowed_extensions: set) -> bool:
    return Path(str(filename or "")).suffix.lower() in allowed_extensions


def _looks_like_onedrive_or_sharepoint(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return any(domain in host for domain in ("1drv.ms", "onedrive.live.com", "sharepoint.com"))


def _encode_share_url(url: str) -> str:
    encoded = base64.urlsafe_b64encode(url.encode("utf-8")).decode("utf-8").rstrip("=")
    return f"u!{encoded}" if encoded else ""

