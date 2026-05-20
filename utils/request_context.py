from __future__ import annotations

import re

from fastapi import Request


_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def _strip_ip_port(value: str) -> str:
    candidate = str(value or "").strip().strip('"')
    if not candidate or candidate.lower() == "unknown":
        return ""
    if candidate.startswith("[") and "]" in candidate:
        return candidate[1:candidate.index("]")]
    if candidate.count(":") == 1 and "." in candidate:
        host, _, _ = candidate.partition(":")
        return host.strip()
    return candidate


def _forwarded_header_ip(header_value: str | None) -> str:
    raw = str(header_value or "").strip()
    if not raw:
        return ""
    for part in raw.split(","):
        match = re.search(r'for=(?:"?\[?)([^;\]",]+)', part, flags=re.IGNORECASE)
        if match:
            return _strip_ip_port(match.group(1))
    return ""


def _forwarded_for_ip(header_value: str | None) -> str:
    raw = str(header_value or "").strip()
    if not raw:
        return ""
    for part in raw.split(","):
        candidate = _strip_ip_port(part)
        if candidate:
            return candidate
    return ""


def get_forwarded_client_ip(request: Request) -> str:
    headers = request.headers
    for header_name in ("x-forwarded-for", "x-original-for", "x-real-ip"):
        candidate = _forwarded_for_ip(headers.get(header_name))
        if candidate:
            return candidate
    return _forwarded_header_ip(headers.get("forwarded"))


def get_client_ip(request: Request) -> str:
    direct_client = getattr(getattr(request, "client", None), "host", "") or ""
    forwarded_client = get_forwarded_client_ip(request)

    if direct_client and direct_client not in _LOOPBACK_HOSTS:
        return direct_client
    if forwarded_client:
        return forwarded_client
    if direct_client:
        return direct_client
    return "unknown"
