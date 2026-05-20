from __future__ import annotations

import re
from datetime import datetime, timedelta

from config import settings

DEFAULT_TEMP_RETENTION_HOURS = max(
    1,
    settings.get_int("TEMP_DOCUMENT_RETENTION_HOURS", 24),
)
CLEANUP_INTERVAL_SECONDS = max(
    60,
    settings.get_int("TEMP_DOCUMENT_CLEANUP_INTERVAL_SECONDS", 3600),
)

EMP_ID_KEYS = {
    "empid",
    "employeeid",
    "employee_id",
    "emp_id",
    "employeecode",
    "employee_code",
    "staffid",
    "staff_id",
}
EMP_NAME_KEYS = {
    "empname",
    "employee_name",
    "employeename",
    "emp_name",
    "staffname",
    "staff_name",
    "name",
}
def clean_text(value) -> str:
    return " ".join(str(value or "").strip().split())


def normalize_choice(value: str | None) -> str:
    normalized = clean_text(value).lower()
    if normalized not in {"standard", "permanent", "temporary"}:
        return "standard"
    return normalized


def resolve_retention_hours(value) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        resolved = DEFAULT_TEMP_RETENTION_HOURS
    return max(1, resolved)


def build_expiry_at(retention_hours: int | None = None) -> datetime:
    hours = resolve_retention_hours(retention_hours)
    return datetime.utcnow() + timedelta(hours=hours)



def _normalized_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", clean_text(key).lower())


def extract_employee_identity(metadata: dict | None) -> tuple[str, str]:
    if not isinstance(metadata, dict):
        return "", ""

    emp_id = ""
    emp_name = ""

    for key, value in metadata.items():
        normalized_key = _normalized_key(key)
        text_value = clean_text(value)
        if not text_value:
            continue
        if not emp_id and normalized_key in EMP_ID_KEYS:
            emp_id = text_value
        if not emp_name and normalized_key in EMP_NAME_KEYS:
            emp_name = text_value

    return emp_id, emp_name


def build_profile_suggestion(emp_id: str | None, emp_name: str | None) -> dict:
    return {
        "empID": clean_text(emp_id),
        "empName": clean_text(emp_name),
    }
