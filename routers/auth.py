from fastapi import Request, Form, HTTPException, Query
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from utils.logger import get_logger
from fastapi.templating import Jinja2Templates
from config import settings
from db.mongo_db import MongoDBManager
from utils.microsoft_graph_auth import (
    build_authorize_url,
    exchange_code_for_token,
    get_me_profile,
    is_configured as microsoft_oauth_configured,
    is_token_expired,
    normalize_token_payload,
    refresh_access_token,
)
from services.models import (
    CreateUserRequest,
    CreateUserResponse,
    FaceAuthResponse,
    FaceEnrollRequest,
    FaceLoginRequest,
    ForgotPasswordRequest,
    ForgotPasswordRequestResponse,
    ForgotPasswordResetRequest,
    ForgotPasswordResetResponse,
    ForgotPasswordVerifyRequest,
    ForgotPasswordVerifyResponse,
    MicrosoftDisconnectResponse,
    MicrosoftStatusResponse,
    UpdateUserRequest,
    UpdateUserResponse,
    UserOnboardingStateResponse,
    UserOnboardingUpdateRequest,
    UsersListResponse,
)
from services.face_auth import FaceAuthError, build_face_auth_record, extract_face_embeddings, find_best_face_match
from services.rbac import (
    can_create_role,
    get_creatable_roles,
    get_role_permissions,
    get_session_role,
    role_to_user_type,
    validate_role_string,
)
from services.qa_hf_answerer import clear_cache
from utils.email_utils import is_email_delivery_configured, send_password_reset_otp_email
from utils.request_context import get_client_ip
from datetime import datetime, timedelta
import base64
import hashlib
import hmac
import json
import re
import secrets
from uuid import uuid4
from urllib.parse import urlencode, urlparse, parse_qsl, urlunparse

# Session management
SESSION_STORE = {}  # session_id -> {"user_id": int, "username": str, "email": str, "user_type": str, "start": datetime, "last": datetime}
PASSWORD_RESET_OTP_CACHE = {}  # email -> {"user_id": str, "otp_hash": str, "expires_at": datetime, "attempts": int}
PASSWORD_RESET_TOKEN_CACHE = {}  # reset_token -> {"user_id": str, "email": str, "expires_at": datetime}
FACE_LOGIN_ATTEMPT_CACHE = {}  # client_ip -> {"attempts": int, "locked_until": datetime | None, "updated_at": datetime}
SESSION_TTL_HOURS = settings.session_ttl_hours
INACTIVITY_TTL_HOURS = settings.inactivity_ttl_hours
JWT_ALGORITHM = "HS256"
MAX_SUPER_ADMIN_USERS = 2
FACE_LOGIN_MAX_ATTEMPTS = 5
FACE_LOGIN_LOCK_MINUTES = 5


def _normalize_allowed_email_domain(value: str | None) -> str:
    domain = str(value or "").strip().lower()
    if domain.startswith("@"):
        domain = domain[1:]
    return domain


def _validate_strong_password(password: str) -> str | None:
    raw = str(password or "")
    if len(raw) < 8:
        return "Password must be at least 8 characters long"
    if not re.search(r"[A-Z]", raw):
        return "Password must include at least one uppercase letter"
    if not re.search(r"[a-z]", raw):
        return "Password must include at least one lowercase letter"
    if not re.search(r"\d", raw):
        return "Password must include at least one number"
    if not re.search(r"[^A-Za-z0-9]", raw):
        return "Password must include at least one special character"
    return None


def _generate_password_reset_otp() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"


def _cleanup_password_reset_cache(now: datetime | None = None) -> None:
    current_time = now or datetime.utcnow()
    for email, record in list(PASSWORD_RESET_OTP_CACHE.items()):
        expires_at = record.get("expires_at")
        if not isinstance(expires_at, datetime) or expires_at <= current_time:
            PASSWORD_RESET_OTP_CACHE.pop(email, None)
    for token, record in list(PASSWORD_RESET_TOKEN_CACHE.items()):
        expires_at = record.get("expires_at")
        if not isinstance(expires_at, datetime) or expires_at <= current_time:
            PASSWORD_RESET_TOKEN_CACHE.pop(token, None)


def _password_reset_otp_ttl_seconds() -> int:
    return max(1, settings.get_int("PASSWORD_RESET_OTP_TTL_SECONDS", 30))


def _password_reset_token_ttl_seconds() -> int:
    return max(1, settings.get_int("PASSWORD_RESET_TOKEN_TTL_MINUTES", 15)) * 60


def _store_password_reset_otp_in_cache(db_manager: MongoDBManager, email: str, user_id: str, otp_code: str) -> None:
    now = datetime.utcnow()
    _cleanup_password_reset_cache(now)
    PASSWORD_RESET_OTP_CACHE[email] = {
        "user_id": str(user_id or "").strip(),
        "email": email,
        "otp_hash": db_manager._hash_password(str(otp_code)),
        "attempts": 0,
        "created_at": now,
        "expires_at": now + timedelta(seconds=_password_reset_otp_ttl_seconds()),
    }
    for token, record in list(PASSWORD_RESET_TOKEN_CACHE.items()):
        if str((record or {}).get("email") or "").strip().lower() == email:
            PASSWORD_RESET_TOKEN_CACHE.pop(token, None)


def _verify_password_reset_otp_from_cache(db_manager: MongoDBManager, email: str, otp_code: str) -> dict | None:
    now = datetime.utcnow()
    _cleanup_password_reset_cache(now)
    record = PASSWORD_RESET_OTP_CACHE.get(email)
    if not record:
        return None
    if record.get("expires_at") <= now:
        PASSWORD_RESET_OTP_CACHE.pop(email, None)
        return {"error": "expired"}

    max_attempts = max(1, settings.get_int("PASSWORD_RESET_MAX_ATTEMPTS", 5))
    if int(record.get("attempts") or 0) >= max_attempts:
        return {"error": "locked"}

    if not db_manager._verify_password(str(otp_code), record.get("otp_hash")):
        record["attempts"] = int(record.get("attempts") or 0) + 1
        record["updated_at"] = now
        if record["attempts"] >= max_attempts:
            return {"error": "locked"}
        return {"error": "invalid"}

    PASSWORD_RESET_OTP_CACHE.pop(email, None)
    reset_token = uuid4().hex + uuid4().hex
    expires_at = now + timedelta(seconds=_password_reset_token_ttl_seconds())
    PASSWORD_RESET_TOKEN_CACHE[reset_token] = {
        "user_id": str(record.get("user_id") or "").strip(),
        "email": email,
        "created_at": now,
        "expires_at": expires_at,
    }
    return {
        "user_id": str(record.get("user_id") or "").strip(),
        "email": email,
        "reset_token": reset_token,
        "reset_token_expires_at": _to_utc_iso(expires_at),
    }


def _reset_password_with_cached_token(db_manager: MongoDBManager, email: str, reset_token: str, new_password: str) -> dict | None:
    now = datetime.utcnow()
    _cleanup_password_reset_cache(now)
    record = PASSWORD_RESET_TOKEN_CACHE.get(reset_token)
    if not record:
        return {"error": "invalid"}
    if str(record.get("email") or "").strip().lower() != email:
        return {"error": "invalid"}
    if record.get("expires_at") <= now:
        PASSWORD_RESET_TOKEN_CACHE.pop(reset_token, None)
        return {"error": "expired"}

    user_id = str(record.get("user_id") or "").strip()
    updated_user = db_manager.update_user_password(user_id, new_password)
    if not updated_user:
        return {"error": "user_not_found"}

    PASSWORD_RESET_TOKEN_CACHE.pop(reset_token, None)
    for token, token_record in list(PASSWORD_RESET_TOKEN_CACHE.items()):
        if str((token_record or {}).get("user_id") or "").strip() == user_id:
            PASSWORD_RESET_TOKEN_CACHE.pop(token, None)

    return {
        "user_id": user_id,
        "email": email,
        "user": updated_user,
    }


def _get_face_login_lock_error(client_ip: str, now: datetime | None = None) -> str | None:
    current_time = now or datetime.utcnow()
    record = FACE_LOGIN_ATTEMPT_CACHE.get(str(client_ip or "unknown"))
    if not record:
        return None
    locked_until = record.get("locked_until")
    if isinstance(locked_until, datetime) and locked_until > current_time:
        minutes = max(1, int((locked_until - current_time).total_seconds() // 60) + 1)
        return f"Too many face login attempts. Try again in {minutes} minute(s)."
    if isinstance(locked_until, datetime) and locked_until <= current_time:
        FACE_LOGIN_ATTEMPT_CACHE.pop(str(client_ip or "unknown"), None)
    return None


def _record_face_login_failure(client_ip: str, now: datetime | None = None) -> None:
    current_time = now or datetime.utcnow()
    key = str(client_ip or "unknown")
    record = FACE_LOGIN_ATTEMPT_CACHE.setdefault(
        key,
        {"attempts": 0, "locked_until": None, "updated_at": current_time},
    )
    record["attempts"] = int(record.get("attempts") or 0) + 1
    record["updated_at"] = current_time
    if record["attempts"] >= FACE_LOGIN_MAX_ATTEMPTS:
        record["locked_until"] = current_time + timedelta(minutes=FACE_LOGIN_LOCK_MINUTES)


def _clear_face_login_failures(client_ip: str) -> None:
    FACE_LOGIN_ATTEMPT_CACHE.pop(str(client_ip or "unknown"), None)


def _invalidate_sessions_for_user(user_id: str | None) -> int:
    target_user_id = str(user_id or "").strip()
    if not target_user_id:
        return 0
    expired_sessions = [
        sid
        for sid, session_data in list(SESSION_STORE.items())
        if str((session_data or {}).get("user_id") or "").strip() == target_user_id
    ]
    for sid in expired_sessions:
        destroy_session(sid)
    return len(expired_sessions)

def _should_auto_audit(path: str) -> bool:
    """Avoid noisy/duplicate entries for explicit audit and logout events."""
    if path == "/logout":
        return False
    if path == "/api/activity/log":
        return False
    if path == "/api/activity-log" or path.startswith("/api/activity-log/"):
        return False
    if path.startswith("/api/status/"):
        return False
    return True

def _build_auto_action(method: str, path: str) -> str:
    clean_path = (path or "/").strip("/").replace("-", "_").replace("/", "_")
    if not clean_path:
        clean_path = "ROOT"
    return f"{method.upper()}_{clean_path.upper()}"

def _to_utc_iso(value: datetime) -> str:
    if not isinstance(value, datetime):
        value = datetime.utcnow()
    return value.replace(microsecond=0).isoformat() + "Z"


def _jwt_secret_bytes() -> bytes:
    secret = str(settings.jwt_secret or "").strip()
    if not secret:
        raise RuntimeError("JWT_SECRET is not configured")
    return secret.encode("utf-8")


def _normalize_onboarding_status(value: str | None) -> str:
    status = str(value or "").strip().lower()
    if status in {"completed", "skipped"}:
        return status
    return "pending"


def _serialize_user_onboarding_state(raw_state: dict | None, force_should_show: bool = False) -> dict:
    state = raw_state if isinstance(raw_state, dict) else {}
    status = _normalize_onboarding_status(state.get("status"))
    auto_shown_at = state.get("auto_shown_at")
    return {
        "status": status,
        "current_step": str(state.get("current_step") or "").strip(),
        "should_show": bool(force_should_show) or (status == "pending" and not auto_shown_at),
        "last_seen_at": state.get("last_seen_at"),
        "completed_at": state.get("completed_at"),
        "skipped_at": state.get("skipped_at"),
    }


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("Empty base64url payload")
    padding = "=" * ((4 - (len(raw) % 4)) % 4)
    return base64.urlsafe_b64decode((raw + padding).encode("ascii"))


def _encode_auth_token(session_id: str, session_data: dict) -> str:
    header = {"alg": JWT_ALGORITHM, "typ": "JWT"}
    session_started_at = session_data.get("start") if isinstance(session_data, dict) else None
    issued_at = datetime.utcnow()
    session_role = get_session_role(session_data) or validate_role_string("employee")
    payload = {
        "iss": settings.jwt_issuer,
        "session_id": str(session_id or "").strip(),
        "user_type": str((session_data or {}).get("user_type") or role_to_user_type(session_role)).strip(),
        "role": session_role.value,
        "email": str((session_data or {}).get("email") or (session_data or {}).get("username") or "").strip(),
        "timestamp": _to_utc_iso(session_started_at or issued_at),
        "iat": int(issued_at.timestamp()),
    }
    header_segment = _b64url_encode(json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    payload_segment = _b64url_encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    signing_input = f"{header_segment}.{payload_segment}".encode("ascii")
    signature = hmac.new(_jwt_secret_bytes(), signing_input, hashlib.sha256).digest()
    signature_segment = _b64url_encode(signature)
    return f"{header_segment}.{payload_segment}.{signature_segment}"


def _decode_auth_token(token: str) -> dict:
    raw_token = str(token or "").strip()
    parts = raw_token.split(".")
    if len(parts) != 3:
        raise ValueError("Malformed JWT token")

    header_segment, payload_segment, signature_segment = parts
    signing_input = f"{header_segment}.{payload_segment}".encode("ascii")
    expected_signature = hmac.new(_jwt_secret_bytes(), signing_input, hashlib.sha256).digest()
    actual_signature = _b64url_decode(signature_segment)
    if not hmac.compare_digest(expected_signature, actual_signature):
        raise ValueError("JWT signature mismatch")

    header = json.loads(_b64url_decode(header_segment).decode("utf-8"))
    if str(header.get("alg") or "").upper() != JWT_ALGORITHM:
        raise ValueError("Unsupported JWT algorithm")

    payload = json.loads(_b64url_decode(payload_segment).decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JWT payload must be a JSON object")
    if str(payload.get("iss") or "").strip() != str(settings.jwt_issuer or "").strip():
        raise ValueError("JWT issuer mismatch")
    return payload


def _set_auth_cookies(response, session_id: str) -> None:
    session_data = SESSION_STORE.get(session_id) or {}
    max_age_seconds = SESSION_TTL_HOURS * 3600
    response.set_cookie("session_id", session_id, httponly=True, samesite="lax", max_age=max_age_seconds)
    response.set_cookie(
        settings.jwt_cookie_name,
        _encode_auth_token(session_id, session_data),
        httponly=True,
        samesite="lax",
        max_age=max_age_seconds,
    )


def _clear_auth_cookies(response) -> None:
    response.delete_cookie("session_id")
    jwt_cookie_name = str(settings.jwt_cookie_name or "").strip()
    if jwt_cookie_name:
        response.delete_cookie(jwt_cookie_name)


def create_session(
    user_id: int,
    username: str | None = None,
    email: str | None = None,
    user_type: str | None = None,
    role: str | None = None,
    created_by: str | None = None,  # NEW: Track who created this user
    onboarding_should_show: bool = False,
):
    """Create a new session with role information"""
    sid = secrets.token_urlsafe(32)
    now = datetime.utcnow()
    try:
        safe_role = validate_role_string(role or user_type or "employee")
    except ValueError:
        safe_role = validate_role_string("employee")
    safe_user_type = str(user_type or role_to_user_type(safe_role)).strip() or role_to_user_type(safe_role)
    SESSION_STORE[sid] = {
        "user_id": user_id,
        "username": username,
        "email": str(email or username or "").strip() or None,
        "user_type": safe_user_type,
        "role": safe_role.value,
        "created_by": str(created_by or "").strip() or None,  # NEW: Track creator
        "onboarding_should_show": bool(onboarding_should_show),
        "start": now,
        "last": now,
    }
    return sid

def destroy_session(sid: str):
    SESSION_STORE.pop(sid, None)


# Password reset endpoints invalidate active sessions after a successful reset.


def get_session_data(session_id: str | None):
    if not session_id:
        return None
    return SESSION_STORE.get(session_id)


def _safe_local_path(path: str | None, default: str = "/") -> str:
    candidate = str(path or default).strip()
    parsed = urlparse(candidate)
    if parsed.scheme or parsed.netloc:
        return default
    if not candidate.startswith("/"):
        candidate = "/" + candidate.lstrip("/")
    return candidate or default


def _get_request_root_path(request: Request | None) -> str:
    if request is None:
        return ""
    return str(request.scope.get("root_path") or "").rstrip("/")


def _strip_root_path(path: str | None, root_path: str | None) -> str:
    candidate = _safe_local_path(path, default="/")
    prefix = str(root_path or "").rstrip("/")
    if not prefix:
        return candidate

    parsed = urlparse(candidate)
    normalized_path = parsed.path or "/"
    if normalized_path == prefix or normalized_path.startswith(f"{prefix}/"):
        normalized_path = normalized_path[len(prefix):] or "/"
        candidate = urlunparse(parsed._replace(path=normalized_path))
    return candidate


def _safe_next_path(next_path: str | None, root_path: str | None = None) -> str:
    candidate = _strip_root_path(next_path or "/upload-&-process", root_path)
    return _safe_local_path(candidate, default="/upload-&-process")


def _app_path(
    path: str | None,
    request: Request | None = None,
    root_path: str | None = None,
    default: str = "/",
) -> str:
    candidate = _safe_local_path(path, default=default)
    prefix = _get_request_root_path(request) if root_path is None else str(root_path or "").rstrip("/")
    if not prefix:
        return candidate

    parsed = urlparse(candidate)
    normalized_path = parsed.path or "/"
    if normalized_path == prefix or normalized_path.startswith(f"{prefix}/"):
        return candidate

    prefixed_path = prefix if normalized_path == "/" else f"{prefix}{normalized_path}"
    return urlunparse(parsed._replace(path=prefixed_path))


def _with_query_params(path: str, params: dict) -> str:
    parsed = urlparse(_safe_next_path(path))
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.update({key: value for key, value in params.items() if value is not None})
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))


def _build_auth_redirect_response(
    request: Request,
    detail: str = "Unauthorized",
    api_status_code: int = 401,
):
    path = str(request.scope.get("path") or request.url.path or "")
    if path.startswith("/api"):
        response = JSONResponse({"detail": detail}, status_code=api_status_code)
    else:
        response = RedirectResponse(_app_path("/login", request=request), status_code=302)
    _clear_auth_cookies(response)
    return response


def clear_microsoft_session(session_id: str | None):
    session = get_session_data(session_id)
    if not session:
        return
    session.pop("microsoft_auth", None)
    session.pop("microsoft_oauth_state", None)
    session.pop("microsoft_oauth_next", None)


def get_microsoft_access_token_for_session(session_id: str | None):
    session = get_session_data(session_id)
    if not session:
        return None

    token_data = session.get("microsoft_auth")
    if not token_data:
        return None

    if not is_token_expired(token_data):
        return token_data.get("access_token")

    refresh_token_value = token_data.get("refresh_token")
    if not refresh_token_value or not microsoft_oauth_configured():
        return None

    try:
        refreshed = refresh_access_token(refresh_token_value)
        normalized = normalize_token_payload(
            refreshed,
            existing_refresh_token=refresh_token_value,
        )
        account = token_data.get("account") or {}
        session["microsoft_auth"] = {
            **normalized,
            "account": account,
            "connected_at": token_data.get("connected_at") or datetime.utcnow().isoformat() + "Z",
        }
        return normalized.get("access_token")
    except Exception:
        clear_microsoft_session(session_id)
        return None


from fastapi.responses import FileResponse
from pathlib import Path


def get_auth_routes(app):
    base_dir = Path(__file__).resolve().parent.parent
    templates = Jinja2Templates(directory=str(base_dir / "templates"))
    db_manager = MongoDBManager()
    logger = get_logger("auth")

    @app.get("/login")
    def login_form(request: Request):
        logger.info("Login page requested")
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "error": None,
                "base_path": _get_request_root_path(request),
                "face_auth_enabled": settings.face_auth_enabled,
            },
        )

    def _complete_login_for_user(request: Request, user: dict, username: str, action: str = "LOGIN"):
        client_ip = get_client_ip(request)
        db_manager.log_activity(
            username=username,
            action=action,
            details=f"User {username} successfully logged in",
            client_ip=client_ip,
        )
        onboarding_state = user.get("onboarding") if isinstance(user.get("onboarding"), dict) else {}
        serialized_onboarding = _serialize_user_onboarding_state(onboarding_state)
        onboarding_should_show = bool(serialized_onboarding.get("should_show"))
        if onboarding_should_show:
            next_onboarding_state = {
                **onboarding_state,
                "status": _normalize_onboarding_status(onboarding_state.get("status")),
                "auto_shown_at": _to_utc_iso(datetime.utcnow()),
            }
            db_manager.update_user_fields(str(user["_id"]), {"onboarding": next_onboarding_state})

        sid = create_session(
            str(user["_id"]),
            username=username,
            email=user.get("email") or username,
            user_type=user.get("user_type"),
            role=user.get("role"),
            created_by=user.get("created_by"),
            onboarding_should_show=onboarding_should_show,
        )
        user_role = get_session_role({"role": user.get("role"), "user_type": user.get("user_type")})
        landing_path = "/dashboard"
        if user_role and user_role.value == "employee":
            landing_path = "/qa"
        elif user_role and user_role.value == "manager":
            landing_path = "/upload-&-process"
        return sid, _app_path(landing_path, request=request)

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        # Use the normalized ASGI path so mounted/root_path deployments
        # still match public allow-list routes like /login.
        path = str(request.scope.get("path") or request.url.path or "")
        allow_prefixes = (
            "/static",
            "/login",
            "/favicon.ico",
            "/public-transcript",
            "/generate-transcript",
            "/api/public-transcript",
            "/api/generate-transcript",
            "/api/auth/forgot-password",
            "/api/auth/face/login",
        )
        if path.startswith(allow_prefixes):
            return await call_next(request)

        sid = request.cookies.get("session_id")
        auth_token = request.cookies.get(settings.jwt_cookie_name)
        now = datetime.utcnow()
        if not sid or sid not in SESSION_STORE or not auth_token:
            return _build_auth_redirect_response(request)

        session = SESSION_STORE[sid]
        try:
            token_payload = _decode_auth_token(auth_token)
        except Exception as token_err:
            logger.warning(f"JWT validation failed for session {sid}: {token_err}")
            destroy_session(sid)
            return _build_auth_redirect_response(request)

        if str(token_payload.get("session_id") or "").strip() != str(sid).strip():
            logger.warning(f"JWT session mismatch detected for session {sid}")
            destroy_session(sid)
            return _build_auth_redirect_response(request)

        token_email = str(token_payload.get("email") or "").strip()
        session_email = str(session.get("email") or session.get("username") or "").strip()
        if session_email and token_email and token_email.lower() != session_email.lower():
            logger.warning(f"JWT email mismatch detected for session {sid}")
            destroy_session(sid)
            return _build_auth_redirect_response(request)

        token_user_type = str(token_payload.get("user_type") or "").strip()
        session_user_type = str(session.get("user_type") or "").strip()
        if session_user_type and token_user_type and token_user_type != session_user_type:
            logger.warning(f"JWT user_type mismatch detected for session {sid}")
            destroy_session(sid)
            return _build_auth_redirect_response(request)

        token_role = get_session_role(token_payload)
        session_role = get_session_role(session)
        if session_role and token_role and token_role != session_role:
            logger.warning(f"JWT role mismatch detected for session {sid}")
            destroy_session(sid)
            return _build_auth_redirect_response(request)

        resolved_role = session_role or token_role
        if not resolved_role:
            destroy_session(sid)
            return _build_auth_redirect_response(request)

        session["role"] = resolved_role.value
        session["user_type"] = session_user_type or role_to_user_type(resolved_role)
        if now - session["last"] > timedelta(hours=INACTIVITY_TTL_HOURS) or now - session["start"] > timedelta(hours=SESSION_TTL_HOURS):
            destroy_session(sid)
            return _build_auth_redirect_response(request, detail="Session expired")

        session["last"] = now
        request.state.user_id = session["user_id"]
        request.state.username = session.get("username")
        request.state.email = session.get("email")
        request.state.user_type = session.get("user_type")
        request.state.role = resolved_role.value
        request.state.created_by = session.get("created_by")
        request.state.permissions = sorted(get_role_permissions(resolved_role))
        request.state.session_data = session
        request.state.auth_token_payload = token_payload
        request.state.client_ip = get_client_ip(request)
        response = await call_next(request)

        # Auto-audit every authenticated request so actions are visible in Audit Log.
        if _should_auto_audit(path):
            try:
                username = session.get("username") or "unknown"
                query_str = f"?{request.url.query}" if request.url.query else ""
                db_manager.log_activity(
                    username=username,
                    action=_build_auto_action(request.method, path),
                    details=f"{request.method.upper()} {path}{query_str} -> {response.status_code}",
                    client_ip=request.state.client_ip,
                )
            except Exception as audit_err:
                logger.warning(f"Auto audit logging failed for {request.method} {path}: {audit_err}")

        return response

    @app.post("/login")
    def login(request: Request, username: str = Form(...), password: str = Form(...)):
        logger.info(f"Login attempt for user: {username}")
        client_ip = get_client_ip(request)
        user = db_manager.find_user(username, password)
        if user:
            logger.info(f"Login successful for user: {username}")
            sid, landing_url = _complete_login_for_user(request, user, username, action="LOGIN")
            resp = RedirectResponse(landing_url, status_code=302)
            _set_auth_cookies(resp, sid)
            return resp
        logger.warning(f"Login failed for user: {username}")
        # Log failed login attempt
        db_manager.log_activity(
            username=username,
            action="LOGIN_FAILED",
            details=f"Failed login attempt for user {username}",
            client_ip=client_ip,
        )
        # Render login.html with error message
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "error": "Invalid credentials. Please try again.",
                "base_path": _get_request_root_path(request),
                "face_auth_enabled": settings.face_auth_enabled,
            },
        )

    @app.post("/api/auth/face/enroll", response_model=FaceAuthResponse)
    async def enroll_face_login(request: Request, payload: FaceEnrollRequest):
        if not settings.face_auth_enabled:
            raise HTTPException(status_code=403, detail="Face login is disabled by system settings.")
        session_email = str(getattr(request.state, "email", None) or getattr(request.state, "username", None) or "").strip().lower()
        email = str(payload.email or session_email or "").strip().lower()
        password = str(payload.password or "").strip()
        if not email or not password:
            raise HTTPException(status_code=400, detail="Enter your email and password before setting up face login.")
        if session_email and email != session_email:
            raise HTTPException(status_code=403, detail="You can set up face login only for your own account.")

        user = db_manager.find_user(email, password)
        if not user:
            db_manager.log_activity(
                username=email or "unknown",
                action="FACE_ENROLL_FAILED",
                details="Face enrollment failed because credentials were invalid",
                client_ip=get_client_ip(request),
            )
            raise HTTPException(status_code=401, detail="Email or password is incorrect.")

        face_images = payload.images or [payload.image]
        try:
            face_auth_record = build_face_auth_record(face_images)
        except FaceAuthError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        updated_user = db_manager.update_user_face_enrollment(str(user["_id"]), face_auth_record)
        if not updated_user:
            raise HTTPException(status_code=500, detail="Face login could not be set up.")

        db_manager.log_activity(
            username=email,
            action="FACE_ENROLL",
            details=f"Face login enrolled for user {email}",
            client_ip=get_client_ip(request),
        )
        return {
            "status": "success",
            "message": "Face login is ready. You can use Login with face next time.",
            "email": email,
        }

    @app.post("/api/auth/face/login", response_model=FaceAuthResponse)
    async def login_with_face(request: Request, payload: FaceLoginRequest):
        if not settings.face_auth_enabled:
            raise HTTPException(status_code=403, detail="Face login is disabled by system settings.")
        client_ip = get_client_ip(request)
        lock_error = _get_face_login_lock_error(client_ip)
        if lock_error:
            raise HTTPException(status_code=429, detail=lock_error)

        face_images = payload.images or [payload.image]
        try:
            probe_embeddings = extract_face_embeddings(face_images)
        except FaceAuthError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        candidates = db_manager.list_face_enabled_users()
        match = find_best_face_match(probe_embeddings, candidates)
        if not match:
            _record_face_login_failure(client_ip)
            db_manager.log_activity(
                username="unknown",
                action="FACE_LOGIN_FAILED",
                details="Face login failed because no enrolled face matched",
                client_ip=client_ip,
            )
            raise HTTPException(status_code=401, detail="Face not recognized. Use password, then set up face login from Settings.")

        user = match.user
        username = str(user.get("email") or user.get("username") or "").strip()
        _clear_face_login_failures(client_ip)
        sid, redirect_url = _complete_login_for_user(request, user, username, action="FACE_LOGIN")
        response = JSONResponse(
            {
                "status": "success",
                "message": "Face recognized. Signing you in.",
                "redirect_url": redirect_url,
                "email": username,
            }
        )
        _set_auth_cookies(response, sid)
        return response

    @app.post("/api/auth/forgot-password/request", response_model=ForgotPasswordRequestResponse)
    async def forgot_password_request(request: Request, payload: ForgotPasswordRequest):
        email = str(payload.email or "").strip().lower()
        generic_message = "If an account exists for this email, a verification code has been sent."
        if not email or "@" not in email:
            raise HTTPException(status_code=400, detail="Email Id must be a valid email address")
        if not is_email_delivery_configured():
            raise HTTPException(status_code=503, detail="Password reset email is not configured on the server.")

        user = db_manager.get_user_by_email(email)
        if not user:
            logger.info(f"Password reset requested for unknown email: {email}")
            return {"status": "success", "message": generic_message}

        otp_ttl_seconds = _password_reset_otp_ttl_seconds()
        otp_code = _generate_password_reset_otp()
        _store_password_reset_otp_in_cache(db_manager, email, str(user.get("_id")), otp_code)

        try:
            send_password_reset_otp_email(email, otp_code, otp_ttl_seconds)
        except Exception as exc:
            PASSWORD_RESET_OTP_CACHE.pop(email, None)
            logger.error(f"Password reset email failed for {email}: {exc}")
            raise HTTPException(status_code=502, detail="Unable to send password reset email right now.")

        db_manager.log_activity(
            username=email,
            action="PASSWORD_RESET_REQUEST",
            details=f"Password reset OTP issued for {email}",
            client_ip=get_client_ip(request),
        )
        logger.info(f"Password reset OTP issued for {email} in in-memory cache")
        return {"status": "success", "message": generic_message}

    @app.post("/api/auth/forgot-password/verify", response_model=ForgotPasswordVerifyResponse)
    async def forgot_password_verify(request: Request, payload: ForgotPasswordVerifyRequest):
        email = str(payload.email or "").strip().lower()
        otp = str(payload.otp or "").strip()
        if not email or "@" not in email:
            raise HTTPException(status_code=400, detail="Email Id must be a valid email address")
        if len(otp) != 6 or not otp.isdigit():
            raise HTTPException(status_code=400, detail="Enter the 6-digit verification code.")

        verified = _verify_password_reset_otp_from_cache(db_manager, email, otp)
        if not verified:
            raise HTTPException(status_code=400, detail="Invalid verification code.")
        if verified.get("error") == "expired":
            raise HTTPException(status_code=400, detail="This verification code has expired. Request a new code.")
        if verified.get("error") == "locked":
            raise HTTPException(status_code=429, detail="Too many incorrect attempts. Request a new code.")
        if verified.get("error") == "invalid":
            raise HTTPException(status_code=400, detail="Invalid verification code.")

        db_manager.log_activity(
            username=email,
            action="PASSWORD_RESET_VERIFY",
            details=f"Password reset OTP verified for {email}",
            client_ip=get_client_ip(request),
        )
        return {
            "status": "success",
            "message": "Code verified. Set your new password.",
            "reset_token": verified["reset_token"],
        }

    @app.post("/api/auth/forgot-password/reset", response_model=ForgotPasswordResetResponse)
    async def forgot_password_reset(request: Request, payload: ForgotPasswordResetRequest):
        email = str(payload.email or "").strip().lower()
        reset_token = str(payload.reset_token or "").strip()
        new_password = str(payload.new_password or "").strip()
        confirm_password = str(payload.confirm_password or "").strip()

        if not email or "@" not in email:
            raise HTTPException(status_code=400, detail="Email Id must be a valid email address")
        if not reset_token:
            raise HTTPException(status_code=400, detail="Password reset session is missing. Verify your code again.")
        if not new_password:
            raise HTTPException(status_code=400, detail="New password is required")
        if new_password != confirm_password:
            raise HTTPException(status_code=400, detail="New password and confirm password must match")

        password_error = _validate_strong_password(new_password)
        if password_error:
            raise HTTPException(status_code=400, detail=password_error)

        reset_result = _reset_password_with_cached_token(db_manager, email, reset_token, new_password)
        if not reset_result:
            raise HTTPException(status_code=400, detail="Password reset session is invalid. Start again.")
        if reset_result.get("error") == "invalid":
            raise HTTPException(status_code=400, detail="Password reset session is invalid. Start again.")
        if reset_result.get("error") == "expired":
            raise HTTPException(status_code=400, detail="Password reset session has expired. Verify your code again.")
        if reset_result.get("error") == "user_not_found":
            raise HTTPException(status_code=404, detail="User not found")

        invalidated_sessions = _invalidate_sessions_for_user(reset_result.get("user_id"))
        clear_cache()
        db_manager.log_activity(
            username=email,
            action="PASSWORD_RESET",
            details=f"Password reset completed for {email}; invalidated_sessions={invalidated_sessions}",
            client_ip=get_client_ip(request),
        )
        return {
            "status": "success",
            "message": "Password updated successfully. Sign in with your new password.",
        }

    @app.post("/api/users", response_model=CreateUserResponse)
    async def create_user(request: Request, payload: CreateUserRequest):
        """Create a new application user from dashboard UI."""
        email = str(payload.email or "").strip().lower()
        password = str(payload.password or "").strip()
        employee_code = str(payload.employee_code or "").strip()
        allowed_domain = _normalize_allowed_email_domain(settings.allowed_user_email_domain)
        try:
            target_role = validate_role_string(payload.role or payload.user_type or "employee")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        normalized_user_type = role_to_user_type(target_role)

        if not email:
            raise HTTPException(status_code=400, detail="Email Id is required")
        if not password:
            raise HTTPException(status_code=400, detail="Password is required")
        password_error = _validate_strong_password(password)
        if password_error:
            raise HTTPException(status_code=400, detail=password_error)
        if not employee_code:
            raise HTTPException(status_code=400, detail="Employee Code is required")
        if not employee_code.isdigit():
            raise HTTPException(status_code=400, detail="Employee Code must contain digits only")
        if "@" not in email:
            raise HTTPException(status_code=400, detail="Email Id must be a valid email address")
        if allowed_domain and email.rsplit("@", 1)[-1].lower() != allowed_domain:
            raise HTTPException(
                status_code=400,
                detail=f"Email Id must end with @{allowed_domain}",
            )

        creator_role = getattr(request.state, "role", None)
        if not creator_role or not can_create_role(creator_role, target_role):
            raise HTTPException(
                status_code=403,
                detail=f"Your role '{creator_role or 'unknown'}' cannot create users with role '{target_role.value}'",
            )
        if target_role.value == "super_admin":
            existing_super_admins = db_manager.count_users_by_role("super_admin")
            if existing_super_admins >= MAX_SUPER_ADMIN_USERS:
                raise HTTPException(
                    status_code=409,
                    detail=f"Only {MAX_SUPER_ADMIN_USERS} super admin users are allowed.",
                )

        conflict = db_manager.find_user_conflict(
            username=email,
            email=email,
            employee_code=employee_code,
        )
        if conflict:
            friendly = {
                "username": "Email Id",
                "email": "Email Id",
                "employee_code": "Employee Code",
            }
            field_name = friendly.get(conflict.get("field"), "User field")
            raise HTTPException(status_code=409, detail=f"{field_name} already exists")

        created_by = getattr(request.state, "email", None) or getattr(request.state, "username", None)
        user_id = db_manager.create_user(
            username=email,
            password=password,
            email=email,
            employee_code=employee_code,
            user_type=normalized_user_type,
            created_by=created_by,
            role=target_role.value,
        )
        if not user_id:
            raise HTTPException(status_code=500, detail="Failed to create user")

        db_manager.log_activity(
            username=created_by or "unknown",
            action="CREATE_USER",
            details=f"Created user {email} (role: {target_role.value})",
            client_ip=getattr(request.state, "client_ip", None) or get_client_ip(request),
        )

        return {
            "status": "success",
            "user_id": user_id,
            "email": email,
            "employee_code": employee_code,
            "user_type": normalized_user_type,
            "role": target_role.value,
        }

    @app.get("/api/users", response_model=UsersListResponse)
    def list_users(
        request: Request,  # NEW: RBAC
        limit: int = Query(100, ge=1, le=500),
        include_legacy: bool = Query(False),
    ):
        """List users visible to the current role."""
        current_role = getattr(request.state, "role", None)
        current_email = getattr(request.state, "email", None) or getattr(request.state, "username", None)
        if current_role not in {"manager", "admin", "super_admin"}:
            raise HTTPException(status_code=403, detail="You are not allowed to view users.")

        created_by_filter = current_email if current_role == "manager" else None
        users = db_manager.list_users(
            limit=limit,
            include_legacy=include_legacy,
            created_by=created_by_filter,
        )
        return {
            "total": len(users),
            "users": users,
        }

    @app.patch("/api/users/{user_id}", response_model=UpdateUserResponse)
    async def update_user(request: Request, user_id: str, payload: UpdateUserRequest):
        """Update editable user fields from the Users page."""
        current_role = getattr(request.state, "role", None)
        current_email = getattr(request.state, "email", None) or getattr(request.state, "username", None)
        if current_role not in {"admin", "super_admin"}:
            raise HTTPException(status_code=403, detail="Only admin and super admin can edit users.")

        existing_user = db_manager.get_user_by_id(user_id)
        if not existing_user:
            raise HTTPException(status_code=404, detail="User not found.")

        try:
            existing_role = validate_role_string(existing_user.get("role") or existing_user.get("user_type") or "employee")
            target_role = validate_role_string(payload.role or existing_role.value)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        if current_role == "admin" and existing_role.value == "super_admin":
            raise HTTPException(status_code=403, detail="Admin cannot edit super admin users.")

        if not can_create_role(current_role, target_role):
            raise HTTPException(
                status_code=403,
                detail=f"Your role '{current_role or 'unknown'}' cannot assign role '{target_role.value}'.",
            )

        email = str(payload.email or "").strip().lower()
        employee_code = str(payload.employee_code or "").strip()
        allowed_domain = _normalize_allowed_email_domain(settings.allowed_user_email_domain)

        if not email:
            raise HTTPException(status_code=400, detail="Email Id is required")
        if "@" not in email:
            raise HTTPException(status_code=400, detail="Email Id must be a valid email address")
        if allowed_domain and email.rsplit("@", 1)[-1].lower() != allowed_domain:
            raise HTTPException(status_code=400, detail=f"Email Id must end with @{allowed_domain}")

        if not employee_code:
            raise HTTPException(status_code=400, detail="Employee Code is required")
        if not employee_code.isdigit():
            raise HTTPException(status_code=400, detail="Employee Code must contain digits only")

        previous_role = existing_role.value
        if previous_role != "super_admin" and target_role.value == "super_admin":
            existing_super_admins = db_manager.count_users_by_role("super_admin")
            if existing_super_admins >= MAX_SUPER_ADMIN_USERS:
                raise HTTPException(
                    status_code=409,
                    detail=f"Only {MAX_SUPER_ADMIN_USERS} super admin users are allowed.",
                )

        if previous_role == "super_admin" and target_role.value != "super_admin":
            existing_super_admins = db_manager.count_users_by_role("super_admin")
            if existing_super_admins <= 1:
                raise HTTPException(status_code=409, detail="At least one super admin must remain in the system.")

        existing_email = str(existing_user.get("email") or existing_user.get("username") or "").strip().lower()
        existing_employee_code = str(existing_user.get("employee_code") or "").strip()
        if email != existing_email:
            conflict = db_manager.find_user_conflict(username=email, email=email)
            if conflict and str(conflict.get("user_id") or "") != str(user_id):
                raise HTTPException(status_code=409, detail="Email Id already exists")

        if employee_code != existing_employee_code:
            conflict = db_manager.find_user_conflict(employee_code=employee_code)
            if conflict and str(conflict.get("user_id") or "") != str(user_id):
                raise HTTPException(status_code=409, detail="Employee Code already exists")

        safe_user_type = role_to_user_type(target_role)
        updated_user = db_manager.update_user_fields(
            user_id,
            {
                "username": email,
                "email": email,
                "employee_code": employee_code,
                "role": target_role.value,
                "user_type": safe_user_type,
            },
        )
        if not updated_user:
            raise HTTPException(status_code=500, detail="Failed to update user.")

        actor = current_email or "unknown"
        db_manager.log_activity(
            username=actor,
            action="UPDATE_USER",
            details=f"Updated user {existing_email or user_id} -> {email} (role: {previous_role} -> {target_role.value})",
            client_ip=getattr(request.state, "client_ip", None) or get_client_ip(request),
        )

        sid = request.cookies.get("session_id")
        session = get_session_data(sid)
        if session and str(session.get("user_id") or "") == str(user_id):
            session["username"] = email
            session["email"] = email
            session["user_type"] = safe_user_type
            session["role"] = target_role.value
            session["last"] = datetime.utcnow()
            response = JSONResponse({"status": "success", "user": updated_user})
            _set_auth_cookies(response, sid)
            return response

        return {
            "status": "success",
            "user": updated_user,
        }

    @app.get("/api/me")
    def get_current_user(request: Request):
        """Get current user information including role (NEW: RBAC)"""
        username = getattr(request.state, "username", None)
        email = getattr(request.state, "email", None) or username
        if not email:
            raise HTTPException(status_code=401, detail="Not authenticated")
        
        try:
            # Query user by email to get full user info including role
            if not db_manager._ensure_connection() or db_manager.users is None:
                raise HTTPException(status_code=503, detail="Database connection is not available")
            user = db_manager.users.find_one({"email": email.lower()})
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            
            # Map user_type to role if role field is missing (backward compatibility)
            role = getattr(request.state, "role", None) or user.get("role")
            if not role:
                user_type = user.get("user_type", "").lower()
                role_mapping = {
                    "admin": "admin",
                    "super_admin": "super_admin",
                    "manager": "manager",
                    "viewer": "employee",
                    "external user": "employee",
                }
                role = role_mapping.get(user_type, "employee")
            
            session_data = getattr(request.state, "session_data", {}) or {}
            force_onboarding_should_show = bool(session_data.get("onboarding_should_show"))
            if force_onboarding_should_show:
                session_data["onboarding_should_show"] = False

            return {
                "email": user.get("email"),
                "username": user.get("username"),
                "user_type": user.get("user_type"),
                "role": role,
                "employee_code": user.get("employee_code"),
                "allowed_user_email_domain": settings.allowed_user_email_domain,
                "guided_tour_enabled": True,
                "face_auth_enabled": settings.face_auth_enabled,
                "created_by": user.get("created_by"),
                "created_at": str(user.get("created_at")),
                "permissions": sorted(get_role_permissions(role)),
                "creatable_roles": get_creatable_roles(role),
                "onboarding": _serialize_user_onboarding_state(
                    user.get("onboarding"),
                    force_onboarding_should_show,
                ),
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error fetching current user: {e}")
            raise HTTPException(status_code=500, detail="Failed to fetch user info")

    @app.post("/api/me/onboarding", response_model=UserOnboardingStateResponse)
    def update_current_user_onboarding(request: Request, payload: UserOnboardingUpdateRequest):
        user_id = getattr(request.state, "user_id", None)
        if not user_id:
            raise HTTPException(status_code=401, detail="Not authenticated")

        target_status = _normalize_onboarding_status(payload.status)
        if target_status not in {"completed", "skipped"}:
            raise HTTPException(status_code=400, detail="Onboarding status must be completed or skipped.")

        existing_user = db_manager.get_user_by_id(user_id, projection={"onboarding": 1})
        if not existing_user:
            raise HTTPException(status_code=404, detail="User not found")

        existing_state = _serialize_user_onboarding_state(existing_user.get("onboarding"))
        now_iso = _to_utc_iso(datetime.utcnow())
        next_state = {
            "status": target_status,
            "current_step": str(payload.current_step or existing_state.get("current_step") or "").strip(),
            "last_seen_at": now_iso,
            "completed_at": existing_state.get("completed_at"),
            "skipped_at": existing_state.get("skipped_at"),
        }
        if target_status == "completed":
            next_state["completed_at"] = now_iso
        if target_status == "skipped":
            next_state["skipped_at"] = now_iso

        updated_user = db_manager.update_user_fields(user_id, {"onboarding": next_state})
        if not updated_user:
            raise HTTPException(status_code=500, detail="Failed to update onboarding state")

        return _serialize_user_onboarding_state(next_state)

    @app.get("/auth/microsoft/connect")
    def connect_microsoft(request: Request, next: str = Query("/upload-&-process")):
        if not microsoft_oauth_configured():
            return RedirectResponse(
                url=_app_path(
                    _with_query_params(
                        _safe_next_path(next, _get_request_root_path(request)),
                        {"ms_error": "Microsoft import is not configured on the server"},
                    ),
                    request=request,
                ),
                status_code=302,
            )

        sid = request.cookies.get("session_id")
        session = get_session_data(sid)
        if not session:
            return RedirectResponse(_app_path("/login", request=request), status_code=302)

        state = secrets.token_urlsafe(24)
        session["microsoft_oauth_state"] = state
        session["microsoft_oauth_root_path"] = _get_request_root_path(request)
        session["microsoft_oauth_next"] = _safe_next_path(next, session["microsoft_oauth_root_path"])
        return RedirectResponse(build_authorize_url(state), status_code=302)

    @app.get("/auth/microsoft/callback")
    def microsoft_callback(
        request: Request,
        code: str | None = Query(None),
        state: str | None = Query(None),
        error: str | None = Query(None),
        error_description: str | None = Query(None),
    ):
        sid = request.cookies.get("session_id")
        session = get_session_data(sid)
        if not session:
            return RedirectResponse(_app_path("/login", request=request), status_code=302)

        oauth_root_path = str(session.get("microsoft_oauth_root_path") or _get_request_root_path(request)).rstrip("/")
        next_url = _safe_next_path(session.get("microsoft_oauth_next"), oauth_root_path)
        expected_state = session.get("microsoft_oauth_state")
        session.pop("microsoft_oauth_state", None)
        session.pop("microsoft_oauth_next", None)
        session.pop("microsoft_oauth_root_path", None)

        if not microsoft_oauth_configured():
            return RedirectResponse(
                url=_app_path(
                    _with_query_params(next_url, {"ms_error": "Microsoft import is not configured on the server"}),
                    root_path=oauth_root_path,
                ),
                status_code=302,
            )

        if error:
            return RedirectResponse(
                url=_app_path(
                    _with_query_params(next_url, {"ms_error": error_description or error}),
                    root_path=oauth_root_path,
                ),
                status_code=302,
            )

        if not code or not state or not expected_state or state != expected_state:
            return RedirectResponse(
                url=_app_path(
                    _with_query_params(next_url, {"ms_error": "Microsoft sign-in verification failed"}),
                    root_path=oauth_root_path,
                ),
                status_code=302,
            )

        try:
            token_payload = exchange_code_for_token(code)
            normalized = normalize_token_payload(token_payload)
            profile = get_me_profile(normalized["access_token"])
            session["microsoft_auth"] = {
                **normalized,
                "account": {
                    "display_name": profile.get("displayName"),
                    "email": profile.get("mail") or profile.get("userPrincipalName"),
                    "id": profile.get("id"),
                },
                "connected_at": datetime.utcnow().isoformat() + "Z",
            }
            return RedirectResponse(
                url=_app_path(
                    _with_query_params(next_url, {"ms_connected": "1"}),
                    root_path=oauth_root_path,
                ),
                status_code=302,
            )
        except Exception as exc:
            clear_microsoft_session(sid)
            return RedirectResponse(
                url=_app_path(
                    _with_query_params(next_url, {"ms_error": str(exc)[:300]}),
                    root_path=oauth_root_path,
                ),
                status_code=302,
            )

    @app.get("/api/microsoft/status", response_model=MicrosoftStatusResponse)
    def microsoft_status(request: Request):
        sid = request.cookies.get("session_id")
        session = get_session_data(sid)
        token_data = session.get("microsoft_auth") if session else None
        account = token_data.get("account", {}) if token_data else {}
        connected = bool(token_data and get_microsoft_access_token_for_session(sid))
        return {
            "configured": microsoft_oauth_configured(),
            "connected": connected,
            "account": account,
            "expires_at": token_data.get("expires_at") if token_data else None,
        }

    @app.post("/api/microsoft/disconnect", response_model=MicrosoftDisconnectResponse)
    def microsoft_disconnect(request: Request):
        sid = request.cookies.get("session_id")
        clear_microsoft_session(sid)
        return {"status": "success", "connected": False}

    @app.get("/logout")
    def logout(request: Request):
        sid = request.cookies.get("session_id")
        username = getattr(request.state, "username", "unknown")
        client_ip = getattr(request.state, "client_ip", None) or get_client_ip(request)
        if sid:
            logger.info(f"Logout for session: {sid}")
            # Log the logout activity
            db_manager.log_activity(
                username=username,
                action="LOGOUT",
                details=f"User {username} logged out",
                client_ip=client_ip,
            )
            clear_microsoft_session(sid)
            destroy_session(sid)
        
        # Clear Q&A cache on logout
        clear_cache()
        
        resp = RedirectResponse(_app_path("/login", request=request), status_code=302)
        _clear_auth_cookies(resp)
        return resp

    @app.get("/home")
    def home(request: Request):
        logger.info(f"Home page accessed by user: {request.state.username}")
        logout_url = _app_path("/logout", request=request)
        return HTMLResponse(f"<h2>Welcome, {request.state.username}!</h2><a href='{logout_url}'>Logout</a>")

