from datetime import datetime, timedelta
from urllib.parse import urlencode

import requests

from config import settings


MICROSOFT_GRAPH_BASE = "https://graph.microsoft.com/v1.0"


def get_config() -> dict:
    tenant_id = settings.microsoft_tenant_id.strip() or "common"
    client_id = settings.microsoft_client_id.strip()
    client_secret = settings.microsoft_client_secret.strip()
    redirect_uri = settings.microsoft_redirect_uri.strip()
    scopes = settings.microsoft_scopes.strip()
    authority = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0"
    return {
        "tenant_id": tenant_id,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "scopes": scopes,
        "authority": authority,
        "authorize_url": f"{authority}/authorize",
        "token_url": f"{authority}/token",
    }


def is_configured() -> bool:
    config = get_config()
    return bool(config["client_id"] and config["client_secret"] and config["redirect_uri"])


def is_app_configured() -> bool:
    # Forgot password Office 365 OAuth support is intentionally commented out / disabled.
    return False


def build_authorize_url(state: str) -> str:
    config = get_config()
    params = {
        "client_id": config["client_id"],
        "response_type": "code",
        "redirect_uri": config["redirect_uri"],
        "response_mode": "query",
        "scope": config["scopes"],
        "state": state,
        "prompt": "select_account",
    }
    return f"{config['authorize_url']}?{urlencode(params)}"


def exchange_code_for_token(code: str) -> dict:
    config = get_config()
    response = requests.post(
        config["token_url"],
        data={
            "client_id": config["client_id"],
            "client_secret": config["client_secret"],
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": config["redirect_uri"],
            "scope": config["scopes"],
        },
        timeout=30,
    )
    return _parse_token_response(response)


def refresh_access_token(refresh_token: str) -> dict:
    config = get_config()
    response = requests.post(
        config["token_url"],
        data={
            "client_id": config["client_id"],
            "client_secret": config["client_secret"],
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "redirect_uri": config["redirect_uri"],
            "scope": config["scopes"],
        },
        timeout=30,
    )
    return _parse_token_response(response)


def get_application_access_token(scope: str = "https://graph.microsoft.com/.default") -> str:
    # Forgot password Office 365 OAuth support is intentionally commented out / disabled.
    raise RuntimeError("Application mail token flow is disabled")


def send_mail_as_user(access_token: str, sender_email: str, message: dict, save_to_sent_items: bool = True) -> None:
    # Forgot password Office 365 OAuth support is intentionally commented out / disabled.
    raise RuntimeError("Application mail send flow is disabled")


def get_me_profile(access_token: str) -> dict:
    response = requests.get(
        f"{MICROSOFT_GRAPH_BASE}/me?$select=displayName,userPrincipalName,mail,id",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def normalize_token_payload(payload: dict, existing_refresh_token: str | None = None) -> dict:
    expires_in = int(payload.get("expires_in", 0) or 0)
    access_token = payload.get("access_token")
    refresh_token = payload.get("refresh_token") or existing_refresh_token
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": payload.get("token_type", "Bearer"),
        "scope": payload.get("scope", ""),
        "expires_at": (datetime.utcnow() + timedelta(seconds=max(0, expires_in - 120))).isoformat() + "Z",
        "raw": payload,
    }


def is_token_expired(token_data: dict | None) -> bool:
    if not token_data:
        return True
    expires_at = str(token_data.get("expires_at") or "").strip()
    if not expires_at:
        return True
    try:
        expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    except Exception:
        return True
    return datetime.utcnow() >= expiry.replace(tzinfo=None)


def _parse_token_response(response: requests.Response) -> dict:
    try:
        payload = response.json()
    except Exception:
        payload = {"error_description": response.text[:500]}
    if not response.ok:
        message = payload.get("error_description") or payload.get("error") or "Microsoft token request failed"
        raise RuntimeError(message)
    return payload
