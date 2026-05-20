from contextlib import ExitStack, contextmanager
import asyncio
from unittest.mock import patch

import httpx
from fastapi import FastAPI

import routers.auth as auth


class FakeDBManager:
    def __init__(self):
        self.user = None
        self.logged = []
        self.password_updates = []

    def get_user_by_email(self, email):
        return self.user

    def log_activity(self, username, action, details):
        self.logged.append({"username": username, "action": action, "details": details})

    def _hash_password(self, raw_password, salt_hex=None, iterations=None):
        return f"hash::{raw_password}"

    def _verify_password(self, raw_password, stored_password):
        return stored_password == f"hash::{raw_password}"

    def update_user_password(self, user_id, new_password):
        self.password_updates.append({"user_id": user_id, "new_password": new_password})
        if not self.user:
            return None
        return {"_id": user_id, "email": self.user.get("email")}


@contextmanager
def _build_client(fake_db, email_configured=True, send_error=None):
    sent_messages = []
    with ExitStack() as stack:
        stack.enter_context(patch.object(auth, "MongoDBManager", return_value=fake_db))
        stack.enter_context(patch.object(auth, "is_email_delivery_configured", return_value=email_configured))
        stack.enter_context(patch.object(auth, "clear_cache", return_value=None))

        def fake_send(email, otp_code, ttl_minutes):
            sent_messages.append(
                {
                    "email": email,
                    "otp_code": otp_code,
                    "ttl_minutes": ttl_minutes,
                }
            )
            if send_error:
                raise send_error

        stack.enter_context(patch.object(auth, "send_password_reset_otp_email", side_effect=fake_send))

        auth.PASSWORD_RESET_OTP_CACHE.clear()
        auth.PASSWORD_RESET_TOKEN_CACHE.clear()
        app = FastAPI()
        auth.get_auth_routes(app)
        try:
            yield app, sent_messages
        finally:
            auth.PASSWORD_RESET_OTP_CACHE.clear()
            auth.PASSWORD_RESET_TOKEN_CACHE.clear()


def _post_json(app, path, payload):
    async def _send():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post(path, json=payload)

    return asyncio.run(_send())


def test_forgot_password_request_sends_otp_for_existing_user():
    fake_db = FakeDBManager()
    fake_db.user = {"_id": "user-1", "email": "person@megamaxservices.com"}

    with _build_client(fake_db) as resources:
        app, sent_messages = resources
        response = _post_json(app, "/api/auth/forgot-password/request", {"email": "person@megamaxservices.com"})
        cache_record = auth.PASSWORD_RESET_OTP_CACHE["person@megamaxservices.com"]
        assert cache_record["email"] == "person@megamaxservices.com"
        assert cache_record["attempts"] == 0

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert "verification code has been sent" in payload["message"].lower()
    assert sent_messages == [
        {
            "email": "person@megamaxservices.com",
            "otp_code": sent_messages[0]["otp_code"],
            "ttl_minutes": 30,
        }
    ]
    assert len(sent_messages[0]["otp_code"]) == 6
    assert sent_messages[0]["otp_code"].isdigit()


def test_forgot_password_request_is_public_without_session():
    fake_db = FakeDBManager()
    fake_db.user = {"_id": "user-1", "email": "person@megamaxservices.com"}

    with _build_client(fake_db) as resources:
        app, _sent_messages = resources
        response = _post_json(app, "/api/auth/forgot-password/request", {"email": "person@megamaxservices.com"})

    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_forgot_password_request_supersedes_otp_when_email_fails():
    fake_db = FakeDBManager()
    fake_db.user = {"_id": "user-1", "email": "person@megamaxservices.com"}

    with _build_client(fake_db, send_error=RuntimeError("smtp unavailable")) as resources:
        app, _sent_messages = resources
        response = _post_json(app, "/api/auth/forgot-password/request", {"email": "person@megamaxservices.com"})

    assert response.status_code == 502
    assert "person@megamaxservices.com" not in auth.PASSWORD_RESET_OTP_CACHE


def test_forgot_password_verify_returns_reset_token():
    fake_db = FakeDBManager()
    fake_db.user = {"_id": "user-1", "email": "person@megamaxservices.com"}

    with _build_client(fake_db) as resources:
        app, sent_messages = resources
        request_response = _post_json(app, "/api/auth/forgot-password/request", {"email": "person@megamaxservices.com"})
        assert request_response.status_code == 200
        response = _post_json(
            app,
            "/api/auth/forgot-password/verify",
            {"email": "person@megamaxservices.com", "otp": sent_messages[0]["otp_code"]},
        )
        assert "person@megamaxservices.com" not in auth.PASSWORD_RESET_OTP_CACHE

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["reset_token"]


def test_forgot_password_reset_invalidates_existing_sessions():
    fake_db = FakeDBManager()
    fake_db.user = {"_id": "user-1", "email": "person@megamaxservices.com"}
    auth.SESSION_STORE.clear()
    auth.SESSION_STORE["session-a"] = {
        "user_id": "user-1",
        "email": "person@megamaxservices.com",
    }
    auth.SESSION_STORE["session-b"] = {
        "user_id": "someone-else",
        "email": "other@megamaxservices.com",
    }

    try:
        with _build_client(fake_db) as resources:
            app, sent_messages = resources
            request_response = _post_json(app, "/api/auth/forgot-password/request", {"email": "person@megamaxservices.com"})
            assert request_response.status_code == 200
            verify_response = _post_json(
                app,
                "/api/auth/forgot-password/verify",
                {"email": "person@megamaxservices.com", "otp": sent_messages[0]["otp_code"]},
            )
            assert verify_response.status_code == 200
            response = _post_json(
                app,
                "/api/auth/forgot-password/reset",
                {
                    "email": "person@megamaxservices.com",
                    "reset_token": verify_response.json()["reset_token"],
                    "new_password": "NewPass@123",
                    "confirm_password": "NewPass@123",
                },
            )
    finally:
        remaining_sessions = dict(auth.SESSION_STORE)
        auth.SESSION_STORE.clear()

    assert response.status_code == 200
    assert fake_db.password_updates == [{"user_id": "user-1", "new_password": "NewPass@123"}]
    assert "session-a" not in remaining_sessions
    assert "session-b" in remaining_sessions
