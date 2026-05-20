from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


class _Settings:
    def __init__(self, env_file: str | None = None) -> None:
        self._env_file = Path(env_file or (Path(__file__).resolve().parent / ".env"))
        load_dotenv(dotenv_path=self._env_file, override=False)
        self._bootstrap()

    def _bootstrap(self) -> None:
        self._mongo_uri = self._read_str("MONGO_URI", "")
        self._db_name = self._read_str("DB_NAME", "")
        self._collection_name = self._read_str("COLLECTION_NAME", "")
        self._gemini_api_key = self._read_str("GEMINI_API_KEY", "").strip().strip('"').strip("'")
        self._gemini_model_name = self._read_str("GEMINI_MODEL_NAME", "gemini-2.5-flash")
        self._version = self._read_str("VERSION", "")
        self._hf_embedding_model = self._read_str("HF_EMBEDDING_MODEL", "")
        self._llm_structurer_max_concurrency = self._read_int("LLM_STRUCTURER_MAX_CONCURRENCY", 2)
        self._session_ttl_hours = self._read_int("SESSION_TTL_HOURS", 12)
        self._inactivity_ttl_hours = self._read_int("INACTIVITY_TTL_HOURS", 2)
        self._jwt_cookie_name = self._read_str("JWT_COOKIE_NAME", "doc_management_token")
        self._jwt_secret = self._read_str("JWT_SECRET", "")
        self._jwt_issuer = self._read_str("JWT_ISSUER", "doc_management")
        self._allowed_file_extensions = self._read_list(
            "ALLOWED_FILE_EXTENSIONS",
            [
                ".pdf",
                ".jpg",
                ".jpeg",
                ".png",
                ".tiff",
                ".tif",
                ".docx",
                ".doc",
                ".docm",
                ".xlsx",
                ".xls",
                ".xlsm",
                ".xlsb",
                ".mp4",
                ".avi",
                ".mov",
                ".mkv",
                ".flv",
                ".wmv",
                ".m4v",
                ".webm",
            ],
        )
        self._qa_response_cache_ttl_seconds = self._read_int("QA_RESPONSE_CACHE_TTL_SECONDS", 900)
        self._qa_response_cache_max_size = self._read_int("QA_RESPONSE_CACHE_MAX_SIZE", 128)
        self._qa_cache_max_size = self._read_int("QA_CACHE_MAX_SIZE", 256)
        self._qa_cache_ttl_seconds = self._read_int("QA_CACHE_TTL_SECONDS", 10800)
        self._qa_history_ttl_hours = self._read_int("QA_HISTORY_TTL_HOURS", 24)
        self._qa_history_max_users = self._read_int("QA_HISTORY_MAX_USERS", 500)
        self._qa_history_max_conversations_per_user = self._read_int("QA_HISTORY_MAX_CONVERSATIONS_PER_USER", 50)
        self._qa_history_max_turns_per_conversation = self._read_int("QA_HISTORY_MAX_TURNS_PER_CONVERSATION", 40)
        self._embedding_max_chars = self._read_int("EMBEDDING_MAX_CHARS", 8000)
        self._embedding_min_similarity = self._read_float("EMBEDDING_MIN_SIMILARITY", 0.0)
        self._enable_embeddings = self._read_bool("ENABLE_EMBEDDINGS", True)
        self._ocr_language = self._read_str("OCR_LANGUAGE", "en")
        self._ocr_use_gpu = self._read_bool("OCR_USE_GPU", False)
        self._ocr_show_log = self._read_bool("OCR_SHOW_LOG", False)
        self._allowed_user_email_domain = self._normalize_email_domain(
            self._read_str("ALLOWED_USER_EMAIL_DOMAIN", "megamaxservices.com")
        )
        self._test_document_path = self._read_str("TEST_DOCUMENT_PATH", "")
        self._microsoft_tenant_id = self._read_str("MICROSOFT_TENANT_ID", "common") or "common"
        self._microsoft_client_id = self._read_str("MICROSOFT_CLIENT_ID", "")
        self._microsoft_client_secret = self._read_str("MICROSOFT_CLIENT_SECRET", "")
        self._microsoft_redirect_uri = self._read_str("MICROSOFT_REDIRECT_URI", "https://mspl.ai/doc-management/auth/microsoft/callback")
        self._microsoft_scopes = self._read_str(
            "MICROSOFT_SCOPES",
            "openid profile offline_access User.Read Files.ReadWrite.All Sites.ReadWrite.All",
        )
        self._chunk_size_chars = self._read_int("CHUNK_SIZE_CHARS", 1400)
        self._chunk_overlap_chars = self._read_int("CHUNK_OVERLAP_CHARS", 200)
        self._chunks_collection_name = self._read_str("CHUNKS_COLLECTION_NAME", "document_chunks")
        self._enable_chunk_text_index = self._read_bool("ENABLE_CHUNK_TEXT_INDEX", True)
        self._enable_langchain = self._read_bool("ENABLE_LANGCHAIN", False)
        self._langchain_prompt_cache_ttl_seconds = self._read_int("LANGCHAIN_PROMPT_CACHE_TTL_SECONDS", 900)
        self._langchain_prompt_cache_max_size = self._read_int("LANGCHAIN_PROMPT_CACHE_MAX_SIZE", 128)
        self._generate_video_transcripts = self._read_bool("GENERATE_VIDEO_TRANSCRIPTS", True)
        self._save_original_documents = self._read_bool("SAVE_ORIGINAL_DOCUMENTS", True)
        self._guided_tour_enabled = self._read_bool("GUIDED_TOUR_ENABLED", True)

    def _normalize_key(self, key: str) -> str:
        return str(key or "").strip().upper()

    def _normalize_email_domain(self, value: Any) -> str:
        domain = str(value or "").strip().lower()
        if domain.startswith("@"):
            domain = domain[1:]
        return domain

    def _read_str(self, key: str, default: str = "") -> str:
        return str(os.getenv(self._normalize_key(key), default) or default).strip()

    def _read_int(self, key: str, default: int = 0) -> int:
        try:
            return int(str(os.getenv(self._normalize_key(key), default)).strip())
        except (TypeError, ValueError):
            return int(default)

    def _read_float(self, key: str, default: float = 0.0) -> float:
        try:
            return float(str(os.getenv(self._normalize_key(key), default)).strip())
        except (TypeError, ValueError):
            return float(default)

    def _read_bool(self, key: str, default: bool = False) -> bool:
        raw = str(os.getenv(self._normalize_key(key), str(default))).strip().lower()
        if raw in {"1", "true", "yes", "y", "on"}:
            return True
        if raw in {"0", "false", "no", "n", "off"}:
            return False
        return bool(default)

    def _read_list(self, key: str, default: list[str] | None = None) -> list[str]:
        fallback = list(default or [])
        raw = str(os.getenv(self._normalize_key(key), "") or "").strip()
        if not raw:
            return fallback
        return [item.strip() for item in raw.split(",") if item.strip()]

    def _set_env(self, key: str, value: Any) -> None:
        env_key = self._normalize_key(key)
        if isinstance(value, bool):
            os.environ[env_key] = "true" if value else "false"
        elif isinstance(value, (list, tuple, set)):
            os.environ[env_key] = ",".join(str(item).strip() for item in value if str(item).strip())
        elif value is None:
            os.environ[env_key] = ""
        else:
            os.environ[env_key] = str(value)

    def _serialize_env_value(self, value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (list, tuple, set)):
            return ",".join(str(item).strip() for item in value if str(item).strip())
        if value is None:
            return ""
        return str(value)

    def _persist_env_value(self, key: str, value: Any) -> None:
        env_key = self._normalize_key(key)
        serialized_value = self._serialize_env_value(value)
        self._env_file.parent.mkdir(parents=True, exist_ok=True)

        existing_lines = []
        if self._env_file.exists():
            existing_lines = self._env_file.read_text(encoding="utf-8").splitlines()

        updated_lines = []
        found = False
        for line in existing_lines:
            stripped = line.strip()
            if "=" in line and not stripped.startswith("#"):
                existing_key = line.split("=", 1)[0].strip()
                if existing_key == env_key:
                    if not found:
                        updated_lines.append(f"{env_key}={serialized_value}")
                        found = True
                    continue
            updated_lines.append(line)

        if not found:
            updated_lines.append(f"{env_key}={serialized_value}")

        self._env_file.write_text("\n".join(updated_lines).rstrip() + "\n", encoding="utf-8")

    def get_string(self, key: str, default: str = "") -> str:
        return self._read_str(key, default)

    def get_int(self, key: str, default: int = 0) -> int:
        return self._read_int(key, default)

    def get_float(self, key: str, default: float = 0.0) -> float:
        return self._read_float(key, default)

    def get_bool(self, key: str, default: bool = False) -> bool:
        return self._read_bool(key, default)

    def get_list(self, key: str, default: list[str] | None = None) -> list[str]:
        return self._read_list(key, default)

    def set_value(self, key: str, value: Any) -> None:
        self._set_env(key, value)
        self._bootstrap()

    def set_persistent_value(self, key: str, value: Any) -> None:
        self._set_env(key, value)
        self._persist_env_value(key, value)
        self._bootstrap()

    @property
    def mongo_uri(self) -> str:
        return self._mongo_uri

    @mongo_uri.setter
    def mongo_uri(self, value: str) -> None:
        self._mongo_uri = str(value or "").strip()
        self._set_env("MONGO_URI", self._mongo_uri)

    @property
    def db_name(self) -> str:
        return self._db_name

    @db_name.setter
    def db_name(self, value: str) -> None:
        self._db_name = str(value or "").strip()
        self._set_env("DB_NAME", self._db_name)

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @collection_name.setter
    def collection_name(self, value: str) -> None:
        self._collection_name = str(value or "").strip()
        self._set_env("COLLECTION_NAME", self._collection_name)

    @property
    def gemini_api_key(self) -> str:
        return self._gemini_api_key

    @gemini_api_key.setter
    def gemini_api_key(self, value: str) -> None:
        self._gemini_api_key = str(value or "").strip()
        self._set_env("GEMINI_API_KEY", self._gemini_api_key)

    @property
    def gemini_model_name(self) -> str:
        return self._gemini_model_name

    @gemini_model_name.setter
    def gemini_model_name(self, value: str) -> None:
        self._gemini_model_name = str(value or "").strip() or "gemini-2.5-flash"
        self._set_env("GEMINI_MODEL_NAME", self._gemini_model_name)

    @property
    def version(self) -> str:
        return self._version

    @version.setter
    def version(self, value: str) -> None:
        self._version = str(value or "").strip()
        self._set_env("VERSION", self._version)

    @property
    def hf_embedding_model(self) -> str:
        return self._hf_embedding_model

    @hf_embedding_model.setter
    def hf_embedding_model(self, value: str) -> None:
        self._hf_embedding_model = str(value or "").strip()
        self._set_env("HF_EMBEDDING_MODEL", self._hf_embedding_model)

    @property
    def llm_structurer_max_concurrency(self) -> int:
        return self._llm_structurer_max_concurrency

    @llm_structurer_max_concurrency.setter
    def llm_structurer_max_concurrency(self, value: int) -> None:
        self._llm_structurer_max_concurrency = int(value or 0)
        self._set_env("LLM_STRUCTURER_MAX_CONCURRENCY", self._llm_structurer_max_concurrency)

    @property
    def session_ttl_hours(self) -> int:
        return self._session_ttl_hours

    @session_ttl_hours.setter
    def session_ttl_hours(self, value: int) -> None:
        self._session_ttl_hours = int(value or 0)
        self._set_env("SESSION_TTL_HOURS", self._session_ttl_hours)

    @property
    def inactivity_ttl_hours(self) -> int:
        return self._inactivity_ttl_hours

    @inactivity_ttl_hours.setter
    def inactivity_ttl_hours(self, value: int) -> None:
        self._inactivity_ttl_hours = int(value or 0)
        self._set_env("INACTIVITY_TTL_HOURS", self._inactivity_ttl_hours)

    @property
    def jwt_cookie_name(self) -> str:
        return self._jwt_cookie_name

    @jwt_cookie_name.setter
    def jwt_cookie_name(self, value: str) -> None:
        self._jwt_cookie_name = str(value or "").strip() or "doc_management_token"
        self._set_env("JWT_COOKIE_NAME", self._jwt_cookie_name)

    @property
    def jwt_secret(self) -> str:
        return self._jwt_secret

    @jwt_secret.setter
    def jwt_secret(self, value: str) -> None:
        self._jwt_secret = str(value or "").strip()
        self._set_env("JWT_SECRET", self._jwt_secret)

    @property
    def jwt_issuer(self) -> str:
        return self._jwt_issuer

    @jwt_issuer.setter
    def jwt_issuer(self, value: str) -> None:
        self._jwt_issuer = str(value or "").strip() or "doc_management"
        self._set_env("JWT_ISSUER", self._jwt_issuer)

    @property
    def allowed_file_extensions(self) -> list[str]:
        return list(self._allowed_file_extensions)

    @allowed_file_extensions.setter
    def allowed_file_extensions(self, value: list[str] | tuple[str, ...] | set[str]) -> None:
        self._allowed_file_extensions = [str(item).strip() for item in value if str(item).strip()]
        self._set_env("ALLOWED_FILE_EXTENSIONS", self._allowed_file_extensions)

    @property
    def qa_response_cache_ttl_seconds(self) -> int:
        return self._qa_response_cache_ttl_seconds

    @qa_response_cache_ttl_seconds.setter
    def qa_response_cache_ttl_seconds(self, value: int) -> None:
        self._qa_response_cache_ttl_seconds = int(value or 0)
        self._set_env("QA_RESPONSE_CACHE_TTL_SECONDS", self._qa_response_cache_ttl_seconds)

    @property
    def qa_response_cache_max_size(self) -> int:
        return self._qa_response_cache_max_size

    @qa_response_cache_max_size.setter
    def qa_response_cache_max_size(self, value: int) -> None:
        self._qa_response_cache_max_size = int(value or 0)
        self._set_env("QA_RESPONSE_CACHE_MAX_SIZE", self._qa_response_cache_max_size)

    @property
    def qa_cache_max_size(self) -> int:
        return self._qa_cache_max_size

    @qa_cache_max_size.setter
    def qa_cache_max_size(self, value: int) -> None:
        self._qa_cache_max_size = int(value or 0)
        self._set_env("QA_CACHE_MAX_SIZE", self._qa_cache_max_size)

    @property
    def qa_cache_ttl_seconds(self) -> int:
        return self._qa_cache_ttl_seconds

    @qa_cache_ttl_seconds.setter
    def qa_cache_ttl_seconds(self, value: int) -> None:
        self._qa_cache_ttl_seconds = int(value or 0)
        self._set_env("QA_CACHE_TTL_SECONDS", self._qa_cache_ttl_seconds)

    @property
    def qa_history_ttl_hours(self) -> int:
        return self._qa_history_ttl_hours

    @qa_history_ttl_hours.setter
    def qa_history_ttl_hours(self, value: int) -> None:
        self._qa_history_ttl_hours = int(value or 0)
        self._set_env("QA_HISTORY_TTL_HOURS", self._qa_history_ttl_hours)

    @property
    def qa_history_max_users(self) -> int:
        return self._qa_history_max_users

    @qa_history_max_users.setter
    def qa_history_max_users(self, value: int) -> None:
        self._qa_history_max_users = int(value or 0)
        self._set_env("QA_HISTORY_MAX_USERS", self._qa_history_max_users)

    @property
    def qa_history_max_conversations_per_user(self) -> int:
        return self._qa_history_max_conversations_per_user

    @qa_history_max_conversations_per_user.setter
    def qa_history_max_conversations_per_user(self, value: int) -> None:
        self._qa_history_max_conversations_per_user = int(value or 0)
        self._set_env("QA_HISTORY_MAX_CONVERSATIONS_PER_USER", self._qa_history_max_conversations_per_user)

    @property
    def qa_history_max_turns_per_conversation(self) -> int:
        return self._qa_history_max_turns_per_conversation

    @qa_history_max_turns_per_conversation.setter
    def qa_history_max_turns_per_conversation(self, value: int) -> None:
        self._qa_history_max_turns_per_conversation = int(value or 0)
        self._set_env("QA_HISTORY_MAX_TURNS_PER_CONVERSATION", self._qa_history_max_turns_per_conversation)

    @property
    def embedding_max_chars(self) -> int:
        return self._embedding_max_chars

    @embedding_max_chars.setter
    def embedding_max_chars(self, value: int) -> None:
        self._embedding_max_chars = int(value or 0)
        self._set_env("EMBEDDING_MAX_CHARS", self._embedding_max_chars)

    @property
    def embedding_min_similarity(self) -> float:
        return self._embedding_min_similarity

    @embedding_min_similarity.setter
    def embedding_min_similarity(self, value: float) -> None:
        self._embedding_min_similarity = float(value or 0.0)
        self._set_env("EMBEDDING_MIN_SIMILARITY", self._embedding_min_similarity)

    @property
    def enable_embeddings(self) -> bool:
        return self._enable_embeddings

    @enable_embeddings.setter
    def enable_embeddings(self, value: bool) -> None:
        self._enable_embeddings = bool(value)
        self._set_env("ENABLE_EMBEDDINGS", self._enable_embeddings)

    @property
    def ocr_language(self) -> str:
        return self._ocr_language

    @ocr_language.setter
    def ocr_language(self, value: str) -> None:
        self._ocr_language = str(value or "").strip() or "en"
        self._set_env("OCR_LANGUAGE", self._ocr_language)

    @property
    def ocr_use_gpu(self) -> bool:
        return self._ocr_use_gpu

    @ocr_use_gpu.setter
    def ocr_use_gpu(self, value: bool) -> None:
        self._ocr_use_gpu = bool(value)
        self._set_env("OCR_USE_GPU", self._ocr_use_gpu)

    @property
    def ocr_show_log(self) -> bool:
        return self._ocr_show_log

    @ocr_show_log.setter
    def ocr_show_log(self, value: bool) -> None:
        self._ocr_show_log = bool(value)
        self._set_env("OCR_SHOW_LOG", self._ocr_show_log)

    @property
    def allowed_user_email_domain(self) -> str:
        return self._allowed_user_email_domain

    @allowed_user_email_domain.setter
    def allowed_user_email_domain(self, value: str) -> None:
        self._allowed_user_email_domain = self._normalize_email_domain(value)
        self._set_env("ALLOWED_USER_EMAIL_DOMAIN", self._allowed_user_email_domain)

    @property
    def test_document_path(self) -> str:
        return self._test_document_path

    @test_document_path.setter
    def test_document_path(self, value: str) -> None:
        self._test_document_path = str(value or "").strip()
        self._set_env("TEST_DOCUMENT_PATH", self._test_document_path)

    @property
    def microsoft_tenant_id(self) -> str:
        return self._microsoft_tenant_id

    @microsoft_tenant_id.setter
    def microsoft_tenant_id(self, value: str) -> None:
        self._microsoft_tenant_id = str(value or "").strip() or "common"
        self._set_env("MICROSOFT_TENANT_ID", self._microsoft_tenant_id)

    @property
    def microsoft_client_id(self) -> str:
        return self._microsoft_client_id

    @microsoft_client_id.setter
    def microsoft_client_id(self, value: str) -> None:
        self._microsoft_client_id = str(value or "").strip()
        self._set_env("MICROSOFT_CLIENT_ID", self._microsoft_client_id)

    @property
    def microsoft_client_secret(self) -> str:
        return self._microsoft_client_secret

    @microsoft_client_secret.setter
    def microsoft_client_secret(self, value: str) -> None:
        self._microsoft_client_secret = str(value or "").strip()
        self._set_env("MICROSOFT_CLIENT_SECRET", self._microsoft_client_secret)

    @property
    def microsoft_redirect_uri(self) -> str:
        return self._microsoft_redirect_uri

    @microsoft_redirect_uri.setter
    def microsoft_redirect_uri(self, value: str) -> None:
        self._microsoft_redirect_uri = str(value or "").strip()
        self._set_env("MICROSOFT_REDIRECT_URI", self._microsoft_redirect_uri)

    @property
    def microsoft_scopes(self) -> str:
        return self._microsoft_scopes

    @microsoft_scopes.setter
    def microsoft_scopes(self, value: str) -> None:
        self._microsoft_scopes = str(value or "").strip()
        self._set_env("MICROSOFT_SCOPES", self._microsoft_scopes)

    @property
    def chunk_size_chars(self) -> int:
        return self._chunk_size_chars

    @chunk_size_chars.setter
    def chunk_size_chars(self, value: int) -> None:
        self._chunk_size_chars = int(value or 0)
        self._set_env("CHUNK_SIZE_CHARS", self._chunk_size_chars)

    @property
    def chunk_overlap_chars(self) -> int:
        return self._chunk_overlap_chars

    @chunk_overlap_chars.setter
    def chunk_overlap_chars(self, value: int) -> None:
        self._chunk_overlap_chars = int(value or 0)
        self._set_env("CHUNK_OVERLAP_CHARS", self._chunk_overlap_chars)

    @property
    def chunks_collection_name(self) -> str:
        return self._chunks_collection_name

    @chunks_collection_name.setter
    def chunks_collection_name(self, value: str) -> None:
        self._chunks_collection_name = str(value or "").strip()
        self._set_env("CHUNKS_COLLECTION_NAME", self._chunks_collection_name)

    @property
    def enable_chunk_text_index(self) -> bool:
        return self._enable_chunk_text_index

    @enable_chunk_text_index.setter
    def enable_chunk_text_index(self, value: bool) -> None:
        self._enable_chunk_text_index = bool(value)
        self._set_env("ENABLE_CHUNK_TEXT_INDEX", self._enable_chunk_text_index)

    @property
    def enable_langchain(self) -> bool:
        return self._enable_langchain

    @enable_langchain.setter
    def enable_langchain(self, value: bool) -> None:
        self._enable_langchain = bool(value)
        self._set_env("ENABLE_LANGCHAIN", self._enable_langchain)

    @property
    def langchain_prompt_cache_ttl_seconds(self) -> int:
        return self._langchain_prompt_cache_ttl_seconds

    @langchain_prompt_cache_ttl_seconds.setter
    def langchain_prompt_cache_ttl_seconds(self, value: int) -> None:
        self._langchain_prompt_cache_ttl_seconds = int(value or 0)
        self._set_env("LANGCHAIN_PROMPT_CACHE_TTL_SECONDS", self._langchain_prompt_cache_ttl_seconds)

    @property
    def langchain_prompt_cache_max_size(self) -> int:
        return self._langchain_prompt_cache_max_size

    @langchain_prompt_cache_max_size.setter
    def langchain_prompt_cache_max_size(self, value: int) -> None:
        self._langchain_prompt_cache_max_size = int(value or 0)
        self._set_env("LANGCHAIN_PROMPT_CACHE_MAX_SIZE", self._langchain_prompt_cache_max_size)

    @property
    def generate_video_transcripts(self) -> bool:
        return self._generate_video_transcripts

    @generate_video_transcripts.setter
    def generate_video_transcripts(self, value: bool) -> None:
        self._generate_video_transcripts = bool(value)
        self._set_env("GENERATE_VIDEO_TRANSCRIPTS", self._generate_video_transcripts)

    @property
    def save_original_documents(self) -> bool:
        return self._save_original_documents

    @save_original_documents.setter
    def save_original_documents(self, value: bool) -> None:
        self._save_original_documents = bool(value)
        self._set_env("SAVE_ORIGINAL_DOCUMENTS", self._save_original_documents)

    @property
    def guided_tour_enabled(self) -> bool:
        return self._guided_tour_enabled

    @guided_tour_enabled.setter
    def guided_tour_enabled(self, value: bool) -> None:
        self._guided_tour_enabled = bool(value)
        self._set_env("GUIDED_TOUR_ENABLED", self._guided_tour_enabled)


settings = _Settings()
