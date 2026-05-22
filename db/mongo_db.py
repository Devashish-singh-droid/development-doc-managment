# utils/mongo_db.py
import os
import re
import json
import math
import time
import hashlib
import hmac
from pathlib import Path
from uuid import uuid4
from collections import defaultdict
from difflib import SequenceMatcher
from datetime import datetime, timedelta
from typing import Optional
from pymongo import MongoClient
import numpy as np
from config import settings
from services.rbac import role_to_user_type, validate_role_string
from services.qa_query_understander import analyze_query
from utils.text_chunker import build_text_chunks, DEFAULT_CHUNK_OVERLAP_CHARS, DEFAULT_CHUNK_SIZE_CHARS
try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None
try:
    import faiss
except Exception:
    faiss = None

# MongoDB Connection String (must be set in .env)
MONGO_URI = settings.mongo_uri
DB_NAME = settings.db_name
COLLECTION_NAME = settings.collection_name
EMPLOYEE_PROFILE_COLLECTION_NAME = settings.get_string("EMPLOYEE_PROFILE_COLLECTION_NAME", "employee_profiles")
PROCESSING_STATUS_COLLECTION_NAME = settings.get_string("PROCESSING_STATUS_COLLECTION_NAME", "processing_status")
EMPLOYEE_MANIFEST_COLLECTION_NAME = settings.get_string("EMPLOYEE_MANIFEST_COLLECTION_NAME", "employee_manifests")
CHUNKS_COLLECTION_NAME = settings.chunks_collection_name
MAX_SUPER_ADMIN_USERS = 2

# Logger setup
from utils.logger import get_logger
logger = get_logger('mongo_db')


class MongoDBManager:
    PASSWORD_SCHEME = "pbkdf2_sha256"
    PASSWORD_ITERATIONS = 260000
    _faiss_cache = {"index": None, "doc_ids": [], "built_at": 0.0, "dim": 0}

    def __init__(self, uri=None, db_name=None, collection_name=None):
        """Initialize MongoDB connection and ensure users collection exists"""
        self._uri = uri or settings.mongo_uri
        self._db_name = db_name or settings.db_name
        self._collection_name = collection_name or settings.collection_name
        self._reset_handles()
        self._connect()

    def _reset_handles(self):
        self.client = None
        self.db = None
        self.collection = None
        self.employee_profiles = None
        self.processing_status = None
        self.employee_manifests = None
        self.document_chunks = None
        self.users = None
        self.activity_log = None

    def _connect(self):
        try:
            self.client = MongoClient(self._uri)
            self.db = self.client[self._db_name]
            self.collection = self.db[self._collection_name]
            self.employee_profiles = self.db[EMPLOYEE_PROFILE_COLLECTION_NAME]
            self.processing_status = self.db[PROCESSING_STATUS_COLLECTION_NAME]
            self.employee_manifests = self.db[EMPLOYEE_MANIFEST_COLLECTION_NAME]
            self.document_chunks = self.db[CHUNKS_COLLECTION_NAME]
            # Ensure 'users' collection exists
            if "users" not in self.db.list_collection_names():
                self.db.create_collection("users")
                logger.info("'users' collection created in database.")
            self.users = self.db["users"]
            # Ensure 'activity_log' collection exists
            if "activity_log" not in self.db.list_collection_names():
                self.db.create_collection("activity_log")
                logger.info("'activity_log' collection created in database.")
            self.activity_log = self.db["activity_log"]
            self._ensure_search_index()
            self._ensure_employee_indexes()
            self._ensure_chunk_indexes()
            self._ensure_user_indexes()  # NEW: User RBAC indexes
            # Test connection
            self.client.admin.command('ping')
            logger.info("MongoDB connected successfully!")
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
            self._reset_handles()

    def _ensure_connection(self) -> bool:
        if self.client:
            return True
        self._connect()
        return self.client is not None

    @staticmethod
    def _safe_datetime(value):
        if isinstance(value, datetime):
            return value.isoformat() + "Z"
        if value in [None, ""]:
            return None
        return str(value)

    def _ensure_search_index(self):
        """Create a wildcard text index so free-text search can match across document fields."""
        try:
            self.collection.create_index([("$**", "text")], name="wildcard_text_search")
        except Exception as e:
            # If another text index already exists with different options, keep app running.
            logger.warning(f"Could not create wildcard text index: {e}")

    def _ensure_employee_indexes(self):
        try:
            self.collection.create_index("empID", name="documents_empid_idx")
            self.collection.create_index("employee_uuid", name="documents_employee_uuid_idx")
            self.collection.create_index(
                "document_uuid",
                unique=True,
                sparse=True,
                name="documents_uuid_idx",
            )
            self.collection.create_index(
                [("isTemporary", 1), ("expiry_at", 1)],
                name="documents_temporary_expiry_idx",
            )
            self.collection.create_index(
                [("kbac_scope", 1), ("kbac_owner", 1), ("saved_at", -1)],
                name="documents_kbac_scope_owner_idx",
            )
            self.employee_profiles.create_index("empID", unique=True, name="employee_empid_idx")
            self.employee_profiles.create_index("uuid", unique=True, name="employee_uuid_idx")
            self.employee_profiles.create_index("created_by", name="employee_created_by_idx")
            self.employee_manifests.create_index(
                "employee_uuid",
                unique=True,
                sparse=True,
                name="manifest_employee_uuid_idx",
            )
            self.employee_manifests.create_index(
                "empID",
                sparse=True,
                name="manifest_empid_idx",
            )
            self.employee_manifests.create_index(
                "empName_normalized",
                sparse=True,
                name="manifest_empname_norm_idx",
            )
            self.employee_manifests.create_index(
                "documents.document_number",
                sparse=True,
                name="manifest_doc_number_idx",
            )
            self.employee_manifests.create_index(
                "documents.search_terms",
                sparse=True,
                name="manifest_doc_terms_idx",
            )
        except Exception as e:
            logger.warning(f"Could not create employee/document indexes: {e}")

    def _ensure_chunk_indexes(self):
        try:
            self.document_chunks.create_index("doc_id", name="chunk_doc_id_idx")
            self.document_chunks.create_index("document_uuid", name="chunk_doc_uuid_idx")
            self.document_chunks.create_index(
                [("doc_id", 1), ("chunk_index", 1)],
                unique=True,
                name="chunk_doc_index_idx",
            )
            self.document_chunks.create_index(
                [("kbac_scope", 1), ("kbac_owner", 1), ("doc_id", 1)],
                name="chunk_kbac_scope_owner_doc_idx",
            )
            if settings.enable_chunk_text_index:
                self.document_chunks.create_index(
                    [("chunk_text", "text")],
                    name="chunk_text_search",
                )
        except Exception as e:
            logger.warning(f"Could not create chunk indexes: {e}")

    def _ensure_user_indexes(self):
        """Create indexes for users collection (RBAC system)"""
        try:
            self.users.create_index("email", unique=True, name="users_email_idx")
            self.users.create_index("role", name="users_role_idx")
            self.users.create_index("created_by", name="users_created_by_idx")
            self.users.create_index([("role", 1), ("created_by", 1)], name="users_role_creator_idx")
            logger.info("User RBAC indexes created successfully")
        except Exception as e:
            logger.warning(f"Could not create user RBAC indexes: {e}")

    @staticmethod
    def _active_documents_query(query: dict = None) -> dict:
        active_filter = {
            "$or": [
                {"isTemporary": {"$ne": True}},
                {"expiry_at": None},
                {"expiry_at": {"$gt": datetime.utcnow()}},
            ]
        }
        if not query:
            return active_filter
        return {"$and": [query, active_filter]}

    @staticmethod
    def _is_document_active(doc: dict) -> bool:
        if not isinstance(doc, dict):
            return False
        if not doc.get("isTemporary"):
            return True
        expiry_at = doc.get("expiry_at")
        if not expiry_at:
            return True
        if isinstance(expiry_at, datetime):
            return expiry_at > datetime.utcnow()
        return True

    def _contains_term(self, value, term: str) -> bool:
        """Recursively check whether term appears in a nested MongoDB document value."""
        if value is None:
            return False
        if isinstance(value, str):
            return term in value.lower()
        if isinstance(value, (int, float, bool)):
            return term in str(value).lower()
        if isinstance(value, datetime):
            return term in value.isoformat().lower()
        if isinstance(value, list):
            return any(self._contains_term(item, term) for item in value)
        if isinstance(value, dict):
            for key, nested_value in value.items():
                if term in str(key).lower() or self._contains_term(nested_value, term):
                    return True
            return False
        return term in str(value).lower()

    def _extract_match_preview(self, doc: dict, term: str, window: int = 80) -> str:
        """Build a compact preview around the first matched occurrence."""
        if not term:
            return ""

        term_lower = term.lower()
        metadata = doc.get("high_level_metadata") if isinstance(doc.get("high_level_metadata"), dict) else {}
        content = doc.get("content") if isinstance(doc.get("content"), dict) else {}
        full_text = self._preferred_document_text(content)

        sources = []
        if isinstance(full_text, str) and full_text.strip():
            sources.append(full_text)
        if metadata:
            try:
                sources.append(json.dumps(metadata, ensure_ascii=False))
            except Exception:
                sources.append(str(metadata))
        if doc.get("document_type"):
            sources.append(str(doc.get("document_type")))

        for source in sources:
            clean_source = source.replace("\r", " ").replace("\n", " ")
            idx = clean_source.lower().find(term_lower)
            if idx == -1:
                continue

            start = max(0, idx - window)
            end = min(len(clean_source), idx + len(term) + window)
            snippet = clean_source[start:end].strip()

            if start > 0:
                snippet = "..." + snippet
            if end < len(clean_source):
                snippet = snippet + "..."
            return snippet

        return ""

    def _normalize_text(self, value) -> str:
        """Normalize text for matching/similarity checks."""
        if value is None:
            return ""
        text = str(value).replace("\r", " ").replace("\n", " ").lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _normalize_kbac_owner(self, value) -> Optional[str]:
        owner = str(value or "").strip().lower()
        return owner or None

    def _normalize_kbac_scope(self, value) -> str:
        raw = str(value or "").strip().lower()
        if raw == "team":
            return "team"
        return "global"

    def _normalize_access_context(self, access_context: Optional[dict] = None) -> dict:
        if not isinstance(access_context, dict):
            return {"allow_all": True, "team_owner": None}
        return {
            "allow_all": bool(access_context.get("allow_all")),
            "team_owner": self._normalize_kbac_owner(access_context.get("team_owner")),
        }

    def _kbac_query(self, access_context: Optional[dict] = None) -> dict:
        context = self._normalize_access_context(access_context)
        if context.get("allow_all"):
            return {}
        clauses = [
            {"kbac_scope": "global"},
            {"kbac_scope": {"$exists": False}},
        ]
        team_owner = context.get("team_owner")
        if team_owner:
            clauses.append({"kbac_owner": team_owner})
        return {"$or": clauses}

    def _with_document_kbac(self, doc: dict) -> dict:
        if not isinstance(doc, dict):
            return {}
        source = doc.get("source") if isinstance(doc.get("source"), dict) else {}
        raw_scope = doc.get("kbac_scope") or source.get("kbac_scope")
        scope = self._normalize_kbac_scope(raw_scope) if raw_scope else ""
        owner = self._normalize_kbac_owner(doc.get("kbac_owner") or source.get("kbac_owner"))
        uploaded_by = self._normalize_kbac_owner(source.get("uploaded_by") or source.get("uploaded_by_normalized"))
        uploaded_role = None
        raw_uploaded_role = str(source.get("uploaded_by_role") or "").strip()
        if raw_uploaded_role:
            try:
                uploaded_role = validate_role_string(raw_uploaded_role)
            except Exception:
                uploaded_role = None

        if not scope:
            if str(uploaded_role or "") == "manager" and uploaded_by:
                scope = "team"
                owner = owner or uploaded_by
            else:
                scope = "global"

        if scope == "team" and not owner:
            if str(uploaded_role or "") == "manager":
                owner = uploaded_by
            elif str(uploaded_role or "") in {"admin", "super_admin"}:
                scope = "global"
        if scope == "team" and not owner:
            scope = "global"
        enriched = dict(doc)
        enriched["kbac_scope"] = scope
        enriched["kbac_owner"] = owner if scope == "team" else None
        return enriched

    def _manifest_entry_matches_access(self, entry: dict, access_context: Optional[dict] = None) -> bool:
        context = self._normalize_access_context(access_context)
        if context.get("allow_all"):
            return True
        scope = self._normalize_kbac_scope((entry or {}).get("kbac_scope"))
        if scope == "global":
            return True
        team_owner = context.get("team_owner")
        entry_owner = self._normalize_kbac_owner((entry or {}).get("kbac_owner"))
        return bool(team_owner and entry_owner == team_owner)

    def _document_matches_access(self, doc: dict, access_context: Optional[dict] = None) -> bool:
        context = self._normalize_access_context(access_context)
        if context.get("allow_all"):
            return True
        enriched = self._with_document_kbac(doc)
        if enriched.get("kbac_scope") == "global":
            return True
        team_owner = context.get("team_owner")
        return bool(team_owner and self._normalize_kbac_owner(enriched.get("kbac_owner")) == team_owner)

    def _compact_alnum(self, value) -> str:
        """Keep only alphanumeric characters for robust ID comparisons."""
        return re.sub(r"[^a-z0-9]", "", self._normalize_text(value))

    def _normalize_person_name(self, value) -> str:
        """Normalize person names for reliable employee/profile comparisons."""
        return re.sub(r"[^a-z0-9]+", " ", self._normalize_text(value)).strip()

    def _tokenize(self, value) -> list:
        """Tokenize normalized text into searchable terms."""
        return re.findall(r"[a-z0-9]{2,}", self._normalize_text(value))

    def _tfidf_idf(self, docs_tokens: list) -> dict:
        """Compute IDF map for a list of token lists."""
        doc_count = len(docs_tokens)
        if doc_count == 0:
            return {}
        df = {}
        for tokens in docs_tokens:
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1
        return {
            token: math.log((doc_count + 1) / (count + 1)) + 1.0
            for token, count in df.items()
        }

    def _tfidf_vector(self, tokens: list, idf: dict) -> dict:
        """Build a sparse TF-IDF vector from tokens."""
        if not tokens:
            return {}
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        denom = float(len(tokens))
        vec = {}
        for token, count in tf.items():
            if token not in idf:
                continue
            vec[token] = (count / denom) * idf[token]
        return vec

    def _cosine_similarity(self, vec_a: dict, vec_b: dict) -> float:
        """Cosine similarity for sparse vectors."""
        if not vec_a or not vec_b:
            return 0.0
        dot = 0.0
        for token, val in vec_a.items():
            other = vec_b.get(token)
            if other is not None:
                dot += val * other
        if dot <= 0:
            return 0.0
        norm_a = math.sqrt(sum(val * val for val in vec_a.values()))
        norm_b = math.sqrt(sum(val * val for val in vec_b.values()))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _dense_cosine_similarity(self, vec_a: list, vec_b: list) -> float:
        """Cosine similarity for dense vectors."""
        if not vec_a or not vec_b:
            return 0.0
        if len(vec_a) != len(vec_b):
            return 0.0
        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for a, b in zip(vec_a, vec_b):
            dot += a * b
            norm_a += a * a
            norm_b += b * b
        if dot <= 0.0 or norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))

    def _faiss_enabled(self) -> bool:
        return faiss is not None and settings.get_bool("ENABLE_FAISS", True)

    def _build_faiss_index(self, projection: dict):
        """Build or refresh FAISS index from stored embeddings."""
        if not self._faiss_enabled():
            return None, []

        now = time.time()
        cache_ttl = settings.get_int("FAISS_CACHE_SECS", 300)
        cache = self._faiss_cache
        if cache["index"] is not None and (now - cache["built_at"]) < cache_ttl:
            return cache["index"], cache["doc_ids"]

        max_docs = settings.get_int("FAISS_MAX_DOCS", 2000)
        cursor = self.collection.find(
            self._active_documents_query({"embedding": {"$exists": True}}),
            projection,
        ).sort("saved_at", -1).limit(max_docs)

        vectors = []
        doc_ids = []
        dim = 0
        for doc in cursor:
            emb = doc.get("embedding")
            if not isinstance(emb, list) or not emb:
                continue
            if dim == 0:
                dim = len(emb)
            if len(emb) != dim:
                continue
            vectors.append(emb)
            doc_ids.append(str(doc.get("_id")))

        if not vectors:
            cache["index"] = None
            cache["doc_ids"] = []
            cache["built_at"] = now
            cache["dim"] = 0
            return None, []

        mat = np.array(vectors, dtype="float32")
        faiss.normalize_L2(mat)
        index = faiss.IndexFlatIP(dim)
        index.add(mat)

        cache["index"] = index
        cache["doc_ids"] = doc_ids
        cache["built_at"] = now
        cache["dim"] = dim
        return index, doc_ids

    def _faiss_search(self, query_vec: list, projection: dict, top_k: int = 50):
        """Search FAISS index and return (doc_ids, scores) ordered by similarity."""
        if not self._faiss_enabled() or not query_vec:
            return [], []

        index, doc_ids = self._build_faiss_index(projection)
        if index is None or not doc_ids:
            return [], []

        vec = np.array([query_vec], dtype="float32")
        faiss.normalize_L2(vec)
        scores, idxs = index.search(vec, min(top_k, len(doc_ids)))
        result_ids = []
        result_scores = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0 or idx >= len(doc_ids):
                continue
            result_ids.append(doc_ids[idx])
            result_scores.append(score)
        return result_ids, result_scores

    def _extract_mixed_id_token(self, value: str) -> str:
        """Extract strongest mixed alpha-numeric token from query (e.g., invoice/document IDs)."""
        query = self._normalize_text(value)
        candidates = re.findall(r"\b[a-z0-9/-]{6,}\b", query, flags=re.IGNORECASE)
        ranked = []
        for token in candidates:
            compact = self._compact_alnum(token)
            if len(compact) < 6:
                continue
            if not any(ch.isalpha() for ch in compact):
                continue
            if not any(ch.isdigit() for ch in compact):
                continue
            ranked.append(token)
        if not ranked:
            return ""
        ranked.sort(key=len, reverse=True)
        return ranked[0]

    def _extract_email_terms(self, value: str) -> list:
        """Extract unique email addresses from a query."""
        query = self._normalize_text(value)
        emails = re.findall(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}", query, flags=re.IGNORECASE)
        seen = set()
        unique = []
        for email in emails:
            normalized_email = self._normalize_text(email)
            if not normalized_email or normalized_email in seen:
                continue
            seen.add(normalized_email)
            unique.append(normalized_email)
        return unique

    def _extract_explicit_id_terms(self, value: str) -> list:
        """Extract explicit mixed alpha-numeric identifiers from a query."""
        query = self._normalize_text(value)
        candidates = re.findall(r"\b[a-z0-9][a-z0-9/-]{5,}\b", query, flags=re.IGNORECASE)
        seen = set()
        identifiers = []
        for token in candidates:
            compact = self._compact_alnum(token)
            if len(compact) < 6:
                continue
            if not any(ch.isalpha() for ch in compact):
                continue
            if not any(ch.isdigit() for ch in compact):
                continue
            if compact in seen:
                continue
            seen.add(compact)
            identifiers.append(token)
        return identifiers

    def _normalize_query_token(self, token: str) -> str:
        """Normalize token to improve singular/plural matching."""
        token = self._normalize_text(token)
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        if token.endswith("s") and len(token) > 4:
            return token[:-1]
        return token

    def _significant_tokens(self, value) -> list:
        """Return meaningful tokens (stop words removed) for strict matching."""
        stop_words = {
            "the", "and", "for", "with", "from", "all", "show", "find", "get",
            "list", "in", "on", "of", "to", "a", "an", "is", "are", "by",
            "this", "that", "these", "those", "please", "documents", "document",
            "report", "reports", "me", "which", "number", "no", "id",
            "tell", "give", "want", "need", "may", "can", "could", "would",
            "know", "about", "kindly", "let", "name",
        }
        tokens = self._tokenize(value)
        normalized_tokens = []
        for token in tokens:
            if len(token) < 3 or token in stop_words:
                continue
            normalized_tokens.append(self._normalize_query_token(token))
        # Keep unique order.
        seen = set()
        out = []
        for token in normalized_tokens:
            if token in seen:
                continue
            seen.add(token)
            out.append(token)
        return out

    def _extract_subject_phrase(self, query: str) -> str:
        """Extract the main entity from queries like 'father name of pratyush mathur'."""
        q = self._normalize_text(query)
        if not q:
            return ""

        patterns = [
            r"\b(?:of|for|about|regarding)\s+([a-z][a-z\s.\-]{2,80}?)(?:\s+in\s+(?:19|20)\d{2}\b|$)",
            r"\b(?:for|about)\s+patient\s+([a-z][a-z\s.\-]{2,80}?)(?:\s+in\s+(?:19|20)\d{2}\b|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, q, flags=re.IGNORECASE)
            if match:
                candidate = match.group(1).strip(" .,-")
                if candidate and len(candidate) >= 3:
                    return candidate
        return ""

    def _extract_employee_identifier_terms(self, query: str) -> list:
        """Extract likely employee identifiers such as employee IDs or employee codes."""
        q = self._normalize_text(query)
        if not q:
            return []

        identifiers = []
        patterns = [
            r"\b(?:emp(?:loyee)?(?:\s+id)?|employee code)\s*(?:is|=|:)?\s*([a-z0-9._/-]{2,40})\b",
            r"\b([a-z]{2,10}[0-9]{2,20}|[0-9]{3,20}[a-z]{0,10})\b",
        ]
        for pattern in patterns:
            for match in re.findall(pattern, q, flags=re.IGNORECASE):
                compact = self._compact_alnum(match)
                if len(compact) < 2 or compact in identifiers:
                    continue
                identifiers.append(compact)
        return identifiers

    def _subject_matches_blob(self, subject: str, blob: str) -> bool:
        """Check whether a named subject truly exists in the document text."""
        subject_norm = self._normalize_text(subject)
        blob_norm = self._normalize_text(blob)
        if not subject_norm or not blob_norm:
            return False

        if subject_norm in blob_norm:
            return True

        subject_tokens = self._significant_tokens(subject_norm)
        if not subject_tokens:
            return False

        blob_tokens = set(self._tokenize(blob_norm))
        return all(self._token_matches_blob_tokens(token, blob_tokens) for token in subject_tokens)

    def _strict_keyword_match(self, doc: dict, raw_query: str) -> bool:
        """Ensure a document truly matches query text, avoiding loose OR matches."""
        query = self._normalize_text(raw_query)
        if not query:
            return True

        blob = self._build_document_blob(doc)
        if not blob:
            return False

        compact_blob = self._document_blob_compact(doc)
        blob_tokens = self._document_blob_tokens(doc)

        # Hard gate 1: explicit emails in query must exist exactly in document text.
        explicit_emails = self._extract_email_terms(query)
        for email in explicit_emails:
            if email not in blob:
                return False

        # Hard gate 2: explicit alpha-numeric IDs in query must exist in document text.
        explicit_ids = self._extract_explicit_id_terms(query)
        for explicit_id in explicit_ids:
            explicit_compact = self._compact_alnum(explicit_id)
            if explicit_compact and explicit_compact not in compact_blob:
                return False

        # If query provided hard entities (email/ID), those are enough for strict acceptance.
        if explicit_emails or explicit_ids:
            return True

        # Exact normalized phrase should match.
        if query in blob:
            return True

        # Compact phrase fallback for symbol-heavy queries.
        compact_query = self._compact_alnum(query)
        if compact_query and len(compact_query) >= 8 and compact_query in compact_blob:
            return True

        tokens = self._significant_tokens(query)
        if not tokens:
            return False

        # Strict AND across meaningful tokens with mild morphology tolerance.
        matched = [False] * len(tokens)
        for idx, token in enumerate(tokens):
            if self._token_matches_blob_tokens(token, blob_tokens):
                matched[idx] = True
                continue
            # Allow long token containment inside merged OCR tokens.
            if len(token) >= 7 and any(token in blob_token for blob_token in blob_tokens):
                matched[idx] = True

        # Handle merged adjacent tokens (e.g., "hexa"+"com" -> "hexacom").
        if len(tokens) >= 2 and not all(matched):
            for i in range(len(tokens) - 1):
                if matched[i] and matched[i + 1]:
                    continue
                combined = tokens[i] + tokens[i + 1]
                if combined in blob_tokens or (len(combined) >= 8 and combined in compact_blob):
                    matched[i] = True
                    matched[i + 1] = True

        return all(matched)

    def _constraint_matches(self, doc: dict, constraints: dict) -> dict:
        """Evaluate whether parsed semantic constraints are actually present in a document."""
        metadata = doc.get("high_level_metadata") if isinstance(doc.get("high_level_metadata"), dict) else {}
        blob = self._build_document_blob(doc)
        doc_type = self._normalize_text(doc.get("document_type", ""))
        vendor_text = self._normalize_text(metadata.get("vendor_name") or metadata.get("company_name") or "")
        patient_text = self._normalize_text(metadata.get("patient_name") or "")

        matches = {}
        if constraints.get("document_type"):
            requested = self._normalize_text(constraints["document_type"])
            matches["document_type"] = bool(requested and (requested in doc_type or doc_type in requested))
        if constraints.get("year"):
            matches["year"] = constraints["year"] in blob
        if constraints.get("vendor"):
            requested_vendor = self._normalize_text(constraints["vendor"])
            matches["vendor"] = bool(requested_vendor and (requested_vendor in vendor_text or requested_vendor in blob))
        if constraints.get("patient"):
            requested_patient = self._normalize_text(constraints["patient"])
            matches["patient"] = bool(requested_patient and (requested_patient in patient_text or requested_patient in blob))
        if constraints.get("subject"):
            matches["subject"] = self._subject_matches_blob(constraints["subject"], blob)
        if constraints.get("document_number"):
            requested_doc_no = self._normalize_text(constraints["document_number"])
            requested_compact = self._compact_alnum(requested_doc_no)
            blob_compact = self._document_blob_compact(doc)
            matches["document_number"] = bool(
                requested_doc_no
                and (
                    requested_doc_no in blob
                    or (requested_compact and requested_compact in blob_compact)
                )
            )
        return matches

    def _matches_document_number(self, doc: dict, requested_number: str) -> bool:
        """Robust match for document/invoice numbers (supports partial ID input)."""
        requested_compact = self._compact_alnum(requested_number)
        if not requested_compact:
            return False

        metadata = doc.get("high_level_metadata") if isinstance(doc.get("high_level_metadata"), dict) else {}
        candidate_fields = [
            metadata.get("invoice_number"),
            metadata.get("bill_number"),
            metadata.get("document_number"),
            metadata.get("po_number"),
            metadata.get("claim_id"),
            metadata.get("bid_number"),
            metadata.get("bid_no"),
            metadata.get("gem_bid_number"),
            metadata.get("gem_number"),
            metadata.get("tender_number"),
            metadata.get("tender_id"),
            metadata.get("rfq_number"),
        ]
        for candidate in candidate_fields:
            candidate_compact = self._compact_alnum(candidate)
            if not candidate_compact:
                continue
            if requested_compact in candidate_compact or candidate_compact in requested_compact:
                return True

        # Metadata can miss values; also match against full OCR blob.
        blob_compact = self._document_blob_compact(doc)
        if blob_compact and requested_compact in blob_compact:
            return True
        return False

    def _document_number_key(self, doc: dict) -> str:
        """Return a normalized primary document-number key for deduplication."""
        metadata = doc.get("high_level_metadata") if isinstance(doc.get("high_level_metadata"), dict) else {}
        for field in (
            "invoice_number",
            "bill_number",
            "document_number",
            "po_number",
            "claim_id",
            "bid_number",
            "bid_no",
            "gem_bid_number",
            "gem_number",
            "tender_number",
            "tender_id",
            "rfq_number",
        ):
            key = self._compact_alnum(metadata.get(field))
            if key:
                return key
        return self._compact_alnum(doc.get("_id"))

    def _token_matches_blob_tokens(self, token: str, blob_tokens: set) -> bool:
        """Check token presence with small morphological tolerance."""
        if token in blob_tokens:
            return True
        if token.endswith("y") and (token[:-1] + "ies") in blob_tokens:
            return True
        if (token + "s") in blob_tokens:
            return True
        return False

    def _metadata_values_to_text(self, value) -> str:
        """Flatten nested metadata to values-only text (no field names)."""
        chunks = []

        def _walk(node):
            if node is None:
                return
            if isinstance(node, dict):
                for nested in node.values():
                    _walk(nested)
                return
            if isinstance(node, list):
                for nested in node:
                    _walk(nested)
                return
            chunks.append(str(node))

        _walk(value)
        return " ".join(chunks)

    def _preferred_document_text(self, content: dict) -> str:
        if not isinstance(content, dict):
            return ""
        for field_name in ("full_text", "retrieval_text", "main_document_text", "preview_text"):
            value = content.get(field_name)
            if isinstance(value, str) and value.strip():
                return value
        return ""

    def _build_document_blob(self, doc: dict) -> str:
        """Combine key fields, metadata, and OCR text into one searchable blob."""
        cached = doc.get("_qa_blob")
        if isinstance(cached, str):
            return cached
        metadata = doc.get("high_level_metadata") if isinstance(doc.get("high_level_metadata"), dict) else {}
        content = doc.get("content") if isinstance(doc.get("content"), dict) else {}
        full_text = self._preferred_document_text(content)
        metadata_text = self._metadata_values_to_text(metadata)
        doc_type = doc.get("document_type")
        parts = [
            "" if doc_type is None else str(doc_type),
            "" if metadata_text is None else str(metadata_text),
            "" if full_text is None else str(full_text),
            str(doc.get("saved_at", "") or ""),
        ]
        blob = self._normalize_text(" ".join(parts))
        doc["_qa_blob"] = blob
        return blob

    def _document_blob_tokens(self, doc: dict) -> set:
        cached = doc.get("_qa_blob_tokens")
        if isinstance(cached, set):
            return cached
        tokens = set(self._tokenize(self._build_document_blob(doc)))
        doc["_qa_blob_tokens"] = tokens
        return tokens

    def _document_blob_compact(self, doc: dict) -> str:
        cached = doc.get("_qa_blob_compact")
        if isinstance(cached, str):
            return cached
        compact = self._compact_alnum(self._build_document_blob(doc))
        doc["_qa_blob_compact"] = compact
        return compact

    def _parse_semantic_constraints(self, query: str) -> dict:
        """Extract simple structured filters from natural-language query."""
        q = self._normalize_text(query)
        analysis = analyze_query(q)
        constraints = {
            "document_type": analysis.get("document_type") or None,
            "year": analysis.get("year") or None,
            "vendor": analysis.get("vendor") or None,
            "patient": analysis.get("patient") or None,
            "subject": analysis.get("subject") or None,
            "document_number": analysis.get("document_number") or None,
        }

        # If the query-understanding layer extracted useful structure, prefer it.
        if any(constraints.values()):
            return constraints

        doc_type_keywords = {
            "invoice": ["invoice", "invoices"],
            "bill": ["bill", "bills", "receipt"],
            "medical_report": ["medical report", "medical reports"],
            "discharge_summary": ["discharge summary", "discharge summaries"],
            "report": ["report", "reports"],
            "bid": ["bid", "tender", "rfq"],
        }
        for canonical_type, variants in doc_type_keywords.items():
            if any(variant in q for variant in variants):
                constraints["document_type"] = canonical_type
                break

        year_match = re.search(r"\b(?:19|20)\d{2}\b", q)
        if year_match:
            constraints["year"] = year_match.group(0)

        vendor_match = re.search(
            r"\bfrom\s+([a-z0-9&.,\-\s]{2,80}?)(?:\s+in\s+(?:19|20)\d{2}\b|\s+for\b|\s+with\b|$)",
            q,
            flags=re.IGNORECASE,
        )
        if vendor_match:
            constraints["vendor"] = vendor_match.group(1).strip(" .,-")

        patient_match = re.search(
            r"\bpatient\s+([a-z][a-z\s.\-]{1,80}?)(?:\s+in\s+(?:19|20)\d{2}\b|$)",
            q,
            flags=re.IGNORECASE,
        )
        if patient_match:
            constraints["patient"] = patient_match.group(1).strip(" .,-")
            constraints["subject"] = constraints["patient"]

        if not constraints["subject"]:
            constraints["subject"] = self._extract_subject_phrase(q)

        explicit_doc_no = re.search(
            r"\b(?:invoice|bill|document|doc)\s*(?:number|no|#)\s*(?:is|=|:)?\s*([a-z0-9][a-z0-9/-]{5,})\b",
            q,
            flags=re.IGNORECASE,
        )
        if explicit_doc_no:
            constraints["document_number"] = explicit_doc_no.group(1)

        doc_no_match = re.search(r"\b[a-z0-9]+(?:[/-][a-z0-9]+){2,}\b", q, flags=re.IGNORECASE)
        if doc_no_match and not constraints["document_number"]:
            constraints["document_number"] = doc_no_match.group(0)

        if not constraints["document_number"]:
            id_candidates = re.findall(r"\b[a-z0-9/-]{8,}\b", q, flags=re.IGNORECASE)
            for candidate in id_candidates:
                compact = self._compact_alnum(candidate)
                if len(compact) >= 8 and any(ch.isalpha() for ch in compact) and any(ch.isdigit() for ch in compact):
                    constraints["document_number"] = candidate
                    break

        return constraints

    def _is_scenario_query(self, query_analysis: dict, query: str = "") -> bool:
        """Identify hypothetical/procedural questions that need broader retrieval."""
        analysis = query_analysis if isinstance(query_analysis, dict) else {}
        if bool(analysis.get("scenario_question")):
            return True

        intent = self._normalize_text(analysis.get("intent"))
        if "scenario" in intent:
            return True

        normalized_query = self._normalize_text(query)
        scenario_markers = (
            "what if",
            "in case",
            "scenario",
            "suppose",
            "assume",
            "assuming",
            "what happens",
            "how should",
            "how do we handle",
            "how to handle",
        )
        return any(marker in normalized_query for marker in scenario_markers)

    def _semantic_score(
        self,
        doc: dict,
        query: str,
        constraints: dict,
        *,
        q_norm: str | None = None,
        query_tokens: set | None = None,
        blob: str | None = None,
        blob_tokens: set | None = None,
    ) -> tuple:
        """Score a document against a natural-language query."""
        q_norm = q_norm if q_norm is not None else self._normalize_text(query)
        blob = blob if blob is not None else self._build_document_blob(doc)
        query_tokens = query_tokens if query_tokens is not None else set(self._tokenize(q_norm))
        blob_tokens = blob_tokens if blob_tokens is not None else self._document_blob_tokens(doc)

        token_overlap = 0.0
        if query_tokens:
            token_overlap = len(query_tokens & blob_tokens) / max(1, len(query_tokens))

        fuzzy_score = 0.0
        if q_norm and blob:
            if fuzz is not None:
                fuzzy_score = fuzz.token_set_ratio(q_norm, blob[:12000]) / 100.0
            else:
                fuzzy_score = SequenceMatcher(None, q_norm, blob[:6000]).ratio()

        sequence_score = 0.0
        if q_norm and blob:
            sequence_score = SequenceMatcher(None, q_norm, blob[:6000]).ratio()

        score = (0.5 * token_overlap) + (0.35 * fuzzy_score) + (0.15 * sequence_score)
        reasons = []

        if q_norm and q_norm in blob:
            score += 0.25
            reasons.append("Exact phrase match")

        metadata = doc.get("high_level_metadata") if isinstance(doc.get("high_level_metadata"), dict) else {}
        doc_type = self._normalize_text(doc.get("document_type", ""))

        if constraints.get("document_type"):
            requested = self._normalize_text(constraints["document_type"])
            if requested in doc_type or doc_type in requested:
                score += 0.6
                reasons.append(f"Type matched ({requested})")
            else:
                score -= 0.4

        if constraints.get("year"):
            year = constraints["year"]
            if year in blob:
                score += 0.35
                reasons.append(f"Year matched ({year})")
            else:
                score -= 0.2

        if constraints.get("vendor"):
            requested_vendor = self._normalize_text(constraints["vendor"])
            vendor_text = self._normalize_text(
                metadata.get("vendor_name") or metadata.get("company_name") or ""
            )
            if requested_vendor and (requested_vendor in vendor_text or requested_vendor in blob):
                score += 0.8
                reasons.append(f"Vendor matched ({constraints['vendor']})")
            else:
                score -= 0.35

        if constraints.get("patient"):
            requested_patient = self._normalize_text(constraints["patient"])
            patient_text = self._normalize_text(metadata.get("patient_name") or "")
            if requested_patient and (requested_patient in patient_text or requested_patient in blob):
                score += 0.85
                reasons.append(f"Patient matched ({constraints['patient']})")
            else:
                score -= 0.35

        if constraints.get("subject"):
            requested_subject = self._normalize_text(constraints["subject"])
            if requested_subject and self._subject_matches_blob(requested_subject, blob):
                score += 0.95
                reasons.append(f"Subject matched ({constraints['subject']})")
            else:
                score -= 0.55

        if constraints.get("document_number"):
            requested_doc_no = self._normalize_text(constraints["document_number"])
            if requested_doc_no and requested_doc_no in blob:
                score += 0.7
                reasons.append(f"Document number matched ({constraints['document_number']})")
            else:
                score -= 0.35

        return score, reasons

    def _to_float_amount(self, value) -> float:
        """Parse loosely formatted amount strings into float."""
        if value is None:
            return 0.0
        cleaned = re.sub(r"[^0-9.]", "", str(value))
        if cleaned.count(".") > 1:
            first_dot = cleaned.find(".")
            cleaned = cleaned[:first_dot + 1] + cleaned[first_dot + 1:].replace(".", "")
        try:
            return float(cleaned) if cleaned else 0.0
        except ValueError:
            return 0.0

    def _doc_key_fields(self, doc: dict) -> dict:
        """Extract key fields used in duplicate/fraud heuristics."""
        metadata = doc.get("high_level_metadata") if isinstance(doc.get("high_level_metadata"), dict) else {}
        doc_number = (
            metadata.get("invoice_number")
            or metadata.get("bill_number")
            or metadata.get("document_number")
            or metadata.get("po_number")
            or metadata.get("claim_id")
            or metadata.get("bid_number")
            or metadata.get("bid_no")
            or metadata.get("gem_bid_number")
            or metadata.get("gem_number")
            or metadata.get("tender_number")
            or metadata.get("tender_id")
            or metadata.get("rfq_number")
        )
        vendor = metadata.get("vendor_name") or metadata.get("company_name")
        patient = metadata.get("patient_name")
        amount_raw = metadata.get("total_amount") or metadata.get("amount")
        report_date = metadata.get("report_date") or metadata.get("date")
        return {
            "document_number": str(doc_number or "").strip(),
            "vendor": str(vendor or "").strip(),
            "patient": str(patient or "").strip(),
            "amount_raw": str(amount_raw or "").strip(),
            "amount_value": self._to_float_amount(amount_raw),
            "report_date": str(report_date or "").strip(),
        }

    def _duplicate_fingerprint(self, doc: dict) -> str:
        """Create a stable fingerprint to identify true re-uploads of the same document."""
        key_fields = self._doc_key_fields(doc)
        doc_type = self._normalize_text(doc.get("document_type", "")) or "unknown"
        doc_number_key = self._compact_alnum(key_fields.get("document_number"))
        vendor_key = self._normalize_text(key_fields.get("vendor", ""))
        patient_key = self._normalize_text(key_fields.get("patient", ""))
        amount_value = key_fields.get("amount_value")
        amount_key = f"{float(amount_value):.2f}" if isinstance(amount_value, (int, float)) and float(amount_value) > 0 else ""
        date_key = self._normalize_text(key_fields.get("report_date", ""))

        # Document number alone is too risky because many vendors/systems reuse IDs like INV-001.
        # Require a supporting identity field before treating a matching number as a true duplicate.
        if doc_number_key:
            supporting_parts = [part for part in (vendor_key, patient_key, amount_key, date_key) if part]
            if supporting_parts:
                composite = "|".join([doc_type, doc_number_key, *supporting_parts[:3]])
                composite_hash = hashlib.sha256(composite.encode("utf-8")).hexdigest()
                return f"docno:{composite_hash}"

        content = doc.get("content") if isinstance(doc.get("content"), dict) else {}
        full_text = self._normalize_text(self._preferred_document_text(content))
        if full_text:
            text_hash = hashlib.sha256(full_text[:50000].encode("utf-8")).hexdigest()
            return f"text:{doc_type}:{text_hash}"

        # Last-resort fallback based on key metadata fields.
        amount_raw_key = self._normalize_text(key_fields.get("amount_raw", ""))
        meta_blob = f"{doc_type}|{vendor_key}|{patient_key}|{amount_key or amount_raw_key}|{date_key}"
        meta_hash = hashlib.sha256(meta_blob.encode("utf-8")).hexdigest()
        return f"meta:{meta_hash}"

    def _doc_brief(self, doc: dict) -> dict:
        """Compact document payload for duplicate-analysis responses."""
        key_fields = self._doc_key_fields(doc)
        return {
            "id": str(doc.get("_id")),
            "document_type": doc.get("document_type", "unknown"),
            "saved_at": str(doc.get("saved_at", "")),
            "document_number": key_fields["document_number"] or None,
            "vendor": key_fields["vendor"] or None,
            "patient": key_fields["patient"] or None,
            "amount": key_fields["amount_raw"] or None,
            "report_date": key_fields["report_date"] or None,
        }

    def _duplicate_match_payload(self, doc: dict, reason: str = "", file_hash: str = None) -> dict:
        brief = self._doc_brief(doc)
        return {
            "document_id": brief.get("id"),
            "document_uuid": str(doc.get("document_uuid") or "").strip() or None,
            "file_name": str(doc.get("file_name") or "").strip() or None,
            "tracking_filename": str(doc.get("tracking_filename") or "").strip() or None,
            "document_type": brief.get("document_type"),
            "saved_at": brief.get("saved_at"),
            "reason": reason or None,
            "file_hash": file_hash or str((((doc.get("source") or {}) if isinstance(doc.get("source"), dict) else {}).get("file_hash") or "")).strip() or None,
        }

    def _duplicate_owner_query(self, duplicate_owner: Optional[str]) -> Optional[dict]:
        safe_owner = self._normalize_kbac_owner(duplicate_owner)
        if not safe_owner:
            return None
        return {
            "$or": [
                {"source.uploaded_by_normalized": safe_owner},
                {
                    "source.uploaded_by": {
                        "$regex": f"^{re.escape(safe_owner)}$",
                        "$options": "i",
                    }
                },
            ]
        }

    def find_duplicate_by_file_hash(self, file_hash: str, duplicate_owner: Optional[str] = None) -> Optional[dict]:
        """Fast-path duplicate lookup for exact re-uploads using binary file hash."""
        if not self.client:
            return None
        safe_hash = str(file_hash or "").strip().lower()
        if not safe_hash:
            return None
        try:
            projection = {
                "_id": 1,
                "document_uuid": 1,
                "document_type": 1,
                "file_name": 1,
                "tracking_filename": 1,
                "saved_at": 1,
                "source.file_hash": 1,
                "source.original_filename": 1,
                "high_level_metadata": 1,
            }
            query = self._active_documents_query({"source.file_hash": safe_hash})
            owner_query = self._duplicate_owner_query(duplicate_owner)
            if owner_query:
                query = {"$and": [query, owner_query]}
            doc = self.collection.find_one(
                query,
                projection,
                sort=[("saved_at", -1)],
            )
            if not doc:
                return None
            return self._duplicate_match_payload(
                doc,
                reason="exact_file_hash_match",
                file_hash=safe_hash,
            )
        except Exception as exc:
            logger.warning(f"File-hash duplicate lookup failed: {exc}")
            return None

    def find_duplicate_for_document(
        self,
        document: dict,
        limit_scan: int = 1500,
        duplicate_owner: Optional[str] = None,
    ) -> Optional[dict]:
        """Detect duplicates for a structured document before final MongoDB insert."""
        if not self.client or not isinstance(document, dict):
            return None

        fingerprint = str(document.get("duplicate_fingerprint") or self._duplicate_fingerprint(document) or "").strip()
        if not fingerprint:
            return None

        projection = {
            "_id": 1,
            "document_uuid": 1,
            "document_type": 1,
            "file_name": 1,
            "tracking_filename": 1,
            "saved_at": 1,
            "source.file_hash": 1,
            "high_level_metadata": 1,
            "content.full_text": 1,
        }

        try:
            indexed_query = self._active_documents_query({"duplicate_fingerprint": fingerprint})
            owner_query = self._duplicate_owner_query(duplicate_owner)
            if owner_query:
                indexed_query = {"$and": [indexed_query, owner_query]}
            indexed_match = self.collection.find_one(
                indexed_query,
                projection,
                sort=[("saved_at", -1)],
            )
            if indexed_match:
                return self._duplicate_match_payload(indexed_match, reason="structured_document_fingerprint_match")

            # Fallback for legacy documents saved before duplicate_fingerprint existed.
            candidates = list(
                self.collection.find(
                    {"$and": [self._active_documents_query({}), owner_query]} if owner_query else self._active_documents_query({}),
                    projection,
                ).sort("saved_at", -1).limit(max(200, int(limit_scan)))
            )
            for candidate in candidates:
                try:
                    if self._duplicate_fingerprint(candidate) == fingerprint:
                        return self._duplicate_match_payload(candidate, reason="legacy_document_fingerprint_match")
                except Exception:
                    continue
        except Exception as exc:
            logger.warning(f"Structured duplicate lookup failed: {exc}")

        return None

    def _infer_field_type(self, value) -> str:
        """Infer data type: string, number, boolean, date, array"""
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, (int, float)):
            return "number"
        if isinstance(value, list):
            return "array"
        if isinstance(value, dict):
            return "object"
        if isinstance(value, str):
            value_lower = value.lower().strip()
            if re.match(r'^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', value_lower):
                return "date"
            if re.match(r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}', value_lower):
                return "date"
            if re.match(r'^[-+]?[\d,.]+$', value_lower):
                return "number"
            return "string"
        return "string"

    def _extract_sample_value(self, value, max_length: int = 100) -> str:
        """Extract safe sample value"""
        if value is None:
            return None
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, list):
            if not value:
                return None
            if isinstance(value[0], str):
                return str(value[0])[:max_length]
            return str(value[0])
        if isinstance(value, str):
            return value[:max_length]
        return str(value)[:max_length]

    def _manifest_doc_fields_enhanced(self, metadata: dict) -> list:
        """Extract detailed field info"""
        fields = []
        if not isinstance(metadata, dict):
            return fields
        
        seen_field_names = set()
        for key, value in metadata.items():
            if not key or key.startswith("_"):
                continue
            clean_key = str(key).strip()
            if clean_key in seen_field_names:
                continue
            seen_field_names.add(clean_key)
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            
            field_type = self._infer_field_type(value)
            sample_value = self._extract_sample_value(value)
            field_obj = {
                "name": clean_key,
                "type": field_type,
                "indexed": True,
            }
            if sample_value:
                field_obj["sample_value"] = sample_value
            if field_type == "number" and isinstance(value, (int, float)):
                field_obj["value"] = value
            
            fields.append(field_obj)
            if len(fields) >= 30:
                break
        
        return fields

    def _manifest_doc_labels(self, metadata: dict) -> list:
        labels = []
        if isinstance(metadata, dict):
            for key in metadata.keys():
                clean_key = str(key or "").strip()
                if clean_key and clean_key not in labels:
                    labels.append(clean_key)
                if len(labels) >= 16:
                    break
        return labels

    def _manifest_doc_search_terms(
        self,
        doc: dict,
        metadata: dict,
        key_fields: dict,
        summary_blob: str,
        headings: list,
        titles: list,
    ) -> list:
        terms = []
        seen = set()

        def _add(value):
            for candidate in [str(value or "").strip(), self._compact_alnum(value)]:
                text = str(candidate or "").strip().lower()
                if len(text) < 2 or text in seen:
                    continue
                seen.add(text)
                terms.append(text)

        _add(doc.get("document_type"))
        _add(doc.get("file_name"))
        _add(doc.get("tracking_filename"))
        source = doc.get("source") if isinstance(doc.get("source"), dict) else {}
        _add(source.get("original_filename"))
        for value in [
            key_fields.get("document_number"),
            key_fields.get("vendor"),
            key_fields.get("patient"),
        ]:
            _add(value)

        if isinstance(metadata, dict):
            for key, value in list(metadata.items())[:20]:
                _add(key)
                if isinstance(value, (str, int, float)):
                    _add(value)

        for title in titles or []:
            _add(title)

        for heading in headings or []:
            _add(heading)

        for token in self._significant_tokens(summary_blob)[:40]:
            _add(token)
        return terms

    def _extract_document_titles(self, metadata: dict) -> list:
        if not isinstance(metadata, dict):
            return []
        title_keys = [
            "title",
            "document_title",
            "doc_title",
            "subject",
            "heading",
            "header",
            "name",
        ]
        titles = []
        seen = set()
        for key in title_keys:
            value = metadata.get(key)
            if not isinstance(value, str):
                continue
            cleaned = " ".join(value.split()).strip()
            if not cleaned:
                continue
            if len(cleaned) > 160:
                cleaned = cleaned[:160].rstrip()
            key_norm = cleaned.lower()
            if key_norm in seen:
                continue
            seen.add(key_norm)
            titles.append(cleaned)
        return titles

    def _extract_headings_from_text(self, text: str, max_headings: int = 14) -> list:
        raw = str(text or "").strip()
        if not raw:
            return []
        headings = []
        seen = set()
        for line in raw.splitlines():
            candidate = " ".join(line.strip().split())
            if len(candidate) < 4 or len(candidate) > 140:
                continue
            if candidate.endswith(":"):
                candidate = candidate[:-1].strip()
            if not candidate:
                continue

            is_heading = False
            if candidate.isupper() and len(candidate.split()) <= 10:
                is_heading = True
            elif re.match(r"^\d+(\.\d+)*\s+.+", candidate):
                is_heading = True
            elif re.match(r"^[A-Z][A-Za-z0-9 /\-&(),.]{3,}$", candidate) and len(candidate.split()) <= 12:
                is_heading = True

            if not is_heading:
                continue

            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            headings.append(candidate)
            if len(headings) >= max_headings:
                break
        return headings

    def _build_manifest_document_entry(self, doc: dict) -> dict:
        doc = self._with_document_kbac(doc)
        metadata = doc.get("high_level_metadata") if isinstance(doc.get("high_level_metadata"), dict) else {}
        content = doc.get("content") if isinstance(doc.get("content"), dict) else {}
        source = doc.get("source") if isinstance(doc.get("source"), dict) else {}
        key_fields = self._doc_key_fields(doc)
        metadata_summary = self._qa_format_metadata(metadata, max_items=12)
        text_preview_source = ""
        for candidate in [
            content.get("main_document_text"),
            content.get("preview_text"),
            content.get("retrieval_text"),
            content.get("full_text"),
        ]:
            if isinstance(candidate, str) and candidate.strip():
                text_preview_source = candidate
                break
        text_preview = " ".join(text_preview_source.split())[:1000]
        titles = self._extract_document_titles(metadata)
        headings = self._extract_headings_from_text(text_preview_source)
        summary_parts = [
            str(doc.get("document_type") or ""),
            str(doc.get("file_name") or ""),
            str(source.get("original_filename") or ""),
            metadata_summary,
            key_fields.get("document_number") or "",
            key_fields.get("vendor") or "",
            key_fields.get("patient") or "",
            " ".join(titles),
            " ".join(headings[:6]),
            text_preview,
        ]
        summary_blob = " ".join(part for part in summary_parts if part).strip()

        return {
            "doc_id": str(doc.get("_id")),
            "document_uuid": str(doc.get("document_uuid") or "").strip() or None,
            "file_name": str(doc.get("file_name") or "").strip() or None,
            "tracking_filename": str(doc.get("tracking_filename") or "").strip() or None,
            "document_type": str(doc.get("document_type") or "unknown").strip() or "unknown",
            "saved_at": doc.get("saved_at"),
            "updated_at": doc.get("updated_at"),
            "storage_mode": str(doc.get("storage_mode") or "").strip() or None,
            "isTemporary": bool(doc.get("isTemporary")),
            "expiry_at": doc.get("expiry_at"),
            "assignment_status": str(doc.get("assignment_status") or "").strip() or None,
            "kbac_scope": str(doc.get("kbac_scope") or "global").strip() or "global",
            "kbac_owner": str(doc.get("kbac_owner") or "").strip() or None,
            "document_number": key_fields.get("document_number") or None,
            "vendor": key_fields.get("vendor") or None,
            "patient": key_fields.get("patient") or None,
            "metadata_summary": metadata_summary,
            "labels": self._manifest_doc_labels(metadata),
            "fields": self._manifest_doc_fields_enhanced(metadata),
            "titles": titles,
            "headings": headings,
            "search_terms": self._manifest_doc_search_terms(
                doc,
                metadata,
                key_fields,
                summary_blob,
                headings=headings,
                titles=titles,
            ),
            "summary_blob": summary_blob,
        }

    def _text_similarity(self, text_a: str, text_b: str) -> float:
        """Compute blended text similarity for near-duplicate detection."""
        if not text_a or not text_b:
            return 0.0
        a = text_a[:12000]
        b = text_b[:12000]
        if fuzz is not None:
            fuzzy_ratio = fuzz.token_set_ratio(a, b) / 100.0
        else:
            fuzzy_ratio = SequenceMatcher(None, a[:5000], b[:5000]).ratio()
        seq_ratio = SequenceMatcher(None, a[:5000], b[:5000]).ratio()
        return (0.7 * fuzzy_ratio) + (0.3 * seq_ratio)
    # ------------------- AUDIT LOGGING -------------------
    def log_activity(self, username, action, claim_id=None, details=None, client_ip=None):
        """Log a user activity to the activity_log collection."""
        if not self._ensure_connection():
            logger.error("No database connection")
            return False
        activity = {
            "username": username,
            "action": action,
            "claim_id": claim_id,
            "details": details,
            "client_ip": str(client_ip or "").strip() or None,
            "timestamp": datetime.utcnow()
        }
        try:
            self.activity_log.insert_one(activity)
            logger.info(f"Activity logged: {username} - {action}")
            return True
        except Exception as e:
            logger.error(f"Error logging activity: {e}")
            return False

    def get_activity_log(self, days=7, username=None, limit=200, offset=0):
        """Retrieve activity logs from the last N days with optional username filter and pagination."""
        if not self.client:
            logger.error("No database connection")
            return []
        from datetime import timedelta
        since = datetime.utcnow() - timedelta(days=days)
        query = {"timestamp": {"$gte": since}}
        if username:
            query["username"] = username
        try:
            # Keep result size bounded so large audit tables do not freeze the browser UI.
            safe_limit = max(1, min(int(limit or 200), 1000))
            safe_offset = max(0, int(offset or 0))
            activities = list(
                self.activity_log.find(query)
                .sort("timestamp", -1)
                .skip(safe_offset)
                .limit(safe_limit)
            )
            for act in activities:
                act["_id"] = str(act["_id"])
            return activities
        except Exception as e:
            logger.error(f"Error retrieving activity log: {e}")
            return []

    # ------------------- USER MANAGEMENT FOR LOGIN -------------------
    def _hash_password(self, raw_password: str, salt_hex: str = None, iterations: int = None) -> str:
        """Hash password using PBKDF2-SHA256 with per-user random salt."""
        if raw_password is None:
            return ""
        safe_iterations = int(iterations or self.PASSWORD_ITERATIONS)
        safe_salt_hex = (salt_hex or os.urandom(16).hex()).lower()
        try:
            salt_bytes = bytes.fromhex(safe_salt_hex)
        except ValueError:
            salt_bytes = os.urandom(16)
            safe_salt_hex = salt_bytes.hex()
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            str(raw_password).encode("utf-8"),
            salt_bytes,
            safe_iterations,
        )
        return f"{self.PASSWORD_SCHEME}${safe_iterations}${safe_salt_hex}${digest.hex()}"

    def _is_hashed_password(self, stored_password: str) -> bool:
        """Check whether password is in current hashed format."""
        if not isinstance(stored_password, str):
            return False
        parts = stored_password.split("$")
        return len(parts) == 4 and parts[0] == self.PASSWORD_SCHEME

    def _verify_password(self, raw_password: str, stored_password: str) -> bool:
        """Verify password against hashed format; fallback to legacy plain-text check."""
        if not isinstance(stored_password, str):
            return False
        if self._is_hashed_password(stored_password):
            try:
                _, iterations, salt_hex, _ = stored_password.split("$", 3)
                recalculated = self._hash_password(
                    raw_password=raw_password,
                    salt_hex=salt_hex,
                    iterations=int(iterations),
                )
                return hmac.compare_digest(recalculated, stored_password)
            except Exception:
                return False
        # Legacy plain-text fallback for old records.
        return hmac.compare_digest(str(raw_password), stored_password)

    def find_user_conflict(self, username=None, email=None, employee_code=None):
        """Find whether username/email/employee code is already in use."""
        if not self._ensure_connection():
            logger.error("No database connection")
            return None
        try:
            normalized_username = (username or "").strip()
            normalized_email = (email or "").strip().lower()
            normalized_employee_code = (employee_code or "").strip()

            if normalized_username:
                existing = self.users.find_one({
                    "username": {"$regex": f"^{re.escape(normalized_username)}$", "$options": "i"}
                })
                if existing:
                    return {"field": "username", "value": normalized_username, "user_id": str(existing.get("_id"))}

            if normalized_email:
                existing = self.users.find_one({
                    "email": {"$regex": f"^{re.escape(normalized_email)}$", "$options": "i"}
                })
                if existing:
                    return {"field": "email", "value": normalized_email, "user_id": str(existing.get("_id"))}

            if normalized_employee_code:
                existing = self.users.find_one({
                    "employee_code": {"$regex": f"^{re.escape(normalized_employee_code)}$", "$options": "i"}
                })
                if existing:
                    return {"field": "employee_code", "value": normalized_employee_code, "user_id": str(existing.get("_id"))}
            return None
        except Exception as e:
            logger.error(f"Error checking user conflicts: {e}")
            return None

    def create_user(self, username, password, email=None, employee_code=None, user_type="Viewer", created_by=None, role=None):
        """Create a new login user with hashed password."""
        if not self._ensure_connection():
            logger.error("No database connection")
            return None
        normalized_username = (username or "").strip()
        normalized_password = (password or "").strip()
        normalized_email = (email or "").strip().lower() or None
        normalized_employee_code = (employee_code or "").strip() or None
        try:
            safe_role = validate_role_string(role or user_type or "employee").value
        except ValueError:
            safe_role = "employee"
        safe_user_type = (user_type or role_to_user_type(safe_role)).strip() or role_to_user_type(safe_role)

        if not normalized_username or not normalized_password:
            logger.error("Username and password are required to create user")
            return None
        if normalized_employee_code and not normalized_employee_code.isdigit():
            logger.error("Employee code must contain digits only")
            return None
        if safe_role == "super_admin" and self.count_users_by_role("super_admin") >= MAX_SUPER_ADMIN_USERS:
            logger.warning(f"Super admin creation blocked because the limit of {MAX_SUPER_ADMIN_USERS} has been reached")
            return None

        conflict = self.find_user_conflict(
            username=normalized_username,
            email=normalized_email,
            employee_code=normalized_employee_code,
        )
        if conflict:
            logger.warning(f"Duplicate user detected on {conflict.get('field')}: {conflict.get('value')}")
            return None

        hashed_password = self._hash_password(normalized_password)
        user = {
            "username": normalized_username,
            "password": hashed_password,
            "email": normalized_email,
            "employee_code": normalized_employee_code,
            "user_type": safe_user_type,
            "role": safe_role,  # NEW: RBAC role field
            "created_by": created_by,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        try:
            result = self.users.insert_one(user)
            logger.info(f"User created with _id: {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return None

    def count_users_by_role(self, role):
        """Count users for a given RBAC role."""
        if not self._ensure_connection():
            logger.error("No database connection")
            return 0
        try:
            safe_role = validate_role_string(role).value
        except ValueError:
            return 0
        try:
            return int(self.users.count_documents({"role": safe_role}))
        except Exception as e:
            logger.error(f"Error counting users by role '{safe_role}': {e}")
            return 0

    def find_user(self, username, password):
        """Find a user by username and verify password hash."""
        if not self._ensure_connection():
            logger.error("No database connection")
            return None
        try:
            normalized_username = (username or "").strip()
            normalized_password = (password or "").strip()
            user = self.users.find_one({
                "username": {"$regex": f"^{re.escape(normalized_username)}$", "$options": "i"},
            })
            if not user:
                return None

            stored_password = user.get("password")
            if not self._verify_password(normalized_password, stored_password):
                return None

            # Auto-migrate legacy plain-text passwords to hashed format on successful login.
            if isinstance(stored_password, str) and not self._is_hashed_password(stored_password):
                try:
                    new_hash = self._hash_password(normalized_password)
                    self.users.update_one(
                        {"_id": user.get("_id")},
                        {"$set": {"password": new_hash, "updated_at": datetime.utcnow()}},
                    )
                    user["password"] = new_hash
                except Exception as migration_err:
                    logger.warning(f"Password migration failed for user {normalized_username}: {migration_err}")
            return user
        except Exception as e:
            logger.error(f"Error finding user: {e}")
            return None

    def get_user_by_email(self, email):
        """Fetch a user by email/username for password reset flows."""
        if not self._ensure_connection():
            logger.error("No database connection")
            return None
        normalized_email = (email or "").strip().lower()
        if not normalized_email:
            return None
        try:
            return self.users.find_one(
                {
                    "$or": [
                        {"email": {"$regex": f"^{re.escape(normalized_email)}$", "$options": "i"}},
                        {"username": {"$regex": f"^{re.escape(normalized_email)}$", "$options": "i"}},
                    ]
                }
            )
        except Exception as e:
            logger.error(f"Error finding user by email '{normalized_email}': {e}")
            return None

    def get_user_by_id(self, user_id, projection=None):
        """Fetch a single user by Mongo _id without exposing password by default."""
        if not self._ensure_connection():
            logger.error("No database connection")
            return None
        try:
            from bson import ObjectId

            object_id = ObjectId(str(user_id))
            safe_projection = projection or {
                "username": 1,
                "email": 1,
                "employee_code": 1,
                "user_type": 1,
                "role": 1,
                "created_by": 1,
                "created_at": 1,
                "updated_at": 1,
            }
            user = self.users.find_one({"_id": object_id}, safe_projection)
            if not user:
                return None
            user["_id"] = str(user.get("_id"))
            user["created_at"] = self._safe_datetime(user.get("created_at"))
            user["updated_at"] = self._safe_datetime(user.get("updated_at"))
            return user
        except Exception as e:
            logger.error(f"Error fetching user by id '{user_id}': {e}")
            return None

    def update_user_fields(self, user_id, updates):
        """Update editable user fields and return the updated user record."""
        if not self._ensure_connection():
            logger.error("No database connection")
            return None
        try:
            from bson import ObjectId

            object_id = ObjectId(str(user_id))
            safe_updates = dict(updates or {})
            if not safe_updates:
                return self.get_user_by_id(user_id)
            safe_updates["updated_at"] = datetime.utcnow()

            result = self.users.update_one({"_id": object_id}, {"$set": safe_updates})
            if result.matched_count == 0:
                return None
            return self.get_user_by_id(user_id)
        except Exception as e:
            logger.error(f"Error updating user '{user_id}': {e}")
            return None

    def update_user_password(self, user_id, new_password):
        """Update a user's password hash and return the updated user."""
        if not self._ensure_connection():
            logger.error("No database connection")
            return None
        normalized_password = str(new_password or "").strip()
        if not normalized_password:
            return None
        try:
            from bson import ObjectId

            object_id = ObjectId(str(user_id))
            hashed_password = self._hash_password(normalized_password)
            result = self.users.update_one(
                {"_id": object_id},
                {"$set": {"password": hashed_password, "updated_at": datetime.utcnow()}},
            )
            if result.matched_count == 0:
                return None
            return self.get_user_by_id(user_id)
        except Exception as e:
            logger.error(f"Error updating password for user '{user_id}': {e}")
            return None

    def update_user_face_enrollment(self, user_id, face_auth_record):
        """Store encrypted face-login descriptors for a user."""
        if not self._ensure_connection():
            logger.error("No database connection")
            return None
        try:
            from bson import ObjectId

            object_id = ObjectId(str(user_id))
            safe_record = dict(face_auth_record or {})
            if not safe_record.get("embedding_ciphertext"):
                return None

            result = self.users.update_one(
                {"_id": object_id},
                {
                    "$set": {
                        "face_auth": {**safe_record, "updated_at": datetime.utcnow()},
                        "updated_at": datetime.utcnow(),
                    }
                },
            )
            if result.matched_count == 0:
                return None
            return self.get_user_by_id(user_id)
        except Exception as e:
            logger.error(f"Error updating face enrollment for user '{user_id}': {e}")
            return None

    def list_face_enabled_users(self, limit=500):
        """Return users with face-login samples for server-side matching."""
        if not self._ensure_connection():
            logger.error("No database connection")
            return []
        try:
            safe_limit = max(1, min(int(limit or 500), 1000))
            projection = {
                "username": 1,
                "email": 1,
                "employee_code": 1,
                "user_type": 1,
                "role": 1,
                "created_by": 1,
                "onboarding": 1,
                "face_auth": 1,
            }
            return list(
                self.users.find(
                    {
                        "face_auth.enabled": True,
                        "$or": [
                            {"face_auth.embedding_ciphertext": {"$exists": True, "$ne": ""}},
                            {"face_auth.samples": {"$exists": True, "$ne": []}},
                        ],
                    },
                    projection,
                ).limit(safe_limit)
            )
        except Exception as e:
            logger.error(f"Error listing face-enabled users: {e}")
            return []

    def list_users(self, limit=100, include_legacy=False, created_by=None):
        """List users without exposing passwords."""
        if not self._ensure_connection():
            logger.error("No database connection")
            return []
        try:
            safe_limit = max(1, min(int(limit or 100), 500))
            query = {}
            if not include_legacy:
                # Keep UI focused on users created via current Create User flow.
                query = {
                    "employee_code": {"$exists": True, "$nin": [None, ""]},
                    "role": {"$in": ["employee", "manager", "admin", "super_admin"]},
                }
            if created_by:
                query["created_by"] = str(created_by).strip()
            projection = {
                "username": 1,
                "email": 1,
                "employee_code": 1,
                "user_type": 1,
                "role": 1,
                "created_by": 1,
                "created_at": 1,
                "updated_at": 1,
            }
            users = list(self.users.find(query, projection).sort("created_at", -1).limit(safe_limit))
            for user in users:
                user["_id"] = str(user.get("_id"))
                user["created_at"] = self._safe_datetime(user.get("created_at"))
                user["updated_at"] = self._safe_datetime(user.get("updated_at"))
            return users
        except Exception as e:
            logger.error(f"Error listing users: {e}")
            return []

    def get_employee_profile(self, emp_id: str = None, profile_uuid: str = None):
        if not self.client:
            return None

        query = {}
        if emp_id:
            query["empID"] = str(emp_id).strip()
        elif profile_uuid:
            query["uuid"] = str(profile_uuid).strip()
        else:
            return None

        profile = self.employee_profiles.find_one(query)
        if not profile:
            return None
        profile["_id"] = str(profile["_id"])
        return profile

    def get_employee_manifest(self, profile_uuid: str = None, emp_id: str = None):
        if not self.client:
            return None

        query = {}
        if profile_uuid:
            query["employee_uuid"] = str(profile_uuid).strip()
        elif emp_id:
            query["empID"] = str(emp_id).strip()
        else:
            return None

        manifest = self.employee_manifests.find_one(query)
        if manifest:
            manifest["_id"] = str(manifest["_id"])
            return manifest

        refresh_uuid = str(profile_uuid or "").strip()
        if not refresh_uuid and emp_id:
            profile = self.get_employee_profile(emp_id=emp_id)
            refresh_uuid = str((profile or {}).get("uuid") or "").strip()

        if refresh_uuid:
            self.refresh_employee_manifest(refresh_uuid)
            manifest = self.employee_manifests.find_one({"employee_uuid": refresh_uuid})
            if manifest:
                manifest["_id"] = str(manifest["_id"])
                return manifest
        return None

    def refresh_employee_manifest(self, profile_uuid: str):
        if not self.client or not profile_uuid:
            return None

        safe_uuid = str(profile_uuid).strip()
        profile = self.get_employee_profile(profile_uuid=safe_uuid)
        projection = {
            "document_uuid": 1,
            "file_name": 1,
            "tracking_filename": 1,
            "document_type": 1,
            "saved_at": 1,
            "updated_at": 1,
            "storage_mode": 1,
            "isTemporary": 1,
            "expiry_at": 1,
            "assignment_status": 1,
            "high_level_metadata": 1,
            "content.full_text": 1,
            "content.main_document_text": 1,
            "source.original_filename": 1,
            "kbac_scope": 1,
            "kbac_owner": 1,
        }
        docs = list(
            self.collection.find(
                self._active_documents_query({"employee_uuid": safe_uuid}),
                projection,
            ).sort("saved_at", -1)
        )
        now = datetime.utcnow()
        document_entries = [self._build_manifest_document_entry(doc) for doc in docs]
        manifest_payload = {
            "employee_uuid": safe_uuid,
            "empID": str((profile or {}).get("empID") or "").strip() or None,
            "empName": str((profile or {}).get("empName") or "").strip() or None,
            "created_by": str((profile or {}).get("created_by") or "").strip() or None,
            "empName_normalized": self._normalize_person_name((profile or {}).get("empName") or ""),
            "documents_count": len(document_entries),
            "document_ids": [entry["doc_id"] for entry in document_entries],
            "document_types": sorted(
                {entry["document_type"] for entry in document_entries if entry.get("document_type")}
            ),
            "documents": document_entries,
            "last_document_at": docs[0].get("saved_at") if docs else None,
            "updated_at": now,
        }

        if not profile and not docs:
            self.employee_manifests.delete_one({"employee_uuid": safe_uuid})
            return None

        self.employee_manifests.update_one(
            {"employee_uuid": safe_uuid},
            {"$set": manifest_payload, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )
        refreshed = self.employee_manifests.find_one({"employee_uuid": safe_uuid})
        if refreshed:
            refreshed["_id"] = str(refreshed["_id"])
        return refreshed

    def refresh_all_employee_manifests(self, limit: int | None = None) -> dict:
        if not self.client:
            return {"refreshed": 0}

        projection = {"uuid": 1}
        cursor = self.employee_profiles.find({}, projection)
        if limit:
            cursor = cursor.limit(max(1, int(limit)))

        refreshed = 0
        for profile in cursor:
            profile_uuid = str(profile.get("uuid") or "").strip()
            if not profile_uuid:
                continue
            self.refresh_employee_manifest(profile_uuid)
            refreshed += 1

        return {"refreshed": refreshed}

    def _refresh_employee_indexes(self, profile_uuid: str):
        safe_uuid = str(profile_uuid or "").strip()
        if not safe_uuid:
            return
        self.refresh_employee_profile_stats(safe_uuid)
        self.refresh_employee_manifest(safe_uuid)

    def create_or_update_employee_profile(self, emp_id: str, emp_name: str, extra: dict = None):
        if not self.client:
            return None

        emp_id = str(emp_id or "").strip()
        emp_name = str(emp_name or "").strip()
        if not emp_id or not emp_name:
            return None

        now = datetime.utcnow()
        existing = self.employee_profiles.find_one({"empID": emp_id})
        payload = {
            "empID": emp_id,
            "empName": emp_name,
            "updated_at": now,
        }
        if isinstance(extra, dict):
            for key, value in extra.items():
                if value is not None:
                    payload[key] = value

        if existing:
            self.employee_profiles.update_one(
                {"_id": existing["_id"]},
                {"$set": payload},
            )
            profile = self.employee_profiles.find_one({"_id": existing["_id"]})
        else:
            payload.update(
                {
                    "uuid": str(uuid4()),
                    "status": "active",
                    "documents_count": 0,
                    "last_document_at": None,
                    "created_at": now,
                }
            )
            inserted = self.employee_profiles.insert_one(payload)
            profile = self.employee_profiles.find_one({"_id": inserted.inserted_id})

        if profile:
            profile["_id"] = str(profile["_id"])
            self.refresh_employee_manifest(profile.get("uuid"))
        return profile

    def delete_employee_profile(self, emp_id: str = None, profile_uuid: str = None):
        if not self.client:
            return None

        profile = self.get_employee_profile(emp_id=emp_id, profile_uuid=profile_uuid)
        if not profile:
            return None

        safe_uuid = str(profile.get("uuid") or "").strip()
        safe_emp_id = str(profile.get("empID") or "").strip()
        document_filters = []
        if safe_uuid:
            document_filters.append({"employee_uuid": safe_uuid})
        if safe_emp_id:
            document_filters.append({"empID": safe_emp_id})

        document_query = {}
        if len(document_filters) == 1:
            document_query = document_filters[0]
        elif document_filters:
            document_query = {"$or": document_filters}

        deleted_documents = 0
        deleted_chunks = 0

        if document_query:
            active_document_query = self._active_documents_query(document_query)
            linked_docs = list(self.collection.find(active_document_query, {"_id": 1}))
            for doc in linked_docs:
                doc_id = str(doc.get("_id") or "").strip()
                if not doc_id:
                    continue
                try:
                    chunk_delete_result = self.document_chunks.delete_many({"doc_id": doc_id})
                    deleted_chunks += int(chunk_delete_result.deleted_count or 0)
                except Exception as chunk_err:
                    logger.warning(f"Could not delete chunks for doc {doc_id}: {chunk_err}")
                if self.delete_document(doc_id):
                    deleted_documents += 1

        if safe_uuid:
            self.employee_manifests.delete_one({"employee_uuid": safe_uuid})

        profile_delete_query = {"uuid": safe_uuid} if safe_uuid else {"empID": safe_emp_id}
        delete_result = self.employee_profiles.delete_one(profile_delete_query)
        if not delete_result.deleted_count:
            return None

        logger.info(
            f"Employee profile deleted: empID={safe_emp_id or '-'} uuid={safe_uuid or '-'} "
            f"deleted_documents={deleted_documents} deleted_chunks={deleted_chunks}"
        )
        return {
            "status": "deleted",
            "empID": safe_emp_id,
            "uuid": safe_uuid or None,
            "deleted_documents": deleted_documents,
            "deleted_chunks": deleted_chunks,
        }

    def search_employee_profiles(self, search_term: str = None, limit: int = 20, created_by: str = None):
        if not self.client:
            return []

        safe_limit = max(1, min(int(limit or 20), 100))
        query = {}
        if created_by:
            query["created_by"] = str(created_by).strip()
        normalized_term = str(search_term or "").strip()
        if normalized_term:
            regex = {"$regex": re.escape(normalized_term), "$options": "i"}
            query["$or"] = [
                {"empID": regex},
                {"empName": regex},
                {"uuid": regex},
            ]

        profiles = list(self.employee_profiles.find(query).sort("updated_at", -1).limit(safe_limit))
        for profile in profiles:
            profile["_id"] = str(profile["_id"])
            profile.setdefault(
                "documents_count",
                self.collection.count_documents(
                    self._active_documents_query({"employee_uuid": profile.get("uuid")})
                ),
            )
        return profiles

    def get_documents_by_employee(
        self,
        emp_id: str = None,
        profile_uuid: str = None,
        limit: int = 50,
        access_context: Optional[dict] = None,
    ):
        if not self.client:
            return []

        query = {}
        if emp_id:
            query["empID"] = str(emp_id).strip()
        if profile_uuid:
            query["employee_uuid"] = str(profile_uuid).strip()
        if not query:
            return []

        results = list(
            self.collection.find(self._active_documents_query(query))
            .sort("saved_at", -1)
            .limit(max(1, min(int(limit or 50), 200)))
        )
        visible_results = []
        for doc in results:
            if access_context and not self._document_matches_access(doc, access_context):
                continue
            enriched = self._with_document_kbac(doc)
            enriched["_id"] = str(enriched["_id"])
            visible_results.append(enriched)
        return visible_results

    def refresh_employee_profile_stats(self, profile_uuid: str):
        if not self.client or not profile_uuid:
            return

        active_docs = list(
            self.collection.find(
                self._active_documents_query({"employee_uuid": str(profile_uuid).strip()}),
                {"saved_at": 1},
            ).sort("saved_at", -1)
        )
        documents_count = len(active_docs)
        last_document_at = active_docs[0].get("saved_at") if active_docs else None
        self.employee_profiles.update_one(
            {"uuid": str(profile_uuid).strip()},
            {
                "$set": {
                    "documents_count": documents_count,
                    "last_document_at": last_document_at,
                    "updated_at": datetime.utcnow(),
                }
            },
        )

    def cleanup_expired_temporary_documents(self):
        if not self.client:
            return {"deleted_documents": 0, "deleted_files": 0}

        now = datetime.utcnow()
        expired_docs = list(
            self.collection.find({"isTemporary": True, "expiry_at": {"$lte": now}})
        )
        deleted_documents = 0
        deleted_files = 0
        touched_profiles = set()

        for doc in expired_docs:
            file_path = str(doc.get("file_path") or "").strip()
            if file_path:
                try:
                    path_obj = Path(file_path)
                    if path_obj.exists():
                        path_obj.unlink()
                        deleted_files += 1
                except Exception as e:
                    logger.warning(f"Could not delete expired file '{file_path}': {e}")

            try:
                self.document_chunks.delete_many({"doc_id": str(doc.get("_id"))})
            except Exception as chunk_err:
                logger.warning(f"Could not delete chunks for expired doc {doc.get('_id')}: {chunk_err}")

            touched_profiles.add(str(doc.get("employee_uuid") or "").strip())
            self.collection.delete_one({"_id": doc["_id"]})
            deleted_documents += 1

        for profile_uuid in touched_profiles:
            if profile_uuid:
                self._refresh_employee_indexes(profile_uuid)

        return {
            "deleted_documents": deleted_documents,
            "deleted_files": deleted_files,
        }
    
    def save_document(self, data: dict) -> str:
        """Save structured document to MongoDB"""
        if not self.client:
            logger.error("No database connection")
            return None
        
        try:
            data = self._with_document_kbac(data)
            source = data.get("source") if isinstance(data.get("source"), dict) else {}
            if source:
                data["source"] = dict(source)
            if not data.get("duplicate_fingerprint"):
                data["duplicate_fingerprint"] = self._duplicate_fingerprint(data)

            # Add timestamp
            data["saved_at"] = datetime.now()
            data["updated_at"] = datetime.now()
            
            # Insert document
            result = self.collection.insert_one(data)
            logger.info(f"Document saved to MongoDB: {result.inserted_id}")
            profile_uuid = str(data.get("employee_uuid") or "").strip()
            if profile_uuid:
                self._refresh_employee_indexes(profile_uuid)
            try:
                self.save_document_chunks(str(result.inserted_id), data)
            except Exception as chunk_err:
                logger.warning(f"Failed to save document chunks: {chunk_err}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error saving document: {e}")
            return None

    def save_document_chunks(self, doc_id: str, document: dict) -> int:
        """Persist chunked text for a document into the chunks collection."""
        if not self.client:
            return 0

        payload = document if isinstance(document, dict) else {}
        content = payload.get("content") if isinstance(payload.get("content"), dict) else {}
        full_text = str(self._preferred_document_text(content) or "").strip()
        if not full_text:
            return 0

        chunk_size = max(200, settings.get_int("CHUNK_SIZE_CHARS", DEFAULT_CHUNK_SIZE_CHARS))
        overlap = max(0, settings.get_int("CHUNK_OVERLAP_CHARS", DEFAULT_CHUNK_OVERLAP_CHARS))
        chunks = build_text_chunks(full_text, chunk_size_chars=chunk_size, overlap_chars=overlap)
        if not chunks:
            return 0

        try:
            self.document_chunks.delete_many({"doc_id": str(doc_id)})
        except Exception as exc:
            logger.warning(f"Could not clear previous chunks for {doc_id}: {exc}")

        now = datetime.utcnow()
        common = {
            "doc_id": str(doc_id),
            "document_uuid": str(payload.get("document_uuid") or "").strip() or None,
            "document_type": str(payload.get("document_type") or "unknown").strip() or "unknown",
            "employee_uuid": str(payload.get("employee_uuid") or "").strip() or None,
            "empID": str(payload.get("empID") or "").strip() or None,
            "empName": str(payload.get("empName") or "").strip() or None,
            "storage_mode": str(payload.get("storage_mode") or "").strip() or None,
            "isTemporary": bool(payload.get("isTemporary")),
            "expiry_at": payload.get("expiry_at"),
            "saved_at": payload.get("saved_at"),
            "kbac_scope": self._normalize_kbac_scope(payload.get("kbac_scope")),
            "kbac_owner": self._normalize_kbac_owner(payload.get("kbac_owner")),
        }

        batch = []
        saved = 0
        batch_size = 200
        for chunk in chunks:
            batch.append(
                {
                    **common,
                    "chunk_index": int(chunk.index),
                    "chunk_text": chunk.text,
                    "char_start": int(chunk.char_start),
                    "char_end": int(chunk.char_end),
                    "chunk_size_chars": int(chunk_size),
                    "overlap_chars": int(overlap),
                    "created_at": now,
                }
            )
            if len(batch) >= batch_size:
                self.document_chunks.insert_many(batch, ordered=False)
                saved += len(batch)
                batch = []

        if batch:
            self.document_chunks.insert_many(batch, ordered=False)
            saved += len(batch)

        return saved

    def _build_chunk_search_terms(self, query: str) -> list:
        normalized = self._normalize_text(query)
        if not normalized:
            return []

        analysis = analyze_query(normalized)
        terms = []
        for term in analysis.get("must_match_terms") or []:
            if term and term not in terms:
                terms.append(term)
        for term in analysis.get("search_terms") or []:
            if term and term not in terms:
                terms.append(term)

        if not terms:
            terms = self._significant_tokens(normalized)

        if not terms:
            terms = [normalized]

        return terms[:12]

    def _qa_required_terms(self, query_analysis: dict, fallback_tokens: list | None = None) -> list:
        """Build strong entity terms that must appear in relevant QA sources."""
        required = []
        seen = set()
        scenario_query = self._is_scenario_query(query_analysis)

        def _add(term: str):
            normalized = self._normalize_text(term)
            if not normalized or normalized in seen:
                return
            seen.add(normalized)
            required.append(term)

        for raw in query_analysis.get("must_match_terms") or []:
            text = str(raw or "").strip()
            if not text:
                continue
            tokens = self._significant_tokens(text)
            if tokens:
                for token in tokens:
                    _add(token)
            else:
                _add(text)

        for explicit_field in ("subject", "patient", "vendor", "document_number"):
            explicit_value = str(query_analysis.get(explicit_field) or "").strip()
            if not explicit_value:
                continue
            explicit_tokens = self._significant_tokens(explicit_value)
            if explicit_tokens:
                for token in explicit_tokens:
                    _add(token)
            else:
                _add(explicit_value)

        if not required and fallback_tokens and not scenario_query:
            for token in fallback_tokens[:2]:
                _add(token)

        return required[:6]

    def _qa_blob_matches_required_terms(self, blob: str, required_terms: list[str]) -> bool:
        """Require strong subject/entity overlap before accepting QA sources."""
        normalized_blob = self._normalize_text(blob)
        if not required_terms:
            return True
        if not normalized_blob:
            return False

        blob_tokens = set(self._tokenize(normalized_blob))
        compact_blob = self._compact_alnum(normalized_blob)
        hits = 0

        for raw_term in required_terms:
            term = self._normalize_text(raw_term)
            if not term:
                continue

            matched = False
            compact_term = self._compact_alnum(term)
            if " " in term and term in normalized_blob:
                matched = True
            elif compact_term and len(compact_term) >= 6 and compact_term in compact_blob:
                matched = True
            else:
                term_tokens = self._significant_tokens(term)
                if term_tokens and all(self._token_matches_blob_tokens(token, blob_tokens) for token in term_tokens):
                    matched = True
                elif self._token_matches_blob_tokens(term, blob_tokens):
                    matched = True

            if matched:
                hits += 1

        min_hits = len(required_terms) if len(required_terms) <= 2 else max(2, math.ceil(len(required_terms) * 0.7))
        return hits >= min_hits

    def get_document_chunks(
        self,
        doc_id: str,
        limit: int = 20,
        offset: int = 0,
        access_context: Optional[dict] = None,
    ):
        """Return chunked text for a document with pagination."""
        if not self.client:
            return {"results": [], "total": 0}

        try:
            base_query = self._active_documents_query({"doc_id": str(doc_id)})
            kbac_query = self._kbac_query(access_context)
            if kbac_query:
                base_query = {"$and": [base_query, kbac_query]}
            total = self.document_chunks.count_documents(base_query)
            projection = {
                "doc_id": 1,
                "document_uuid": 1,
                "document_type": 1,
                "chunk_index": 1,
                "chunk_text": 1,
                "char_start": 1,
                "char_end": 1,
                "chunk_size_chars": 1,
                "overlap_chars": 1,
                "kbac_scope": 1,
                "kbac_owner": 1,
            }
            cursor = (
                self.document_chunks.find(base_query, projection)
                .sort("chunk_index", 1)
                .skip(max(0, int(offset or 0)))
                .limit(max(1, int(limit or 20)))
            )
            results = list(cursor)
            for chunk in results:
                chunk["_id"] = str(chunk.get("_id"))
            return {"results": results, "total": total}
        except Exception as e:
            logger.error(f"Error fetching document chunks: {e}")
            return {"results": [], "total": 0}

    def search_document_chunks(
        self,
        query: str,
        doc_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
        access_context: Optional[dict] = None,
    ):
        """Search chunked text using MongoDB text search with pagination."""
        if not self.client:
            return {"results": [], "total": 0}

        search_terms = self._build_chunk_search_terms(query)
        if not search_terms:
            return {"results": [], "total": 0}

        try:
            base_query = {"$text": {"$search": " ".join(search_terms)}}
            if doc_id:
                base_query["doc_id"] = str(doc_id)
            base_query = self._active_documents_query(base_query)
            kbac_query = self._kbac_query(access_context)
            if kbac_query:
                base_query = {"$and": [base_query, kbac_query]}
            projection = {
                "doc_id": 1,
                "document_uuid": 1,
                "document_type": 1,
                "chunk_index": 1,
                "chunk_text": 1,
                "char_start": 1,
                "char_end": 1,
                "chunk_size_chars": 1,
                "overlap_chars": 1,
                "kbac_scope": 1,
                "kbac_owner": 1,
                "score": {"$meta": "textScore"},
            }
            total = self.document_chunks.count_documents(base_query)
            cursor = (
                self.document_chunks.find(base_query, projection)
                .sort([("score", {"$meta": "textScore"}), ("chunk_index", 1)])
                .skip(max(0, int(offset or 0)))
                .limit(max(1, int(limit or 20)))
            )
            results = list(cursor)
            for chunk in results:
                chunk["_id"] = str(chunk.get("_id"))
            return {"results": results, "total": total}
        except Exception as e:
            logger.error(f"Error searching document chunks: {e}")
            return {"results": [], "total": 0}

    def qa_retrieve_chunks(
        self,
        query: str,
        limit: int = 6,
        allowed_doc_ids: list[str] | None = None,
        access_context: Optional[dict] = None,
    ):
        """Retrieve the most relevant chunks for QA."""
        if not self.client:
            return []

        safe_limit = max(1, min(int(limit or 6), 12))
        normalized_query = self._normalize_text(query)
        query_analysis = analyze_query(normalized_query)
        search_terms = self._build_chunk_search_terms(normalized_query)
        if not search_terms:
            return []

        allowed_ids = []
        seen_allowed_ids = set()
        for raw_doc_id in allowed_doc_ids or []:
            doc_id = str(raw_doc_id or "").strip()
            if not doc_id or doc_id in seen_allowed_ids:
                continue
            seen_allowed_ids.add(doc_id)
            allowed_ids.append(doc_id)
        required_terms = self._qa_required_terms(query_analysis, fallback_tokens=search_terms)

        try:
            base_query = {"$text": {"$search": " ".join(search_terms)}}
            if allowed_ids:
                base_query["doc_id"] = {"$in": allowed_ids}
            kbac_query = self._kbac_query(access_context)
            base_query = (
                {"$and": [self._active_documents_query(base_query), kbac_query]}
                if kbac_query
                else self._active_documents_query(base_query)
            )
            projection = {
                "doc_id": 1,
                "document_uuid": 1,
                "document_type": 1,
                "chunk_index": 1,
                "chunk_text": 1,
                "char_start": 1,
                "char_end": 1,
                "kbac_scope": 1,
                "kbac_owner": 1,
                "score": {"$meta": "textScore"},
            }
            cursor = (
                self.document_chunks.find(base_query, projection)
                .sort([("score", {"$meta": "textScore"}), ("chunk_index", 1)])
                .limit(max(12, safe_limit * 8))
            )
            candidates = list(cursor)
        except Exception as e:
            logger.error(f"Chunk retrieval failed: {e}")
            return []

        if not candidates:
            return []

        from bson import ObjectId

        mongo_ids = []
        for chunk in candidates:
            try:
                mongo_ids.append(ObjectId(str(chunk.get("doc_id"))))
            except Exception:
                continue

        doc_map = {}
        if mongo_ids:
            projection = {
                "document_type": 1,
                "high_level_metadata": 1,
                "confidence": 1,
                "saved_at": 1,
                "content.full_text": 1,
            }
            docs = list(
                self.collection.find(
                    (
                        {"$and": [self._active_documents_query({"_id": {"$in": mongo_ids}}), self._kbac_query(access_context)]}
                        if self._kbac_query(access_context)
                        else self._active_documents_query({"_id": {"$in": mongo_ids}})
                    ),
                    projection,
                )
            )
            doc_map = {
                str(doc.get("_id")): self._with_document_kbac(doc)
                for doc in docs
                if self._document_matches_access(doc, access_context)
            }

        per_doc_limit = max(1, settings.get_int("QA_CHUNK_MAX_PER_DOC", 2))
        selected = []
        seen_per_doc: dict[str, int] = {}
        for chunk in candidates:
            doc_id = str(chunk.get("doc_id") or "")
            if not doc_id:
                continue

            doc = doc_map.get(doc_id, {})
            if access_context and not doc:
                continue
            if access_context and doc and not self._document_matches_access(doc, access_context):
                continue
            validation_blob = str(chunk.get("chunk_text") or "")
            if doc:
                validation_blob = f"{validation_blob}\n{self._build_document_blob(doc)}"
            if required_terms and not self._qa_blob_matches_required_terms(validation_blob, required_terms):
                continue
            if seen_per_doc.get(doc_id, 0) >= per_doc_limit:
                continue

            seen_per_doc[doc_id] = seen_per_doc.get(doc_id, 0) + 1
            selected.append(chunk)
            if len(selected) >= safe_limit:
                break

        if not selected:
            return []

        sources = []
        for chunk in selected:
            doc_id = str(chunk.get("doc_id") or "")
            if not doc_id:
                continue
            doc = doc_map.get(doc_id, {})
            metadata = doc.get("high_level_metadata") if isinstance(doc.get("high_level_metadata"), dict) else {}
            items = metadata.get("items", [])
            if not isinstance(items, list):
                items = []

            chunk_text = str(chunk.get("chunk_text") or "")
            snippet = chunk_text[:400].strip()
            if chunk_text and len(chunk_text) > 400:
                snippet += "..."

            doc_type = doc.get("document_type") or chunk.get("document_type") or "unknown"
            score_val = chunk.get("score")
            score = round(float(score_val), 3) if score_val is not None else 0.0

            sources.append(
                {
                    "doc_id": doc_id,
                    "document_type": doc_type,
                    "metadata": self._qa_format_metadata(metadata, max_items=30),
                    "snippet": snippet or "",
                    "items": items[:6],
                    "full_text": (chunk_text[:1600] if chunk_text else ""),
                    "confidence": doc.get("confidence", {}),
                    "score": score,
                    "reasons": ["chunk_match"],
                    "chunk": {
                        "chunk_index": chunk.get("chunk_index"),
                        "char_start": chunk.get("char_start"),
                        "char_end": chunk.get("char_end"),
                    },
                }
            )

        return sources
    
    def find_documents(
        self,
        query: dict = None,
        limit: int = 10,
        offset: int = 0,
        access_context: Optional[dict] = None,
    ):
        """Find documents in MongoDB with pagination"""
        if not self.client:
            return {"results": [], "total": 0}
        
        try:
            query = self._active_documents_query(query or {})
            kbac_query = self._kbac_query(access_context)
            if kbac_query:
                query = {"$and": [query, kbac_query]}
            scan_limit = max(int(limit or 10) + int(offset or 0), int(limit or 10) * 4, 50)
            candidates = list(self.collection.find(query).sort("saved_at", -1).limit(scan_limit))
            results = []
            for doc in candidates:
                if access_context and not self._document_matches_access(doc, access_context):
                    continue
                enriched = self._with_document_kbac(doc)
                enriched["_id"] = str(enriched["_id"])
                results.append(enriched)
            total = len(results)
            return {"results": results[offset:offset + limit], "total": total}
        except Exception as e:
            logger.error(f"Error finding documents: {e}")
            return {"results": [], "total": 0}
    
    def find_by_document_type(
        self,
        doc_type: str,
        limit: int = 10,
        access_context: Optional[dict] = None,
    ):
        """Find documents by type"""
        return self.find_documents({"document_type": doc_type}, limit, access_context=access_context).get("results", [])

    def search_documents(
        self,
        search_term: str = None,
        document_type: str = None,
        limit: int = 10,
        offset: int = 0,
        access_context: Optional[dict] = None,
    ):
        """Search documents by term across content/metadata with optional type filtering and pagination."""
        if not self.client:
            return {"results": [], "total": 0}

        base_query = self._active_documents_query({})
        kbac_query = self._kbac_query(access_context)
        if document_type:
            base_query["document_type"] = {
                "$regex": f"^{re.escape(document_type.strip())}$",
                "$options": "i",
            }

        try:
            # No search term: return filtered list using find_documents with pagination
            if not search_term:
                result = self.find_documents(base_query, limit, offset, access_context=access_context)
                return result

            normalized_term = self._normalize_text(search_term)
            parsed_constraints = self._parse_semantic_constraints(normalized_term)
            requested_doc_number = parsed_constraints.get("document_number") or self._extract_mixed_id_token(normalized_term)
            parsed_doc_type = parsed_constraints.get("document_type")
            if not document_type and parsed_doc_type:
                base_query["document_type"] = {"$regex": re.escape(parsed_doc_type), "$options": "i"}
            scan_limit = max(limit * 60, 600)
            candidates_map = {}

            # If query includes document/invoice number, enforce number-only matching.
            if requested_doc_number:
                id_candidates = list(
                    self.collection.find(
                        {"$and": [base_query, kbac_query]} if kbac_query else base_query
                    ).sort("saved_at", -1).limit(scan_limit)
                )
                id_matches = []
                for doc in id_candidates:
                    if access_context and not self._document_matches_access(doc, access_context):
                        continue
                    if not self._matches_document_number(doc, requested_doc_number):
                        continue
                    doc = self._with_document_kbac(doc)
                    doc["_id"] = str(doc["_id"])
                    preview = self._extract_match_preview(doc, requested_doc_number)
                    if preview:
                        doc["match_preview"] = preview
                    id_matches.append(doc)
                    if len(id_matches) >= limit + offset:
                        break

                # Mongo regex fallback for explicit ID text in indexed fields.
                if not id_matches:
                    escaped = re.escape(requested_doc_number)
                    id_regex = {"$regex": escaped, "$options": "i"}
                    id_query = dict(base_query)
                    id_query["$or"] = [
                        {"high_level_metadata.invoice_number": id_regex},
                        {"high_level_metadata.bill_number": id_regex},
                        {"high_level_metadata.document_number": id_regex},
                        {"high_level_metadata.po_number": id_regex},
                        {"high_level_metadata.claim_id": id_regex},
                    ]
                    regex_candidates = list(
                        self.collection.find(
                            {"$and": [self._active_documents_query(id_query), kbac_query]}
                            if kbac_query
                            else self._active_documents_query(id_query)
                        ).sort("saved_at", -1).limit(limit + offset)
                    )
                    for doc in regex_candidates:
                        if access_context and not self._document_matches_access(doc, access_context):
                            continue
                        doc = self._with_document_kbac(doc)
                        doc["_id"] = str(doc["_id"])
                        preview = self._extract_match_preview(doc, requested_doc_number)
                        if preview:
                            doc["match_preview"] = preview
                        id_matches.append(doc)

                unique_matches = []
                seen_keys = set()
                requested_compact = self._compact_alnum(requested_doc_number)
                for doc in id_matches:
                    key = self._document_number_key(doc)
                    # Explicit ID search should not return repeated copies of the same requested ID.
                    if requested_compact:
                        blob_compact = self._compact_alnum(self._build_document_blob(doc))
                        if requested_compact in blob_compact:
                            key = f"requested:{requested_compact}"
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    unique_matches.append(doc)
                    if len(unique_matches) >= limit + offset:
                        break

                total = len(unique_matches)
                results = unique_matches[offset:offset + limit]
                return {"results": results, "total": total}

            # Candidate Stage 1: Text index lookup (broad), strict filtered later.
            try:
                text_query = dict(base_query)
                text_query["$text"] = {"$search": normalized_term}
                text_candidates = list(
                    self.collection.find(
                        {"$and": [text_query, kbac_query]} if kbac_query else text_query
                    ).sort("saved_at", -1).limit(scan_limit)
                )
                for doc in text_candidates:
                    candidates_map[str(doc.get("_id"))] = doc
            except Exception as text_search_error:
                logger.warning(f"Text search unavailable, falling back to scan: {text_search_error}")

            # Candidate Stage 2: Recent docs fallback (ensures symbol-heavy IDs are still matched).
            if not candidates_map:
                recent_candidates = list(
                    self.collection.find(
                        {"$and": [base_query, kbac_query]} if kbac_query else base_query
                    ).sort("saved_at", -1).limit(scan_limit)
                )
                for doc in recent_candidates:
                    candidates_map[str(doc.get("_id"))] = doc

            matched_docs = []
            for doc in candidates_map.values():
                if access_context and not self._document_matches_access(doc, access_context):
                    continue
                if not self._strict_keyword_match(doc, normalized_term):
                    continue
                doc = self._with_document_kbac(doc)
                doc["_id"] = str(doc["_id"])
                preview = self._extract_match_preview(doc, normalized_term)
                if preview:
                    doc["match_preview"] = preview
                matched_docs.append(doc)
                if len(matched_docs) >= limit + offset:
                    break

            if not matched_docs:
                fallback_candidates = list(
                    self.collection.find(
                        {"$and": [base_query, kbac_query]} if kbac_query else base_query
                    ).sort("saved_at", -1).limit(scan_limit)
                )
                for doc in fallback_candidates:
                    if access_context and not self._document_matches_access(doc, access_context):
                        continue
                    if not self._strict_keyword_match(doc, normalized_term):
                        continue
                    doc = self._with_document_kbac(doc)
                    doc["_id"] = str(doc["_id"])
                    preview = self._extract_match_preview(doc, normalized_term)
                    if preview:
                        doc["match_preview"] = preview
                    matched_docs.append(doc)
                    if len(matched_docs) >= limit + offset:
                        break

            total = len(matched_docs)
            results = matched_docs[offset:offset + limit]
            return {"results": results, "total": total}
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return {"results": [], "total": 0}

    def _qa_format_metadata(self, metadata: dict, max_items: int = 8) -> str:
        """Build a compact metadata string for Q&A context - dynamically from actual metadata."""
        if not isinstance(metadata, dict):
            return ""
        
        # Extract ALL fields from metadata, sorted by priority (non-null values first)
        items = []
        
        # First pass: collect all non-null/non-empty fields
        for key, value in metadata.items():
            if value is None or value == "":
                continue
            # Skip deeply nested structures
            if isinstance(value, (dict, list)):
                continue
            items.append((key, str(value)))
        
        # Sort by key name for consistency, but keep order for readability
        # Prioritize fields with common keywords in their names
        priority_keywords = ["name", "id", "number", "amount", "date", "cmd", "ceo", "director", "address", "email", "phone"]
        
        def priority_score(item):
            key, _ = item
            key_lower = key.lower()
            # Keywords that should appear first
            for i, kw in enumerate(priority_keywords):
                if kw in key_lower:
                    return i
            return len(priority_keywords)
        
        items.sort(key=priority_score)
        
        # Format as "key: value" strings, limited to max_items
        formatted = []
        for key, value in items[:max_items]:
            formatted.append(f"{key}: {value}")
        
        return "; ".join(formatted)

    def _resolve_employee_profile_from_query(self, query: str, query_analysis: dict, access_context: Optional[dict] = None):
        if not self.client:
            return None

        candidate_texts = []
        for raw in [
            query_analysis.get("subject"),
            query_analysis.get("patient"),
            self._extract_subject_phrase(query),
        ]:
            value = str(raw or "").strip()
            if value and value not in candidate_texts:
                candidate_texts.append(value)

        identifier_terms = self._extract_employee_identifier_terms(query)
        if not candidate_texts and not identifier_terms:
            return None

        scored_profiles = []
        seen_profile_ids = set()
        context = self._normalize_access_context(access_context)
        team_owner = context.get("team_owner")
        profile_scope_query = {}
        if not context.get("allow_all") and team_owner:
            profile_scope_query["created_by"] = team_owner

        for identifier in identifier_terms:
            regex = {"$regex": f"^{re.escape(identifier)}$", "$options": "i"}
            identifier_query = {"empID": regex}
            if profile_scope_query:
                identifier_query = {"$and": [identifier_query, profile_scope_query]}
            for profile in self.employee_profiles.find(identifier_query).limit(10):
                profile_key = str(profile.get("_id"))
                if profile_key in seen_profile_ids:
                    continue
                seen_profile_ids.add(profile_key)
                scored_profiles.append(
                    (
                        220,
                        [f"employee id matched ({profile.get('empID')})"],
                        profile,
                    )
                )

        for candidate_text in candidate_texts:
            regex = {"$regex": re.escape(candidate_text), "$options": "i"}
            query_filter = {"$or": [{"empName": regex}, {"empID": regex}]}
            if profile_scope_query:
                query_filter = {"$and": [query_filter, profile_scope_query]}
            for profile in self.employee_profiles.find(query_filter).limit(25):
                profile_key = str(profile.get("_id"))
                if profile_key in seen_profile_ids:
                    continue
                seen_profile_ids.add(profile_key)

                profile_name_norm = self._normalize_person_name(profile.get("empName"))
                profile_id_norm = self._compact_alnum(profile.get("empID"))
                candidate_name_norm = self._normalize_person_name(candidate_text)
                candidate_id_norm = self._compact_alnum(candidate_text)
                score = 0
                reasons = []

                if candidate_id_norm and candidate_id_norm == profile_id_norm:
                    score += 220
                    reasons.append(f"employee id matched ({profile.get('empID')})")
                if candidate_name_norm and candidate_name_norm == profile_name_norm:
                    score += 200
                    reasons.append(f"employee name matched ({profile.get('empName')})")
                elif candidate_name_norm and (
                    candidate_name_norm in profile_name_norm or profile_name_norm in candidate_name_norm
                ):
                    score += 140
                    reasons.append("employee name overlap")

                candidate_tokens = set(self._tokenize(candidate_name_norm))
                profile_tokens = set(self._tokenize(profile_name_norm))
                if candidate_tokens and profile_tokens:
                    overlap = len(candidate_tokens & profile_tokens)
                    if overlap:
                        score += overlap * 35
                        reasons.append(f"name token overlap {overlap}")

                if score >= 90:
                    scored_profiles.append((score, reasons, profile))

        if not scored_profiles:
            return None

        scored_profiles.sort(key=lambda item: item[0], reverse=True)
        best_score, reasons, best_profile = scored_profiles[0]
        if best_score < 90:
            return None

        best_profile["_id"] = str(best_profile["_id"])
        return {
            "score": best_score,
            "reasons": reasons,
            "profile": best_profile,
        }

    def _resolve_manifest_from_document_query(
        self,
        query: str,
        query_analysis: dict,
        constraints: dict,
        access_context: Optional[dict] = None,
    ):
        if not self.client:
            return None

        requested_doc_number = constraints.get("document_number") or self._extract_mixed_id_token(query)
        requested_doc_number = str(requested_doc_number or "").strip()
        if not requested_doc_number:
            return None

        requested_compact = self._compact_alnum(requested_doc_number)
        manifest_query = {
            "documents.document_number": {
                "$regex": re.escape(requested_doc_number),
                "$options": "i",
            }
        }
        manifests = list(
            self.employee_manifests.find(
                manifest_query,
                {
                    "employee_uuid": 1,
                    "empID": 1,
                    "empName": 1,
                    "documents": 1,
                },
            ).limit(20)
        )

        if not manifests and requested_compact:
            manifests = list(
                self.employee_manifests.find(
                    {"documents.search_terms": requested_compact},
                    {
                        "employee_uuid": 1,
                        "empID": 1,
                        "empName": 1,
                        "documents": 1,
                    },
                ).limit(20)
            )

        scored = []
        for manifest in manifests:
            if any("kbac_scope" not in entry for entry in (manifest.get("documents") or [])):
                refreshed = self.refresh_employee_manifest(str(manifest.get("employee_uuid") or "").strip())
                if refreshed:
                    manifest = refreshed
            matched_entries = []
            for entry in manifest.get("documents") or []:
                if access_context and not self._manifest_entry_matches_access(entry, access_context):
                    continue
                entry_number = self._compact_alnum(entry.get("document_number"))
                entry_terms = {str(term or "").strip().lower() for term in entry.get("search_terms") or []}
                if requested_compact and entry_number and (
                    requested_compact in entry_number or entry_number in requested_compact
                ):
                    matched_entries.append(entry)
                    continue
                if requested_doc_number.lower() in entry_terms or requested_compact in entry_terms:
                    matched_entries.append(entry)

            if matched_entries:
                score = max(
                    240 if self._compact_alnum(entry.get("document_number")) == requested_compact else 200
                    for entry in matched_entries
                )
                scored.append(
                    {
                        "manifest": manifest,
                        "score": score,
                        "reasons": [f"document mapped in manifest ({requested_doc_number})"],
                        "matched_entries": matched_entries,
                    }
                )

        if not scored:
            return None

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[0]

    def _resolve_manifest_from_terms(
        self,
        query: str,
        query_analysis: dict,
        constraints: dict,
        query_tokens: list,
        access_context: Optional[dict] = None,
    ):
        if not self.client:
            return None

        required_terms = self._qa_required_terms(query_analysis, fallback_tokens=query_tokens)
        tokens = []
        for token in query_tokens or []:
            cleaned = str(token or "").strip().lower()
            if not cleaned:
                continue
            tokens.append(cleaned)
            compact = self._compact_alnum(cleaned)
            if compact and compact not in tokens:
                tokens.append(compact)

        tokens = [t for t in tokens if len(t) >= 2]
        tokens = list(dict.fromkeys(tokens))[:16]
        if not tokens:
            return None

        manifests = list(
            self.employee_manifests.find(
                {"documents.search_terms": {"$in": tokens}},
                {
                    "employee_uuid": 1,
                    "empID": 1,
                    "empName": 1,
                    "documents": 1,
                },
            ).limit(40)
        )
        if not manifests:
            return None

        best_match = None
        for manifest in manifests:
            if any("kbac_scope" not in entry for entry in (manifest.get("documents") or [])):
                refreshed = self.refresh_employee_manifest(str(manifest.get("employee_uuid") or "").strip())
                if refreshed:
                    manifest = refreshed
            matched_entries = []
            for entry in manifest.get("documents") or []:
                if access_context and not self._manifest_entry_matches_access(entry, access_context):
                    continue
                entry_terms = {str(term or "").strip().lower() for term in entry.get("search_terms") or []}
                if not entry_terms:
                    continue
                if not any(token in entry_terms for token in tokens):
                    continue
                if required_terms and not self._qa_blob_matches_required_terms(entry.get("summary_blob") or "", required_terms):
                    continue
                score, reasons = self._score_manifest_entry(
                    entry,
                    query=query,
                    query_analysis=query_analysis,
                    constraints=constraints,
                    query_tokens=query_tokens,
                )
                if score <= 0:
                    score = 0.2
                    reasons = reasons + ["search_terms matched"]
                matched_entries.append(
                    {
                        "entry": entry,
                        "score": score,
                        "reasons": reasons,
                    }
                )

            if not matched_entries:
                continue

            matched_entries.sort(key=lambda item: item["score"], reverse=True)
            top = matched_entries[0]
            manifest_score = top["score"]
            if not best_match or manifest_score > best_match["score"]:
                best_match = {
                    "manifest": manifest,
                    "score": manifest_score,
                    "reasons": ["search_terms matched"],
                    "matched_entries": [item["entry"] for item in matched_entries[:10]],
                }

        return best_match

    def _query_requests_multiple_documents(self, query: str, query_analysis: dict) -> bool:
        q = self._normalize_text(query)
        if not q:
            return False
        if not query_analysis.get("simple_lookup"):
            return True
        multi_doc_markers = [
            "all documents",
            "all files",
            "all records",
            "summary",
            "history",
            "timeline",
            "compare",
            "comparison",
            "documents",
            "files",
            "records",
            "across",
            "together",
            "both",
        ]
        return any(marker in q for marker in multi_doc_markers)

    def _qa_is_broad_purchase_order_listing(self, query: str, query_analysis: dict) -> bool:
        normalized = self._normalize_text(query)
        normalized = re.sub(r"\bpo['’]s\b", "po", normalized, flags=re.IGNORECASE)
        if not normalized:
            return False
        if not any(marker in normalized for marker in ["po", "purchase order", "purchase orders"]):
            return False
        if not self._query_requests_multiple_documents(normalized, query_analysis):
            return False
        broad_patterns = [
            r"\b(?:all|every)\s+(?:the\s+)?(?:po|purchase order|purchase orders)\b",
            r"\b(?:list|show|tell me about)\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?(?:po|purchase order|purchase orders)\b",
        ]
        return any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in broad_patterns)

    def _document_mentions_purchase_order(self, doc: dict) -> bool:
        if not isinstance(doc, dict):
            return False

        metadata = doc.get("high_level_metadata") if isinstance(doc.get("high_level_metadata"), dict) else {}
        if any(
            str(metadata.get(key) or "").strip()
            for key in ("po_number", "purchase_order_number", "linked_po_number")
        ):
            return True

        doc_type = self._normalize_text(doc.get("document_type"))
        if "purchase order" in doc_type or doc_type == "po":
            return True

        blob = self._normalize_text(self._build_document_blob(doc))
        return bool(
            re.search(r"\b(?:po|purchase\s*order)\s*[-#:/]?\s*\d{1,6}\b", blob, flags=re.IGNORECASE)
        )

    def _extract_purchase_order_refs_from_doc(self, doc: dict) -> list[str]:
        if not isinstance(doc, dict):
            return []

        refs = set()
        metadata = doc.get("high_level_metadata") if isinstance(doc.get("high_level_metadata"), dict) else {}
        for key in ("po_number", "purchase_order_number", "linked_po_number", "document_number"):
            raw_value = str(metadata.get(key) or "").strip()
            digits = re.sub(r"[^0-9]", "", raw_value)
            if digits:
                refs.add(f"PO-{digits.zfill(3)}")

        blob = self._build_document_blob(doc)
        for match in re.finditer(
            r"\b(?:po|purchase\s*order)\s*[-#:/]?\s*(\d{1,6})\b",
            blob,
            flags=re.IGNORECASE,
        ):
            refs.add(f"PO-{match.group(1).zfill(3)}")

        return sorted(refs)

    def _qa_retrieve_all_purchase_order_documents(
        self,
        query: str,
        safe_limit: int,
        access_context: Optional[dict] = None,
    ) -> list[dict]:
        if not self.client:
            return []

        projection = {
            "document_type": 1,
            "high_level_metadata": 1,
            "content.full_text": 1,
            "saved_at": 1,
            "confidence": 1,
        }
        scan_limit = max(safe_limit * 20, 240)
        try:
            candidates = list(
                self.collection.find(
                    (
                        {"$and": [self._active_documents_query({}), self._kbac_query(access_context)]}
                        if self._kbac_query(access_context)
                        else self._active_documents_query({})
                    ),
                    projection,
                ).sort("saved_at", -1).limit(scan_limit)
            )
        except Exception as exc:
            logger.error(f"[QA] Broad PO retrieval failed during scan: {exc}")
            return []

        ranked_docs = []
        for doc in candidates:
            if access_context and not self._document_matches_access(doc, access_context):
                continue
            if not self._document_mentions_purchase_order(doc):
                continue
            refs = self._extract_purchase_order_refs_from_doc(doc)
            doc_type = self._normalize_text(doc.get("document_type"))
            priority = 2 if ("purchase order" in doc_type or doc_type == "po") else 1
            ranked_docs.append(
                {
                    "doc": doc,
                    "refs": refs,
                    "priority": priority,
                }
            )

        if not ranked_docs:
            return []

        ranked_docs.sort(
            key=lambda item: (
                item["priority"],
                len(item["refs"]),
                str(item["doc"].get("saved_at") or ""),
            ),
            reverse=True,
        )

        selected = []
        seen_doc_ids = set()
        seen_refs = set()
        target_doc_count = min(max(safe_limit, 12), 48)

        for item in ranked_docs:
            doc = item["doc"]
            doc_id = str(doc.get("_id") or "")
            refs = set(item["refs"])
            if not doc_id or doc_id in seen_doc_ids:
                continue
            if refs and refs.issubset(seen_refs):
                continue
            seen_doc_ids.add(doc_id)
            seen_refs.update(refs)
            selected.append(doc)
            if len(selected) >= target_doc_count:
                break

        if len(selected) < target_doc_count:
            for item in ranked_docs:
                doc = item["doc"]
                doc_id = str(doc.get("_id") or "")
                if not doc_id or doc_id in seen_doc_ids:
                    continue
                seen_doc_ids.add(doc_id)
                selected.append(doc)
                if len(selected) >= target_doc_count:
                    break

        sources = [
            self._build_qa_source_payload(
                doc=doc,
                query=query,
                score=2.4 if "purchase order" in self._normalize_text(doc.get("document_type")) else 1.8,
                reasons=["broad purchase-order retrieval"],
            )
            for doc in selected
        ]
        logger.info(
            f"[QA] Broad PO retrieval selected {len(sources)} source(s) covering {len(seen_refs)} unique PO reference(s)"
        )
        return sources

    def _manifest_candidate_count(self, safe_limit: int, query: str, query_analysis: dict) -> int:
        if self._query_requests_multiple_documents(query, query_analysis):
            return max(safe_limit, min(8, safe_limit + 2))
        if query_analysis.get("simple_lookup"):
            return min(3, max(2, safe_limit))
        return min(5, max(3, safe_limit))

    def _score_manifest_entry(
        self,
        entry: dict,
        query: str,
        query_analysis: dict,
        constraints: dict,
        query_tokens: list,
    ) -> tuple:
        summary_blob = self._normalize_text(entry.get("summary_blob"))
        if not summary_blob:
            return 0.0, []

        blob_tokens = set(self._tokenize(summary_blob))
        score = 0.0
        reasons = []

        if query_tokens:
            token_hits = sum(1 for token in query_tokens if self._token_matches_blob_tokens(token, blob_tokens))
            token_overlap = token_hits / max(1, len(query_tokens))
            score += 0.9 * token_overlap
            if token_hits:
                reasons.append(f"token overlap {token_hits}/{len(query_tokens)}")

        if query and query in summary_blob:
            score += 0.35
            reasons.append("exact manifest phrase match")

        requested_doc_type = self._normalize_text(constraints.get("document_type"))
        entry_doc_type = self._normalize_text(entry.get("document_type"))
        if requested_doc_type:
            if requested_doc_type in entry_doc_type or entry_doc_type in requested_doc_type:
                score += 0.85
                reasons.append(f"type matched ({requested_doc_type})")
            else:
                score -= 0.15

        requested_doc_number = self._compact_alnum(constraints.get("document_number"))
        entry_doc_number = self._compact_alnum(entry.get("document_number"))
        if requested_doc_number:
            if entry_doc_number and (
                requested_doc_number in entry_doc_number or entry_doc_number in requested_doc_number
            ):
                score += 1.2
                reasons.append("document number matched")
            elif requested_doc_number in self._compact_alnum(summary_blob):
                score += 1.0
                reasons.append("document number matched in manifest")
            else:
                score -= 0.25

        attribute = self._normalize_text(query_analysis.get("attribute"))
        if attribute and attribute in summary_blob:
            score += 0.3
            reasons.append(f"attribute matched ({attribute})")

        labels = [self._normalize_text(label) for label in entry.get("labels") or []]
        if attribute and any(attribute in label for label in labels):
            score += 0.12

        if self._normalize_text(entry.get("vendor")) and constraints.get("vendor"):
            requested_vendor = self._normalize_text(constraints.get("vendor"))
            if requested_vendor in self._normalize_text(entry.get("vendor")):
                score += 0.45
                reasons.append("vendor matched")

        return score, reasons

    def _build_qa_source_payload(self, doc: dict, query: str, score: float, reasons: list) -> dict:
        metadata = doc.get("high_level_metadata") if isinstance(doc.get("high_level_metadata"), dict) else {}
        content = doc.get("content") if isinstance(doc.get("content"), dict) else {}
        full_text = self._preferred_document_text(content)
        snippet = self._extract_match_preview(doc, query, window=150)
        if not snippet and full_text:
            snippet = full_text[:400].strip()
            if len(full_text) > 400:
                snippet += "..."

        items = metadata.get("items", [])
        if not isinstance(items, list):
            items = []

        return {
            "doc_id": str(doc.get("_id")),
            "document_type": doc.get("document_type", "unknown"),
            "metadata": self._qa_format_metadata(metadata, max_items=30),  # Increased from 8 to 30 for Q&A
            "snippet": snippet or "",
            "items": items[:6],
            "full_text": (full_text[:1800] if isinstance(full_text, str) else ""),
            "confidence": doc.get("confidence", {}),
            "score": round(score, 3),
            "reasons": reasons,
        }

    def _qa_retrieve_by_document_number(self, query: str, safe_limit: int, requested_number: str):
        requested_number = str(requested_number or "").strip()
        if not requested_number:
            return None

        projection = {
            "document_type": 1,
            "high_level_metadata": 1,
            "content.full_text": 1,
            "saved_at": 1,
            "confidence": 1,
        }
        requested_regex = {"$regex": re.escape(requested_number), "$options": "i"}
        metadata_query = self._active_documents_query(
            {
                "$or": [
                    {"high_level_metadata.invoice_number": requested_regex},
                    {"high_level_metadata.bill_number": requested_regex},
                    {"high_level_metadata.document_number": requested_regex},
                    {"high_level_metadata.po_number": requested_regex},
                    {"high_level_metadata.claim_id": requested_regex},
                ]
            }
        )

        matched_docs = []
        seen_ids = set()
        metadata_matches = list(
            self.collection.find(metadata_query, projection).sort("saved_at", -1).limit(max(safe_limit * 4, 12))
        )
        for doc in metadata_matches:
            doc_id = str(doc.get("_id"))
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            matched_docs.append(doc)

        if len(matched_docs) < safe_limit:
            scan_limit = max(safe_limit * 30, 300)
            recent_candidates = list(
                self.collection.find(self._active_documents_query({}), projection).sort("saved_at", -1).limit(scan_limit)
            )
            for doc in recent_candidates:
                doc_id = str(doc.get("_id"))
                if doc_id in seen_ids:
                    continue
                if not self._matches_document_number(doc, requested_number):
                    continue
                seen_ids.add(doc_id)
                matched_docs.append(doc)
                if len(matched_docs) >= max(safe_limit * 3, 8):
                    break

        if not matched_docs:
            return None

        sources = []
        for doc in matched_docs[:safe_limit]:
            sources.append(
                self._build_qa_source_payload(
                    doc=doc,
                    query=requested_number,
                    score=2.0,
                    reasons=[f"document number matched ({requested_number})"],
                )
            )
        logger.info(
            f"[QA] Direct document-number retrieval matched {len(sources)} source(s) for {requested_number}"
        )
        return sources

    def _qa_retrieve_from_employee_manifest(
        self,
        query: str,
        safe_limit: int,
        query_analysis: dict,
        constraints: dict,
        query_tokens: list,
        access_context: Optional[dict] = None,
    ):
        employee_match = self._resolve_employee_profile_from_query(query, query_analysis, access_context=access_context)
        manifest = None
        manifest_seed_reasons = []
        matched_entry_ids = set()

        if employee_match:
            profile = employee_match["profile"]
            manifest = self.get_employee_manifest(profile_uuid=profile.get("uuid"))
            if not manifest:
                manifest = self.refresh_employee_manifest(profile.get("uuid"))
            if manifest and any("kbac_scope" not in entry for entry in (manifest.get("documents") or [])):
                manifest = self.refresh_employee_manifest(profile.get("uuid"))
            manifest_seed_reasons = employee_match.get("reasons", [])
        else:
            manifest_match = self._resolve_manifest_from_document_query(
                query,
                query_analysis,
                constraints,
                access_context=access_context,
            )
            if not manifest_match:
                manifest_match = self._resolve_manifest_from_terms(
                    query,
                    query_analysis,
                    constraints,
                    query_tokens,
                    access_context=access_context,
                )
                if not manifest_match:
                    return None
            manifest = manifest_match.get("manifest")
            manifest_seed_reasons = manifest_match.get("reasons", [])
            matched_entry_ids = {
                str(entry.get("doc_id") or "").strip()
                for entry in (manifest_match.get("matched_entries") or [])
                if str(entry.get("doc_id") or "").strip()
            }

        if not manifest:
            return None
        manifest_documents = list((manifest or {}).get("documents") or [])
        if not manifest_documents:
            logger.info(
                f"[QA] Employee manifest matched {manifest.get('empName')} but contains no documents"
            )
            return []

        candidate_entries = []
        for entry in manifest_documents:
            if access_context and not self._manifest_entry_matches_access(entry, access_context):
                continue
            score, reasons = self._score_manifest_entry(
                entry,
                query=query,
                query_analysis=query_analysis,
                constraints=constraints,
                query_tokens=query_tokens,
            )
            candidate_entries.append(
                {
                    "entry": entry,
                    "score": score + (1.5 if str(entry.get("doc_id") or "").strip() in matched_entry_ids else 0.0),
                    "reasons": reasons + manifest_seed_reasons,
                }
            )

        candidate_entries.sort(key=lambda item: item["score"], reverse=True)
        top_entry_score = candidate_entries[0]["score"] if candidate_entries else 0.0
        candidate_count = self._manifest_candidate_count(safe_limit, query, query_analysis)
        selected_entries = []
        for item in candidate_entries:
            if item["score"] <= 0 and top_entry_score <= 0:
                continue
            if (
                item["score"] < max(0.1, top_entry_score - 0.45)
                and not self._query_requests_multiple_documents(query, query_analysis)
            ):
                continue
            selected_entries.append(item)
            if len(selected_entries) >= candidate_count:
                break

        if not selected_entries:
            if self._query_requests_multiple_documents(query, query_analysis):
                selected_entries = candidate_entries[:candidate_count]
            else:
                logger.info(
                    f"[QA] Employee manifest matched {manifest.get('empName')} but no document scored high enough"
                )
                return None

        from bson import ObjectId

        mongo_ids = []
        for item in selected_entries:
            doc_id = str(item["entry"].get("doc_id") or "").strip()
            if not doc_id:
                continue
            try:
                mongo_ids.append(ObjectId(doc_id))
            except Exception:
                continue

        if not mongo_ids:
            return None

        projection = {
            "document_type": 1,
            "high_level_metadata": 1,
            "content.full_text": 1,
            "saved_at": 1,
            "confidence": 1,
        }
        docs = list(
            self.collection.find(
                (
                    {"$and": [self._active_documents_query({"_id": {"$in": mongo_ids}}), self._kbac_query(access_context)]}
                    if self._kbac_query(access_context)
                    else self._active_documents_query({"_id": {"$in": mongo_ids}})
                ),
                projection,
            )
        )
        docs_by_id = {
            str(doc.get("_id")): self._with_document_kbac(doc)
            for doc in docs
            if self._document_matches_access(doc, access_context)
        }

        sources = []
        for item in selected_entries:
            doc_id = str(item["entry"].get("doc_id") or "").strip()
            doc = docs_by_id.get(doc_id)
            if not doc:
                continue
            sources.append(
                self._build_qa_source_payload(
                    doc=doc,
                    query=query,
                    score=item["score"],
                    reasons=item["reasons"],
                )
            )
            if len(sources) >= safe_limit:
                break

        if not sources:
            return None

        logger.info(
            f"[QA] Employee manifest hit for {manifest.get('empName')} ({manifest.get('empID')}); "
            f"selected {len(sources)}/{len(manifest_documents)} documents"
        )
        return sources

    def qa_retrieve_documents(self, query: str, limit: int = 6, access_context: Optional[dict] = None):
        """Hybrid: Manifest fast-path â†’ Semantic fallback"""
        if not self.client:
            return []
        qa_started_at = time.perf_counter()
        safe_limit = max(1, min(int(limit or 6), 12))
        query_norm = self._normalize_text(query)
        query_analysis = analyze_query(query_norm)
        broad_po_listing = self._qa_is_broad_purchase_order_listing(query_norm, query_analysis)
        retrieval_limit = min(max(safe_limit * 3, 18), 24) if broad_po_listing else safe_limit
        retrieval_term_source = " ".join(query_analysis.get("search_terms") or [query_norm])
        query_tokens = self._significant_tokens(retrieval_term_source or query_norm)
        if not query_tokens:
            return []
        constraints = self._parse_semantic_constraints(query_norm)

        if broad_po_listing:
            broad_po_sources = self._qa_retrieve_all_purchase_order_documents(
                query=query_norm,
                safe_limit=retrieval_limit,
                access_context=access_context,
            )
            if broad_po_sources:
                elapsed = time.perf_counter() - qa_started_at
                logger.info(f"[QA] Broad PO scan: {len(broad_po_sources)} sources in {elapsed:.2f}s")
                return broad_po_sources
        
        # PHASE 1: Manifest
        manifest_sources = self._qa_retrieve_from_employee_manifest(
            query=query_norm,
            safe_limit=retrieval_limit,
            query_analysis=query_analysis,
            constraints=constraints,
            query_tokens=query_tokens,
            access_context=access_context,
        )
        if manifest_sources is not None and len(manifest_sources) > 0:
            if broad_po_listing:
                supplement_docs = self.semantic_search(
                    query=f"purchase order po invoice {query_norm}".strip(),
                    document_type=None,
                    limit=retrieval_limit,
                    access_context=access_context,
                )
                supplemental_sources = []
                for doc in supplement_docs:
                    if not self._document_mentions_purchase_order(doc):
                        continue
                    supplemental_sources.append(
                        self._build_qa_source_payload(
                            doc=doc,
                            query=query_norm,
                            score=float(doc.get("semantic_score", 0.45)),
                            reasons=[doc.get("semantic_reason", "broad purchase-order supplement")],
                        )
                    )
                merged_sources = []
                seen_ids = set()
                for source in list(manifest_sources) + supplemental_sources:
                    doc_id = str(source.get("doc_id") or "").strip()
                    if not doc_id or doc_id in seen_ids:
                        continue
                    seen_ids.add(doc_id)
                    merged_sources.append(source)
                manifest_sources = merged_sources[:retrieval_limit]
            elapsed = time.perf_counter() - qa_started_at
            logger.info(f"[QA] Manifest: {len(manifest_sources)} sources in {elapsed:.2f}s")
            return manifest_sources
        
        # PHASE 2: Fallback to semantic search
        logger.info("[QA] Manifest: 0 results. Fallback to semantic search...")
        fallback_started = time.perf_counter()
        fallback_sources = self.semantic_search(
            query=(f"purchase order po invoice {query_norm}".strip() if broad_po_listing else query_norm),
            document_type=constraints.get("document_type"),
            limit=retrieval_limit,
            access_context=access_context,
        )
        
        if fallback_sources:
            elapsed_manifest = fallback_started - qa_started_at
            elapsed_fallback = time.perf_counter() - fallback_started
            logger.info(
                f"[QA] Fallback: {len(fallback_sources)} sources "
                f"(manifest: {elapsed_manifest:.2f}s, fallback: {elapsed_fallback:.2f}s)"
            )
            qa_sources = []
            filtered_fallback_sources = (
                [doc for doc in fallback_sources if self._document_mentions_purchase_order(doc)]
                if broad_po_listing
                else fallback_sources
            )
            for doc in (filtered_fallback_sources or fallback_sources)[:retrieval_limit]:
                qa_sources.append(self._build_qa_source_payload(
                    doc=doc,
                    query=query_norm,
                    score=doc.get("semantic_score", 0.5),
                    reasons=[doc.get("semantic_reason", "Fallback semantic")]
                ))
            return qa_sources
        
        elapsed = time.perf_counter() - qa_started_at
        logger.info(f"[QA] No results after {elapsed:.2f}s")
        return []

    def semantic_search(
        self,
        query: str,
        document_type: str = None,
        limit: int = 20,
        access_context: Optional[dict] = None,
    ):
        """Search documents using natural-language intent + relevance scoring."""
        if not self.client:
            return []

        normalized_query = self._normalize_text(query)
        if not normalized_query:
            return self.find_documents(limit=limit, access_context=access_context).get("results", [])

        query_analysis = analyze_query(normalized_query)
        constraints = self._parse_semantic_constraints(normalized_query)
        effective_type = document_type or constraints.get("document_type")

        base_query = self._active_documents_query({})
        kbac_query = self._kbac_query(access_context)
        if effective_type:
            base_query["document_type"] = {"$regex": re.escape(effective_type), "$options": "i"}

        projection = {
            "document_type": 1,
            "high_level_metadata": 1,
            "confidence": 1,
            "saved_at": 1,
            "content.full_text": 1,
        }

        scan_limit = max(limit * 30, 250)
        try:
            candidates = list(
                self.collection.find(
                    {"$and": [base_query, kbac_query]} if kbac_query else base_query,
                    projection,
                ).sort("saved_at", -1).limit(scan_limit)
            )
        except Exception as e:
            logger.error(f"Semantic search query failed: {e}")
            return []

        scored = []
        token_source = " ".join(query_analysis.get("search_terms") or [normalized_query])
        query_tokens = self._significant_tokens(token_source or normalized_query)
        query_token_set = set(self._tokenize(normalized_query))
        required_terms = self._qa_required_terms(query_analysis, fallback_tokens=query_tokens)
        scenario_query = self._is_scenario_query(query_analysis, normalized_query)
        for doc in candidates:
            if access_context and not self._document_matches_access(doc, access_context):
                continue
            blob = self._build_document_blob(doc)
            blob_tokens = self._document_blob_tokens(doc)
            score, reasons = self._semantic_score(
                doc,
                normalized_query,
                constraints,
                q_norm=normalized_query,
                query_tokens=query_token_set,
                blob=blob,
                blob_tokens=blob_tokens,
            )
            match_map = self._constraint_matches(doc, constraints)
            requested_constraints = len(match_map)
            matched_constraints = sum(1 for matched in match_map.values() if matched)

            # Hard constraint gate: if query asks for specific fields, enforce them.
            if requested_constraints > 0:
                if scenario_query and not constraints.get("document_number"):
                    required_matches = 1
                else:
                    required_matches = 1 if requested_constraints == 1 else min(2, requested_constraints)
                if matched_constraints < required_matches:
                    continue

            # Hard text gate: at least one significant query token must exist in document text/blob.
            blob = self._build_document_blob(doc)
            if query_tokens:
                blob_tokens = set(self._tokenize(blob))
                token_hits = sum(
                    1 for token in query_tokens if self._token_matches_blob_tokens(token, blob_tokens)
                )
                if scenario_query:
                    min_hits = 1 if len(query_tokens) <= 4 else max(1, int(len(query_tokens) * 0.3))
                else:
                    min_hits = 1 if len(query_tokens) <= 2 else max(2, int(len(query_tokens) * 0.5))
                if token_hits < min_hits:
                    continue

            if required_terms and not self._qa_blob_matches_required_terms(blob, required_terms):
                continue

            has_constraints = requested_constraints > 0
            if scenario_query:
                threshold = 0.38 if has_constraints else 0.32
            else:
                threshold = 0.52 if has_constraints else 0.62
            if score < threshold:
                continue

            doc = self._with_document_kbac(doc)
            doc["_id"] = str(doc["_id"])
            doc["semantic_score"] = round(score, 4)
            if reasons:
                doc["semantic_reason"] = "; ".join(reasons[:3])

            preview_term = (
                constraints.get("document_number")
                or constraints.get("subject")
                or constraints.get("vendor")
                or constraints.get("patient")
                or (max(self._tokenize(normalized_query), key=len, default=normalized_query))
            )
            preview = self._extract_match_preview(doc, preview_term)
            if preview:
                doc["match_preview"] = preview
            scored.append(doc)

        scored.sort(key=lambda item: item.get("semantic_score", 0.0), reverse=True)
        return scored[:limit]

    def detect_duplicates(
        self,
        limit: int = 120,
        similarity_threshold: float = 0.86,
        search_term: str = None,
        document_type: str = None,
    ):
        """Detect true duplicate re-uploads and fraud-risk copies."""
        if not self.client:
            return {
                "summary": {
                    "scanned_documents": 0,
                    "duplicate_uploads": 0,
                    "compared_pairs": 0,
                    "exact_duplicate_groups": 0,
                    "similar_pairs": 0,
                    "fraud_risk_pairs": 0,
                },
                "exact_duplicates": [],
                "similar_versions": [],
                "fraud_risk": [],
            }

        safe_limit = max(10, min(int(limit or 40), 120))
        projection = {
            "document_type": 1,
            "high_level_metadata": 1,
            "saved_at": 1,
            "content.full_text": 1,
        }
        if search_term or document_type:
            docs = self.search_documents(
                search_term=search_term,
                document_type=document_type,
                limit=safe_limit,
            )
        else:
            docs = list(
                self.collection.find(self._active_documents_query({}), projection).sort("saved_at", -1).limit(safe_limit)
            )
        if len(docs) < 2:
            return {
                "summary": {
                    "scanned_documents": len(docs),
                    "duplicate_uploads": 0,
                    "compared_pairs": 0,
                    "exact_duplicate_groups": 0,
                    "similar_pairs": 0,
                    "fraud_risk_pairs": 0,
                },
                "exact_duplicates": [],
                "similar_versions": [],
                "fraud_risk": [],
            }

        prepared = []
        duplicate_groups = defaultdict(list)
        for doc in docs:
            doc["_id"] = str(doc["_id"])
            blob = self._build_document_blob(doc)
            key_fields = self._doc_key_fields(doc)
            fingerprint = self._duplicate_fingerprint(doc)
            prepared_doc = {
                "doc": doc,
                "blob": blob,
                "fingerprint": fingerprint,
                "key_fields": key_fields,
                "doc_type": self._normalize_text(doc.get("document_type", "")),
            }
            prepared.append(prepared_doc)
            if fingerprint:
                duplicate_groups[fingerprint].append(doc)

        exact_duplicates = []
        duplicate_uploads = 0
        for idx, (fingerprint, grouped_docs) in enumerate(duplicate_groups.items(), start=1):
            if len(grouped_docs) < 2:
                continue
            upload_count = len(grouped_docs)
            duplicate_uploads += max(0, upload_count - 1)
            display_hash = fingerprint.split(":")[-1][:12]
            exact_duplicates.append({
                "group_id": f"exact-{idx}",
                "hash": display_hash,
                "upload_count": upload_count,
                "duplicate_copies": max(0, upload_count - 1),
                "documents": [self._doc_brief(doc) for doc in grouped_docs],
            })

        fraud_risk = []
        pair_counter = 1
        max_pairs = 250
        compared_pairs = 0

        for i in range(len(prepared)):
            if pair_counter > max_pairs:
                break
            for j in range(i + 1, len(prepared)):
                if pair_counter > max_pairs:
                    break

                left = prepared[i]
                right = prepared[j]

                left_keys = left["key_fields"]
                right_keys = right["key_fields"]
                same_doc_number = (
                    left_keys["document_number"]
                    and right_keys["document_number"]
                    and self._normalize_text(left_keys["document_number"]) == self._normalize_text(right_keys["document_number"])
                )

                # Fraud checks only make sense for pairs with same explicit document number.
                if not same_doc_number:
                    continue

                compared_pairs += 1
                similarity = self._text_similarity(left["blob"], right["blob"])

                reasons = []
                risk = "high"

                amount_delta = abs(left_keys["amount_value"] - right_keys["amount_value"])
                both_have_amount = left_keys["amount_value"] > 0 and right_keys["amount_value"] > 0
                vendor_mismatch = (
                    left_keys["vendor"]
                    and right_keys["vendor"]
                    and self._normalize_text(left_keys["vendor"]) != self._normalize_text(right_keys["vendor"])
                )

                if same_doc_number and both_have_amount and amount_delta > 1:
                    reasons.append("Same document number but amount differs")

                if vendor_mismatch:
                    reasons.append("Same document number but vendor differs")

                if similarity >= max(0.96, float(similarity_threshold)):
                    reasons.append("Text is near-identical with same document number")

                if not reasons:
                    continue

                pair_payload = {
                    "pair_id": f"pair-{pair_counter}",
                    "similarity": round(similarity, 4),
                    "risk": risk,
                    "reasons": reasons,
                    "document_a": self._doc_brief(left["doc"]),
                    "document_b": self._doc_brief(right["doc"]),
                }
                pair_counter += 1
                fraud_risk.append(pair_payload)

        fraud_risk.sort(key=lambda item: item["similarity"], reverse=True)

        return {
            "summary": {
                "scanned_documents": len(docs),
                "duplicate_uploads": duplicate_uploads,
                "compared_pairs": compared_pairs,
                "exact_duplicate_groups": len(exact_duplicates),
                "similar_pairs": 0,
                "fraud_risk_pairs": len(fraud_risk),
            },
            "exact_duplicates": exact_duplicates,
            "similar_versions": [],
            "fraud_risk": fraud_risk,
        }
    
    def update_document(self, doc_id: str, update_data: dict):
        """Update document in MongoDB"""
        if not self.client:
            return False
        
        try:
            from bson import ObjectId
            object_id = ObjectId(doc_id)
            existing_doc = self.collection.find_one({"_id": object_id}, {"employee_uuid": 1})
            previous_profile_uuid = str((existing_doc or {}).get("employee_uuid") or "").strip()
            result = self.collection.update_one(
                {"_id": object_id},
                {"$set": {**update_data, "updated_at": datetime.now()}}
            )
            logger.info(f"Document updated: {result.modified_count} document(s)")
            if result.modified_count:
                updated_doc = self.collection.find_one({"_id": object_id}, {"employee_uuid": 1})
                touched_profile_uuids = {
                    previous_profile_uuid,
                    str((updated_doc or {}).get("employee_uuid") or "").strip(),
                }
                for profile_uuid in touched_profile_uuids:
                    if profile_uuid:
                        self._refresh_employee_indexes(profile_uuid)
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating document: {e}")
            return False
    
    def delete_document(self, doc_id: str):
        """Delete document from MongoDB"""
        if not self.client:
            return False
        
        try:
            from bson import ObjectId
            existing_doc = self.collection.find_one({"_id": ObjectId(doc_id)})
            if not existing_doc:
                return False
            result = self.collection.delete_one({"_id": ObjectId(doc_id)})
            logger.info(f"Document deleted: {result.deleted_count} document(s)")
            if result.deleted_count:
                try:
                    self.document_chunks.delete_many({"doc_id": str(doc_id)})
                except Exception as chunk_err:
                    logger.warning(f"Could not delete chunks for doc {doc_id}: {chunk_err}")
                file_path = str(existing_doc.get("file_path") or "").strip()
                if file_path:
                    try:
                        path_obj = Path(file_path)
                        if path_obj.exists():
                            path_obj.unlink()
                    except Exception as e:
                        logger.warning(f"Could not delete document file '{file_path}': {e}")
                profile_uuid = str(existing_doc.get("employee_uuid") or "").strip()
                if profile_uuid:
                    self._refresh_employee_indexes(profile_uuid)
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
    
    def get_statistics(self):
        """Get database statistics"""
        if not self.client:
            return {}
        
        try:
            active_query = self._active_documents_query({})
            total_docs = self.collection.count_documents(active_query)
            doc_types = list(self.collection.aggregate([
                {"$match": active_query},
                {"$group": {"_id": "$document_type", "count": {"$sum": 1}}}
            ]))
            confidence_stats = list(self.collection.aggregate([
                {
                    "$match": self._active_documents_query({
                        "confidence.confidence_percent": {"$type": "number"}
                    })
                },
                {
                    "$group": {
                        "_id": None,
                        "average_confidence": {"$avg": "$confidence.confidence_percent"}
                    }
                }
            ]))
            average_confidence = round(
                float(confidence_stats[0].get("average_confidence", 0.0)),
                1
            ) if confidence_stats else 0.0
            
            return {
                "total_documents": total_docs,
                "document_types": doc_types,
                "average_confidence": average_confidence
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}

    def get_dashboard_insights(self):
        """Get dashboard insights for manager-created employees and manager upload storage."""
        if not self.client:
            return {
                "employee_managers": [],
                "upload_managers": [],
                "total_employee_profiles_created": 0,
                "total_manager_upload_bytes": 0,
                "manager_count": 0,
            }

        def _safe_datetime(value):
            if isinstance(value, datetime):
                return value.isoformat() + "Z"
            if value in [None, ""]:
                return None
            return str(value)

        managers: dict[str, dict] = {}

        def _ensure_manager(key: str, email: Optional[str] = None, username: Optional[str] = None) -> Optional[dict]:
            normalized_key = self._normalize_kbac_owner(key or email or username)
            if not normalized_key:
                return None

            entry = managers.get(normalized_key)
            if not entry:
                safe_email = str(email or normalized_key).strip() or normalized_key
                safe_username = str(username or "").strip()
                entry = {
                    "manager_key": normalized_key,
                    "manager_email": safe_email,
                    "manager_name": safe_username or safe_email,
                    "employee_count": 0,
                    "employees": [],
                    "upload_count": 0,
                    "upload_bytes": 0,
                    "files": [],
                }
                managers[normalized_key] = entry
            else:
                if email:
                    entry["manager_email"] = str(email).strip() or entry["manager_email"]
                if username:
                    entry["manager_name"] = str(username).strip() or entry["manager_name"]
            return entry

        try:
            manager_projection = {"email": 1, "username": 1, "created_at": 1}
            for user in self.users.find({"role": "manager"}, manager_projection).sort("created_at", 1):
                _ensure_manager(
                    str(user.get("email") or user.get("username") or "").strip(),
                    email=str(user.get("email") or "").strip(),
                    username=str(user.get("username") or "").strip(),
                )

            employee_projection = {
                "empID": 1,
                "empName": 1,
                "uuid": 1,
                "created_by": 1,
                "created_at": 1,
            }
            for profile in self.employee_profiles.find({}, employee_projection).sort("created_at", -1):
                entry = _ensure_manager(str(profile.get("created_by") or "").strip())
                if not entry:
                    continue
                entry["employees"].append(
                    {
                        "empID": str(profile.get("empID") or "").strip(),
                        "empName": str(profile.get("empName") or "").strip(),
                        "uuid": str(profile.get("uuid") or "").strip(),
                        "created_at": _safe_datetime(profile.get("created_at")),
                    }
                )

            upload_projection = {
                "file_name": 1,
                "document_type": 1,
                "saved_at": 1,
                "size_bytes": 1,
                "source.uploaded_by": 1,
                "source.uploaded_by_normalized": 1,
                "source.uploaded_by_role": 1,
            }
            upload_query = self._active_documents_query({"source.uploaded_by_role": "manager"})
            for doc in self.collection.find(upload_query, upload_projection).sort("saved_at", -1):
                source = doc.get("source") if isinstance(doc.get("source"), dict) else {}
                owner = (
                    source.get("uploaded_by_normalized")
                    or source.get("uploaded_by")
                    or ""
                )
                entry = _ensure_manager(str(owner).strip())
                if not entry:
                    continue
                try:
                    size_bytes = max(0, int(doc.get("size_bytes") or 0))
                except (TypeError, ValueError):
                    size_bytes = 0
                entry["upload_count"] += 1
                entry["upload_bytes"] += size_bytes
                entry["files"].append(
                    {
                        "_id": str(doc.get("_id") or "").strip(),
                        "file_name": str(doc.get("file_name") or "").strip(),
                        "document_type": str(doc.get("document_type") or "unknown").strip() or "unknown",
                        "saved_at": _safe_datetime(doc.get("saved_at")),
                        "size_bytes": size_bytes,
                    }
                )

            employee_managers = []
            upload_managers = []
            total_employee_profiles_created = 0
            total_manager_upload_bytes = 0

            for entry in managers.values():
                employee_count = len(entry.get("employees") or [])
                upload_bytes = int(entry.get("upload_bytes") or 0)
                total_employee_profiles_created += employee_count
                total_manager_upload_bytes += upload_bytes

                employee_managers.append(
                    {
                        "manager_key": entry["manager_key"],
                        "manager_email": entry["manager_email"],
                        "manager_name": entry["manager_name"],
                        "employee_count": employee_count,
                        "employees": entry.get("employees") or [],
                    }
                )
                upload_managers.append(
                    {
                        "manager_key": entry["manager_key"],
                        "manager_email": entry["manager_email"],
                        "manager_name": entry["manager_name"],
                        "upload_count": int(entry.get("upload_count") or 0),
                        "upload_bytes": upload_bytes,
                        "files": entry.get("files") or [],
                    }
                )

            employee_managers.sort(
                key=lambda item: (-int(item.get("employee_count") or 0), str(item.get("manager_email") or item.get("manager_name") or "").lower())
            )
            upload_managers.sort(
                key=lambda item: (-int(item.get("upload_bytes") or 0), str(item.get("manager_email") or item.get("manager_name") or "").lower())
            )

            return {
                "employee_managers": employee_managers,
                "upload_managers": upload_managers,
                "total_employee_profiles_created": total_employee_profiles_created,
                "total_manager_upload_bytes": total_manager_upload_bytes,
                "manager_count": len(managers),
            }
        except Exception as e:
            logger.error(f"Error getting dashboard insights: {e}")
            return {
                "employee_managers": [],
                "upload_managers": [],
                "total_employee_profiles_created": 0,
                "total_manager_upload_bytes": 0,
                "manager_count": 0,
            }
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")

