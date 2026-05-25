from utils.logger import get_logger
logger = get_logger('api_database')
from config import settings
from db.mongo_db import MongoDBManager
from typing import List, Dict, Any, Optional
import hashlib
import threading
import time
import re
from collections import OrderedDict
from datetime import datetime


_qa_retrieval_cache = OrderedDict()
_qa_retrieval_cache_lock = threading.Lock()
_QA_RETRIEVAL_CACHE_TTL = max(1, settings.get_int("QA_RETRIEVAL_CACHE_TTL_SECONDS", 900))
_QA_RETRIEVAL_CACHE_MAX = max(1, settings.get_int("QA_RETRIEVAL_CACHE_MAX_SIZE", 128))


def clear_qa_retrieval_cache() -> None:
    with _qa_retrieval_cache_lock:
        _qa_retrieval_cache.clear()
    logger.info("[QA] Retrieval cache cleared")


class DatabaseService:
    """Service layer for database operations"""

    def __init__(self):
        self.db = MongoDBManager()

    def _ensure_db_connection(self):
        """Reconnect MongoDB manager if initial startup connection failed."""
        client = getattr(self.db, "client", None)
        if client:
            return
        self.db = MongoDBManager()

    @staticmethod
    def _access_scope_key(access_context: Optional[Dict[str, Any]] = None) -> str:
        if not isinstance(access_context, dict):
            return "all"
        if access_context.get("allow_all"):
            return "all"
        team_owner = str(access_context.get("team_owner") or "").strip().lower()
        return f"team:{team_owner}" if team_owner else "global"

    @classmethod
    def _qa_retrieval_cache_key(
        cls,
        query: str,
        limit: int,
        access_context: Optional[Dict[str, Any]] = None,
        preferred_doc_ids: Optional[List[str]] = None,
    ) -> str:
        normalized = " ".join(str(query or "").strip().lower().split())
        preferred_scope = ",".join(
            sorted(str(doc_id or "").strip() for doc_id in (preferred_doc_ids or []) if str(doc_id or "").strip())
        ) or "none"
        payload = (
            f"{normalized}|{int(limit or 0)}|scope:{cls._access_scope_key(access_context)}|"
            f"preferred:{preferred_scope}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _qa_retrieval_cache_get(self, key: str):
        now = time.time()
        with _qa_retrieval_cache_lock:
            item = _qa_retrieval_cache.get(key)
            if not item:
                return None
            expires_at, value = item
            if expires_at <= now:
                _qa_retrieval_cache.pop(key, None)
                return None
            _qa_retrieval_cache.move_to_end(key)
            return value

    def _qa_retrieval_cache_set(self, key: str, value):
        expires_at = time.time() + _QA_RETRIEVAL_CACHE_TTL
        with _qa_retrieval_cache_lock:
            _qa_retrieval_cache[key] = (expires_at, value)
            _qa_retrieval_cache.move_to_end(key)
            while len(_qa_retrieval_cache) > _QA_RETRIEVAL_CACHE_MAX:
                _qa_retrieval_cache.popitem(last=False)

    @staticmethod
    def _is_broad_purchase_order_listing_query(query: str) -> bool:
        normalized = " ".join(str(query or "").strip().lower().split())
        normalized = re.sub(r"\bpo['’]s\b", "po", normalized, flags=re.IGNORECASE)
        if not normalized:
            return False
        if not any(marker in normalized for marker in ["po", "purchase order", "purchase orders"]):
            return False
        return bool(
            re.search(
                r"\b(?:all|every)\s+(?:the\s+)?(?:po|purchase order|purchase orders)\b",
                normalized,
                flags=re.IGNORECASE,
            )
            or re.search(
                r"\b(?:list|show|tell me about)\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?(?:po|purchase order|purchase orders)\b",
                normalized,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _safe_string(value: Any, default: str = "") -> str:
        """Return a string value safe for response models."""
        if isinstance(value, str):
            return value
        if value is None:
            return default
        return str(value)

    @staticmethod
    def _safe_dict(value: Any, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return a dict value safe for response models."""
        if isinstance(value, dict):
            return value
        return default or {}

    @staticmethod
    def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
        """Return numeric values as float, else default."""
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes"}
        return bool(value)

    @staticmethod
    def _safe_datetime(value: Any) -> Optional[str]:
        if isinstance(value, datetime):
            return value.isoformat() + "Z"
        if value in [None, ""]:
            return None
        return str(value)

    def _format_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format documents for API response"""
        formatted = []
        for doc in documents:
            formatted_doc = {
                "_id": self._safe_string(doc.get("_id")),
                "document_type": self._safe_string(doc.get("document_type"), "unknown"),
                "high_level_metadata": self._safe_dict(doc.get("high_level_metadata"), {}),
                "confidence": self._safe_dict(
                    doc.get("confidence"),
                    {"ocr_engine": "PaddleOCR", "confidence_percent": 0},
                ),
                "saved_at": self._safe_string(doc.get("saved_at")),
                "match_preview": self._safe_string(doc.get("match_preview")) if doc.get("match_preview") is not None else None,
                "semantic_score": self._safe_float(doc.get("semantic_score")),
                "semantic_reason": self._safe_string(doc.get("semantic_reason")) if doc.get("semantic_reason") is not None else None,
                "empID": self._safe_string(doc.get("empID")) if doc.get("empID") is not None else None,
                "empName": self._safe_string(doc.get("empName")) if doc.get("empName") is not None else None,
                "employee_uuid": self._safe_string(doc.get("employee_uuid")) if doc.get("employee_uuid") is not None else None,
                "document_uuid": self._safe_string(doc.get("document_uuid")) if doc.get("document_uuid") is not None else None,
                "file_name": self._safe_string(doc.get("file_name")) if doc.get("file_name") is not None else None,
                "file_path": self._safe_string(doc.get("file_path")) if doc.get("file_path") is not None else None,
                "storage_mode": self._safe_string(doc.get("storage_mode")) if doc.get("storage_mode") is not None else None,
                "isTemporary": self._safe_bool(doc.get("isTemporary"), False),
                "expiry_at": self._safe_datetime(doc.get("expiry_at")),
                "assignment_status": self._safe_string(doc.get("assignment_status")) if doc.get("assignment_status") is not None else None,
                "kbac_scope": self._safe_string(doc.get("kbac_scope")) if doc.get("kbac_scope") is not None else None,
                "kbac_owner": self._safe_string(doc.get("kbac_owner")) if doc.get("kbac_owner") is not None else None,
            }
            formatted.append(formatted_doc)
        return formatted

    def _format_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        formatted = []
        for chunk in chunks:
            formatted.append(
                {
                    "_id": self._safe_string(chunk.get("_id")),
                    "doc_id": self._safe_string(chunk.get("doc_id")),
                    "document_uuid": self._safe_string(chunk.get("document_uuid")) if chunk.get("document_uuid") is not None else None,
                    "document_type": self._safe_string(chunk.get("document_type"), "unknown"),
                    "chunk_index": int(chunk.get("chunk_index") or 0),
                    "chunk_text": self._safe_string(chunk.get("chunk_text"), ""),
                    "char_start": int(chunk.get("char_start") or 0),
                    "char_end": int(chunk.get("char_end") or 0),
                    "chunk_size_chars": int(chunk.get("chunk_size_chars") or 0),
                    "overlap_chars": int(chunk.get("overlap_chars") or 0),
                    "score": self._safe_float(chunk.get("score")),
                }
            )
        return formatted

    def get_all_documents(
        self,
        limit: int = 10,
        offset: int = 0,
        access_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get all documents with pagination"""
        self._ensure_db_connection()
        result = self.db.find_documents(limit=limit, offset=offset, access_context=access_context)
        documents = result.get("results", [])
        total = result.get("total", 0)
        formatted = self._format_documents(documents)
        return {"total": total, "documents": formatted, "offset": offset, "limit": limit}

    def get_documents_by_type(
        self,
        doc_type: str,
        limit: int = 10,
        access_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get documents filtered by type"""
        self._ensure_db_connection()
        documents = self.db.find_by_document_type(doc_type, limit, access_context=access_context)
        formatted = self._format_documents(documents)
        return {"total": len(formatted), "documents": formatted}

    def get_document_by_id(
        self,
        doc_id: str,
        access_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get single document by ID"""
        self._ensure_db_connection()
        from bson import ObjectId

        try:
            doc = self.db.collection.find_one({"_id": ObjectId(doc_id)})
            if doc and self.db._is_document_active(doc):
                doc = self.db._with_document_kbac(doc)
                if access_context and not self.db._document_matches_access(doc, access_context):
                    return None
                return {
                    "_id": self._safe_string(doc.get("_id")),
                    "document_type": self._safe_string(doc.get("document_type"), "unknown"),
                    "high_level_metadata": self._safe_dict(doc.get("high_level_metadata"), {}),
                    "confidence": self._safe_dict(doc.get("confidence"), {}),
                    "content": self._safe_dict(doc.get("content"), {}),
                    "source": self._safe_dict(doc.get("source"), {}),
                    "saved_at": self._safe_string(doc.get("saved_at")),
                    "empID": self._safe_string(doc.get("empID")) if doc.get("empID") is not None else None,
                    "empName": self._safe_string(doc.get("empName")) if doc.get("empName") is not None else None,
                    "employee_uuid": self._safe_string(doc.get("employee_uuid")) if doc.get("employee_uuid") is not None else None,
                    "document_uuid": self._safe_string(doc.get("document_uuid")) if doc.get("document_uuid") is not None else None,
                    "file_name": self._safe_string(doc.get("file_name")) if doc.get("file_name") is not None else None,
                    "file_path": self._safe_string(doc.get("file_path")) if doc.get("file_path") is not None else None,
                    "storage_mode": self._safe_string(doc.get("storage_mode")) if doc.get("storage_mode") is not None else None,
                    "isTemporary": self._safe_bool(doc.get("isTemporary"), False),
                    "expiry_at": self._safe_datetime(doc.get("expiry_at")),
                    "assignment_status": self._safe_string(doc.get("assignment_status")) if doc.get("assignment_status") is not None else None,
                    "kbac_scope": self._safe_string(doc.get("kbac_scope")) if doc.get("kbac_scope") is not None else None,
                    "kbac_owner": self._safe_string(doc.get("kbac_owner")) if doc.get("kbac_owner") is not None else None,
                    "suggested_profile": self._safe_dict(doc.get("suggested_profile"), {}),
                }
            return None
        except Exception as e:
            print(f"Error fetching document: {e}")
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        self._ensure_db_connection()
        return self.db.get_statistics()

    def get_dashboard_insights(self) -> Dict[str, Any]:
        """Get dashboard insights for manager activity charts."""
        self._ensure_db_connection()
        return self.db.get_dashboard_insights()

    def search_documents(
        self,
        search_term: Optional[str] = None,
        document_type: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
        access_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Search documents by free text and optional document type with pagination."""
        self._ensure_db_connection()
        result = self.db.search_documents(
            search_term=search_term,
            document_type=document_type,
            limit=limit,
            offset=offset,
            access_context=access_context,
        )
        documents = result.get("results", [])
        total = result.get("total", 0)
        formatted = self._format_documents(documents)
        return {"total": total, "documents": formatted, "offset": offset, "limit": limit}

    def semantic_search(
        self,
        query: str,
        document_type: Optional[str] = None,
        limit: int = 10,
        access_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Semantic/natural-language search over OCR documents."""
        self._ensure_db_connection()
        documents = self.db.semantic_search(
            query=query,
            document_type=document_type,
            limit=limit,
            access_context=access_context,
        )
        formatted = self._format_documents(documents)
        return {"total": len(formatted), "documents": formatted}

    def qa_retrieve_documents(
        self,
        query: str,
        limit: int = 6,
        access_context: Optional[Dict[str, Any]] = None,
        preferred_doc_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve Q&A source documents using the shared MongoDB manager."""
        self._ensure_db_connection()
        bypass_cache = self._is_broad_purchase_order_listing_query(query)
        cache_key = self._qa_retrieval_cache_key(query, limit, access_context, preferred_doc_ids)
        if not bypass_cache:
            cached = self._qa_retrieval_cache_get(cache_key)
            if cached is not None:
                logger.info("[QA] Retrieval cache hit")
                return cached

        sources = self.db.qa_retrieve_documents(
            query=query,
            limit=limit,
            access_context=access_context,
            preferred_doc_ids=preferred_doc_ids,
        )
        if not bypass_cache:
            self._qa_retrieval_cache_set(cache_key, sources)
        return sources

    def qa_retrieve_chunks(
        self,
        query: str,
        limit: int = 6,
        allowed_doc_ids: Optional[List[str]] = None,
        access_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve chunk-level Q&A sources."""
        self._ensure_db_connection()
        return self.db.qa_retrieve_chunks(
            query=query,
            limit=limit,
            allowed_doc_ids=allowed_doc_ids,
            access_context=access_context,
        )

    def detect_duplicates(
        self,
        limit: int = 120,
        similarity_threshold: float = 0.86,
        search_term: Optional[str] = None,
        document_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Detect duplicate, similar, and fraud-risk document copies."""
        self._ensure_db_connection()
        return self.db.detect_duplicates(
            limit=limit,
            similarity_threshold=similarity_threshold,
            search_term=search_term,
            document_type=document_type,
        )

    def find_duplicate_by_file_hash(
        self,
        file_hash: str,
        duplicate_owner: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        self._ensure_db_connection()
        return self.db.find_duplicate_by_file_hash(file_hash, duplicate_owner=duplicate_owner)

    def find_duplicate_for_document(
        self,
        document: Dict[str, Any],
        duplicate_owner: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        self._ensure_db_connection()
        return self.db.find_duplicate_for_document(document, duplicate_owner=duplicate_owner)

    def delete_document(self, doc_id: str) -> bool:
        """Delete document by ID"""
        self._ensure_db_connection()
        return self.db.delete_document(doc_id)

    def delete_documents(self, doc_ids: List[str]) -> Dict[str, Any]:
        """Delete multiple documents by ID."""
        self._ensure_db_connection()
        deleted_ids = []
        not_found_ids = []

        for doc_id in doc_ids:
            if self.db.delete_document(doc_id):
                deleted_ids.append(doc_id)
            else:
                not_found_ids.append(doc_id)

        return {
            "requested_count": len(doc_ids),
            "deleted_count": len(deleted_ids),
            "deleted_ids": deleted_ids,
            "not_found_ids": not_found_ids,
        }

    def search_employee_profiles(
        self,
        search_term: Optional[str] = None,
        limit: int = 20,
        created_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._ensure_db_connection()
        employees = self.db.search_employee_profiles(
            search_term=search_term,
            limit=limit,
            created_by=created_by,
        )
        formatted = []
        for employee in employees:
            formatted.append(
                {
                    "_id": self._safe_string(employee.get("_id")),
                    "empID": self._safe_string(employee.get("empID")),
                    "empName": self._safe_string(employee.get("empName")),
                    "uuid": self._safe_string(employee.get("uuid")),
                    "status": self._safe_string(employee.get("status"), "active"),
                    "documents_count": int(employee.get("documents_count") or 0),
                    "last_document_at": self._safe_datetime(employee.get("last_document_at")),
                    "created_by": self._safe_string(employee.get("created_by")) if employee.get("created_by") is not None else None,
                    "created_at": self._safe_datetime(employee.get("created_at")),
                    "updated_at": self._safe_datetime(employee.get("updated_at")),
                }
            )
        return {"total": len(formatted), "employees": formatted}

    def get_document_chunks(
        self,
        doc_id: str,
        limit: int = 20,
        offset: int = 0,
        access_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._ensure_db_connection()
        result = self.db.get_document_chunks(
            doc_id=doc_id,
            limit=limit,
            offset=offset,
            access_context=access_context,
        )
        chunks = self._format_chunks(result.get("results", []))
        return {"total": result.get("total", 0), "chunks": chunks, "offset": offset, "limit": limit}

    def search_document_chunks(
        self,
        query: str,
        doc_id: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
        access_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._ensure_db_connection()
        result = self.db.search_document_chunks(
            query=query,
            doc_id=doc_id,
            limit=limit,
            offset=offset,
            access_context=access_context,
        )
        chunks = self._format_chunks(result.get("results", []))
        return {"total": result.get("total", 0), "chunks": chunks, "offset": offset, "limit": limit}

    def get_documents_by_employee(
        self,
        emp_id: Optional[str] = None,
        profile_uuid: Optional[str] = None,
        limit: int = 50,
        access_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._ensure_db_connection()
        documents = self.db.get_documents_by_employee(
            emp_id=emp_id,
            profile_uuid=profile_uuid,
            limit=limit,
            access_context=access_context,
        )
        formatted = self._format_documents(documents)
        return {"total": len(formatted), "documents": formatted}

    def create_or_update_employee_profile(self, emp_id: str, emp_name: str, extra: Optional[Dict[str, Any]] = None):
        self._ensure_db_connection()
        return self.db.create_or_update_employee_profile(emp_id=emp_id, emp_name=emp_name, extra=extra or {})

    def delete_employee_profile(self, emp_id: Optional[str] = None, profile_uuid: Optional[str] = None) -> Optional[Dict[str, Any]]:
        self._ensure_db_connection()
        return self.db.delete_employee_profile(emp_id=emp_id, profile_uuid=profile_uuid)

    def delete_employee_profiles(self, emp_ids: List[str]) -> Dict[str, Any]:
        self._ensure_db_connection()
        deleted_ids = []
        not_found_ids = []
        deleted_documents = 0
        deleted_chunks = 0

        for raw_emp_id in emp_ids:
            emp_id = str(raw_emp_id or "").strip()
            if not emp_id:
                continue
            result = self.db.delete_employee_profile(emp_id=emp_id)
            if result:
                deleted_ids.append(emp_id)
                deleted_documents += int(result.get("deleted_documents") or 0)
                deleted_chunks += int(result.get("deleted_chunks") or 0)
            else:
                not_found_ids.append(emp_id)

        return {
            "requested_count": len(emp_ids),
            "deleted_count": len(deleted_ids),
            "deleted_ids": deleted_ids,
            "not_found_ids": not_found_ids,
            "deleted_documents": deleted_documents,
            "deleted_chunks": deleted_chunks,
        }

    def cleanup_expired_temporary_documents(self) -> Dict[str, Any]:
        self._ensure_db_connection()
        return self.db.cleanup_expired_temporary_documents()

    def refresh_employee_manifests(self, limit: Optional[int] = None) -> Dict[str, Any]:
        self._ensure_db_connection()
        return self.db.refresh_all_employee_manifests(limit=limit)

    def close(self):
        """Close database connection"""
        self.db.close()

