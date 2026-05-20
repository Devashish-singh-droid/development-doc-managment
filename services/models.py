from utils.logger import get_logger

logger = get_logger("api_models")

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from enum import Enum


class ApiModel(BaseModel):
    class Config:
        from_attributes = True
        populate_by_name = True


class ConfidenceInfo(ApiModel):
    ocr_engine: str
    confidence_percent: Optional[float] = None


class SuggestedProfileResponse(ApiModel):
    empID: str = ""
    empName: str = ""


class DocumentResponse(ApiModel):
    id: str = Field(alias="_id")
    document_type: str
    high_level_metadata: Dict[str, Any]
    confidence: ConfidenceInfo
    saved_at: Optional[str] = None
    match_preview: Optional[str] = None
    semantic_score: Optional[float] = None
    semantic_reason: Optional[str] = None
    empID: Optional[str] = None
    empName: Optional[str] = None
    employee_uuid: Optional[str] = None
    document_uuid: Optional[str] = None
    file_name: Optional[str] = None
    file_path: Optional[str] = None
    storage_mode: Optional[str] = None
    isTemporary: bool = False
    expiry_at: Optional[str] = None
    assignment_status: Optional[str] = None


class DocumentDetailResponse(ApiModel):
    id: str = Field(alias="_id")
    document_type: str = "unknown"
    high_level_metadata: Dict[str, Any] = Field(default_factory=dict)
    confidence: Dict[str, Any] = Field(default_factory=dict)
    content: Dict[str, Any] = Field(default_factory=dict)
    source: Dict[str, Any] = Field(default_factory=dict)
    saved_at: Optional[str] = None
    empID: Optional[str] = None
    empName: Optional[str] = None
    employee_uuid: Optional[str] = None
    document_uuid: Optional[str] = None
    file_name: Optional[str] = None
    file_path: Optional[str] = None
    storage_mode: Optional[str] = None
    isTemporary: bool = False
    expiry_at: Optional[str] = None
    assignment_status: Optional[str] = None
    suggested_profile: Dict[str, Any] = Field(default_factory=dict)


class DocumentsListResponse(ApiModel):
    total: int
    documents: List[DocumentResponse] = Field(default_factory=list)
    offset: int = 0
    limit: Optional[int] = None
    pages: Optional[int] = None


class StatisticsResponse(ApiModel):
    total_documents: int
    document_types: List[Dict[str, Any]] = Field(default_factory=list)
    average_confidence: float = 0.0


class EmployeeProfileResponse(ApiModel):
    id: str = Field(alias="_id")
    empID: str
    empName: str
    uuid: str
    status: Optional[str] = "active"
    documents_count: int = 0
    last_document_at: Optional[str] = None
    created_by: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class EmployeeProfilesListResponse(ApiModel):
    total: int
    employees: List[EmployeeProfileResponse] = Field(default_factory=list)


class EmployeeSummaryResponse(ApiModel):
    empID: str
    empName: str
    uuid: str


class EmployeeProfileSaveResponse(ApiModel):
    status: str
    employee: EmployeeSummaryResponse


class EmployeeProfileDeleteResponse(ApiModel):
    status: str
    empID: str
    uuid: Optional[str] = None
    deleted_documents: int = 0
    deleted_chunks: int = 0


class BulkDeleteEmployeeProfilesRequest(ApiModel):
    emp_ids: List[str] = Field(default_factory=list)


class BulkDeleteEmployeeProfilesResponse(ApiModel):
    status: str
    requested_count: int
    deleted_count: int
    deleted_ids: List[str] = Field(default_factory=list)
    not_found_ids: List[str] = Field(default_factory=list)
    deleted_documents: int = 0
    deleted_chunks: int = 0


class EmployeeDocumentProfileResolutionResponse(ApiModel):
    status: str
    document_id: str
    expiry_at: Optional[str] = None
    employee: Optional[EmployeeSummaryResponse] = None


class AuditLogEntryResponse(ApiModel):
    id: Optional[str] = Field(default=None, alias="_id")
    username: str = "unknown"
    action: str
    claim_id: Optional[str] = None
    details: Optional[str] = None
    client_ip: Optional[str] = None
    timestamp: Optional[str] = None


class ActivityLogResponse(ApiModel):
    status: str
    days: int
    limit: int
    offset: int
    count: int
    has_more: bool
    activities: List[AuditLogEntryResponse] = Field(default_factory=list)
    username: Optional[str] = None


class SimpleStatusResponse(ApiModel):
    status: str


class SystemSettingsResponse(ApiModel):
    generate_video_transcripts: bool
    save_original_documents: bool
    guided_tour_enabled: bool
    allowed_user_email_domain: str


class UpdateSystemSettingsRequest(ApiModel):
    generate_video_transcripts: bool
    save_original_documents: bool
    guided_tour_enabled: bool = True
    allowed_user_email_domain: str


class HealthResponse(ApiModel):
    status: str
    message: str


class ProcessingStatusResponse(ApiModel):
    status: str
    progress: int = 0
    message: str = ""
    document_id: Optional[str] = None
    document_uuid: Optional[str] = None
    storage_mode: Optional[str] = None
    empID: Optional[str] = None
    empName: Optional[str] = None
    employee_uuid: Optional[str] = None
    isTemporary: Optional[bool] = None
    expiry_at: Optional[str] = None
    assignment_status: Optional[str] = None
    employee_action_required: Optional[bool] = None
    suggested_profile: Optional[SuggestedProfileResponse] = None
    duplicate_of: Optional[Dict[str, Any]] = None
    duplicate_reason: Optional[str] = None


class CleanupTemporaryDocumentsResponse(ApiModel):
    status: str
    deleted_documents: int
    deleted_files: int


class BulkDeleteResponse(ApiModel):
    status: str
    requested_count: int
    deleted_count: int
    deleted_ids: List[str] = Field(default_factory=list)
    not_found_ids: List[str] = Field(default_factory=list)


class DeleteDocumentResponse(ApiModel):
    status: str
    doc_id: str


class DuplicateMatchResponse(ApiModel):
    document_id: Optional[str] = None
    document_uuid: Optional[str] = None
    file_name: Optional[str] = None
    tracking_filename: Optional[str] = None
    document_type: Optional[str] = None
    saved_at: Optional[str] = None
    reason: Optional[str] = None
    file_hash: Optional[str] = None


class DuplicateSkippedItemResponse(ApiModel):
    filename: str
    original_filename: Optional[str] = None
    source_url: Optional[str] = None
    size_bytes: Optional[int] = None
    message: str = ""
    duplicate_of: Optional[DuplicateMatchResponse] = None


class UploadQueuedResponse(ApiModel):
    status: str
    filename: str
    message: str
    duplicate_of: Optional[DuplicateMatchResponse] = None


class MultiUploadQueuedResponse(ApiModel):
    status: str
    uploaded_count: int
    files: List[str] = Field(default_factory=list)
    message: str
    duplicate_skipped_count: int = 0
    skipped_files: List[DuplicateSkippedItemResponse] = Field(default_factory=list)


class ImportedFileResponse(ApiModel):
    filename: str
    original_filename: str
    source_url: str
    size_bytes: int


class ImportFromLinkResponse(ApiModel):
    status: str
    imported_count: int
    files: List[ImportedFileResponse] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    message: str
    duplicate_skipped_count: int = 0
    skipped_files: List[DuplicateSkippedItemResponse] = Field(default_factory=list)


class QACitationResponse(ApiModel):
    doc_id: str
    snippet: str = ""


class QASourceResponse(ApiModel):
    doc_id: str
    document_type: str = "unknown"
    metadata: str = ""
    snippet: str = ""
    items: List[Dict[str, Any]] = Field(default_factory=list)
    full_text: str = ""
    confidence: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None
    reasons: List[str] = Field(default_factory=list)


class QASectionResponse(ApiModel):
    heading: str = ""
    body: str = ""
    bullets: List[str] = Field(default_factory=list)


class QAStructuredAnswerResponse(ApiModel):
    style: str = "paragraph"
    title: str = ""
    summary: str = ""
    highlights: List[str] = Field(default_factory=list)
    sections: List[QASectionResponse] = Field(default_factory=list)
    closing: str = ""


class QAResponse(ApiModel):
    answer: str
    suggestion: str = ""
    conversation_id: str = ""
    turn_id: str = ""
    revision_count: int = 1
    structured_answer: QAStructuredAnswerResponse = Field(default_factory=QAStructuredAnswerResponse)
    citations: List[QACitationResponse] = Field(default_factory=list)
    sources: List[QASourceResponse] = Field(default_factory=list)
    chunk_sources: List[QASourceResponse] = Field(default_factory=list)
    graph_id: str = ""
    chart_options: List[str] = Field(default_factory=list)
    follow_up_suggestions: List[str] = Field(default_factory=list)


class QAHistoryConversationItemResponse(ApiModel):
    conversation_id: str
    title: str = ""
    preview: str = ""
    updated_at: Optional[str] = None
    turn_count: int = 0


class QAHistoryTurnResponse(ApiModel):
    turn_id: str = ""
    question: str = ""
    answer: str = ""
    suggestion: str = ""
    revision_count: int = 1
    structured_answer: QAStructuredAnswerResponse = Field(default_factory=QAStructuredAnswerResponse)
    citations: List[QACitationResponse] = Field(default_factory=list)
    sources: List[QASourceResponse] = Field(default_factory=list)
    chunk_sources: List[QASourceResponse] = Field(default_factory=list)
    graph_id: str = ""
    chart_options: List[str] = Field(default_factory=list)
    follow_up_suggestions: List[str] = Field(default_factory=list)
    created_at: Optional[str] = None


class QAHistoryListResponse(ApiModel):
    status: str
    ttl_hours: int = 24
    conversations: List[QAHistoryConversationItemResponse] = Field(default_factory=list)


class QAHistoryDetailResponse(ApiModel):
    status: str
    conversation_id: str
    title: str = ""
    updated_at: Optional[str] = None
    turns: List[QAHistoryTurnResponse] = Field(default_factory=list)


class UserResponse(ApiModel):
    id: Optional[str] = Field(default=None, alias="_id")
    username: Optional[str] = None
    email: Optional[str] = None
    employee_code: Optional[str] = None
    user_type: Optional[str] = None
    role: Optional[str] = None
    created_by: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class UsersListResponse(ApiModel):
    total: int
    users: List[UserResponse] = Field(default_factory=list)


class CreateUserResponse(ApiModel):
    status: str
    user_id: str
    email: str
    employee_code: str
    user_type: str
    role: Optional[str] = "employee"  # NEW: RBAC role field


class UpdateUserRequest(ApiModel):
    email: str = ""
    employee_code: str = ""
    role: Optional[str] = "employee"


class UpdateUserResponse(ApiModel):
    status: str
    user: UserResponse


class MicrosoftStatusResponse(ApiModel):
    configured: bool
    connected: bool
    account: Dict[str, Any] = Field(default_factory=dict)
    expires_at: Optional[str] = None


class MicrosoftDisconnectResponse(ApiModel):
    status: str
    connected: bool


class ImportFromLinkRequest(ApiModel):
    link: str = Field(..., min_length=5)
    empID: Optional[str] = None
    empName: Optional[str] = None
    choose: Optional[str] = "permanent"
    temporaryRetentionHours: Optional[int] = None


class ResolveEmployeeProfileRequest(ApiModel):
    action: str = Field(..., min_length=4)
    empID: Optional[str] = None
    empName: Optional[str] = None
    temporaryRetentionHours: Optional[int] = None


class EmployeeProfileRequest(ApiModel):
    empID: str = Field(..., min_length=1)
    empName: str = Field(..., min_length=1)


class BulkDeleteDocumentsRequest(ApiModel):
    doc_ids: List[str] = Field(..., min_length=1)


class QARequest(ApiModel):
    question: str = ""
    limit: int = 6
    use_chunks: bool = True
    chunk_limit: Optional[int] = None
    conversation_id: Optional[str] = None
    replace_turn_id: Optional[str] = None


class QAPresentationRequest(ApiModel):
    question: str = ""
    answer: str = ""
    suggestion: str = ""
    structured_answer: QAStructuredAnswerResponse = Field(default_factory=QAStructuredAnswerResponse)
    citations: List[QACitationResponse] = Field(default_factory=list)
    sources: List[QASourceResponse] = Field(default_factory=list)
    graph_data: Dict[str, Any] = Field(default_factory=dict)
    chart_options: List[str] = Field(default_factory=list)
    follow_up_suggestions: List[str] = Field(default_factory=list)


class RefreshManifestRequest(ApiModel):
    limit: Optional[int] = None


class StructurerKeyValueEntry(ApiModel):
    key: str = ""
    value: str = ""


class StructurerItemResponse(ApiModel):
    fields: List[StructurerKeyValueEntry] = Field(default_factory=list)


class StructurerChunkResponse(ApiModel):
    document_type: Optional[str] = None
    metadata_entries: List[StructurerKeyValueEntry] = Field(default_factory=list)
    items: List[StructurerItemResponse] = Field(default_factory=list)
    chunk_summary: str = ""


class CreateUserRequest(ApiModel):
    email: str = ""
    password: str = ""
    employee_code: str = ""
    user_type: str = ""
    role: Optional[str] = "employee"  # NEW: For RBAC system


class ForgotPasswordRequest(ApiModel):
    email: str = ""


class ForgotPasswordVerifyRequest(ApiModel):
    email: str = ""
    otp: str = ""


class ForgotPasswordResetRequest(ApiModel):
    email: str = ""
    reset_token: str = ""
    new_password: str = ""
    confirm_password: str = ""


class ForgotPasswordRequestResponse(ApiModel):
    status: str
    message: str


class ForgotPasswordVerifyResponse(ApiModel):
    status: str
    message: str
    reset_token: str


class ForgotPasswordResetResponse(ApiModel):
    status: str
    message: str


class UserOnboardingUpdateRequest(ApiModel):
    status: str = ""
    current_step: Optional[str] = None


class UserOnboardingStateResponse(ApiModel):
    status: str = "pending"
    current_step: str = ""
    should_show: bool = True
    last_seen_at: Optional[str] = None
    completed_at: Optional[str] = None
    skipped_at: Optional[str] = None


class UserRoleEnum(str, Enum):
    """User role options for RBAC system"""
    EMPLOYEE = "employee"
    MANAGER = "manager"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


class UserRoleResponse(ApiModel):
    """User with role information"""
    id: Optional[str] = Field(default=None, alias="_id")
    email: str
    username: Optional[str] = None
    role: str
    created_at: Optional[str] = None
    created_by: Optional[str] = None
    last_login: Optional[str] = None


class UsersRoleListResponse(ApiModel):
    """List of users with role information"""
    status: str
    users: List[UserRoleResponse] = Field(default_factory=list)
    total: int = 0
