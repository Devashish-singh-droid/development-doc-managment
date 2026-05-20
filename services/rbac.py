"""
Role-Based Access Control (RBAC) System for Kasturi-BIS.

Roles:
1. EMPLOYEE - AI Buzz only, without sources
2. MANAGER - AI Buzz with sources, upload/process, create Employee/Manager,
   and manage only employee profiles they created
3. ADMIN - Manager capabilities plus dashboard/documents and all employee profiles,
   and can create users up to Admin
4. SUPER_ADMIN - Full access to the application
"""

from enum import Enum
from typing import List, Optional, Set
from functools import wraps
from fastapi import HTTPException, status, Request
from utils.logger import get_logger

logger = get_logger("rbac")


class UserRole(str, Enum):
    """User role definitions"""
    EMPLOYEE = "employee"
    MANAGER = "manager"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"

    def __str__(self):
        return self.value

    def __repr__(self):
        return f"UserRole.{self.name}"


ROLE_DISPLAY_NAMES = {
    UserRole.EMPLOYEE: "Employee",
    UserRole.MANAGER: "Manager",
    UserRole.ADMIN: "Admin",
    UserRole.SUPER_ADMIN: "Super Admin",
}

ROLE_TO_USER_TYPE = {
    UserRole.EMPLOYEE: "Viewer",
    UserRole.MANAGER: "Manager",
    UserRole.ADMIN: "Admin",
    UserRole.SUPER_ADMIN: "Super Admin",
}

LEGACY_ROLE_MAP = {
    "viewer": UserRole.EMPLOYEE,
    "external user": UserRole.EMPLOYEE,
    "external-user": UserRole.EMPLOYEE,
    "employee": UserRole.EMPLOYEE,
    "manager": UserRole.MANAGER,
    "admin": UserRole.ADMIN,
    "super admin": UserRole.SUPER_ADMIN,
    "super_admin": UserRole.SUPER_ADMIN,
    "super-admin": UserRole.SUPER_ADMIN,
}

# Role hierarchy (for role-based creation restrictions)
ROLE_HIERARCHY = {
    UserRole.SUPER_ADMIN: 4,
    UserRole.ADMIN: 3,
    UserRole.MANAGER: 2,
    UserRole.EMPLOYEE: 1,
}

# Maximum role a user can create
MAX_CREATABLE_ROLE = {
    UserRole.SUPER_ADMIN: UserRole.SUPER_ADMIN,
    UserRole.ADMIN: UserRole.ADMIN,
    UserRole.MANAGER: UserRole.MANAGER,
    UserRole.EMPLOYEE: UserRole.EMPLOYEE,
}

# Permission matrix: role -> allowed features
PERMISSIONS = {
    UserRole.EMPLOYEE: {
        "view_ai_buzz",
        "view_ai_buzz_without_source",
    },
    UserRole.MANAGER: {
        "view_ai_buzz",
        "view_ai_buzz_with_source",
        "upload_document",
        "process_document",
        "manage_settings",
        "view_users",
        "create_user",
        "view_own_employees",
        "view_own_employee_profiles",
        "view_own_employee_documents",
        "create_employee_profile",
        "resolve_employee_profile",
        "delete_own_employees",
        "view_upload_queue",
    },
    UserRole.ADMIN: {
        "view_ai_buzz",
        "view_ai_buzz_with_source",
        "upload_document",
        "process_document",
        "manage_settings",
        "view_dashboard",
        "view_documents",
        "view_all_documents",
        "view_document_chunks",
        "delete_document",
        "view_statistics",
        "view_users",
        "create_user",
        "edit_user",
        "view_all_employees",
        "view_all_employee_profiles",
        "view_all_employee_documents",
        "create_employee_profile",
        "resolve_employee_profile",
        "delete_employee",
        "view_upload_queue",
    },
    UserRole.SUPER_ADMIN: {
        "view_ai_buzz",
        "view_ai_buzz_with_source",
        "upload_document",
        "process_document",
        "manage_settings",
        "view_dashboard",
        "view_documents",
        "view_all_documents",
        "view_document_chunks",
        "delete_document",
        "view_statistics",
        "view_users",
        "create_user",
        "edit_user",
        "delete_user",
        "view_all_employees",
        "view_all_employee_profiles",
        "view_all_employee_documents",
        "create_employee_profile",
        "resolve_employee_profile",
        "delete_employee",
        "view_activity_log",
        "modify_activity_log",
        "manage_system",
        "view_upload_queue",
        "manage_upload_queue",
    },
}


def normalize_role_value(role: object) -> Optional[str]:
    raw = str(role or "").strip().lower()
    if not raw:
        return None
    mapped = LEGACY_ROLE_MAP.get(raw)
    if mapped:
        return mapped.value
    try:
        return UserRole(raw).value
    except ValueError:
        return None


def normalize_role(role: object) -> Optional[UserRole]:
    normalized = normalize_role_value(role)
    return UserRole(normalized) if normalized else None


def role_to_user_type(role: object) -> str:
    role_obj = normalize_role(role) or UserRole.EMPLOYEE
    return ROLE_TO_USER_TYPE.get(role_obj, "Viewer")


def get_role_display_name(role: object) -> str:
    role_obj = normalize_role(role) or UserRole.EMPLOYEE
    return ROLE_DISPLAY_NAMES.get(role_obj, "Employee")


def get_role_permissions(role: object) -> Set[str]:
    role_obj = normalize_role(role)
    if not role_obj:
        return set()
    return set(PERMISSIONS.get(role_obj, set()))


def has_permission(role: UserRole, permission: str) -> bool:
    """Check if a role has a specific permission"""
    role_obj = normalize_role(role)
    if not role_obj:
        return False
    return permission in PERMISSIONS.get(role_obj, set())


def has_any_permission(role: UserRole, permissions: List[str]) -> bool:
    """Check if role has ANY of the given permissions"""
    return any(has_permission(role, perm) for perm in permissions)


def has_all_permissions(role: UserRole, permissions: List[str]) -> bool:
    """Check if role has ALL given permissions"""
    return all(has_permission(role, perm) for perm in permissions)


def can_create_role(creator_role: UserRole, target_role: UserRole) -> bool:
    """Check if creator_role can create a user with target_role"""
    creator_obj = normalize_role(creator_role)
    target_obj = normalize_role(target_role)
    if not creator_obj or not target_obj or not has_permission(creator_obj, "create_user"):
        return False

    max_role = MAX_CREATABLE_ROLE.get(creator_obj)
    if max_role is None:
        return False

    creator_hierarchy = ROLE_HIERARCHY.get(max_role, 0)
    target_hierarchy = ROLE_HIERARCHY.get(target_obj, 0)
    return target_hierarchy <= creator_hierarchy


def get_creatable_roles(creator_role: object) -> List[str]:
    creator_obj = normalize_role(creator_role)
    if not creator_obj or not has_permission(creator_obj, "create_user"):
        return []
    creator_hierarchy = ROLE_HIERARCHY.get(MAX_CREATABLE_ROLE.get(creator_obj), 0)
    return [
        role.value
        for role in UserRole
        if ROLE_HIERARCHY.get(role, 0) <= creator_hierarchy
    ]


def validate_role_string(role_str: str) -> UserRole:
    """Validate and normalize role string"""
    normalized = normalize_role(role_str)
    if normalized:
        return normalized
    valid_roles = ", ".join([r.value for r in UserRole])
    raise ValueError(f"Invalid role: {role_str}. Must be one of: {valid_roles}")


def get_session_role(session_data: dict) -> Optional[UserRole]:
    """Extract and validate role from session"""
    if not session_data:
        return None
    return normalize_role(session_data.get("role") or session_data.get("user_type"))


def require_roles(*allowed_roles: str):
    """
    Decorator to enforce role-based access on FastAPI endpoints
    
    Usage:
        @require_roles("admin", "super_admin")
        async def some_endpoint(request: Request):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, request: Request = None, **kwargs):
            # Extract session from request
            if not request:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Request context missing"
                )
            
            session_id = request.cookies.get("session_id")
            if not session_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Not authenticated"
                )
            
            # Import here to avoid circular imports
            from routers.auth import get_session_data
            session_data = get_session_data(session_id)
            
            if not session_data:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Session expired"
                )
            
            user_role = get_session_role(session_data)
            if not user_role:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="User role not set"
                )
            
            required_roles = [UserRole(r.lower()) for r in allowed_roles]
            if user_role not in required_roles:
                logger.warning(
                    f"Unauthorized access attempt by {session_data.get('email')} "
                    f"(role: {user_role}) to resource requiring {required_roles}"
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied. Required role(s): {', '.join([r.value for r in required_roles])}"
                )
            
            return await func(*args, request=request, **kwargs)
        
        return wrapper
    return decorator


def check_permission(session_data: dict, permission: str) -> bool:
    """Check if session user has permission"""
    user_role = get_session_role(session_data)
    if not user_role:
        return False
    return has_permission(user_role, permission)


def filter_qa_response_by_role(user_role: UserRole, response: dict) -> dict:
    """
    Filter Q&A response based on user role
    Employee role doesn't see source information
    """
    if user_role == UserRole.EMPLOYEE:
        # Remove sensitive fields for employees
        response.pop("sources", None)
        response.pop("chunk_sources", None)
        response.pop("citations", None)
    
    return response
