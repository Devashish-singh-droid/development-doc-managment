from services.rbac import (
    UserRole,
    can_create_role,
    filter_qa_response_by_role,
    get_creatable_roles,
    get_session_role,
    has_permission,
    role_to_user_type,
)


def test_session_role_normalizes_legacy_values():
    assert get_session_role({"user_type": "Viewer"}) == UserRole.EMPLOYEE
    assert get_session_role({"user_type": "Manager"}) == UserRole.MANAGER
    assert get_session_role({"user_type": "Admin"}) == UserRole.ADMIN
    assert get_session_role({"user_type": "Super Admin"}) == UserRole.SUPER_ADMIN


def test_role_creation_hierarchy():
    assert can_create_role("manager", "employee") is True
    assert can_create_role("manager", "manager") is True
    assert can_create_role("manager", "admin") is False
    assert can_create_role("admin", "admin") is True
    assert can_create_role("admin", "super_admin") is False
    assert can_create_role("super_admin", "super_admin") is True


def test_creatable_roles_match_expected_policy():
    assert get_creatable_roles("employee") == []
    assert get_creatable_roles("manager") == ["employee", "manager"]
    assert get_creatable_roles("admin") == ["employee", "manager", "admin"]
    assert get_creatable_roles("super_admin") == ["employee", "manager", "admin", "super_admin"]


def test_permission_matrix_matches_expected_policy():
    assert has_permission("employee", "view_ai_buzz") is True
    assert has_permission("employee", "upload_document") is False
    assert has_permission("manager", "upload_document") is True
    assert has_permission("manager", "view_documents") is False
    assert has_permission("admin", "view_documents") is True
    assert has_permission("super_admin", "view_activity_log") is True
    assert has_permission("admin", "view_activity_log") is False


def test_role_to_user_type_mapping():
    assert role_to_user_type("employee") == "Viewer"
    assert role_to_user_type("manager") == "Manager"
    assert role_to_user_type("admin") == "Admin"
    assert role_to_user_type("super_admin") == "Super Admin"


def test_employee_qa_filter_removes_sources_only_for_employee():
    payload = {
        "answer": "visible",
        "sources": [{"doc_id": "1"}],
        "chunk_sources": [{"doc_id": "1"}],
        "citations": [{"doc_id": "1"}],
    }

    employee_view = filter_qa_response_by_role(UserRole.EMPLOYEE, dict(payload))
    manager_view = filter_qa_response_by_role(UserRole.MANAGER, dict(payload))

    assert "sources" not in employee_view
    assert "chunk_sources" not in employee_view
    assert "citations" not in employee_view
    assert manager_view["sources"] == [{"doc_id": "1"}]
