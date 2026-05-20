import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from utils.logger import get_logger
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pathlib import Path
from config import _Settings
from routers.routes import router, start_cleanup_scheduler, public_transcript_page, generate_transcript

load_dotenv() 
logger = get_logger("app")
settings = _Settings()
DOC_MANAGEMENT_PREFIX = "/doc-management"


class ReverseProxyPrefixMiddleware:
    def __init__(self, app, prefix: str):
        self.app = app
        self.prefix = str(prefix or "").rstrip("/")

    async def __call__(self, scope, receive, send):
        if self.prefix and scope.get("type") in {"http", "websocket"}:
            path = str(scope.get("path") or "")
            if path == self.prefix or path.startswith(f"{self.prefix}/"):
                stripped_path = path[len(self.prefix):] or "/"
                updated_scope = dict(scope)
                current_root_path = str(scope.get("root_path") or "").rstrip("/")
                updated_scope["root_path"] = f"{current_root_path}{self.prefix}"
                updated_scope["path"] = stripped_path
                raw_path = scope.get("raw_path")
                if raw_path is not None:
                    try:
                        updated_scope["raw_path"] = stripped_path.encode("ascii")
                    except UnicodeEncodeError:
                        pass
                scope = updated_scope
            elif scope.get("type") == "http":
                # Keep the public app canonical under /doc-management so
                # browser-visible SPA routes and redirects stay consistent.
                target_path = f"{self.prefix}/" if path == "/" else f"{self.prefix}{path}"
                query_string = scope.get("query_string") or b""
                if query_string:
                    target_path = f"{target_path}?{query_string.decode('latin-1')}"
                response = RedirectResponse(url=target_path, status_code=307)
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)

# Create FastAPI app
app = FastAPI(
    title="Kasturi-BIS API",
    description="API to retrieve and manage AI-extracted structured documents",
    version=settings.version or "1.0.0"
)


# Register authentication endpoints and middleware FIRST
from routers.auth import get_auth_routes
get_auth_routes(app)
app.add_middleware(ReverseProxyPrefixMiddleware, prefix=DOC_MANAGEMENT_PREFIX)


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Serve static files (logo, css, favicon, etc.)
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# Include API routes
app.include_router(router)
app.add_api_route("/public-transcript", public_transcript_page, methods=["GET"], include_in_schema=False)
app.add_api_route("/generate-transcript", generate_transcript, methods=["POST"], include_in_schema=False)


@app.on_event("startup")
async def startup_background_jobs():
    start_cleanup_scheduler()


# Serve templates
TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _build_user_badge(username: str | None) -> tuple[str, str]:
    raw_username = (username or "").strip()
    display_name = raw_username or "User"

    badge_source = raw_username.split("@", 1)[0] if "@" in raw_username else raw_username
    words = [part for part in "".join(ch if ch.isalnum() else " " for ch in badge_source).split() if part]

    if len(words) >= 2:
        initials = (words[0][0] + words[1][0]).upper()
    elif words:
        initials = words[0][:2].upper()
    else:
        initials = "U"

    return display_name, initials


def _get_request_base_path(request: Request) -> str:
    return str(request.scope.get("root_path") or "").rstrip("/")


def render_index(request: Request):
    qa_user_name, qa_user_initials = _build_user_badge(getattr(request.state, "username", None))
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "base_path": _get_request_base_path(request),
            "qa_user_name": qa_user_name,
            "qa_user_initials": qa_user_initials,
            "app_version": settings.version,
        },
    )


# Serve /auditlog and /audit_log for SPA navigation
@app.get("/auditlog")
@app.get("/audit_log")
async def auditlog(request: Request):
    return render_index(request)

# Serve main page for root and all SPA panels
@app.get("/")
async def root(request: Request):
    """Serve the main HTML page"""
    return render_index(request)


@app.get("/dashboard")
async def dashboard(request: Request):
    return render_index(request)


@app.get("/documents")
async def documents(request: Request):
    return render_index(request)


@app.get("/employeeProfiles")
@app.get("/employee-profiles")
async def employee_profiles(request: Request):
    return render_index(request)


@app.get("/qa")
async def qa(request: Request):
    return render_index(request)


@app.get("/users")
async def users(request: Request):
    return render_index(request)


@app.get("/upload")
@app.get("/upload-&-process")
async def upload_and_process(request: Request):
    return render_index(request)


@app.get("/analytics")
async def analytics(request: Request):
    return RedirectResponse(url=f"{_get_request_base_path(request)}/dashboard", status_code=307)


@app.get("/settings")
async def settings_page(request: Request):
    return render_index(request)


@app.get("/gem")
@app.get("/gem(mm)")
@app.get("/GEM")
async def gem(request: Request):
    return render_index(request)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=os.getenv("PORT", 8081),
        reload=False,
        log_level="info",
        proxy_headers=True,
        forwarded_allow_ips=os.getenv("FORWARDED_ALLOW_IPS", "127.0.0.1"),
    )
