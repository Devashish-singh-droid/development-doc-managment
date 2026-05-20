# Kasturi-BIS - Project Documentation

**Last Updated:** March 27, 2026

---

## 📖 Table of Contents

1. [Project Overview](#project-overview)
2. [Project Architecture](#project-architecture)
3. [File Structure & Purpose](#file-structure--purpose)
4. [Core Modules](#core-modules)
5. [Key Workflows](#key-workflows)
6. [Technology Stack](#technology-stack)
7. [Setup & Installation](#setup--installation)
8. [API Endpoints](#api-endpoints)
9. [Database Schema](#database-schema)
10. [How to Extend](#how-to-extend)

---

## 🎯 Project Overview

This is an **enterprise-grade OCR + AI-powered document management system** designed to:

- **Extract text** from documents (PDF, Word, Excel, Images) using PaddleOCR
- **Structure metadata** (title, author, date, keywords) using Google Gemini
- **Store documents** in MongoDB with full-text indexing
- **Search & retrieve** documents using keyword search and fuzzy matching
- **Answer questions** over stored documents using Retrieval-Augmented Generation (RAG)
- **Manage users** with role-based access control and session management
- **Track activity** with auto-audit logging of user actions

**Use Case:** Organizations need to extract, organize, and intelligently query large volumes of documents (contracts, invoices, reports, bids, etc.)

---

## 🏗️ Project Architecture

### High-Level System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                        WEB BROWSER (SPA)                        │
│  Dashboard | Documents | AI Buzz | Upload | Users | Audit Log  │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTP/WebSocket
                         ▼
        ┌─────────────────────────────────────┐
        │      FastAPI Web Server (app.py)    │
        │                                     │
        │  ├─ Auth Routes (auth.py)          │
        │  ├─ API Routes (routes.py)         │
        │  └─ Static Files & Templates       │
        └──┬──────────────────────────────┬──┘
           │                              │
    ┌──────▼──────────┐          ┌──────▼──────────┐
    │  OCR Pipeline   │          │  AI/LLM Layer  │
    │                 │          │                │
    │ • PaddleOCR     │          │• Llama-3 8B    │
    │   (paddle_      │          │  Structurer    │
    │   ocr_engine)   │          │  (llama_hf_    │
    │                 │          │   structurer)  │
    │ • Document      │          │                │
    │   Extraction    │          │• Q&A Answerer  │
    │   (pdf_text_    │          │  (qa_hf_       │
    │   utils, etc.)  │          │   answerer)    │
    │                 │          │                │
    │ • Confidence    │          │• RAG Embedder  │
    │   Scoring       │          │  (rag_         │
    │                 │          │   embedder)    │
    └────────┬────────┘          └────────┬───────┘
             │                            │
             └────────────┬───────────────┘
                          │
                   ┌──────▼──────────┐
                   │   Database      │
                   │   Layer         │
                   │                 │
                   │ • MongoDBMgr    │
                   │ • DatabaseSvc   │
                   │                 │
                   └────────┬────────┘
                            │
                      ┌─────▼─────┐
                      │ MongoDB    │
                      │ Collection │
                      └────────────┘
```

### Data Flow Example: Document Upload

```
User Uploads File (Web UI)
         │
         ▼
  /api/upload endpoint (routes.py)
         │
         ├─→ Save to temp storage
         │
         ├─→ Check for selectable text
         │   ├─ PDF → pdf_text_utils.py
         │   ├─ Word → word_text_utils.py
         │   ├─ Excel → excel_text_utils.py
         │
         ├─→ If no selectable text → PaddleOCR (paddle_ocr_engine.py)
         │
         ├─→ Extract metadata with Gemini (`gemini_structurer.py`)
         │
         ├─→ Store in MongoDB (mongo_db.py)
         │
         └─→ Return status & document info
```

---

## 📂 File Structure & Purpose

### Root Level Files

```
ocr/                          OCR project root
├── app.py                    ⭐ FastAPI app entry point
├── main.py                   Standalone document processor
├── auth.py                   User auth & session management
├── logger.py                 Logging configuration
├── requirements.txt          Python dependencies
├── PANEL_PROCESS.md          UI panel documentation
├── .env                       (Not in repo) Configuration file
└── env3.10/                  Virtual environment
```

---

## 🔧 Core Modules

### 1️⃣ **Application Core** (`app.py`)

**Responsibility:** Web server setup and routing

**Key Features:**
- FastAPI application initialization
- CORS middleware (allow cross-origin requests)
- Static file serving (CSS, JS, images)
- Authentication middleware
- Route registration
- SPA frontend serving

**Endpoints Registered:**
- Auth routes from `auth.py`
- API routes from `api/routes.py`
- Frontend routes (/, /dashboard, /documents, etc.)

```python
# Example flow
app = FastAPI()
get_auth_routes(app)              # Register /login, /logout, /register
app.include_router(router)         # Register /api/* routes
# Serve index.html for SPA navigation
```

---

### 2️⃣ **Authentication & Sessions** (`auth.py`)

**Responsibility:** User authentication, session management, and audit logging

**Key Components:**
- Session creation and validation
- Cookie-based authentication
- Login/logout endpoints
- Auto-audit logging of API calls
- Session timeout management (TTL: 12 hours, inactivity: 2 hours)

**Session Structure:**
```python
SESSION_STORE = {
    "session_id": {
        "user_id": 123,
        "username": "john@example.com",
        "start": datetime,
        "last": datetime
    }
}
```

**Auto-Audit:** Logs every API call with:
- User ID
- Action (GET_/API/DOCUMENTS, POST_/API/UPLOAD, etc.)
- Timestamp
- Stored in MongoDB `activity_log` collection

---

### 3️⃣ **OCR Pipeline** (`ocr/paddle_ocr_engine.py`)

**Responsibility:** Extract text from images using PaddleOCR

**Features:**
- **Thread-safe singleton** - Only one OCR model loaded in memory
- **Multi-language support** - English by default, configurable
- **GPU support** - Optional GPU acceleration
- **Confidence scoring** - Returns text + confidence percentage per line
- **Reusable** - Used by main.py, routes.py, and other modules

**Function Signature:**
```python
def paddle_ocr_extract(image: PIL.Image) -> List[Dict]:
    # Returns: [
    #   {"text": "extracted text", "confidence": 0.95},
    #   ...
    # ]
```

---

### 4️⃣ **AI Metadata Extraction** (`ai/gemini_structurer.py`)

**Responsibility:** Extract structured metadata from OCR text using LLM

**Features:**
- Uses Google Gemini for chunk-based JSON structuring
- Extracts: title, author, date, keywords, summary, document_type
- Handles JSON parsing with error recovery
- Generates automatic titles if not found
- Confidence scoring

**Function Signature:**
```python
def extract_metadata_with_gemini(text: str) -> Dict:
    # Returns: {
    #   "title": "Document Title",
    #   "author": "John Doe",
    #   "date": "2024-01-15",
    #   "keywords": ["keyword1", "keyword2"],
    #   "summary": "Brief summary",
    #   "document_type": "contract",
    #   "confidence": 0.85
    # }
```

---

### 5️⃣ **Q&A System** (`ai/qa_hf_answerer.py`)

**Responsibility:** Answer user questions using RAG (Retrieval-Augmented Generation)

**Features:**
- Uses Google Gemini
- Retrieves relevant documents first
- Answers **only** from provided documents
- **Response caching** - Cache TTL: 120s, max 128 items
- Confidence scoring
- Citation generation

**Function Signature:**
```python
def answer_question(question: str, documents: List[str]) -> Dict:
    # Returns: {
    #   "answer": "Concise one-sentence answer",
    #   "citations": [
    #     {"doc_id": "123", "snippet": "relevant text"}
    #   ],
    #   "confidence": 0.92
    # }
```

---

### 6️⃣ **Document Embeddings** (`ai/rag_embedder.py`)

**Responsibility:** Create semantic embeddings for similarity search

**Features:**
- Uses sentence-transformers (MiniLM-L6)
- Creates embeddings for all documents
- Enables semantic similarity search
- **FAISS indexing** - Fast approximate nearest neighbor search

**Function Signature:**
```python
def embed_document(text: str) -> List[float]:
    # Returns embedding vector (384 dimensions by default)

def find_similar_documents(query: str, k: int = 5) -> List[Dict]:
    # Returns top-k most similar documents
```

---

### 7️⃣ **Database Operations** (`api/database.py` + `utils/mongo_db.py`)

**Responsibility:** All database operations and document management

**Key Services:**

#### `database.py` - High-level business logic
```python
class DatabaseService:
    def save_document(doc: Dict) -> str              # Insert doc, return ID
    def get_documents(limit: int) -> List[Dict]     # Get all docs
    def search_documents(query: str) -> List[Dict]  # Full-text search
    def get_statistics() -> Dict                     # Document counts
    def find_duplicates() -> List[List[Dict]]       # Group similar docs
    def delete_document(doc_id: str) -> bool        # Remove doc
```

#### `mongo_db.py` - Low-level MongoDB operations
```python
class MongoDBManager:
    def insert_document(doc: Dict) -> str           # Save to DB
    def find(query: Dict) -> List[Dict]             # Query
    def update(doc_id: str, update: Dict)           # Update
    def delete(doc_id: str)                         # Delete
    def fuzzy_search(query: str) -> List[Dict]      # Fuzzy matching
    def similarity_score(text1, text2) -> float     # Cosine similarity
```

---

### 8️⃣ **API Routes** (`api/routes.py`)

**Responsibility:** All REST endpoints for client-server communication

**Endpoint Categories:**

#### Documents
- `GET /api/documents` - List all documents
- `GET /api/documents/{id}` - Get document details
- `DELETE /api/documents/{id}` - Delete document
- `GET /api/search?query=text` - Search documents
- `GET /api/documents/duplicates` - Find duplicates

#### Upload & Processing
- `POST /api/upload` - Upload single file
- `POST /api/upload-multiple` - Upload multiple files
- `GET /api/status/{filename}` - Get processing status

#### AI Features
- `POST /api/qa` - Ask question → get answer
- `GET /api/statistics` - Document & user statistics

#### Users & Admin
- `GET /api/users` - List all users
- `POST /api/users` - Create new user
- `DELETE /api/users/{id}` - Delete user

#### Activity Logging
- `GET /api/activity-log` - Get audit log
- `POST /api/activity/log` - Log action

---

### 9️⃣ **Utility Modules** (`utils/`)

#### PDF Utilities (`pdf_text_utils.py`)
```python
def pdf_has_selectable_text(path: str) -> bool
def extract_pdf_text(path: str) -> str
# Avoids OCR for PDFs with selectable text
```

#### Word Utilities (`word_text_utils.py`)
```python
def word_has_selectable_text(path: str) -> bool
def extract_word_text(path: str) -> str
```

#### Excel Utilities (`excel_text_utils.py`)
```python
def excel_has_selectable_text(path: str) -> bool
def extract_excel_text(path: str) -> str
```

#### File Output (`output_writer.py`)
```python
def save_json(data: Dict, filename: str)
def save_txt(text: str, filename: str)
```

---

## 🔄 Key Workflows

### **Workflow 1: Document Upload & Indexing**

```
1. User selects file & clicks "Process"
   ↓
2. Frontend sends POST /api/upload
   ↓
3. Backend saves file to temp directory
   ↓
4. Check for selectable text:
   ├─ PDF → pdf_text_utils.extract_pdf_text()
   ├─ Word → word_text_utils.extract_word_text()
   ├─ Excel → excel_text_utils.extract_excel_text()
   └─ If found → Skip to step 7
   ↓
5. No selectable text found → OCR:
   └─ paddle_ocr_engine.paddle_ocr_extract()
   ↓
6. Extract metadata:
   └─ gemini_structurer.extract_metadata_with_gemini()
   ↓
7. Create embeddings:
   └─ rag_embedder.embed_document()
   ↓
8. Store in MongoDB:
   └─ mongo_db.MongoDBManager.insert_document()
   ↓
9. Update status & return to frontend
   ↓
10. Frontend polls /api/status/{filename} until complete
```

**Status Values:** `pending` → `processing` → `completed` / `failed`

---

### **Workflow 2: Search Documents**

```
1. User enters search query
   ↓
2. Frontend sends GET /api/search?query=<text>
   ↓
3. routes.py receives query
   ↓
4. database.py performs search:
   ├─ Full-text index search
   ├─ Fuzzy matching (if no results)
   └─ Returns ranked results
   ↓
5. mongo_db.py queries MongoDB
   ↓
6. Results returned with score & snippet
   ↓
7. Frontend displays results with highlighting
```

---

### **Workflow 3: Ask Question (AI Buzz Panel)**

```
1. User types question
   ↓
2. Frontend sends POST /api/qa with question
   ↓
3. routes.py receives request
   ↓
4. Check response cache:
   ├─ Hit → Return cached answer (skip to step 8)
   └─ Miss → Continue
   ↓
5. Retrieve relevant documents:
   └─ rag_embedder.find_similar_documents(question, k=6)
   ↓
6. Generate answer with LLM:
   └─ qa_hf_answerer.answer_question(question, documents)
   ↓
7. Cache the response
   ↓
8. Return answer + citations to frontend
   ↓
9. User can click "View Document" links for each citation
```

---

### **Workflow 4: User Authentication**

```
1. Unauthenticated user visits /
   ↓
2. Redirected to /login by middleware
   ↓
3. User enters email + password
   ↓
4. Frontend POST /login
   ↓
5. auth.py verifies credentials against MongoDB users collection
   ↓
6. Session created:
   └─ Session ID generated & stored in SESSION_STORE
   └─ Cookie sent to client
   ↓
7. User redirected to /dashboard
   ↓
8. All subsequent requests include session cookie
   ↓
9. Middleware validates session & injects user_id into request context
```

---

## 🛠️ Technology Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Web Framework** | FastAPI | 0.104.1 | REST API & SPA serving |
| **ASGI Server** | Uvicorn | 0.24.0 | Production-grade async web server |
| **Database** | MongoDB | Latest | Document storage & indexing |
| **OCR** | PaddleOCR | 2.7.0.3 | Text extraction from images |
| **LLM Provider** | Google Gemini | - | Gemini API for document structuring and Q&A |
| **Embeddings** | Sentence Transformers | 2.6.1 | Document similarity search |
| **Vector DB** | FAISS | 1.8.0 | Fast similarity search |
| **Doc Processing** | PyMuPDF, python-docx, openpyxl | Latest | PDF, Word, Excel handling |
| **Machine Learning** | NumPy, scikit-learn, rapidfuzz | Latest | Numeric & fuzzy operations |
| **Frontend** | HTML/CSS/JavaScript | Vanilla | SPA with responsive UI |

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10+
- MongoDB instance (local or cloud)
- Google Gemini API key

### Step 1: Create Virtual Environment
```bash
python -m venv env3.10
.\env3.10\Scripts\Activate.ps1  # Windows
source env3.10/bin/activate     # Linux/Mac
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configuration (.env file)
```env
# MongoDB
MONGO_URI=mongodb://localhost:27017/
DB_NAME=document_management
COLLECTION_NAME=documents
USERS_COLLECTION=users

# Gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL_NAME=gemini-2.5-flash

# Caching
QA_RESPONSE_CACHE_TTL_SECONDS=120
QA_RESPONSE_CACHE_MAX_SIZE=128

# Session Management
SESSION_TTL_HOURS=12
INACTIVITY_TTL_HOURS=2
```

### Step 4: Start MongoDB
```bash
# Local MongoDB
mongod

# Or use MongoDB Atlas (cloud)
# Update MONGO_URI in .env
```

### Step 5: Run Application
```bash
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Access: http://localhost:8000

---

## 📡 API Endpoints

### Authentication
```
POST   /login                    Login with email & password
POST   /logout                   Logout (invalidate session)
POST   /register                 Create new user account
GET    /api/auth/me              Get current user info
```

### Documents
```
GET    /api/documents            List all documents
GET    /api/documents/{id}       Get document details
DELETE /api/documents/{id}       Delete document
GET    /api/search?query=text    Search documents
GET    /api/documents/duplicates Find duplicate documents
```

### Upload
```
POST   /api/upload               Upload single file
POST   /api/upload-multiple      Upload multiple files
GET    /api/status/{filename}    Get processing status
```

### AI Features
```
POST   /api/qa                   Ask question & get answer
GET    /api/statistics           Get document statistics
```

### Users (Admin)
```
GET    /api/users                List all users
POST   /api/users                Create new user
DELETE /api/users/{id}           Delete user
```

### Activity Log
```
GET    /api/activity-log         Get audit log
POST   /api/activity/log         Log an action
```

---

## 🗄️ Database Schema

### Documents Collection
```javascript
{
  "_id": ObjectId,
  "filename": "contract_2024.pdf",
  "file_path": "/path/to/file",
  "file_size": 1024000,
  "file_type": "pdf",
  
  // Extracted Content
  "raw_text": "extracted OCR text...",
  "ocr_confidence": 0.95,
  
  // Structured Metadata
  "title": "Sales Contract",
  "author": "John Doe",
  "date": "2024-01-15",
  "keywords": ["contract", "sales", "2024"],
  "summary": "Contract regarding...",
  "document_type": "contract",
  
  // Embeddings
  "embedding": [0.123, 0.456, ...],  // 384 dimensions
  
  // Metadata
  "uploaded_by": 123,                 // user_id
  "uploaded_at": ISODate,
  "updated_at": ISODate,
  "hash": "sha256_hash",              // For duplicate detection
  
  // Status
  "status": "completed",              // pending, processing, completed, failed
  "error": null
}
```

### Users Collection
```javascript
{
  "_id": ObjectId,
  "email": "user@example.com",
  "password_hash": "bcrypt_hash",
  "user_type": "admin",              // admin, analyst, viewer
  "employee_code": "EMP001",
  "created_at": ISODate,
  "last_login": ISODate,
  "is_active": true
}
```

### Activity Log Collection
```javascript
{
  "_id": ObjectId,
  "user_id": 123,
  "username": "user@example.com",
  "action": "POST_/API/UPLOAD",
  "method": "POST",
  "endpoint": "/api/upload",
  "status_code": 200,
  "timestamp": ISODate,
  "details": {...}                  // Additional context
}
```

---

## 📈 How to Extend

### Adding a New File Format

1. Create new utility file: `utils/newformat_text_utils.py`
```python
def newformat_has_selectable_text(path: str) -> bool:
    # Detect if file has extractable text
    pass

def extract_newformat_text(path: str) -> str:
    # Extract text
    pass
```

2. Update `api/routes.py` upload handler:
```python
elif newformat_has_selectable_text(DOC_PATH):
    all_text = extract_newformat_text(DOC_PATH)
    ocr_confidence = 100.0
```

---

### Adding a New API Endpoint

1. Add function to `api/routes.py`:
```python
@router.get("/api/custom-endpoint")
async def custom_endpoint(request: Request, param: str):
    user_id = request.state.user_id
    # Implement logic
    return {"result": "data"}
```

2. Auto-audit logging happens automatically if path matches

---

### Adding a New AI Feature

1. Create new AI module: `ai/new_feature.py`
2. Implement using the Gemini client helper
3. Add endpoint in `api/routes.py`
4. Add caching if needed (like qa_hf_answerer)

---

### Customizing OCR

Edit `ocr/paddle_ocr_engine.py`:
```python
def get_paddle_ocr(lang="en", use_gpu=False):
    # Change language: lang="zh"  (Chinese)
    # Enable GPU: use_gpu=True
    pass
```

---

## 🔐 Security Considerations

1. **Session Management:** 12-hour TTL, 2-hour inactivity timeout
2. **Password Storage:** Use bcrypt hashing (implement in auth.py)
3. **CORS:** Currently allows all origins (restrict in production)
4. **API Keys:** Gemini API key stored in .env (never commit)
5. **Database Access:** MongoDB URI in .env
6. **Audit Logging:** All user actions tracked

---

## 📊 Performance Optimization

1. **Caching:** Q&A responses cached for 120 seconds
2. **FAISS Indexing:** Fast similarity search with approximate nearest neighbors
3. **Connection Pooling:** MongoDB connection reuse
4. **Lazy Loading:** Paddle OCR model loaded on-demand
5. **Batch Processing:** Support for multiple file uploads

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| OCR not working | Check PaddleOCR installation, GPU memory |
| LLM timeout | Check Gemini API key validity and provider quota |
| MongoDB connection error | Verify MONGO_URI in .env, MongoDB service running |
| Q&A returns generic answer | Insufficient relevant documents, improve query |
| Slow search | Add MongoDB indexes on `raw_text`, `title` |

---

## 📚 References

- **FastAPI Docs:** https://fastapi.tiangolo.com/
- **PaddleOCR:** https://github.com/PaddlePaddle/PaddleOCR
- **Google Gemini:** https://ai.google.dev/
- **MongoDB:** https://docs.mongodb.com/
- **FAISS:** https://github.com/facebookresearch/faiss

---

**Document Version:** 1.0  
**Last Updated:** March 27, 2026  
**Author:** Project Development Team
