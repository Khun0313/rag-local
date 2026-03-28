# RAG Local 구현 계획서

## 1. 프로젝트 개요

로컬 문서 기반 RAG(Retrieval-Augmented Generation) Q&A 시스템

## 2. 기술 스택

| 구성 요소 | 기술 |
|---|---|
| LLM | GPT-5.3-Codex (ChatGPT 구독 OAuth 토큰) |
| RAG 프레임워크 | LlamaIndex |
| 벡터 DB | ChromaDB (로컬) |
| 임베딩 | `upskyy/bge-m3-korean` (한국어 특화, 1024차원) |
| BM25 토크나이저 | Kiwi (한국어 형태소 분석기) |
| 문서 파싱 | PDF, TXT, DOCX (Phase 1) / HWP, HWPX (Phase 2) |
| API | FastAPI |
| CLI | Typer |
| 언어 | Python 3.12+ |

## 3. 프로젝트 구조

```
rag-local/
├── pyproject.toml
├── .env.example
├── .gitignore
├── PLAN.md
├── src/
│   └── rag_local/
│       ├── __init__.py
│       ├── main.py                 # CLI 진입점 (Typer)
│       ├── config.py               # 설정 관리 (pydantic-settings)
│       ├── auth/
│       │   ├── __init__.py
│       │   └── token.py            # ~/.codex/auth.json 토큰 읽기
│       ├── llm/
│       │   ├── __init__.py
│       │   └── provider.py         # LlamaIndex LLM 팩토리
│       ├── ingest/
│       │   ├── __init__.py
│       │   ├── pipeline.py         # 문서 처리 오케스트레이션
│       │   ├── parsers.py          # PDF, TXT, DOCX 리더
│       │   └── chunker.py          # SentenceSplitter
│       ├── store/
│       │   ├── __init__.py
│       │   ├── chroma.py           # ChromaDB 관리
│       │   └── docstore.py         # BM25용 SimpleDocumentStore
│       ├── retrieval/
│       │   ├── __init__.py
│       │   └── hybrid.py           # 하이브리드 검색 (Vector + BM25)
│       ├── rag/
│       │   ├── __init__.py
│       │   └── engine.py           # 쿼리 엔진 (검색 → 생성)
│       └── api/
│           ├── __init__.py
│           └── app.py              # FastAPI REST API
├── data/
│   ├── documents/                  # 문서 저장소
│   └── chroma_db/                  # ChromaDB 영구 저장
└── tests/
    ├── conftest.py
    ├── test_auth.py
    ├── test_ingest.py
    ├── test_retrieval.py
    └── test_rag.py
```

## 4. 핵심 모듈 상세

### 4.1 인증 (`auth/token.py`)

`~/.codex/auth.json`에서 기존 발급 토큰을 읽어 사용한다.
토큰 갱신은 외부 프로그램이 자동 처리하므로, 앱에서는 읽기만 수행한다.

```python
def get_access_token() -> str:
    """~/.codex/auth.json에서 access_token 읽기"""
    auth_path = Path.home() / ".codex" / "auth.json"
    data = json.loads(auth_path.read_text())
    return data["tokens"]["access_token"]

def get_openai_client() -> OpenAI:
    """인증된 OpenAI 클라이언트 반환"""
    return OpenAI(
        api_key=get_access_token(),
        base_url="https://api.openai.com/v1",
    )
```

- OAuth 플로우 직접 수행하지 않음 (codex-auth 의존성 불필요)
- 토큰 만료 시: 401 에러 캐치 → "외부 프로그램으로 토큰 갱신해주세요" 안내
- 토큰 파일 없을 시: 명확한 에러 메시지 출력

**⚠️ 확인 필요**: ChatGPT 구독 OAuth 토큰의 API base_url이
`https://api.openai.com/v1`인지, 별도 엔드포인트인지 확인 필요.
일부 리서치에서 `https://chatgpt.com/backend-api/codex/responses`를 사용하는
사례가 있음. 실제 테스트로 확인 후 확정 예정.

### 4.2 LLM 프로바이더 (`llm/provider.py`)

- codex-auth 인증된 OpenAI 클라이언트를 LlamaIndex OpenAI LLM에 주입
- 모델: `gpt-5.3-codex`

### 4.3 임베딩 모델 — 한국어 특화

**`upskyy/bge-m3-korean`** 사용 (로컬 실행)

| 항목 | 값 |
|------|-----|
| 차원 | 1024 |
| 기반 | BAAI/bge-m3 한국어 파인튜닝 |
| 장점 | 한국어 의미 검색 최적화, 다국어 지원 |

기존 계획의 `all-MiniLM-L6-v2`(영어 중심, 384차원)에서 변경.
한국어 문서 검색 정확도가 크게 향상됨.

### 4.4 문서 처리 (`ingest/`)

#### 파서 (`parsers.py`)

| 포맷 | 파서 | 비고 |
|------|------|------|
| PDF | PyMuPDF + EasyOCR 폴백 | 텍스트 PDF는 PyMuPDF, 스캔 PDF는 OCR 자동 전환 |
| TXT | SimpleDirectoryReader | 기본 텍스트 |
| DOCX | DocxReader | python-docx 기반 |
| HWP/HWPX | (Phase 2) | pyhwp 라이브러리 예정 |

#### 청킹 (`chunker.py`)

- LlamaIndex `SentenceSplitter` 사용
- `chunk_size`: 512 토큰
- `chunk_overlap`: 64 토큰

#### 스캔 PDF OCR (`parsers.py` — EasyOCR 폴백)

스캔 PDF(이미지 기반)는 PyMuPDF 텍스트 추출 결과가 거의 없으므로,
자동으로 EasyOCR 기반 OCR로 전환하여 텍스트를 추출한다.

```
PDF 페이지 로드
    │
    ▼
PyMuPDF 텍스트 추출
    │
    ├── 텍스트 50자 이상 → 정상 텍스트 PDF (그대로 사용)
    │
    └── 텍스트 50자 미만 → 스캔 PDF로 판단
            │
            ▼
        페이지를 이미지(Pixmap)로 렌더링 (300 DPI)
            │
            ▼
        EasyOCR (한국어+영어+한자) → 텍스트 추출
```

- **OCR 엔진**: EasyOCR (딥러닝 기반, 한글+한자+영어 동시 인식)
- **판단 기준**: 페이지당 추출 텍스트 50자 미만이면 스캔 페이지로 간주
- **렌더링 해상도**: 300 DPI (OCR 정확도와 속도 균형)
- **지원 언어**: `['ko', 'en']` (한국어, 영어) — EasyOCR은 한자+한국어 동시 불가, 한국어 모델이 한자도 부분 인식
- **EasyOCR Reader**: 싱글톤 패턴으로 초기화 비용 최소화

#### 파이프라인 (`pipeline.py`)

```
문서 파일 (data/documents/)
    │
    ▼
파일 스캐너 (SHA-256 해시로 변경 감지)
    │
    ▼
파일 라우터 (.pdf → PyMuPDF, .txt → Text, .docx → Docx)
    │
    ▼
SentenceSplitter (512 토큰, 64 오버랩)
    │
    ▼
임베딩 (bge-m3-korean, 로컬)
    │
    ├──→ ChromaDB (벡터 저장)
    └──→ SimpleDocumentStore (BM25용 원문)
```

- 증분 처리: `data/.ingestion_state.json`에 SHA-256 해시 + 수정시간 추적
- 삭제 감지: 없어진 파일은 양쪽 스토어에서 제거

### 4.5 스토어 (`store/`)

#### ChromaDB (`chroma.py`)
- `chromadb.PersistentClient`로 `data/chroma_db/`에 영구 저장
- 컬렉션: `rag_local`

#### DocStore (`docstore.py`)
- `SimpleDocumentStore`로 BM25 검색용 원문 텍스트 보관
- ChromaDB와 함께 `data/` 디렉토리에 영구 저장

### 4.6 하이브리드 검색 (`retrieval/hybrid.py`)

```
사용자 질문
    │
    ├── Vector 검색 (ChromaDB) → top-5 코사인 유사도
    │
    └── BM25 검색 (Kiwi 형태소 분석) → top-5 키워드 매칭
    │
    ▼
Reciprocal Rank Fusion → 최종 top-5 결과
```

**BM25 한국어 형태소 분석**: Kiwi 형태소 분석기를 커스텀 토크나이저로 사용.
기본 영어 토크나이저로는 한국어 BM25 검색 품질이 매우 낮음.

```python
from kiwipiepy import Kiwi
kiwi = Kiwi()

def korean_tokenizer(text: str) -> list[str]:
    tokens = kiwi.tokenize(text)
    return [t.form for t in tokens if t.tag.startswith(("N", "V", "MA"))]

# BM25Retriever에 커스텀 토크나이저 주입
bm25_retriever = BM25Retriever.from_defaults(
    docstore=docstore,
    similarity_top_k=5,
    tokenizer=korean_tokenizer,
)
```

### 4.7 RAG 엔진 (`rag/engine.py`)

1. 사용자 질문 수신
2. `QueryFusionRetriever`로 하이브리드 검색
3. 검색된 문서 조각 + 질문으로 프롬프트 구성
4. GPT-5.3-Codex로 답변 생성
5. 출처 인용 포함하여 반환

시스템 프롬프트:
- 제공된 컨텍스트 기반으로만 답변
- 출처(파일명, 페이지) 인용
- 컨텍스트 부족 시 "정보가 부족합니다" 안내

### 4.8 API (`api/app.py`)

#### REST API (FastAPI)

| Method | Path | 설명 |
|--------|------|------|
| `POST` | `/ingest` | 문서 처리 시작 (파일 업로드 또는 디렉토리 스캔) |
| `POST` | `/query` | 질문 → 답변 + 출처 반환 (SSE 스트리밍 지원) |
| `GET` | `/documents` | 처리된 문서 목록 |
| `DELETE` | `/documents/{doc_id}` | 문서 제거 |
| `GET` | `/health` | 상태 확인 (인증, 인덱스 통계) |

#### CLI (Typer)

```bash
rag-local ingest [--dir PATH]
rag-local query "계약서의 해지 조항은?"
rag-local serve [--port 8000]
```

## 5. Phase 구분

### Phase 1: 핵심 기능

| 단계 | 작업 | 핵심 파일 |
|------|------|-----------|
| 1.1 | 프로젝트 스캐폴딩 (pyproject.toml, 디렉토리) | `pyproject.toml` |
| 1.2 | 설정 모듈 | `config.py` |
| 1.3 | 토큰 인증 (auth.json 읽기) | `auth/token.py` |
| 1.4 | LLM 프로바이더 | `llm/provider.py` |
| 1.5 | 문서 파서 (PDF, TXT, DOCX) | `ingest/parsers.py` |
| 1.6 | 청킹 | `ingest/chunker.py` |
| 1.7 | ChromaDB + DocStore 설정 | `store/` |
| 1.8 | 문서 처리 파이프라인 | `ingest/pipeline.py` |
| 1.9 | 하이브리드 검색 (Kiwi BM25 + Vector) | `retrieval/hybrid.py` |
| 1.10 | RAG 쿼리 엔진 | `rag/engine.py` |
| 1.11 | CLI 인터페이스 | `main.py` |
| 1.12 | REST API | `api/app.py` |
| 1.13 | 테스트 | `tests/` |

### Phase 1.5: 스캔 PDF OCR 지원

| 단계 | 작업 | 핵심 파일 |
|------|------|-----------|
| 1.5.1 | EasyOCR 의존성 추가 | `pyproject.toml` |
| 1.5.2 | OCR 폴백 로직 구현 (스캔 PDF 자동 감지) | `ingest/parsers.py` |
| 1.5.3 | EasyOCR Reader 싱글톤 | `ingest/parsers.py` |
| 1.5.4 | 테스트 (텍스트 PDF / 스캔 PDF 혼합) | 수동 확인 |

### Phase 2: HWP/HWPX 지원 (추후)

| 단계 | 작업 |
|------|------|
| 2.1 | pyhwp 라이브러리 조사 |
| 2.2 | HWPReader, HWPXReader 구현 |
| 2.3 | 파일 라우터에 확장자 등록 |
| 2.4 | 한국어 인코딩/테이블 추출 처리 |
| 2.5 | 테스트 |

## 6. 의존성

```toml
[project]
name = "rag-local"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    # RAG 프레임워크
    "llama-index-core>=0.12",
    "llama-index-llms-openai>=0.4",
    "llama-index-embeddings-huggingface>=0.5",
    "llama-index-readers-file>=0.4",
    "llama-index-vector-stores-chroma>=0.4",
    "llama-index-retrievers-bm25>=0.5",
    # 벡터 DB
    "chromadb>=1.0",
    # OpenAI SDK
    "openai>=1.60",
    # 임베딩 (한국어 로컬)
    "sentence-transformers>=3.0",
    # 한국어 형태소 분석 (BM25)
    "kiwipiepy>=0.18",
    # 문서 파싱
    "pymupdf>=1.25",
    "python-docx>=1.1",
    # API & CLI
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "typer>=0.15",
    # 설정
    "pydantic-settings>=2.7",
    # 유틸리티
    "httpx>=0.28",
    "rich>=13.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "ruff>=0.9",
]
phase2 = [
    "pyhwp>=0.2",
]

[project.scripts]
rag-local = "rag_local.main:app"
```

## 7. 주요 설계 결정

| 결정 | 이유 |
|------|------|
| 기존 토큰 재사용 (OAuth 플로우 X) | 외부 프로그램이 토큰 갱신 담당, 중복 구현 불필요 |
| `bge-m3-korean` 임베딩 | 한국어 문서 대상이므로 한국어 특화 모델 필수 |
| Kiwi 형태소 분석기 | BM25 한국어 토큰화 필수, Kiwi가 속도/정확도 균형 우수 |
| 하이브리드 검색 (BM25 + Vector) | 벡터만으로는 키워드 매칭 누락, 결합 시 검색 정확도 극대화 |
| PyMuPDF + EasyOCR | 텍스트 PDF는 PyMuPDF, 스캔 PDF는 EasyOCR 자동 폴백 |
| 증분 처리 (SHA-256) | 대량 문서 효율적 재처리 |
| `codex-auth` 의존성 제거 | 토큰을 직접 읽으므로 불필요 |

## 8. 확인 완료 사항

1. **API 엔드포인트**: `https://chatgpt.com/backend-api/codex/responses` (Responses API 형식)
   - 표준 `api.openai.com/v1`은 구독 OAuth 토큰으로 사용 불가 (401)
   - SSE 스트리밍 필수 (`stream: true`)
   - `max_output_tokens`, `temperature` 파라미터 미지원
2. **사용 가능 모델**: `gpt-5.3-codex`, `gpt-5-codex`, `gpt-5`, `gpt-5-codex-mini` 확인됨
3. **인증 헤더**: `Authorization: Bearer`, `OpenAI-Beta: responses=experimental`, `originator: codex_cli_rs`, `chatgpt-account-id`

## 9. 잠재적 위험 및 대응

| 위험 | 대응 |
|------|------|
| 임베딩 모델 변경 시 기존 인덱스 무효화 | 컬렉션 메타데이터에 모델명 기록, 불일치 시 경고 |
| 대용량 문서(10K+) 시 BM25 메모리 부족 | Phase 1에서는 인메모리, 추후 Elasticsearch 전환 고려 |
| ChromaDB 단일 프로세스 제한 | FastAPI에서 싱글톤 인스턴스 공유 |
| 토큰 파일 미존재 | 명확한 에러 메시지 + 설정 가이드 출력 |
