# Context-Aware Multimodal Knowledge Retrieval System

> **A production-grade, fully multimodal Retrieval-Augmented Generation (RAG) pipeline for PDF documents.**
> Combines three parallel retrieval pipelines — LLM-summarized embeddings, raw-atomic content, and CLIP visual similarity — to answer questions about any PDF using text passages, tables, figures, formulas, and form fields simultaneously.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Repository Structure](#3-repository-structure)
4. [Technology Stack](#4-technology-stack)
5. [Installation & Setup](#5-installation--setup)
6. [API Keys](#6-api-keys)
7. [Pipeline Deep Dive](#7-pipeline-deep-dive)
   - [Step 1 — PDF Ingestion with Docling](#step-1--pdf-ingestion-with-docling)
   - [Step 2 — Persistent Disk Cache](#step-2--persistent-disk-cache)
   - [Step 3 — Text & Table Summarization (Groq LLaMA)](#step-3--text--table-summarization-groq-llama)
   - [Step 4 — Image Summarization (Gemini Vision)](#step-4--image-summarization-gemini-vision)
   - [Step 5 — CLIP Visual Embeddings (Pipeline C)](#step-5--clip-visual-embeddings-pipeline-c)
   - [Step 6 — Embedding Model & ChromaDB Setup](#step-6--embedding-model--chromadb-setup)
   - [Step 7 — Multi-Vector Retriever Architecture](#step-7--multi-vector-retriever-architecture)
   - [Step 8 — Raw Atomic Index (Pipeline B)](#step-8--raw-atomic-index-pipeline-b)
   - [Step 9 — pdfplumber Supplemental Tables](#step-9--pdfplumber-supplemental-tables)
   - [Step 10 — Query Pipeline](#step-10--query-pipeline)
8. [Advanced Retrieval Features](#8-advanced-retrieval-features)
   - [HyDE — Hypothetical Document Expansion](#hyde--hypothetical-document-expansion)
   - [Cross-Encoder Reranking](#cross-encoder-reranking)
   - [Multi-Query Retrieval](#multi-query-retrieval)
   - [Cosine Similarity Scores](#cosine-similarity-scores)
9. [Multi-Turn Conversation](#9-multi-turn-conversation)
10. [Inline Citations](#10-inline-citations)
11. [Streamlit Application](#11-streamlit-application)
    - [Features](#features)
    - [Running the App](#running-the-app)
    - [Chat Interface](#chat-interface)
    - [Document Explorer](#document-explorer)
    - [Conversation Export](#conversation-export)
12. [Jupyter Notebook Reference](#12-jupyter-notebook-reference)
13. [Caching Strategy](#13-caching-strategy)
14. [Model Configurations & Fallbacks](#14-model-configurations--fallbacks)
15. [Data Flow: End-to-End](#15-data-flow-end-to-end)
16. [Configuration Reference](#16-configuration-reference)
17. [Known Limitations & Design Decisions](#17-known-limitations--design-decisions)
18. [Component References](#18-component-references)

---

## 1. System Overview

This system turns any PDF document into a fully queryable multimodal knowledge base. Unlike traditional RAG systems that only handle text, this pipeline simultaneously processes and retrieves from:

| Modality | Source | Indexed As |
|---|---|---|
| Text passages | Docling paragraph/section extraction | LLM summary → BGE embedding |
| Section headers & titles | Docling structural labels | LLM summary → BGE embedding |
| Captions, footnotes, references | Docling structural labels | LLM summary → BGE embedding |
| Tables | Docling + pdfplumber fallback | LLM analytical summary + raw markdown |
| Figures, charts, diagrams | Docling image extraction | Gemini Vision description + CLIP image embedding |
| Mathematical formulas | Docling formula detection | Raw LaTeX/text in BGE raw-atomic index |
| Form fields / key-value pairs | Docling form detection | Raw key:value text in BGE raw-atomic index |

Every single element from the document is preserved in its **original form** in a docstore. The embeddings are only used for retrieval lookup — the LLM receives the raw original content (actual table HTML, full text, real base64 image) for answer generation.

### The Three Retrieval Pipelines

```
Query ──► Pipeline A ──► BGE embedding of LLM summaries     ──► original elements
       ──► Pipeline B ──► BGE embedding of raw content       ──► original elements
       ──► Pipeline C ──► CLIP text-to-image cosine search   ──► image elements
                    └──► Merge + Deduplicate ──► Gemini answer generation
```

**Pipeline A** (summary-based) catches semantic queries: "what is the main contribution?" matches the LLM-written summary of relevant sections.

**Pipeline B** (raw-atomic) catches exact-value queries: "what is the BLEU score for base model?" matches the raw markdown table containing "41.0" directly.

**Pipeline C** (CLIP visual) catches visual queries: "transformer architecture diagram" retrieves the most visually similar figure to the encoded text query.

---

## 2. Architecture Diagram

```
                         ┌─────────────────────────────────────────┐
                         │           PDF Document                  │
                         └──────────────────┬──────────────────────┘
                                            │
                                     Docling v2.x
                                    (IBM PDF parser)
                                            │
              ┌─────────────────────────────┼──────────────────────────────┐
              │                             │                              │
        Texts (137)                   Tables (4+8)                  Images (6)
        Formulas                      HTML + Markdown               Base64 JPEG
        Form Fields                   Captions                      Page + Caption
              │                             │                              │
    ┌─────────┴──────┐          ┌──────────┴──────┐            ┌──────────┴──────┐
    │  Groq LLaMA    │          │  Groq LLaMA     │            │  Gemini 2.5     │
    │  llama-3.3-70b │          │  llama-3.3-70b  │            │  Flash Vision   │
    │  (text summ)   │          │  (table summ)   │            │  (img describe) │
    └────────┬───────┘          └─────────┬───────┘            └────────┬────────┘
             │ summaries                  │ summaries                   │ descriptions
             └──────────────┬─────────────┘                             │
                            │                                           │
                     BGE-base-en-v1.5                            ┌──────┴──────┐
                     768-dim embeddings                          │  CLIP       │
                     normalize=True                              │  ViT-B-32   │
                            │                                   │  (openai)   │
              ┌─────────────┼──────────────┐                    └──────┬──────┘
              │             │              │                           │ 512-dim
           ChromaDB       ChromaDB    InMemoryStore                   │ image embs
         rag_bge         rag_raw_bge   (docstore)           ChromaDB clip_index
        (summary vecs)  (raw content   uuid → element       clip_ViT_B_32_<hash>
         147 vectors)    4 vectors)
              │             │              │                           │
              └─────────────┼──────────────┘                           │
                            │                                           │
                     ┌──────┴──────────────────────────────────────────┘
                     │               QUERY TIME
                     ▼
            ┌──────────────────────────────────────────────────────────┐
            │   Optional pre-processing                                │
            │   ├─ HyDE: generate hypothetical answer → use as query   │
            │   ├─ Multi-query: Groq generates N query variants         │
            │   └─ Both can combine                                    │
            └──────────┬──────────────────────────────────────────────┘
                       │ query_text (original or HyDE-expanded)
         ┌─────────────┼─────────────────────────────────┐
         │             │                                 │
    Pipeline A    Pipeline B                       Pipeline C
    similarity_  similarity_                      CLIP text→
    search_with_ search_with_                     image cosine
    relevance_   relevance_                       query
    scores(k=6)  scores(k=3)
    + docstore   + docstore
    lookup       lookup
         │             │                                 │
         └─────────────┼─────────────────────────────────┘
                       │ merge + deduplicate
                       │
            Optional post-processing:
            cross-encoder/ms-marco-MiniLM-L-6-v2 reranking
                       │
              context = {texts, tables, images}  (each item with _score)
                       │
            build_rag_prompt():
            ├─ [CONVERSATION HISTORY] (last 3 turns if multi-turn)
            ├─ [SOURCE 1 — TEXT page N]
            ├─ [SOURCE 2 — TABLE page N]
            ├─ ... numbered sources
            └─ images attached as base64 vision inputs
                       │
               Gemini (multimodal LLM)
               with inline citation instructions
                       │
              Answer with [N] / [N, p.X] citations
```

---

## 3. Repository Structure

```
Context-Aware-Multimodal-Knowledge-Retrieval-System/
│
├── multimodal_rag_complete.ipynb   # Full pipeline notebook (63 cells)
├── app.py                          # Streamlit web application (~1700 lines)
├── requirements.txt                # Loose minimum-version requirements
├── requirements-pinned.txt         # Exact pinned versions (Python 3.14.4, 2026-05-31)
├── .env                            # API keys (not committed — create manually)
│
├── content/                        # PDF storage + HTML exports
│   ├── images/                     # extracted image files (optional)
│   └── report.html                 # auto-generated extraction report (Step 18)
│
├── chroma_db/                      # Persistent ChromaDB (Pipelines A + B)
│   ├── chroma.sqlite3              # collection metadata + HNSW references
│   ├── <collection-uuid>/          # HNSW segment files for rag_bge
│   └── <collection-uuid>/          # HNSW segment files for rag_raw_bge
│
├── clip_index/                     # CLIP-dedicated ChromaDB (Pipeline C)
│   ├── chroma.sqlite3
│   └── <collection-uuid>/          # HNSW segment files for CLIP vectors
│
├── cache/                          # diskcache persistent cache (2 GB default)
│   └── <hex>/                      # content-addressed cache files
│       └── <hex>/
│           └── *.val               # serialized Python objects (pickle)
│
└── multimodal_env/                 # Python virtual environment
    └── ...
```

### Why `clip_index/` is separate from `chroma_db/`

ChromaDB's Rust-based HNSW backend serializes segment files per-client. When multiple `PersistentClient` instances share the same directory, only one collection's HNSW files are written correctly — others exist in SQLite metadata but have no matching segment files, causing a `"Nothing found on disk"` error on reload. The fix is a dedicated directory with an isolated client for CLIP, avoiding contention entirely.

---

## 4. Technology Stack

| Layer | Component | Version | Purpose |
|---|---|---|---|
| **PDF Parsing** | docling (IBM) | ≥2.0.0 | Structured extraction — text, tables, images, formulas, forms |
| **PDF Fallback** | pdfplumber | ≥0.11.0 | Supplemental table extraction with heuristic line detection |
| **PDF Metadata** | PyMuPDF (fitz) | ≥1.24.0 | Fast metadata, TOC, hyperlinks, annotations |
| **Image handling** | Pillow | ≥10.0.0 | JPEG conversion + base64 encoding of extracted figures |
| **Text/Table LLM** | Groq / llama-3.3-70b-versatile | — | Fast summarization — 280 tok/s, 131k context, free tier |
| **Image LLM** | Gemini 2.5 Flash | — | Multimodal vision descriptions for all figures |
| **Answer LLM** | Gemini (auto-probed) | — | Final answer generation with vision inputs |
| **Text Embeddings** | BAAI/bge-base-en-v1.5 | — | 768-dim, normalize=True, strong semantic similarity |
| **Alt Embeddings** | all-MiniLM-L6-v2 | — | 384-dim, ~2× faster, slightly lower recall |
| **Visual Embeddings** | CLIP ViT-B-32 (openai) | open-clip ≥2.24.0 | 512-dim cross-modal image↔text embeddings |
| **Vector Store** | ChromaDB | ≥0.5.0 | Persistent HNSW-based vector database |
| **Retriever** | MultiVectorRetriever | langchain-classic | Summary-in-vectorstore, original-in-docstore pattern |
| **Reranker** | cross-encoder/ms-marco-MiniLM-L-6-v2 | sentence-transformers | Cross-encoder reranking after vector retrieval |
| **Disk Cache** | diskcache | ≥5.6.0 | Content-addressed persistent cache for all expensive operations |
| **Orchestration** | LangChain 1.x + langchain-classic | ≥1.3.0 | Chains, retrievers, document objects |
| **UI** | Streamlit | ≥1.35.0 | Full web application with streaming answers |
| **Notebook UI** | ipywidgets | ≥8.0.0 | Interactive Q&A widget in Jupyter |

---

## 5. Installation & Setup

### Prerequisites

| Requirement | Minimum | Tested / Recommended |
|---|---|---|
| Python | 3.11 | **3.14.4** (exact version used in development) |
| RAM | 4 GB | 8 GB+ for large PDFs |
| Disk space | 5 GB | 10 GB+ (models + cache + venv) |
| OS | Any | Windows 11 / Ubuntu 22.04+ / macOS 13+ |
| GPU | Not required | CUDA 12.x optional (speeds up BGE + CLIP) |

> **Reproducibility note:** All exact package versions are pinned in the [Pinned Requirements](#pinned-requirements-exact-versions-for-long-term-reproducibility) section below. Use those versions to guarantee the environment works identically regardless of when you set it up.

---

### Clone & Create Virtual Environment

```powershell
# Windows PowerShell
git clone https://github.com/your-username/Context-Aware-Multimodal-Knowledge-Retrieval-System.git
cd Context-Aware-Multimodal-Knowledge-Retrieval-System

python -m venv multimodal_env
multimodal_env\Scripts\Activate.ps1
```

```bash
# macOS / Linux
git clone https://github.com/your-username/Context-Aware-Multimodal-Knowledge-Retrieval-System.git
cd Context-Aware-Multimodal-Knowledge-Retrieval-System

python3.14 -m venv multimodal_env
source multimodal_env/bin/activate
```

---

### Install Dependencies

**Option A — Loose requirements (lets pip resolve latest compatible versions):**

```powershell
pip install -r requirements.txt
```

**Option B — Exact pinned versions (guaranteed reproducibility):**

```powershell
pip install -r requirements-pinned.txt
```

See the [Pinned Requirements](#pinned-requirements-exact-versions-for-long-term-reproducibility) section to create `requirements-pinned.txt`.

---

### Pinned Requirements — Exact Versions for Long-Term Reproducibility

The following are the **exact versions verified to work together** as of the development environment (Python 3.14.4, Windows 11). Copy the block below into a file named `requirements-pinned.txt` and install with `pip install -r requirements-pinned.txt`.

```
# ============================================================
# Context-Aware Multimodal Knowledge Retrieval System
# Pinned Requirements — verified working environment
# Python: 3.14.4
# Generated: 2026-05-31
# ============================================================

# --- Web UI ---
streamlit==1.58.0

# --- PDF Parsing ---
docling==2.96.0
pdfplumber==0.11.9
PyMuPDF==1.27.2.3
Pillow==12.2.0
lxml==6.1.1
opencv-python-headless==4.13.0.92

# --- LangChain Core ---
langchain==1.3.2
langchain-classic==1.0.7
langchain-core==1.4.0
langchain-community==0.4.2
langchain-chroma==1.1.0

# --- LLM Integrations ---
langchain-google-genai==4.2.4
langchain-groq==1.1.2
langchain-huggingface==1.2.2
google-generativeai==0.8.6

# --- Embeddings & Transformers ---
sentence-transformers==5.5.1
transformers==5.9.0
huggingface_hub==1.17.0
tokenizers==0.22.2
safetensors==0.7.0
accelerate==1.13.0

# --- PyTorch (CPU build — works on all platforms) ---
# For GPU: replace with the appropriate CUDA wheel from https://pytorch.org/get-started/locally/
torch==2.12.0
torchvision==0.27.0

# --- CLIP Visual Embeddings ---
open_clip_torch==3.3.0

# --- Vector Store ---
chromadb==1.5.9

# --- Persistent Disk Cache ---
diskcache==5.6.3

# --- Data & Numerics ---
pandas==3.0.3
numpy==2.4.6
scipy==1.17.1
pyarrow==24.0.0

# --- Tokenization ---
tiktoken==0.13.0

# --- Utilities ---
python-dotenv==1.2.2
tqdm==4.67.3
tabulate==0.10.0
Markdown==3.10.2
matplotlib==3.10.9
PyYAML==6.0.3
packaging==26.2
regex==2026.5.9
rich==15.0.0
click==8.4.1
requests==2.34.2
urllib3==2.7.0
certifi==2026.5.20
charset-normalizer==3.4.7
idna==3.17
filelock==3.29.0
fsspec==2026.4.0
typing_extensions==4.15.0

# --- Async / HTTP ---
aiohttp==3.13.5
anyio==4.13.0
httpx==0.28.1

# --- Pydantic ---
pydantic==2.13.4

# --- gRPC / Protobuf (ChromaDB dependency) ---
grpcio==1.80.0
protobuf==5.29.6

# --- Altair (Streamlit charts) ---
altair==6.1.0
watchdog==6.0.0

# --- Jupyter / Notebook ---
ipykernel==7.2.0
ipython==9.14.0
ipywidgets==8.1.8
jupyterlab_widgets==3.0.16
jupyter_client==8.8.0
jupyter_core==5.9.1

# --- Windows only ---
# pywin32==311   # uncomment on Windows if not auto-installed
```

> **GPU users:** Replace the `torch==2.12.0` and `torchvision==0.27.0` lines with the CUDA-specific wheels from [pytorch.org](https://pytorch.org/get-started/locally/). Everything else stays the same.

> **macOS (Apple Silicon):** `torch==2.12.0` supports MPS (Metal Performance Shaders). No changes needed for CPU; for MPS acceleration, set `device="mps"` in the embedding and CLIP config.

---

### `requirements.txt` (Loose / Minimum Versions)

The project ships with a loose `requirements.txt` for forward compatibility:

```
# --- Streamlit app ---
streamlit>=1.35.0

# --- PDF Parsing & Element Extraction ---
docling>=2.0.0
pdfplumber>=0.11.0
pymupdf>=1.24.0
pillow>=10.0.0
lxml>=5.0.0

# --- LangChain Core ---
langchain>=1.3.0
langchain-classic>=1.0.0
langchain-core>=1.0.0
langchain-community>=0.4.0
langchain-chroma>=0.1.0

# --- LLM Integrations ---
langchain-google-genai>=2.0.0
langchain-groq>=0.2.0
langchain-huggingface>=0.1.0
google-generativeai>=0.8.0

# --- Embeddings ---
sentence-transformers>=3.0.0

# --- Vector Store ---
chromadb>=0.5.0

# --- Tokenization ---
tiktoken>=0.7.0

# --- Data & Utilities ---
pandas>=2.0.0
numpy>=1.26.0
python-dotenv>=1.0.0
opencv-python-headless>=4.9.0
matplotlib>=3.8.0
ipywidgets>=8.0.0
tqdm>=4.66.0

# --- CLIP visual embeddings ---
open-clip-torch>=2.24.0

# --- Persistent disk cache ---
diskcache>=5.6.0

# --- Markdown rendering ---
markdown>=3.5.0
tabulate>=0.9.0

# --- Jupyter (notebook work) ---
ipykernel>=6.29.0
ipython>=8.0.0
jupyter>=1.0.0
notebook>=7.0.0
```

---

### Verify Installation (Notebook Cell 1)

Run the first code cell in the notebook — it checks every required import and reports any missing packages.

---

### Complete Environment Reference

| Package | Pinned Version | Role |
|---|---|---|
| **Python** | **3.14.4** | Runtime |
| streamlit | 1.58.0 | Web UI |
| docling | 2.96.0 | PDF parser (IBM) |
| pdfplumber | 0.11.9 | Supplemental table extractor |
| PyMuPDF | 1.27.2.3 | PDF metadata / fitz |
| Pillow | 12.2.0 | Image processing |
| lxml | 6.1.1 | XML/HTML parsing |
| opencv-python-headless | 4.13.0.92 | Image ops (headless) |
| langchain | 1.3.2 | Orchestration |
| langchain-classic | 1.0.7 | MultiVectorRetriever / InMemoryStore |
| langchain-core | 1.4.0 | Base abstractions |
| langchain-community | 0.4.2 | Community integrations |
| langchain-chroma | 1.1.0 | ChromaDB integration |
| langchain-google-genai | 4.2.4 | Gemini LLM wrapper |
| langchain-groq | 1.1.2 | Groq LLM wrapper |
| langchain-huggingface | 1.2.2 | HuggingFace embeddings wrapper |
| google-generativeai | 0.8.6 | Google AI SDK |
| sentence-transformers | 5.5.1 | BGE embeddings + cross-encoder |
| transformers | 5.9.0 | HuggingFace model hub |
| huggingface_hub | 1.17.0 | Model download / caching |
| tokenizers | 0.22.2 | Fast BPE tokenization |
| safetensors | 0.7.0 | Safe model weight loading |
| accelerate | 1.13.0 | Multi-device training/inference |
| torch | 2.12.0 | PyTorch (CPU/CUDA/MPS) |
| torchvision | 0.27.0 | Torch vision transforms (CLIP) |
| open_clip_torch | 3.3.0 | CLIP ViT-B-32 visual embeddings |
| chromadb | 1.5.9 | Vector database |
| diskcache | 5.6.3 | Persistent disk cache |
| pandas | 3.0.3 | DataFrames |
| numpy | 2.4.6 | Numerics |
| scipy | 1.17.1 | Scientific computing |
| pyarrow | 24.0.0 | Arrow/Parquet (ChromaDB) |
| tiktoken | 0.13.0 | Token counting |
| python-dotenv | 1.2.2 | `.env` loading |
| tqdm | 4.67.3 | Progress bars |
| tabulate | 0.10.0 | DataFrame → Markdown tables |
| Markdown | 3.10.2 | Markdown → HTML rendering |
| matplotlib | 3.10.9 | Plotting (stats dashboard) |
| PyYAML | 6.0.3 | YAML parsing |
| packaging | 26.2 | Version parsing |
| regex | 2026.5.9 | Extended regex |
| rich | 15.0.0 | Rich terminal output |
| click | 8.4.1 | CLI framework (Streamlit dep) |
| requests | 2.34.2 | HTTP |
| urllib3 | 2.7.0 | HTTP transport |
| certifi | 2026.5.20 | TLS certificates |
| charset-normalizer | 3.4.7 | Charset detection |
| idna | 3.17 | Internationalized domain names |
| filelock | 3.29.0 | File locking |
| fsspec | 2026.4.0 | Filesystem abstraction |
| typing_extensions | 4.15.0 | Backported type hints |
| aiohttp | 3.13.5 | Async HTTP |
| anyio | 4.13.0 | Async I/O abstraction |
| httpx | 0.28.1 | Async HTTP client |
| pydantic | 2.13.4 | Data validation |
| grpcio | 1.80.0 | gRPC transport (ChromaDB) |
| protobuf | 5.29.6 | Protobuf serialization (ChromaDB) |
| altair | 6.1.0 | Declarative charts (Streamlit) |
| watchdog | 6.0.0 | File system events (Streamlit) |
| ipykernel | 7.2.0 | Jupyter kernel |
| ipython | 9.14.0 | Interactive Python |
| ipywidgets | 8.1.8 | Notebook widgets |
| jupyterlab_widgets | 3.0.16 | JupyterLab widget extension |
| jupyter_client | 8.8.0 | Jupyter client protocol |
| jupyter_core | 5.9.1 | Jupyter core utilities |

---

## 6. API Keys

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_ai_studio_key
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
```

| Key | Where to get | Used for |
|---|---|---|
| `GOOGLE_API_KEY` | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | Gemini Vision (image descriptions) + Gemini answer generation + HyDE |
| `GROQ_API_KEY` | [console.groq.com/keys](https://console.groq.com/keys) | LLaMA-3.3-70b for text/table summarization + multi-query variant generation |
| `HF_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | Downloading BGE / MiniLM embedding models from HuggingFace Hub |

### Quota Notes

| Model | Free Tier |
|---|---|
| `gemini-2.5-flash` | 20 RPD (used for image summarization only — strict quota) |
| `gemini-2.0-flash` | 1,500 RPD, 15 RPM |
| `gemini-2.5-flash-lite` | 1,500 RPD, 15 RPM |
| `llama-3.3-70b-versatile` (Groq) | ~14,400 RPD, 30 RPM |

The system auto-probes Gemini models in order (`2.5-flash-lite` → `2.0-flash-lite`) and switches to the first one with remaining quota. Text/table summarization always uses Groq to avoid consuming Gemini quota.

---

## 7. Pipeline Deep Dive

### Step 1 — PDF Ingestion with Docling

**Docling** (IBM Research, 2024) is an enterprise-grade document parser that understands document layout without requiring Tesseract or external OCR. It processes PDFs at the layout level — understanding columns, tables, figures, and structural hierarchy.

**Configuration used:**

```python
pipeline_opts = PdfPipelineOptions()
pipeline_opts.do_table_structure      = True   # reconstruct table cells from PDF primitives
pipeline_opts.generate_picture_images = True   # extract embedded images as PIL objects
pipeline_opts.generate_table_images   = True   # also capture table raster if needed
pipeline_opts.images_scale            = 2.0    # 2× resolution for figures
pipeline_opts.do_ocr                  = False  # not needed for machine-generated PDFs
pipeline_opts.do_formula_enrichment   = False  # LaTeX enrichment disabled (uses raw text)
```

**What Docling extracts:**

| DocItemLabel | Stored as | Description |
|---|---|---|
| TEXT, PARAGRAPH | `texts[]` | Body text, paragraphs |
| TITLE, SECTION_HEADER | `texts[]` | Headings (with `heading=True`) |
| LIST_ITEM | `texts[]` | Bullet/numbered list items |
| CAPTION | `texts[]` | Figure/table captions |
| FOOTNOTE | `texts[]` | Footnotes |
| REFERENCE | `texts[]` | Bibliography entries |
| HANDWRITTEN_TEXT | `texts[]` | Any handwritten regions |
| TableItem | `tables[]` | HTML + DataFrame + Markdown + caption |
| PictureItem | `images[]` | JPEG base64 + page + label + caption |
| FORMULA | `formulas[]` | Raw LaTeX / equation text |
| FORM, KEY_VALUE_REGION | `forms[]` | Form fields and key-value pairs |

**Text filtering:** Only elements with > 20 characters after stripping are kept to avoid noise from single-character artifacts.

**Image extraction:** Each `PictureItem` is rendered as a PIL Image at 2× scale and JPEG-compressed at quality=90 before base64-encoding. Width and height are stored for downstream filtering.

**Table extraction:** Tables get three representations:
- `html`: HTML `<table>` with `<thead>` and `<tbody>` — used for visual display
- `df`: pandas DataFrame — for programmatic access
- `markdown`: pipe-table format — used for LLM summarization and raw-atomic indexing

**PDF metadata** (via PyMuPDF/fitz): title, author, page count, table of contents, external hyperlinks, annotations — stored in `pdf_meta` dict.

**Formula/Form merging:** After extraction, formulas and forms are appended to the `texts` list with appropriate labels so they flow through the same summarization and indexing pipeline as regular text.

**Result for "Attention Is All You Need":**
- 137 text chunks (texts + merged formulas/forms)
- 4 Docling tables + 8 pdfplumber supplemental = 12 total
- 6 images

---

### Step 2 — Persistent Disk Cache

All expensive operations are wrapped in a **diskcache.Cache** stored at `./cache` (2 GB default limit). This means:

- The entire pipeline from PDF → indexed vectors runs **once**, taking 2–5 minutes
- Every subsequent run (including notebook restarts) loads results instantly from disk
- Cache keys are always content-addressed using the PDF's SHA-256 hash

**Cache key builder:**

```python
def _ck(*parts) -> str:
    return ":".join(str(p) for p in parts)

PDF_HASH = hashlib.sha256(Path(PDF_PATH).read_bytes()).hexdigest()[:16]
```

**What is cached:**

| Cache Key | Contents | Size (typical) |
|---|---|---|
| `docling_v1:{pdf_hash}` | All extracted elements (texts, tables, images, formulas, forms, metadata) | 5–50 MB |
| `groq_summ_v2:{pdf_hash}:llama-3.3-70b-versatile:text:{i}` | Text summary for chunk i | ~500 bytes each |
| `groq_summ_v2:{pdf_hash}:llama-3.3-70b-versatile:table:{i}` | Table summary for table i | ~800 bytes each |
| `gemini_img_v1:{pdf_hash}:gemini-2.5-flash:{i}` | Image description for figure i | ~1–3 KB each |
| `clip_emb_v1:ViT-B-32:openai:{pdf_hash}` | All CLIP image embeddings as float list | ~12 KB per 6 images |
| `docstore_v1:{pdf_hash}:{embedding_model}` | Full docstore backing dict (uuid → element) | 5–50 MB |

**Cache miss behavior:** Only successful API responses are cached. If Groq/Gemini returns an error or rate-limit fallback, the result is NOT cached — forcing a retry on the next run.

**v1 → v2 migration:** When the table summarization prompt was improved (to use markdown content instead of empty HTML), a migration step automatically copies valid v1 text summaries to v2 keys, avoiding redundant API calls for already-correct summaries.

---

### Step 3 — Text & Table Summarization (Groq LLaMA)

Every text chunk and table is summarized by **Groq llama-3.3-70b-versatile** (280 tok/s, 131k context window). These summaries are what get embedded — the summaries are semantically richer and more search-friendly than the raw content.

**Why summarize instead of embedding raw text?**

Raw text chunks are often too dense, too specific, or contain noise (author names, page numbers, citations). A good summary captures the *semantic intent* of the chunk in a way that matches query language better.

**Text summarization prompt** (simplified):
> You are an expert academic research assistant. Write a concise, information-dense summary (3-6 sentences) that captures: the main topic/argument/finding, key facts/methods/metrics, named entities (authors, datasets, architectures), and context within the paper. Do NOT start with "Here is a summary".

**Table summarization prompt** (simplified):
> You are an expert data analyst. Describe: what the table reports, column/row semantics, key numerical values with exact numbers, trends/patterns, and what conclusion the table supports.

**Batching and rate limiting:** Each chunk is summarized individually with a 0.15 s sleep between calls. On rate limit (HTTP 429), the `_cached_invoke()` function retries up to 3 times with 5 s / 10 s / 15 s backoff. After 3 failures, the raw text is returned as a fallback (not cached — will be retried on next run).

**LangChain chains used:**

```python
text_chain  = {element, label, page} | TEXT_PROMPT  | groq_llm | StrOutputParser()
table_chain = {element, caption, page} | TABLE_PROMPT | groq_llm | StrOutputParser()
```

---

### Step 4 — Image Summarization (Gemini Vision)

Each extracted figure is described by **Gemini 2.5 Flash Vision** with a comprehensive structured prompt. This converts visual information into searchable text.

**The image description prompt covers:**
1. Element type (bar chart, line graph, architecture diagram, flowchart, equation, photograph, etc.)
2. All visible text (axis labels, titles, legends, annotations, callouts)
3. Key numerical values, percentages, comparisons, ranges
4. Trends, patterns, directional observations
5. Main insight — what conclusion or argument this figure supports
6. Layout/structure — multi-panel, color coding, arrows, etc.

The description is stored in `image_summaries[i]` and indexed in ChromaDB alongside text and table summaries. When this image is retrieved at query time, both the text description AND the actual base64 image are sent to Gemini for the final answer — enabling true visual question answering.

**Quota management:** Gemini 2.5 Flash has a strict 20 RPD free-tier limit. Images are summarized once and cached permanently — never re-processed for the same PDF.

---

### Step 5 — CLIP Visual Embeddings (Pipeline C)

**CLIP** (Contrastive Language-Image Pre-training, OpenAI) learns a shared embedding space for images and text, enabling direct cross-modal similarity search. A text query like "transformer encoder-decoder architecture" can directly retrieve the most visually similar figure without going through a text description.

**Model:** `ViT-B-32` with `openai` pretrained weights — 151 MB, runs on CPU, produces 512-dimensional L2-normalized embeddings.

**Image embedding:**
```python
# Each image → PIL → CLIP preprocessor → ViT-B-32 image encoder → 512-dim normalized vector
tensor = _clip_preprocess(pil).unsqueeze(0)
emb = _clip_model.encode_image(tensor)
emb = emb / emb.norm(dim=-1, keepdim=True)   # L2 normalize
```

**Text-to-image retrieval:**
```python
# Query text → CLIP text encoder → 512-dim vector → cosine search in ChromaDB
tokens = _clip_tokenizer([query_text])
emb = _clip_model.encode_text(tokens)
emb = emb / emb.norm(dim=-1, keepdim=True)
results = clip_collection.query(query_embeddings=emb, n_results=k)
```

**Storage:** CLIP vectors are stored in `./clip_index/` (separate ChromaDB client from `./chroma_db/`). The collection is named `clip_ViT_B_32_{pdf_hash[:8]}`.

**ChromaDB space:** `{"hnsw:space": "cosine"}` — cosine similarity (matching CLIP's L2-normalized vectors).

---

### Step 6 — Embedding Model & ChromaDB Setup

**Supported embedding models:**

| Key | Model | Dimensions | Description |
|---|---|---|---|
| `bge` | `BAAI/bge-base-en-v1.5` | 768 | Best semantic quality — recommended |
| `minilm` | `sentence-transformers/all-MiniLM-L6-v2` | 384 | ~2× faster, slightly lower recall |

**Configuration:**
```python
EMBEDDING_MODEL = "bge"    # or "minilm"
EMBEDDING_MODE  = "local"  # or "api" (uses HuggingFace Inference API)
```

**Collection naming:** ChromaDB collection names include the embedding model key: `rag_bge` and `rag_raw_bge`. Switching to MiniLM creates `rag_minilm` and `rag_raw_minilm`. This prevents cross-contamination of vectors with different dimensions, and both can coexist.

**Two ChromaDB collections per embedding model:**
- `rag_{model}` — stores **LLM summary embeddings** (Pipeline A)
- `rag_raw_{model}` — stores **raw content embeddings** (Pipeline B)

Both collections are backed by a single **InMemoryStore docstore** that maps UUIDs to original element dicts. This means that regardless of which collection found the match, the same original element (full text, HTML table, base64 image) is returned.

---

### Step 7 — Multi-Vector Retriever Architecture

The **MultiVectorRetriever** (from `langchain-classic`) implements a two-tier lookup:

```
Query → embed(query) → vectorstore.similarity_search() → doc_ids
                                                             │
                                                    docstore.mget(doc_ids)
                                                             │
                                                    original elements ←─ returned to LLM
```

**What's in ChromaDB (the search layer):**
- `Document.page_content` = LLM-generated summary
- `Document.metadata` = `{doc_id: <uuid>, modality: "text"|"table"|"image", page: N, ...}`

**What's in the docstore (the content layer):**
- Key: UUID string
- Value: original element dict (full text, full HTML, base64 image, etc.)

**The critical insight:** The LLM never sees the summaries. It only sees the original content. Summaries exist purely to make retrieval semantically accurate.

**Docstore persistence:** The docstore is an `InMemoryStore` — it lives only in RAM during a session. To make it persistent, the entire docstore backing dict (`{uuid: element}`) is serialized to diskcache as `docstore_v1:{pdf_hash}:{embedding_model}` and restored on the next run.

**`_add_to_retriever()` function:**
```python
def _add_to_retriever(originals, summaries, retriever, modality, extra_meta_fn=None):
    ids = [str(uuid.uuid4()) for _ in originals]
    summary_docs = [Document(page_content=summary, metadata={DOC_ID_KEY: id, ...})
                    for id, summary in zip(ids, summaries)]
    retriever.vectorstore.add_documents(summary_docs)   # ChromaDB
    retriever.docstore.mset(list(zip(ids, originals)))  # InMemoryStore
    for id, elem in zip(ids, originals):
        DOCSTORE_BACKING[id] = elem                     # for diskcache persistence
    return ids
```

---

### Step 8 — Raw Atomic Index (Pipeline B)

The summary-based index (Pipeline A) is excellent for semantic questions but poor at exact-value lookup. A question like "what is the BLEU score for the base model?" needs to match a table cell containing "41.0" — not its summary saying "BLEU scores for various model sizes".

**Raw Atomic Index** embeds the literal content:

| Content type | What gets embedded |
|---|---|
| Tables | Raw markdown pipe-table format |
| Formulas | Raw LaTeX/text as extracted by Docling |
| Form fields | Raw `key: value` pairs |

**Critical implementation detail:** Raw table documents use the **same UUID** as their corresponding summary document. This means both pipelines ultimately retrieve the same original element from the docstore. The only difference is which query embedding finds it.

```python
# For tables: reuse existing tbl_id (same UUID as Pipeline A)
doc = Document(
    page_content=f"[TABLE markdown — page {page}]\n{markdown_content}",
    metadata={DOC_ID_KEY: tbl_id, "modality": "table", ...}
)
vectorstore_raw.add_documents([doc])
# docstore already has this UUID → returns full original table dict
```

Formulas and forms get **new** UUIDs since they aren't in the summary collection (they're merged into the text pool instead of being individually indexed in Pipeline A).

---

### Step 9 — pdfplumber Supplemental Tables

Even with Docling's excellent table detection, some PDFs contain borderless or pseudo-tables that aren't detected. **pdfplumber** uses heuristic line detection as a supplemental pass.

**How it works:**
1. `pdfplumber.open(pdf_path)` iterates all pages
2. `page.extract_tables()` returns raw cell arrays using coordinate analysis
3. Valid tables (≥2 rows) are cleaned (None → ""), converted to DataFrames, and serialized to HTML + markdown
4. These extra tables are summarized with Groq and indexed alongside docling tables

**For "Attention Is All You Need":** pdfplumber found 8 additional tables not captured by Docling, bringing the total from 4 to 12 indexed tables.

---

### Step 10 — Query Pipeline

The full `query_paper()` function orchestrates all three pipelines:

```python
def query_paper(
    question: str,
    k: int = 6,           # Pipeline A: top-k summary matches
    k_raw: int = 3,       # Pipeline B: top-k raw content matches
    k_clip: int = 2,      # Pipeline C: top-k CLIP image matches
    show_sources: bool = True,
    use_clip: bool = True,
    use_raw: bool = True,
    use_hyde: bool = True,        # HyDE query expansion
    use_rerank: bool = True,      # cross-encoder reranking
    rerank_top_k: int = 6,        # chunks to keep after reranking
    history: list | None = None,  # multi-turn context
    use_multiquery: bool = False, # Groq query variants
    n_queries: int = 3,           # number of variants
) -> str:
```

**Execution order:**
1. **HyDE** (if enabled): generate hypothetical answer → use as retrieval query
2. **Multi-query** (if enabled): generate N Groq variants → use all for retrieval
3. **Pipeline A**: `_retrieve_with_scores(vectorstore, docstore, query, k)` for each query → classify
4. **Pipeline B**: `_retrieve_with_scores(vectorstore_raw, docstore, query, k_raw)` for each query → classify
5. **Pipeline C**: `retrieve_by_clip(query_text, k=k_clip)` → image list
6. **Merge + Deduplicate**: union all three pipelines, deduplicate by first-64-chars of key field
7. **Reranking** (if enabled): cross-encoder reranking on merged text+tables
8. **Prompt construction**: `build_rag_prompt(context, question, history=history)`
9. **Answer generation**: `_gemini_invoke_with_retry(messages)` with 4-attempt backoff
10. **Display**: sources with cosine badges + markdown-rendered answer

**Deduplication key:** For each element, the dedup key is the first 64 chars of `b64` (images), `html` (tables), or `text` (text chunks). Elements from multiple pipelines or multiple query variants that refer to the same original are kept only once.

---

## 8. Advanced Retrieval Features

### HyDE — Hypothetical Document Expansion

**HyDE** (Hypothetical Document Embeddings) is a technique where instead of embedding the raw question (which may be phrased differently from how answers appear in the document), the system first generates a *hypothetical answer passage* and uses that as the retrieval query.

**Why it helps:** Research papers use technical language. A question like "what is the main contribution?" might not semantically match a passage that says "we propose a novel self-attention mechanism...". HyDE generates a passage in the *same style and vocabulary as the document*, dramatically improving recall for vague or broadly-phrased questions.

**Implementation:**
```python
def _hyde_expand_query(question: str) -> str:
    hyp = gemini_answer.invoke([HumanMessage(content=(
        "Write a short (3-5 sentence) hypothetical passage that would be "
        "the perfect answer to this question. Write it as if it were extracted "
        "from a research paper. Do not add any preamble.\n\n"
        f"Question: {question}"
    ))])
    return hyp.content.strip()   # fallback to original on any error
```

The HyDE-expanded text replaces the original query only for **vector retrieval**. The original question is always used for prompt construction, reranking, and multi-query variants.

**Notebook default:** `use_hyde=True`. **App default:** `use_hyde=False` (can be enabled per-query).

---

### Cross-Encoder Reranking

Vector similarity (bi-encoder) retrieval is fast but imprecise — it compares query and passage in separate embedding spaces. **Cross-encoder reranking** feeds the full (query, passage) pair to a smaller BERT-based model that scores relevance directly, with much higher accuracy.

**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` — 22M parameters, CPU-friendly, ~200 ms overhead for 10 chunks, ~65 MB download.

**How it works:**
1. After merging all three pipelines, collect all text + table chunks
2. Pair each chunk with the original question: `[(question, chunk_text), ...]`
3. Feed all pairs to the cross-encoder in one batch: `scores = ce.predict(pairs)`
4. Sort by score descending, keep top `rerank_top_k` items
5. Re-split into `reranked_texts` and `reranked_tables` based on original modality

**Important:** Reranking works on **any single query** — it is NOT limited to multi-turn conversations. History context has no effect on the reranker. HyDE + Rerank together is valid: HyDE improves retrieval recall (more relevant candidates enter the pool), then reranking improves precision (the best of those candidates are selected).

**Images are not reranked** — they enter the context directly since cross-encoders are text-only.

---

### Multi-Query Retrieval

Single-query retrieval can miss relevant documents if the user's phrasing doesn't match the indexed vocabulary. **Multi-query retrieval** generates N alternative phrasings using Groq and takes the union of all results.

**Uses Groq (not Gemini)** to avoid consuming the daily Gemini quota.

**Implementation:**
```python
def _generate_query_variants(question: str, n: int = 3) -> list:
    prompt = f"Generate exactly {n} different phrasings of: {question}\n..."
    resp = groq_llm.invoke([HumanMessage(content=prompt)])
    return [question] + resp.content.strip().splitlines()[:n]
```

When both HyDE and Multi-query are enabled:
- HyDE generates one hypothetical passage → used as the primary retrieval query
- Groq generates N variants of the **original** question → used for additional passes
- All N+1 queries run retrieval independently → results are merged + deduplicated

---

### Cosine Similarity Scores

Every retrieved element has its cosine similarity score attached as `_score` (float 0–1), displayed as a colored badge next to the source number.

**Implementation replaces the standard retriever with direct vectorstore access:**
```python
def _retrieve_with_scores(vs, ds, query: str, k: int) -> list:
    scored = vs.similarity_search_with_relevance_scores(query, k=k)
    # scored = [(Document, float), ...]
    for doc, score in scored:
        doc_id = doc.metadata.get(DOC_ID_KEY)
        elem = ds.mget([doc_id])[0]
        elem = dict(elem)           # shallow copy — never mutate cached data
        elem["_score"] = round(float(score), 3)
    return results
```

**Score color coding:**

| Score | Color | Meaning |
|---|---|---|
| ≥ 0.75 | Green badge | High relevance — strong semantic match |
| 0.55 – 0.74 | Orange badge | Moderate relevance |
| < 0.55 | Red badge | Low relevance — may be noise |

Scores are shown in both the notebook (`display_sources()`) and the Streamlit app (`render_sources()`).

---

## 9. Multi-Turn Conversation

The system maintains conversation history across multiple questions, injecting the last 3 Q&A turns into every subsequent prompt. This enables genuinely contextual follow-up questions.

**Notebook (Step 10.5):**

```python
QUESTIONS = [
    "What is the main contribution?",
    {"question": "What BLEU scores are in Table 2?", "use_rerank": True},
    "How does that compare to LSTM-based models?",  # benefits from history
]
CONVERSATION_HISTORY = []

for turn in QUESTIONS:
    history_ctx = CONVERSATION_HISTORY[-3:] if USE_HISTORY else None
    answer = query_paper(question=q, history=history_ctx, ...)
    CONVERSATION_HISTORY.append({"question": q, "answer": answer})
```

**In `build_rag_prompt()`**, history is injected at the top of the context:
```
[CONVERSATION HISTORY — last 3 turns]
Q: What is the main contribution?
A: The paper proposes the Transformer, a model architecture based entirely on attention...
```

**App:** Each message in `st.session_state.chat_history` stores `{question, answer, sources, model_used, stats, history_used}`. History context is built from the last 3 items when the History toggle is enabled.

**Per-question override:** In the notebook, each QUESTIONS entry can be a dict with `"use_history"`, `"use_hyde"`, `"use_rerank"` overrides. In the app, four toggles above the chat input control each question independently.

---

## 10. Inline Citations

The LLM is instructed to emit inline citations matching the numbered sources displayed in "Retrieved Sources". Every source is numbered `[SOURCE N]` in the prompt context.

**In `build_rag_prompt()`:**
```
[SOURCE 1 — TEXT page 3]
An attention function can be described as mapping a query...

[SOURCE 2 — TABLE page 6]  Caption: Maximum path lengths...
| Layer Type | Complexity per Layer | ...

[SOURCE 3 — TEXT page 1]
Ashish Vaswani, Google Brain...
```

**LLM citation rules (in system prompt):**
- Cite each source inline using `[N]` — e.g. `'The BLEU score is 28.4 [3]'`
- Always include the page — e.g. `'[2, p.8]'`
- Quote exact numbers from tables — do not paraphrase
- For figures attached as images, describe what you see

This creates a verifiable answer where each factual claim traces back to a specific numbered source visible in the output.

---

## 11. Streamlit Application

### Features

The Streamlit app (`app.py`, ~1700 lines) provides a complete web interface with all notebook features plus:

- **Streaming answers** via `st.write_stream()` — text appears token-by-token
- **Auto model fallback** — automatically tries fallback models if quota is exhausted
- **Document Explorer** tab — browse all extracted elements (texts, tables, images, metadata)
- **Per-query feature toggles** — four independent toggles above every question
- **Pipeline contribution stats** — 8-metric dashboard per answer
- **Full conversation export** — download entire chat as self-contained HTML

### Running the App

```powershell
# Activate virtual environment
multimodal_env\Scripts\Activate.ps1

# Start the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

**First run:** Enter API keys in the sidebar (or pre-load from `.env`). Upload a PDF and click "Start Processing". Everything runs automatically with live status updates and a progress bar.

**Subsequent runs:** If the same PDF was processed before (matched by SHA-256 hash), all cached data loads in under 1 second and the app goes straight to the chat interface.

### Chat Interface

**Sidebar — Retrieval Settings:**
```
Pipeline A — summary k:        [2 ────────── 12]    default: 6
Pipeline B — raw atomic k:     [1 ──────── 8]       default: 3
Pipeline C — CLIP visual k:    [1 ──── 6]           default: 2
────────────────────────────────────────────────────
[x] Enable raw atomic pipeline (B)
[x] Enable CLIP visual pipeline (C)
[x] Include conversation history
[ ] HyDE query expansion
[ ] Cross-encoder reranking       → [Rerank top k slider if enabled]
[ ] Multi-query retrieval         → [N variants slider if enabled]
```

**Per-question toggles (above chat input):**
```
💬 History  |  🔮 HyDE  |  🎯 Rerank  |  🔀 MultiQ
```
These override sidebar defaults for each individual question.

**Answer display:**
1. Streaming answer (token-by-token)
2. Model used caption
3. 8-column pipeline stats dashboard
4. Collapsible sources section with numbered expanders + cosine badges

### Pipeline Stats Dashboard (8 columns)

| Metric | Description |
|---|---|
| A texts | Text chunks retrieved from Pipeline A (summary) |
| A tables | Tables retrieved from Pipeline A |
| A images | Images retrieved from Pipeline A |
| B tables | Tables retrieved from Pipeline B (raw atomic) |
| C images | Images retrieved from Pipeline C (CLIP) |
| HyDE | ✓ if HyDE was used and expanded the query |
| Rerank | ✓ if cross-encoder reranking was applied |
| MultiQ | ✓ if multi-query retrieval ran multiple variants |

### Document Explorer

A second tab allows browsing all extracted elements:

- **Text tab:** Filter by keyword and page number; shows text + LLM summary for each chunk
- **Tables tab:** Renders each table as HTML with caption
- **Images tab:** Shows all figures with captions and Gemini descriptions
- **Metadata tab:** PDF title, author, pages, cache info, vector counts

### Conversation Export

A "Download Conversation" button appears in the sidebar as soon as the first question is answered. Downloads a self-contained HTML file with:

- All Q&A turns with turn numbers
- Markdown → HTML conversion (bold, code, headers, lists)
- Per-turn source list in collapsible `<details>` elements
- Pipeline flags (HyDE/Rerank/History/MultiQ) per turn
- Model name per turn
- Export timestamp

---

## 12. Jupyter Notebook Reference

The notebook (`multimodal_rag_complete.ipynb`) has 63 cells organized into numbered steps:

| Step | Description |
|---|---|
| 1 | Package verification — check all imports |
| 2 | API key loading from `.env` |
| 3 | PDF loading (path copy / widget uploader / direct path) |
| — | diskcache setup + PDF SHA-256 hash |
| 4 | Docling extraction (with caching) + formula/form merging |
| 5 | Preview extracted images, tables, text chunks |
| 6 | Groq summarization chains + batch summarization |
| 7 | Gemini Vision image descriptions |
| 8 | CLIP ViT-B-32 embeddings + isolated clip_index |
| — | Embedding model config + ChromaDB + MultiVectorRetriever setup |
| — | Multi-vector indexing (`_add_to_retriever`) |
| — | Raw atomic index (Pipeline B) |
| — | Retrieval smoke test |
| 9 | `classify_docs`, `_retrieve_with_scores`, `_score_badge`, `build_rag_prompt`, `gemini_answer` |
| — | Model probe cell — auto-switches to working Gemini model |
| — | `_gemini_invoke_with_retry`, `display_sources`, `_rerank_docs`, `_hyde_expand_query`, `_generate_query_variants`, `query_paper` |
| 10 | Example questions Q1–Q5 |
| 10.5 | Multi-turn conversation session runner |
| — | `export_conversation_html()` + auto-export |
| 11 | Interactive ipywidgets Q&A widget |
| 12 | Statistics dashboard (HTML table) |
| 13 | ChromaDB reload instructions |
| 14 | PyMuPDF deep metadata extraction |
| 15 | pdfplumber supplemental table extraction |
| 16 | Index supplemental pdfplumber tables |
| 17 | Batch Q&A runner (`run_batch_qa`) |
| 18 | Full HTML report generator (`build_html_report`) |
| — | Component references table |

---

## 13. Caching Strategy

The caching system is designed so that **each operation runs exactly once per PDF** and each **parameter combination**. The cache key always includes the PDF hash, ensuring different PDFs never share cached data.

```
First run (cold):
  PDF hash → docling extraction (30–120 s) → cache
  Text chunks → Groq summarization (1–3 min) → cache individually per chunk
  Images → Gemini Vision (20 RPD limited, ~10 s each) → cache individually per image
  CLIP embeddings → computed once, cached as list → stored in clip_index
  Docstore → serialized dict → cache after indexing

Subsequent runs (warm):
  All of the above → cache HIT → loaded in < 1 s total
  ChromaDB → already on disk → SKIP_INDEXING = True
  clip_index → already populated → SKIP_CLIP_INDEX = True
```

**Invalidation:** To force a full re-process of a PDF, delete its cache entries or clear the entire cache. The Streamlit app provides a "Clear cache for this PDF" button in the sidebar.

---

## 14. Model Configurations & Fallbacks

### Gemini Model Probe (Notebook Cell 33 / App auto-probe)

Because Gemini free-tier quotas are per-model and per-day, the system auto-probes available models and switches to the first one with quota:

```python
_candidates = ["gemini-2.5-flash-lite", "gemini-3.1-flash-lite", "gemini-2.0-flash-lite"]
for model in _candidates:
    try:
        llm = ChatGoogleGenerativeAI(model=model, ...)
        llm.invoke([HumanMessage(content="Reply with just OK.")])
        gemini_answer = llm  # switch to working model
        break
    except Exception:
        continue
```

In the app, `_stream_gemini()` probes inline at query time — the probe request itself is a minimal message to avoid wasting quota.

### Retrieval Fallback (Cosine Scores)

If `similarity_search_with_relevance_scores()` fails for any reason, `_retrieve_with_scores()` falls back to `similarity_search()` without scores (all `_score = None`), and the score badge is simply omitted.

### Cross-Encoder Fallback

If `sentence-transformers` or the cross-encoder model is unavailable, `_rerank_docs()` returns the original lists unchanged without error.

### HyDE Fallback

If Gemini is rate-limited during HyDE expansion, the original question is used unchanged.

### Multi-Query Fallback

If Groq errors during variant generation, the system falls back to single-query behavior automatically.

---

## 15. Data Flow: End-to-End

```
User uploads PDF
       │
       ▼
compute_pdf_hash()           — SHA-256[:16] content fingerprint
       │
       ├─ cache HIT ──────────────────────────────────────────► skip to query
       │
       ▼
_run_docling_extraction()    — ~30–120 s
       │  texts[], tables[], images[], formulas[], forms[], metadata{}
       ▼
texts += formulas + forms    — merge into unified text pool
       │
       ├─ Groq text summarization ──────────────────────────► text_summaries[]
       ├─ Groq table summarization ─────────────────────────► table_summaries[]
       └─ Gemini Vision image description ──────────────────► image_summaries[]
                                                │
                                                ▼
                                       BGE embed summaries
                                                │
                              ┌─────────────────┼─────────────────┐
                              │                 │                 │
                           ChromaDB          ChromaDB          ChromaDB
                           rag_bge           rag_raw_bge       clip_index
                           (summary          (raw content      (CLIP 512-dim
                            768-dim)          768-dim)          vectors)
                              │                 │                 │
                              └─────────────────┼─────────────────┘
                                                │
                                         InMemoryStore
                                         (uuid → original element)
                                         + diskcache persistence
                                                │
                                        ════════════════
                                           QUERY TIME
                                        ════════════════
                                                │
User submits question "q"
       │
       ├─ [HyDE] generate hypothetical answer → query_text
       ├─ [MultiQ] generate N Groq variants → queries[]
       │
       ├─ Pipeline A: similarity_search_with_relevance_scores(query_text, k=6)
       │              + docstore.mget(doc_ids) → texts+tables+images with _score
       │
       ├─ Pipeline B: similarity_search_with_relevance_scores(query_text, k=3)
       │              + docstore.mget(doc_ids) → tables+formulas+forms with _score
       │
       ├─ Pipeline C: CLIP text embed → cosine search → images[]
       │
       ├─ Merge + Deduplicate (by content key[:64])
       │
       ├─ [Rerank] cross-encoder/ms-marco rerank text+tables → keep top-k
       │
       ├─ build_rag_prompt(context, question, history=last_3_turns)
       │   ├─ [CONVERSATION HISTORY] optional last-3-turns prefix
       │   ├─ [SOURCE 1 — TEXT page N] raw text
       │   ├─ [SOURCE 2 — TABLE page N] raw HTML/markdown
       │   └─ base64 images attached as vision inputs
       │
       └─ Gemini.invoke(messages) → answer with [N] / [N, p.X] citations
              │
              └─ display: sources (with cosine badges) + markdown answer
```

---

## 16. Configuration Reference

### `query_paper()` Parameters (Notebook)

| Parameter | Default | Description |
|---|---|---|
| `question` | required | The user's question |
| `k` | 6 | Pipeline A: top-k summary results |
| `k_raw` | 3 | Pipeline B: top-k raw content results |
| `k_clip` | 2 | Pipeline C: top-k CLIP image results |
| `show_sources` | True | Display retrieved sources before answer |
| `use_clip` | True | Enable CLIP visual pipeline |
| `use_raw` | True | Enable raw atomic pipeline |
| `use_hyde` | True | HyDE query expansion |
| `use_rerank` | True | Cross-encoder reranking |
| `rerank_top_k` | 6 | Chunks to keep after reranking |
| `history` | None | List of `{question, answer}` dicts for multi-turn |
| `use_multiquery` | False | Groq multi-query variant generation |
| `n_queries` | 3 | Number of query variants to generate |

### Embedding Model Switch

To switch embedding models, change `EMBEDDING_MODEL = "minilm"` and re-run from the embedding setup cell. A fresh ChromaDB collection (`rag_minilm`) will be created automatically. Old `rag_bge` data is unaffected.

### Cache Size

Default diskcache limit is 2 GB. To change:
```python
cache = diskcache.Cache("./cache", size_limit=int(4e9))  # 4 GB
```

---

## 17. Known Limitations & Design Decisions

**Free-tier Gemini quotas:** The `gemini-2.5-flash` model used for image descriptions has a 20 RPD limit on the free tier. For a paper with 6 figures, this uses 6 of those 20 requests. The caching system ensures these are consumed only once per PDF. For production use or large document batches, a billing account removes this limit.

**CPU-only inference:** All models (BGE, MiniLM, CLIP, cross-encoder) run on CPU by default. For faster embedding and reranking, change `device: "cpu"` to `"cuda"` in the HuggingFaceEmbeddings config if a CUDA GPU is available.

**InMemoryStore RAM usage:** The docstore holds all extracted elements (full text, HTML tables, base64 images) in RAM. For a typical academic paper, this is ~50–200 MB. For very large documents (100+ pages with many figures), consider limiting image resolution or the number of indexed elements.

**ChromaDB HNSW warm-up:** On first query after a cold start, ChromaDB loads the HNSW graph from disk into memory. This may add 1–2 seconds to the first query. Subsequent queries are fast.

**Deduplication is content-based:** Two elements from different pipelines that share the same first 64 characters of their key field are considered duplicates. In practice this works well because each element has a unique content fingerprint.

**Multi-query and HyDE together:** When both are enabled, HyDE expands the original question into one hypothetical passage, and Groq generates N variants of the *original* question (not the HyDE passage). This gives the broadest coverage — one semantically rich retrieval query from HyDE plus N surface-variation queries from multi-query.

**Summary vs raw trade-off:** Pipeline A (summary) and Pipeline B (raw) serve different query types. For factual number lookups use `use_raw=True` (default). For broad semantic questions, Pipeline A dominates. For visual questions, Pipeline C (CLIP) adds recall that neither text pipeline can provide.

---

## 18. Component References

| Component | Reference |
|---|---|
| [docling (IBM, 2024)](https://github.com/DS4SD/docling) | PDF/Office parser — text, tables, charts, figures with layout understanding |
| [LangChain 1.x](https://python.langchain.com/) | Orchestration framework for LLM applications |
| [langchain-classic 1.0.x](https://github.com/langchain-ai/langchain) | Multi-vector retriever + InMemoryStore for LangChain 1.x |
| [ChromaDB 1.5.x](https://www.trychroma.com/) | Persistent vector database via langchain-chroma |
| [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | 768-dim embedding model, strong semantic similarity for retrieval |
| [Groq / llama-3.3-70b-versatile](https://console.groq.com/docs/models) | Fast text & table summarization — 280 tok/s, 131k context |
| [Google Gemini 2.x Flash](https://ai.google.dev/gemini-api/docs/models) | Multimodal final-answer model — vision + 1M context window |
| [CLIP ViT-B-32 (OpenAI)](https://github.com/mlfoundations/open_clip) | Cross-modal image-text embeddings for visual retrieval |
| [cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) | Cross-encoder reranking model |
| [pdfplumber](https://github.com/jsvine/pdfplumber) | Heuristic PDF table extraction as docling supplement |
| [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/) | Fast PDF metadata, TOC, hyperlinks, annotations |
| [diskcache](https://grantjenks.com/docs/diskcache/) | Content-addressed persistent disk cache |
| [Streamlit](https://streamlit.io/) | Web application framework |
| [sentence-transformers](https://www.sbert.net/) | HuggingFace sentence embedding library + cross-encoder support |
| [ipywidgets](https://ipywidgets.readthedocs.io/) | Interactive notebook UI widgets |

---

*Built with Python 3.14 · LangChain 1.x · ChromaDB 1.5 · Docling 2.x · Streamlit 1.35+*
