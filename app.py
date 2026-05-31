"""
Context-Aware Multimodal Knowledge Retrieval System
Streamlit Application — Full 3-Pipeline RAG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pipelines:
  A — LLM-summary MultiVectorRetriever (BGE/MiniLM embeddings)
  B — Raw-atomic MultiVectorRetriever   (exact table markdown + formulas)
  C — CLIP visual retriever             (text→image cosine similarity)

New vs notebook:
  • Streaming Gemini answers via st.write_stream
  • True multi-turn conversation with last-3-turns context
  • Inline source citations  [TEXT·p4] [TABLE·p8] [FIG·1]
  • Auto Gemini model fallback (no manual probe cell needed)
  • Document explorer tab (browse all extracted elements)
  • Download conversation as HTML
  • Per-query pipeline contribution stats
  • API keys from sidebar or auto-loaded from .env
  • HyDE query expansion (optional — generates hypothetical answer for better retrieval)
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import streamlit as st
import os, io, sys, base64, hashlib, uuid, time, re, warnings, textwrap, json
from pathlib import Path
from typing import Optional, Generator

warnings.filterwarnings("ignore")

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Multimodal RAG",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get Help": "https://github.com/DS4SD/docling", "About": "Multimodal RAG System"},
)

# ── Path constants ─────────────────────────────────────────────────────────────
_BASE          = Path(__file__).parent
CONTENT_DIR    = _BASE / "content"
CHROMA_DIR     = str(_BASE / "chroma_db")
CLIP_INDEX_DIR = str(_BASE / "clip_index")
CACHE_DIR      = str(_BASE / "cache")

for _d in [CONTENT_DIR, CONTENT_DIR / "images",
           Path(CHROMA_DIR), Path(CLIP_INDEX_DIR), Path(CACHE_DIR)]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Config constants ───────────────────────────────────────────────────────────
EMBEDDING_CONFIGS = {
    "bge": {
        "name":      "BAAI/bge-base-en-v1.5",
        "dim":       768,
        "normalize": True,
        "desc":      "BGE 768-dim — best semantic quality (recommended)",
    },
    "minilm": {
        "name":      "sentence-transformers/all-MiniLM-L6-v2",
        "dim":       384,
        "normalize": True,
        "desc":      "MiniLM 384-dim — ~2× faster, slightly lower recall",
    },
}

ANSWER_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

DOC_ID_KEY = "doc_id"

LABEL_COLORS = {
    "text":             "#007acc",
    "paragraph":        "#007acc",
    "title":            "#b00020",
    "section_header":   "#d84315",
    "table":            "#2e7d32",
    "picture":          "#6a1b9a",
    "chart":            "#ad1457",
    "formula":          "#00695c",
    "form":             "#4527a0",
    "key_value_region": "#283593",
    "list_item":        "#1565c0",
    "caption":          "#558b2f",
    "footnote":         "#795548",
    "reference":        "#546e7a",
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SESSION STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_SS_DEFAULTS: dict = {
    # API keys
    "google_api_key":  "",
    "groq_api_key":    "",
    "hf_token":        "",
    "api_keys_valid":  False,
    # Model settings
    "embedding_model": "bge",
    "answer_model":    "gemini-2.5-flash-lite",
    # Retrieval parameters
    "k_summary":    6,
    "k_raw":        3,
    "k_clip":       2,
    "use_raw":      True,
    "use_clip":     True,
    "use_history":  True,
    "use_hyde":     False,
    "use_rerank":   False,
    "rerank_top_k": 6,
    # Document state
    "rag_ready":    False,
    "pdf_path":     None,
    "pdf_hash":     None,
    "pdf_name":     None,
    "pdf_pages":    0,
    # Extracted elements
    "texts":        [],
    "tables":       [],
    "images":       [],
    "formulas":     [],
    "forms":        [],
    "pdf_meta":     {},
    # Summaries
    "text_summaries":  [],
    "table_summaries": [],
    "image_summaries": [],
    # IDs
    "text_ids":   [],
    "table_ids":  [],
    "image_ids":  [],
    # Docstore backing (persisted for retriever reconstruction)
    "docstore_backing": {},
    # Chat
    "chat_history": [],   # [{question, answer, sources, model_used, stats}]
    # Processing
    "processing_log":   [],
    "processing_error": None,
    # UI state
    "active_tab": "chat",
}

for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CACHED RESOURCES  (heavy objects loaded once per process lifetime)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_resource(show_spinner=False)
def _disk_cache():
    import diskcache
    return diskcache.Cache(CACHE_DIR, size_limit=2 * 1024 ** 3)


@st.cache_resource(show_spinner=False)
def _load_embedding_model(key: str):
    from langchain_huggingface import HuggingFaceEmbeddings
    cfg = EMBEDDING_CONFIGS[key]
    return HuggingFaceEmbeddings(
        model_name=cfg["name"],
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": cfg["normalize"], "batch_size": 32},
    )


@st.cache_resource(show_spinner=False)
def _load_clip():
    import open_clip, torch
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tok = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    return model, preprocess, tok


@st.cache_resource(show_spinner=False)
def _clip_db_client():
    import chromadb
    return chromadb.PersistentClient(path=CLIP_INDEX_DIR)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UTILITY FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _ck(*parts) -> str:
    return ":".join(str(p) for p in parts)


def compute_pdf_hash(pdf_path: str) -> str:
    h = hashlib.sha256()
    p = Path(pdf_path)
    with open(p, "rb") as f:
        h.update(f.read(65536))
    h.update(str(p.stat().st_size).encode())
    return h.hexdigest()[:16]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLIP HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def clip_embed_images_batch(img_dicts: list) -> list:
    import torch
    from PIL import Image as PILImage
    clip_model, clip_preprocess, _ = _load_clip()
    out = []
    for img in img_dicts:
        pil = PILImage.open(io.BytesIO(base64.b64decode(img["b64"]))).convert("RGB")
        t = clip_preprocess(pil).unsqueeze(0)
        with torch.no_grad():
            e = clip_model.encode_image(t)
            e = e / e.norm(dim=-1, keepdim=True)
        out.append(e.squeeze().numpy().tolist())
    return out


def clip_embed_text(queries: list) -> list:
    import torch
    clip_model, _, clip_tok = _load_clip()
    tokens = clip_tok(queries)
    with torch.no_grad():
        e = clip_model.encode_text(tokens)
        e = e / e.norm(dim=-1, keepdim=True)
    return e.numpy().tolist()


def _get_clip_col(pdf_hash_val: str):
    client = _clip_db_client()
    return client.get_or_create_collection(
        f"clip_ViT_B_32_{pdf_hash_val[:8]}",
        metadata={"hnsw:space": "cosine"},
    )


def retrieve_by_clip(query: str, k: int, images: list, pdf_hash_val: str) -> list:
    col = _get_clip_col(pdf_hash_val)
    n = col.count()
    if n == 0:
        return []
    q_emb = clip_embed_text([query])
    results = col.query(query_embeddings=q_emb, n_results=min(k, n))
    retrieved = []
    for meta in results["metadatas"][0]:
        idx = meta.get("img_idx", -1)
        if 0 <= idx < len(images):
            retrieved.append(images[idx])
    return retrieved


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DOCLING EXTRACTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_docling_extraction(pdf_path: str, log_fn=None) -> dict:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling_core.types.doc import PictureItem, TableItem, DocItemLabel
    import fitz

    if log_fn:
        log_fn("Running Docling PDF extraction (30-120 s first time)…")

    pipeline_opts = PdfPipelineOptions()
    pipeline_opts.do_table_structure      = True
    pipeline_opts.generate_picture_images = True
    pipeline_opts.generate_table_images   = True
    pipeline_opts.images_scale            = 2.0
    pipeline_opts.do_ocr                  = False
    pipeline_opts.do_formula_enrichment   = False

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)}
    )
    result = converter.convert(pdf_path)
    doc    = result.document

    texts, tables, images, formulas, forms = [], [], [], [], []

    TEXT_LABELS = {
        DocItemLabel.TEXT, DocItemLabel.PARAGRAPH, DocItemLabel.TITLE,
        DocItemLabel.SECTION_HEADER, DocItemLabel.LIST_ITEM, DocItemLabel.CAPTION,
        DocItemLabel.FOOTNOTE, DocItemLabel.REFERENCE, DocItemLabel.HANDWRITTEN_TEXT,
    }
    FORM_LABELS = {
        DocItemLabel.FORM, DocItemLabel.KEY_VALUE_REGION,
        DocItemLabel.FIELD_REGION, DocItemLabel.FIELD_ITEM,
    }

    for item, _ in doc.iterate_items():
        page = item.prov[0].page_no if item.prov else None

        if isinstance(item, TableItem):
            try:
                html  = item.export_to_html(doc)
                df    = item.export_to_dataframe(doc)
                mdown = item.export_to_markdown(doc)
                cap   = item.caption_text(doc) or ""
                tables.append({
                    "html": html, "df": df, "markdown": mdown,
                    "page": page, "caption": cap, "label": "table",
                })
            except Exception:
                pass

        elif isinstance(item, PictureItem):
            try:
                pil = item.get_image(doc)
                if pil:
                    buf = io.BytesIO()
                    pil.convert("RGB").save(buf, format="JPEG", quality=90)
                    b64 = base64.b64encode(buf.getvalue()).decode()
                    cap = item.caption_text(doc) or ""
                    w, h = pil.size
                    images.append({
                        "b64": b64, "page": page,
                        "label": item.label.value, "caption": cap,
                        "width": w, "height": h,
                    })
            except Exception:
                pass

        elif item.label == DocItemLabel.FORMULA:
            text = getattr(item, "text", "") or ""
            if text.strip():
                formulas.append({"text": text, "page": page})

        elif item.label in FORM_LABELS:
            text = getattr(item, "text", "") or ""
            if text.strip():
                forms.append({"text": text, "page": page, "label": item.label.value})

        elif item.label in TEXT_LABELS:
            text = getattr(item, "text", "") or ""
            if len(text.strip()) > 20:
                texts.append({
                    "text":    text,
                    "page":    page,
                    "label":   item.label.value,
                    "heading": item.label in {DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER},
                })

    fitz_doc = fitz.open(pdf_path)
    meta     = fitz_doc.metadata
    toc      = fitz_doc.get_toc()
    pages    = fitz_doc.page_count
    fitz_doc.close()

    return {
        "texts": texts, "tables": tables, "images": images,
        "formulas": formulas, "forms": forms,
        "metadata": {
            "title":  meta.get("title", Path(pdf_path).stem),
            "author": meta.get("author", ""),
            "pages":  pages, "toc": toc, "source": str(pdf_path),
        },
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GROQ TEXT / TABLE SUMMARIZATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def summarize_texts_and_tables(
    texts: list, tables: list, pdf_hash_val: str,
    groq_key: str, log_fn=None
) -> tuple[list, list]:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    cache = _disk_cache()
    llm   = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, api_key=groq_key)

    TEXT_P = ChatPromptTemplate.from_template(
        "You are an expert academic research assistant.\n"
        "Write a concise, information-dense summary (3-6 sentences) of the following "
        "passage from a research paper. Capture: main topic, key findings, named entities, "
        "context within the paper. Do NOT start with 'Here is a summary'.\n\n"
        "Element type: {label}\nPage: {page}\nContent:\n{element}"
    )
    TABLE_P = ChatPromptTemplate.from_template(
        "You are an expert data analyst reviewing a table from a research paper.\n"
        "Describe: (1) what the table reports, (2) column/row headers, "
        "(3) key numerical values and exact numbers, (4) trends, "
        "(5) what conclusion it supports. Be specific with numbers.\n"
        "Do NOT start with 'Here is a'.\n\n"
        "Caption: {caption}\nPage: {page}\nTable:\n{element}"
    )

    text_chain  = ({"element": lambda x: x["text"], "label": lambda x: x.get("label",""), "page": lambda x: x.get("page","?")} | TEXT_P  | llm | StrOutputParser())
    table_chain = ({"element": lambda x: x.get("markdown") or x.get("html",""), "caption": lambda x: x.get("caption",""), "page": lambda x: x.get("page","?")} | TABLE_P | llm | StrOutputParser())

    _PFX    = _ck("groq_summ_v2", pdf_hash_val, "llama-3.3-70b-versatile")
    _PFX_V1 = _ck("groq_summ_v1", pdf_hash_val, "llama-3.3-70b-versatile")

    # Migrate v1 → v2 cache keys (no API calls)
    for i in range(len(texts)):
        v1k = _ck(_PFX_V1, "text", i)
        v2k = _ck(_PFX,    "text", i)
        if v1k in cache and v2k not in cache:
            cache[v2k] = cache[v1k]

    def _invoke_cached(chain, payload, ck):
        if ck in cache:
            return cache[ck]
        for attempt in range(3):
            try:
                res = chain.invoke(payload)
                cache[ck] = res
                return res
            except Exception as exc:
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
                else:
                    fb = (payload.get("text") or payload.get("markdown") or str(payload)[:200]) if isinstance(payload, dict) else str(payload)[:200]
                    return fb  # NOT cached → retried next run

    text_summaries = []
    for i, chunk in enumerate(texts):
        if log_fn:
            log_fn(f"Summarizing text {i+1}/{len(texts)}…")
        ck = _ck(_PFX, "text", i)
        text_summaries.append(_invoke_cached(text_chain, chunk, ck))
        if ck not in cache:
            time.sleep(0.15)

    table_summaries = []
    for i, tbl in enumerate(tables):
        if log_fn:
            log_fn(f"Summarizing table {i+1}/{len(tables)}…")
        ck = _ck(_PFX, "table", i)
        table_summaries.append(_invoke_cached(table_chain, tbl, ck))
        if ck not in cache:
            time.sleep(0.15)

    return text_summaries, table_summaries


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GEMINI IMAGE SUMMARIZATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def summarize_images(
    images: list, pdf_hash_val: str, google_key: str, log_fn=None
) -> list:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage

    cache       = _disk_cache()
    vision_llm  = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_key, temperature=0.1)
    _PFX        = _ck("gemini_img_v1", pdf_hash_val, "gemini-2.5-flash")

    IMG_PROMPT = (
        "This image is extracted from a research paper. Describe it thoroughly:\n"
        "1. Element type (chart / diagram / figure / photo / other)\n"
        "2. All visible text (axis labels, legends, titles, annotations)\n"
        "3. Key data values, percentages, comparisons\n"
        "4. Trends / patterns\n"
        "5. Main insight / conclusion this figure supports\n"
        "6. Layout / structure (multi-panel, color coding, arrows, etc.)\n\n"
        "Be thorough — your description is the only textual representation of this figure."
    )

    summaries = []
    for i, img in enumerate(images):
        ck = _ck(_PFX, i)
        if ck in cache:
            summaries.append(cache[ck])
            continue
        if log_fn:
            log_fn(f"Summarizing image {i+1}/{len(images)} (page {img.get('page','?')})…")
        try:
            cap     = img.get("caption", "")
            prompt  = IMG_PROMPT + (f'\n\nCaption from paper: "{cap}"' if cap else "")
            msg     = HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img['b64']}"}},
            ])
            resp   = vision_llm.invoke([msg])
            result = resp.content
            cache[ck] = result
        except Exception as exc:
            result = f"[Vision description unavailable: {exc}]"
        summaries.append(result)
        time.sleep(0.5)

    return summaries


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PDFPLUMBER SUPPLEMENTAL TABLES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def extract_pdfplumber_tables(pdf_path: str) -> list:
    try:
        import pdfplumber, pandas as pd
        extra = []
        with pdfplumber.open(pdf_path) as pdf:
            for pg_no, page in enumerate(pdf.pages, 1):
                for raw_tbl in page.extract_tables() or []:
                    if not raw_tbl:
                        continue
                    header = raw_tbl[0]
                    rows   = raw_tbl[1:]
                    if header and any(h for h in header if h):
                        df = pd.DataFrame(rows, columns=header)
                        md = df.to_markdown(index=False)
                        extra.append({
                            "markdown": md, "df": df, "page": pg_no,
                            "caption": "", "html": "", "label": "table",
                        })
        return extra
    except Exception:
        return []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INDEXING  (ChromaDB multi-vector + raw atomic + CLIP)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_all_indexes(
    texts, tables, images, formulas, forms,
    text_summaries, table_summaries, image_summaries,
    pdf_hash_val: str, embedding_key: str,
    log_fn=None,
) -> tuple[list, list, list, dict]:
    """
    Build summary index (Pipeline A) + raw atomic index (Pipeline B) + CLIP index (Pipeline C).
    Returns (text_ids, table_ids, image_ids, docstore_backing).
    Idempotent: if indexes already exist, returns existing IDs from cache.
    """
    import uuid as _uuid
    from langchain_chroma import Chroma
    from langchain_classic.storage import InMemoryStore
    from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
    from langchain_core.documents import Document

    cache        = _disk_cache()
    docstore_key = _ck("docstore_v1", pdf_hash_val, embedding_key)

    # ── Restore from cache if already indexed ──────────────────────────────
    existing = dict(cache.get(docstore_key, {}))
    emb      = _load_embedding_model(embedding_key)
    vs       = Chroma(collection_name=f"rag_{embedding_key}", embedding_function=emb, persist_directory=CHROMA_DIR)
    vs_raw   = Chroma(collection_name=f"rag_raw_{embedding_key}", embedding_function=emb, persist_directory=CHROMA_DIR)

    if existing and vs._collection.count() > 0:
        if log_fn:
            log_fn(f"Index already exists ({vs._collection.count()} vectors) — skipping")
        text_ids  = [k for k, v in existing.items() if isinstance(v, dict) and "text"  in v and "b64" not in v and "html" not in v]
        table_ids = [k for k, v in existing.items() if isinstance(v, dict) and "html"  in v]
        image_ids = [k for k, v in existing.items() if isinstance(v, dict) and "b64"   in v]
        return text_ids, table_ids, image_ids, existing

    # ── Full indexing ───────────────────────────────────────────────────────
    docstore = InMemoryStore()
    backing  = {}

    def _idx(originals, summaries, modality, extra_fn=None):
        ids  = [str(_uuid.uuid4()) for _ in originals]
        docs = []
        for i, (did, s) in enumerate(zip(ids, summaries)):
            meta = {DOC_ID_KEY: did, "modality": modality}
            if extra_fn:
                meta.update(extra_fn(i, originals[i]))
            docs.append(Document(page_content=s, metadata=meta))
        vs.add_documents(docs)
        docstore.mset(list(zip(ids, originals)))
        for did, el in zip(ids, originals):
            backing[did] = el
        return ids

    if log_fn: log_fn("Indexing text chunks into summary collection…")
    text_ids = _idx(texts, text_summaries, "text",
        lambda i, el: {"page": el.get("page", -1), "label": el.get("label", "text")})

    if log_fn: log_fn("Indexing tables into summary collection…")
    table_ids = _idx(tables, table_summaries, "table",
        lambda i, el: {"page": el.get("page", -1), "caption": el.get("caption", "")})

    if log_fn: log_fn("Indexing images into summary collection…")
    image_ids = _idx(images, image_summaries, "image",
        lambda i, el: {"page": el.get("page", -1), "label": el.get("label", "picture"), "caption": el.get("caption", "")})

    # ── Raw atomic indexing (Pipeline B) ──────────────────────────────────
    if log_fn: log_fn("Building raw atomic index (table markdown + formulas)…")

    for tbl_id, tbl in zip(table_ids, tables):
        md = tbl.get("markdown") or tbl.get("html", "")
        if not md.strip():
            continue
        vs_raw.add_documents([Document(
            page_content=f"[TABLE markdown — page {tbl.get('page',-1)}]\n{md}",
            metadata={DOC_ID_KEY: tbl_id, "modality": "table", "page": tbl.get("page",-1), "raw_type": "table_markdown"},
        )])

    for i, f in enumerate(formulas):
        text = f.get("text","").strip()
        if not text: continue
        did  = str(_uuid.uuid4())
        elem = {"text": text, "page": f.get("page",-1), "label": "formula", "heading": False}
        docstore.mset([(did, elem)]); backing[did] = elem
        vs_raw.add_documents([Document(
            page_content=f"[FORMULA — page {f.get('page',-1)}] {text}",
            metadata={DOC_ID_KEY: did, "modality": "formula", "raw_type": "formula"},
        )])

    for i, f in enumerate(forms):
        text = f.get("text","").strip()
        if not text: continue
        did   = str(_uuid.uuid4())
        label = f.get("label", "form")
        elem  = {"text": text, "page": f.get("page",-1), "label": label, "heading": False}
        docstore.mset([(did, elem)]); backing[did] = elem
        vs_raw.add_documents([Document(
            page_content=f"[{label.upper()} — page {f.get('page',-1)}] {text}",
            metadata={DOC_ID_KEY: did, "modality": "form", "raw_type": "form"},
        )])

    cache[docstore_key] = backing
    if log_fn:
        log_fn(f"Indexed: {len(text_ids)} texts | {len(table_ids)} tables | {len(image_ids)} images | {vs_raw._collection.count()} raw vectors")
    return text_ids, table_ids, image_ids, backing


def build_clip_index(images: list, pdf_hash_val: str, log_fn=None) -> None:
    """CLIP embeddings into isolated ./clip_index directory."""
    cache = _disk_cache()
    ck    = _ck("clip_emb_v1", "ViT-B-32", "openai", pdf_hash_val)
    col   = _get_clip_col(pdf_hash_val)

    if col.count() > 0:
        if log_fn: log_fn(f"CLIP index already built ({col.count()} images) — skipping")
        return
    if not images:
        return

    if log_fn: log_fn(f"Computing CLIP embeddings for {len(images)} images…")
    embs = cache.get(ck) or clip_embed_images_batch(images)
    cache[ck] = embs

    col.add(
        ids=[f"clip_img_{i}" for i in range(len(images))],
        embeddings=embs,
        documents=[f"Figure {i+1} (page {img.get('page','?')}): {img.get('caption','')}" for i, img in enumerate(images)],
        metadatas=[{"page": img.get("page",-1), "label": img.get("label","picture"), "caption": img.get("caption",""), "img_idx": i} for i, img in enumerate(images)],
    )
    if log_fn: log_fn(f"CLIP: indexed {col.count()} images ✓")


def index_pdfplumber_tables(
    extra_tables: list, table_ids: list, pdf_hash_val: str,
    groq_key: str, embedding_key: str, log_fn=None,
) -> None:
    """Summarize and index pdfplumber supplemental tables."""
    if not extra_tables:
        return
    from langchain_chroma import Chroma
    from langchain_classic.storage import InMemoryStore
    from langchain_core.documents import Document
    import uuid as _uuid

    cache        = _disk_cache()
    docstore_key = _ck("docstore_v1", pdf_hash_val, embedding_key)
    backing      = dict(cache.get(docstore_key, {}))

    emb = _load_embedding_model(embedding_key)
    vs  = Chroma(collection_name=f"rag_{embedding_key}", embedding_function=emb, persist_directory=CHROMA_DIR)

    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, api_key=groq_key)
    TABLE_P = ChatPromptTemplate.from_template(
        "Analyse this supplemental table from a PDF. Describe its content concisely "
        "including exact values.\nPage: {page}\n{element}"
    )
    chain = ({"element": lambda x: x.get("markdown",""), "page": lambda x: x.get("page","?")} | TABLE_P | llm | StrOutputParser())

    added = 0
    for i, tbl in enumerate(extra_tables):
        if log_fn: log_fn(f"Indexing pdfplumber table {i+1}/{len(extra_tables)}…")
        _pfx = _ck("groq_summ_v2", pdf_hash_val, "llama-3.3-70b-versatile")
        ck = _ck(_pfx, "extra_table", i)
        if ck in cache:
            summ = cache[ck]
        else:
            try:
                summ = chain.invoke(tbl)
                cache[ck] = summ
            except Exception:
                summ = tbl.get("markdown", "")[:300]

        did = str(_uuid.uuid4())
        doc = Document(page_content=summ, metadata={DOC_ID_KEY: did, "modality": "table", "page": tbl.get("page",-1), "caption": "", "source": "pdfplumber"})
        vs.add_documents([doc])
        backing[did] = tbl
        added += 1

    cache[docstore_key] = backing
    if log_fn: log_fn(f"pdfplumber: indexed {added} supplemental tables")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RAG RETRIEVAL + QUERY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_retriever(backing: dict, embedding_key: str):
    """Reconstruct MultiVectorRetriever from disk state. Fast — no re-indexing."""
    from langchain_chroma import Chroma
    from langchain_classic.storage import InMemoryStore
    from langchain_classic.retrievers.multi_vector import MultiVectorRetriever

    emb = _load_embedding_model(embedding_key)
    vs     = Chroma(collection_name=f"rag_{embedding_key}",     embedding_function=emb, persist_directory=CHROMA_DIR)
    vs_raw = Chroma(collection_name=f"rag_raw_{embedding_key}", embedding_function=emb, persist_directory=CHROMA_DIR)
    ds     = InMemoryStore()
    ds.mset(list(backing.items()))

    r     = MultiVectorRetriever(vectorstore=vs,     docstore=ds, id_key=DOC_ID_KEY, search_kwargs={"k": 6})
    r_raw = MultiVectorRetriever(vectorstore=vs_raw, docstore=ds, id_key=DOC_ID_KEY, search_kwargs={"k": 4})
    return r, r_raw


def classify_docs(raw_docs: list) -> dict:
    out = {"texts": [], "tables": [], "images": []}
    for doc in raw_docs:
        if isinstance(doc, dict):
            if "b64"  in doc:                   out["images"].append(doc)
            elif "html" in doc or "df" in doc:  out["tables"].append(doc)
            else:                               out["texts"].append(doc)
        elif isinstance(doc, str):
            try:
                base64.b64decode(doc[:128])
                out["images"].append({"b64": doc, "page": None, "label": "picture", "caption": ""})
            except Exception:
                out["texts"].append({"text": doc, "page": None, "label": "text"})
        else:
            out["texts"].append(doc)
    return out


def _dedup(lst: list, seen: set) -> list:
    result = []
    for el in lst:
        if isinstance(el, dict):
            key = el.get("b64","")[:64] or el.get("html","")[:64] or el.get("text","")[:64]
        else:
            key = str(el)[:64]
        if key not in seen:
            seen.add(key); result.append(el)
    return result


def build_rag_prompt(context: dict, question: str, history: list | None = None) -> list:
    from langchain_core.messages import HumanMessage

    parts = []

    if history:
        hstr = "\n".join(f"Q: {h['question']}\nA: {str(h['answer'])[:400]}…" for h in history[-3:])
        parts.append(f"[CONVERSATION HISTORY — last {min(3,len(history))} turns]\n{hstr}")

    # Number all sources so the LLM can reference them
    src_num = 1
    for chunk in context["texts"]:
        text  = chunk.get("text","") if isinstance(chunk, dict) else str(chunk)
        page  = chunk.get("page","?") if isinstance(chunk, dict) else "?"
        label = chunk.get("label","text") if isinstance(chunk, dict) else "text"
        if text.strip():
            parts.append(f"[SOURCE {src_num} — {label.upper()} page {page}]\n{text}")
            src_num += 1

    for tbl in context["tables"]:
        content = (tbl.get("html") or tbl.get("markdown") or tbl.get("text","")) if isinstance(tbl, dict) else str(tbl)
        page    = tbl.get("page","?") if isinstance(tbl, dict) else "?"
        caption = tbl.get("caption","") if isinstance(tbl, dict) else ""
        parts.append(f"[SOURCE {src_num} — TABLE page {page}{' | '+caption if caption else ''}]\n{content}")
        src_num += 1

    ctx_str = "\n\n---\n\n".join(parts) if parts else "(no text or table context retrieved)"

    system = (
        "You are an expert research assistant. Answer the question using ONLY the provided context "
        "(text passages, tables, and any attached figures).\n\n"
        "CITATION RULES:\n"
        "- Cite each source inline using [N] where N is the source number, e.g. 'BLEU score is 28.4 [3]'\n"
        "- Always cite the page: e.g. '[2, p.8]'\n"
        "- For figures you can see, describe what you observe\n"
        "- Quote exact numbers from tables — do not paraphrase\n"
        "- If context is insufficient, say so clearly\n"
        "- Use markdown headers and bullet points when helpful\n\n"
        f"=== CONTEXT ===\n{ctx_str}\n\n"
        f"=== QUESTION ===\n{question}"
    )

    content = [{"type": "text", "text": system}]
    for img in context["images"]:
        b64 = img["b64"] if isinstance(img, dict) else img
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    return [HumanMessage(content=content)]


def _stream_gemini(messages, google_key: str, preferred_model: str) -> tuple[Generator, str]:
    """Stream Gemini answer. Auto-falls back through model list on quota errors."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    candidates = [preferred_model] + [m for m in ANSWER_MODELS if m != preferred_model]

    for model in candidates:
        try:
            llm = ChatGoogleGenerativeAI(model=model, google_api_key=google_key, temperature=0.1)

            def _gen(llm=llm):
                for chunk in llm.stream(messages):
                    if chunk.content:
                        yield chunk.content

            # Quick probe: check the model works before returning the generator
            # (We don't want the generator to fail mid-stream for quota reasons)
            probe = ChatGoogleGenerativeAI(model=model, google_api_key=google_key, temperature=0)
            probe.invoke([messages[0].__class__(content="Reply with just OK.")])

            return _gen(), model
        except Exception as exc:
            err = str(exc)
            if "429" in err or "RESOURCE_EXHAUSTED" in err or "404" in err or "NOT_FOUND" in err:
                time.sleep(2)
                continue
            raise

    def _fallback():
        yield "⚠️ All Gemini models are rate-limited. Please wait a few minutes and try again."
    return _fallback(), preferred_model


def _rerank_docs_app(query: str, texts: list, tables: list, top_k: int = 6) -> tuple[list, list]:
    """Cross-encoder reranking. Falls back silently if unavailable."""
    if not texts and not tables:
        return texts, tables
    try:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
        pairs = []
        items = []
        for t in texts:
            passage = (t.get("text", "") if isinstance(t, dict) else str(t))[:512]
            pairs.append((query, passage))
            items.append(("text", t))
        for tbl in tables:
            passage = (tbl.get("markdown") or tbl.get("html") or tbl.get("text", ""))[:512] if isinstance(tbl, dict) else str(tbl)[:512]
            pairs.append((query, passage))
            items.append(("table", tbl))
        scores   = ce.predict(pairs)
        ranked   = sorted(zip(scores, items), key=lambda x: -x[0])
        top_items= [item for _, item in ranked[:top_k]]
        new_texts  = [el for mod, el in top_items if mod == "text"]
        new_tables = [el for mod, el in top_items if mod == "table"]
        return new_texts, new_tables
    except Exception:
        return texts, tables


def query_rag(
    question: str,
    images: list,
    pdf_hash_val: str,
    docstore_backing: dict,
    embedding_key: str,
    google_key: str,
    answer_model: str,
    k: int = 6, k_raw: int = 3, k_clip: int = 2,
    use_raw: bool = True, use_clip: bool = True,
    history: list | None = None,
    use_hyde: bool = False,
    use_rerank: bool = False,
    rerank_top_k: int = 6,
) -> tuple[Generator, str, dict, dict]:
    """
    Full 3-pipeline RAG query.
    Returns (answer_stream, model_used, sources_dict, pipeline_stats).
    """
    retriever, raw_retriever = _build_retriever(docstore_backing, embedding_key)

    query_text = question

    # ── Optional: HyDE — expand query via hypothetical document ──────────
    if use_hyde and google_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import HumanMessage as HM
            hyde_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=google_key, temperature=0.7)
            hyde_resp = hyde_llm.invoke([HM(content=f"Write a short hypothetical passage that would answer: {question}")])
            query_text = hyde_resp.content
        except Exception:
            query_text = question  # fall back silently

    # ── Pipeline A: summary-based ─────────────────────────────────────────
    retriever.search_kwargs = {"k": k}
    raw_a = retriever.invoke(query_text)
    ctx_a = classify_docs(raw_a)

    # ── Pipeline B: raw atomic ────────────────────────────────────────────
    ctx_b = {"texts": [], "tables": [], "images": []}
    if use_raw:
        try:
            raw_retriever.search_kwargs = {"k": k_raw}
            raw_b = raw_retriever.invoke(query_text)
            ctx_b = classify_docs(raw_b)
        except Exception:
            pass

    # ── Pipeline C: CLIP visual ───────────────────────────────────────────
    clip_imgs = []
    if use_clip and images:
        try:
            clip_imgs = retrieve_by_clip(query_text, k_clip, images, pdf_hash_val)
        except Exception:
            pass

    # ── Merge + deduplicate ───────────────────────────────────────────────
    seen = set()
    merged_texts  = _dedup(ctx_a["texts"]  + ctx_b["texts"],  seen)
    merged_tables = _dedup(ctx_a["tables"] + ctx_b["tables"], seen)
    merged_images = _dedup(ctx_a["images"] + ctx_b["images"] + clip_imgs, seen)

    # ── Optional: cross-encoder reranking ────────────────────────────────
    if use_rerank:
        merged_texts, merged_tables = _rerank_docs_app(
            question, merged_texts, merged_tables, top_k=rerank_top_k
        )

    context = {
        "texts":  merged_texts,
        "tables": merged_tables,
        "images": merged_images,
    }

    stats = {
        "A_texts": len(ctx_a["texts"]), "A_tables": len(ctx_a["tables"]), "A_images": len(ctx_a["images"]),
        "B_texts": len(ctx_b["texts"]), "B_tables": len(ctx_b["tables"]),
        "C_images": len(clip_imgs),
        "total_texts": len(context["texts"]), "total_tables": len(context["tables"]), "total_images": len(context["images"]),
        "hyde_used":    use_hyde and query_text != question,
        "rerank_used":  use_rerank,
    }

    messages     = build_rag_prompt(context, question, history=history)
    stream, model = _stream_gemini(messages, google_key, answer_model)
    return stream, model, context, stats


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FULL PIPELINE RUNNER  (called on PDF upload + Start Processing click)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_full_pipeline(pdf_path: str, pdf_hash_val: str, status_container) -> bool:
    """
    Run all pipeline steps with live Streamlit status updates.
    Returns True on success, False on error.
    """
    cache       = _disk_cache()
    google_key  = st.session_state.google_api_key
    groq_key    = st.session_state.groq_api_key
    emb_key     = st.session_state.embedding_model

    log_box = status_container.empty()
    prog    = status_container.progress(0, "Starting…")

    def _log(msg):
        log_box.markdown(f"⏳ {msg}")

    try:
        # Step 1: Docling extraction
        _log("Step 1/7 — PDF extraction with Docling…")
        prog.progress(5, "Extracting PDF elements…")
        docling_key = _ck("docling_v1", pdf_hash_val)
        if docling_key in cache:
            _log("Step 1/7 — Loaded from cache ✓ (skipping PDF parse)")
            extracted = cache[docling_key]
        else:
            extracted = run_docling_extraction(pdf_path, log_fn=_log)
            cache[docling_key] = extracted
        prog.progress(18)

        texts    = extracted["texts"]
        tables   = extracted["tables"]
        images   = extracted["images"]
        formulas = extracted["formulas"]
        forms    = extracted["forms"]

        # Merge formulas + forms into texts pool
        all_texts = texts + [
            {"text": f["text"], "page": f["page"], "label": "formula", "heading": False}
            for f in formulas
        ] + [
            {"text": f["text"], "page": f["page"], "label": f["label"], "heading": False}
            for f in forms
        ]

        # Step 2: Text + Table summarization
        _log(f"Step 2/7 — Summarizing {len(all_texts)} text chunks with Groq…")
        prog.progress(20, "Summarizing text & tables…")
        text_summaries, table_summaries = summarize_texts_and_tables(
            all_texts, tables, pdf_hash_val, groq_key, log_fn=_log
        )
        prog.progress(45)

        # Step 3: Image summarization
        _log(f"Step 3/7 — Summarizing {len(images)} images with Gemini Vision…")
        prog.progress(47, "Describing images…")
        image_summaries = summarize_images(images, pdf_hash_val, google_key, log_fn=_log)
        prog.progress(62)

        # Step 4: CLIP embeddings
        _log("Step 4/7 — CLIP visual embeddings…")
        prog.progress(63, "Building CLIP index…")
        build_clip_index(images, pdf_hash_val, log_fn=_log)
        prog.progress(70)

        # Step 5: ChromaDB multi-vector indexing
        _log("Step 5/7 — Building ChromaDB multi-vector indexes…")
        prog.progress(72, "Indexing into ChromaDB…")
        text_ids, table_ids, image_ids, backing = build_all_indexes(
            all_texts, tables, images, formulas, forms,
            text_summaries, table_summaries, image_summaries,
            pdf_hash_val, emb_key, log_fn=_log,
        )
        prog.progress(88)

        # Step 6: pdfplumber supplemental tables
        _log("Step 6/7 — Extracting supplemental tables with pdfplumber…")
        prog.progress(89, "Supplemental tables…")
        extra_tables = extract_pdfplumber_tables(pdf_path)
        if extra_tables:
            index_pdfplumber_tables(extra_tables, table_ids, pdf_hash_val, groq_key, emb_key, log_fn=_log)
        prog.progress(95)

        # Step 7: Save to session state
        _log("Step 7/7 — Finalizing…")
        st.session_state.texts          = all_texts
        st.session_state.tables         = tables
        st.session_state.images         = images
        st.session_state.formulas       = formulas
        st.session_state.forms          = forms
        st.session_state.text_summaries = text_summaries
        st.session_state.table_summaries= table_summaries
        st.session_state.image_summaries= image_summaries
        st.session_state.text_ids       = text_ids
        st.session_state.table_ids      = table_ids
        st.session_state.image_ids      = image_ids
        st.session_state.docstore_backing = backing
        st.session_state.pdf_meta       = extracted["metadata"]
        st.session_state.pdf_pages      = extracted["metadata"].get("pages", 0)

        prog.progress(100, "Complete ✓")
        log_box.markdown("✅ All done! Document is ready for questions.")
        return True

    except Exception as exc:
        st.session_state.processing_error = str(exc)
        status_container.error(f"Pipeline failed: {exc}")
        return False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UI RENDERERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _badge(label: str, size: str = "12px") -> str:
    color = LABEL_COLORS.get(label.lower(), "#555")
    return (f'<span style="background:{color};color:white;font-size:{size};'
            f'padding:2px 8px;border-radius:10px;font-weight:600">{label.upper()}</span>')


def render_sources(sources: dict, src_offset: int = 1) -> int:
    """Render all sources with sequential numbering. Returns next source number."""
    n = src_offset

    if sources.get("texts"):
        st.markdown("**📝 Text Passages**")
        for t in sources["texts"]:
            page  = t.get("page","?") if isinstance(t, dict) else "?"
            label = t.get("label","text") if isinstance(t, dict) else "text"
            text  = t.get("text","") if isinstance(t, dict) else str(t)
            color = LABEL_COLORS.get(label,"#555")
            with st.expander(f"[{n}] {label.upper()} · Page {page}", expanded=False):
                st.markdown(f'<span style="background:{color};color:white;padding:2px 8px;border-radius:10px;font-size:11px">{label.upper()}</span> **Page {page}**', unsafe_allow_html=True)
                st.text(text[:1500] + ("…" if len(text) > 1500 else ""))
            n += 1

    if sources.get("tables"):
        st.markdown("**📊 Tables**")
        for tbl in sources["tables"]:
            page    = tbl.get("page","?") if isinstance(tbl, dict) else "?"
            caption = tbl.get("caption","") if isinstance(tbl, dict) else ""
            with st.expander(f"[{n}] TABLE · Page {page}" + (f" — {caption[:60]}" if caption else ""), expanded=False):
                st.markdown(f'<span style="background:#2e7d32;color:white;padding:2px 8px;border-radius:10px;font-size:11px">TABLE</span> **Page {page}**', unsafe_allow_html=True)
                if caption:
                    st.caption(caption)
                md = tbl.get("markdown","") if isinstance(tbl, dict) else ""
                if md:
                    try:
                        st.markdown(md)
                    except Exception:
                        st.code(md)
                else:
                    html = tbl.get("html","") if isinstance(tbl, dict) else ""
                    if html:
                        st.markdown(html, unsafe_allow_html=True)
            n += 1

    if sources.get("images"):
        st.markdown("**🖼️ Figures / Charts**")
        for img in sources["images"]:
            page    = img.get("page","?") if isinstance(img, dict) else "?"
            label   = img.get("label","figure") if isinstance(img, dict) else "figure"
            caption = img.get("caption","") if isinstance(img, dict) else ""
            with st.expander(f"[{n}] {label.upper()} · Page {page}" + (f" — {caption[:50]}" if caption else ""), expanded=True):
                st.markdown(f'<span style="background:#6a1b9a;color:white;padding:2px 8px;border-radius:10px;font-size:11px">{label.upper()}</span> **Page {page}**', unsafe_allow_html=True)
                if caption:
                    st.caption(caption)
                b64 = img.get("b64","") if isinstance(img, dict) else ""
                if b64:
                    st.image(base64.b64decode(b64), use_container_width=True)
            n += 1

    return n


def render_pipeline_stats(stats: dict):
    cols = st.columns(6)
    labels = [
        ("A texts", stats.get("A_texts",0), "#007acc"),
        ("A tables", stats.get("A_tables",0), "#2e7d32"),
        ("A images", stats.get("A_images",0), "#6a1b9a"),
        ("B tables", stats.get("B_tables",0), "#ad1457"),
        ("C images", stats.get("C_images",0), "#00695c"),
        ("HyDE", "✓" if stats.get("hyde_used") else "✗", "#795548"),
    ]
    for col, (label, val, color) in zip(cols, labels):
        col.markdown(
            f'<div style="background:#f5f5f5;border-left:4px solid {color};padding:8px;border-radius:4px;text-align:center">'
            f'<div style="font-size:18px;font-weight:700">{val}</div>'
            f'<div style="font-size:11px;color:#555">{label}</div></div>',
            unsafe_allow_html=True
        )


def build_html_export(chat_history: list) -> str:
    import html as _html, re as _re, datetime as _dt

    def _md2html(text: str) -> str:
        text = _html.escape(str(text))
        text = _re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)
        text = _re.sub(r"`(.*?)`",        r"<code>\1</code>",    text)
        text = _re.sub(r"^#{1,3} (.+)$",  r"<h4>\1</h4>",       text, flags=_re.MULTILINE)
        text = _re.sub(r"^[-*] (.+)$",    r"<li>\1</li>",       text, flags=_re.MULTILINE)
        text = text.replace("\n\n", "<br><br>").replace("\n", "<br>")
        return text

    rows = ""
    for i, msg in enumerate(chat_history, 1):
        q      = _html.escape(msg.get('question',''))
        a_html = _md2html(msg.get('answer',''))
        model  = msg.get('model_used','?')
        stats  = msg.get('stats', {})
        flags  = []
        if stats.get('hyde_used'):   flags.append('HyDE')
        if stats.get('rerank_used'): flags.append('Rerank')
        hist   = msg.get('history_used', False)
        if hist: flags.append('History')
        flag_str = ' | '.join(flags)

        # Build numbered sources block
        srcs       = msg.get('sources', {})
        src_blocks = ""
        src_n      = 1
        for t in srcs.get('texts', []):
            page  = t.get('page','?') if isinstance(t, dict) else '?'
            label = (t.get('label','text') if isinstance(t, dict) else 'text').upper()
            text  = (t.get('text','') if isinstance(t, dict) else str(t))[:300]
            src_blocks += (f'<div class="src-item"><small>[{src_n}] <b>{label}</b> · p.{page}</small><br>'
                           f'<span class="src-text">{_html.escape(text)}…</span></div>')
            src_n += 1
        for tbl in srcs.get('tables', []):
            page = tbl.get('page','?') if isinstance(tbl, dict) else '?'
            cap  = (tbl.get('caption','') if isinstance(tbl, dict) else '')[:60]
            src_blocks += (f'<div class="src-item"><small>[{src_n}] <b>TABLE</b> · p.{page}'
                           f'{" — "+cap if cap else ""}</small></div>')
            src_n += 1
        for img in srcs.get('images', []):
            page = img.get('page','?') if isinstance(img, dict) else '?'
            lbl  = (img.get('label','figure') if isinstance(img, dict) else 'figure').upper()
            src_blocks += f'<div class="src-item"><small>[{src_n}] <b>{lbl}</b> · p.{page}</small></div>'
            src_n += 1

        pipeline_note = (
            f"A:{stats.get('A_texts',0)}t/{stats.get('A_tables',0)}tbl "
            f"B:{stats.get('B_tables',0)}tbl C:{stats.get('C_images',0)}img"
        ) if stats else ""

        rows += f"""
        <div class="turn">
          <div class="turn-hdr">Turn {i}  <small style='font-weight:400;color:#888'>{flag_str}</small></div>
          <div class="msg user"><span class="pill upill">Q</span>
            <span class="msg-body">{q}</span></div>
          <div class="msg asst"><span class="pill apill">A <small>{model}</small></span>
            <div class="msg-body">{a_html}</div></div>
          {'<details><summary style="cursor:pointer;font-size:12px;color:#666;margin:6px 0">' + str(src_n-1) + ' sources  (' + pipeline_note + ')</summary><div class="srcs">' + src_blocks + '</div></details>' if src_blocks else ''}
        </div>"""

    ts = _dt.datetime.now().strftime('%Y-%m-%d %H:%M')
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Multimodal RAG — Conversation</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:920px;margin:40px auto;background:#f8f9fa;color:#1a1a2e;padding:0 20px;line-height:1.6}}
h1{{border-bottom:2px solid #1976d2;padding-bottom:8px}}
.meta{{color:#888;font-size:.85rem;margin-bottom:24px}}
.turn{{background:white;border-radius:10px;padding:16px;margin-bottom:20px;box-shadow:0 1px 4px rgba(0,0,0,.1)}}
.turn-hdr{{font-size:.7rem;font-weight:700;color:#aaa;text-transform:uppercase;margin-bottom:10px}}
.msg{{display:flex;gap:10px;margin:6px 0;align-items:flex-start}}
.pill{{min-width:28px;height:24px;border-radius:12px;font-size:11px;font-weight:700;display:flex;align-items:center;justify-content:center;color:white;flex-shrink:0;padding:0 8px}}
.upill{{background:#1976d2}}.apill{{background:#388e3c}}
.msg-body{{flex:1}}
code{{background:#f0f0f0;padding:1px 4px;border-radius:3px;font-size:.9em}}
h4{{margin:6px 0}}
.src-item{{padding:4px 8px;margin:3px 0;border-left:3px solid #ccc;background:#fafafa;font-size:.85rem}}
.src-text{{color:#555}}
.srcs{{padding:6px 0}}
</style></head>
<body>
<h1>🔬 Multimodal RAG — Conversation Export</h1>
<div class="meta">Exported: {ts} &nbsp;|&nbsp; Turns: {len(chat_history)}</div>
{rows}
</body></html>"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("""
<style>
.main-hdr {font-size:2rem;font-weight:800;color:#1a1a2e;letter-spacing:-0.5px}
.sub-hdr  {font-size:1rem;color:#666;margin-bottom:1.5rem}
.pipeline-pill {display:inline-block;padding:3px 12px;border-radius:14px;
                font-size:12px;font-weight:700;margin:2px;color:white}
.source-card {border:1px solid #e0e0e0;border-radius:8px;padding:10px;margin:4px 0;background:#fafafa}
div[data-testid="stChatMessage"] .stMarkdown p {margin:0.25rem 0}
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ── SIDEBAR ──────────────────────────────────────────────────────────────────
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with st.sidebar:
    st.markdown("### 🔬 Multimodal RAG")
    st.caption("3-Pipeline Knowledge Retrieval")
    st.divider()

    # ── API Keys ───────────────────────────────────────────────────────────
    _keys_expanded = not st.session_state.api_keys_valid
    with st.expander("🔑 API Keys", expanded=_keys_expanded):
        # Auto-load from .env (never override user-entered values)
        if not st.session_state.api_keys_valid:
            try:
                from dotenv import load_dotenv
                load_dotenv(".env", override=False)
                _gk = os.getenv("GOOGLE_API_KEY","")
                _qk = os.getenv("GROQ_API_KEY","")
                _hk = os.getenv("HF_TOKEN","")
                if _gk: st.session_state.google_api_key = st.session_state.google_api_key or _gk
                if _qk: st.session_state.groq_api_key   = st.session_state.groq_api_key   or _qk
                if _hk: st.session_state.hf_token        = st.session_state.hf_token       or _hk
                if all([_gk, _qk, _hk]):
                    for k, v in [("GOOGLE_API_KEY",_gk),("GROQ_API_KEY",_qk),("HF_TOKEN",_hk),("HUGGINGFACEHUB_API_TOKEN",_hk)]:
                        os.environ[k] = v
                    st.session_state.api_keys_valid = True
            except Exception:
                pass

        _g = st.text_input("Google API Key (Gemini)", value=st.session_state.google_api_key, type="password",  key="_g_inp", help="https://aistudio.google.com/apikey")
        _q = st.text_input("Groq API Key",            value=st.session_state.groq_api_key,   type="password",  key="_q_inp", help="https://console.groq.com/keys")
        _h = st.text_input("HuggingFace Token",       value=st.session_state.hf_token,       type="password",  key="_h_inp", help="https://huggingface.co/settings/tokens")

        if st.button("💾 Save Keys", use_container_width=True):
            st.session_state.google_api_key = _g
            st.session_state.groq_api_key   = _q
            st.session_state.hf_token       = _h
            for k, v in [("GOOGLE_API_KEY",_g),("GROQ_API_KEY",_q),("HF_TOKEN",_h),("HUGGINGFACEHUB_API_TOKEN",_h)]:
                os.environ[k] = v
            missing = [n for n, v in [("Google",_g),("Groq",_q),("HuggingFace",_h)] if not v]
            if not missing:
                st.session_state.api_keys_valid = True
                st.success("Keys saved ✓")
                st.rerun()
            else:
                st.warning(f"Missing: {', '.join(missing)}")

        if st.session_state.api_keys_valid:
            st.success("All keys loaded ✓")

    # ── Model Settings ─────────────────────────────────────────────────────
    with st.expander("🤖 Model Settings", expanded=False):
        emb_choice = st.selectbox(
            "Embedding Model",
            options=list(EMBEDDING_CONFIGS.keys()),
            format_func=lambda k: EMBEDDING_CONFIGS[k]["desc"],
            index=list(EMBEDDING_CONFIGS.keys()).index(st.session_state.embedding_model),
        )
        if emb_choice != st.session_state.embedding_model:
            st.session_state.embedding_model = emb_choice
            st.warning("Embedding model changed — new documents will use this model. Existing index unaffected.")

        ans_idx = ANSWER_MODELS.index(st.session_state.answer_model) if st.session_state.answer_model in ANSWER_MODELS else 0
        st.session_state.answer_model = st.selectbox(
            "Answer Model (Gemini)", ANSWER_MODELS, index=ans_idx,
            help="Auto-fallback: if this model is rate-limited, next model in list is tried automatically",
        )

    # ── Retrieval Settings ─────────────────────────────────────────────────
    with st.expander("⚙️ Retrieval Settings", expanded=False):
        st.session_state.k_summary  = st.slider("Pipeline A — summary k",   2, 12, st.session_state.k_summary)
        st.session_state.k_raw      = st.slider("Pipeline B — raw atomic k", 1, 8,  st.session_state.k_raw)
        st.session_state.k_clip     = st.slider("Pipeline C — CLIP visual k",1, 6,  st.session_state.k_clip)
        st.divider()
        st.session_state.use_raw     = st.toggle("Enable raw atomic pipeline (B)",  st.session_state.use_raw,    help="Exact table markdown + formulas")
        st.session_state.use_clip    = st.toggle("Enable CLIP visual pipeline (C)",  st.session_state.use_clip,   help="Text→image cosine similarity")
        st.session_state.use_history = st.toggle("Include conversation history",     st.session_state.use_history, help="Last 3 Q&A turns used as context")
        st.session_state.use_hyde    = st.toggle("HyDE query expansion",             st.session_state.use_hyde,   help="Generate hypothetical answer first, use it for retrieval (may improve recall)")
        st.session_state.use_rerank  = st.toggle("Cross-encoder reranking",          st.session_state.use_rerank, help="Rerank retrieved chunks with cross-encoder/ms-marco-MiniLM-L-6-v2 (better precision, ~200ms overhead)")
        if st.session_state.use_rerank:
            st.session_state.rerank_top_k = st.slider("Rerank — keep top k", 2, 12, st.session_state.rerank_top_k)

    # ── System stats ───────────────────────────────────────────────────────
    if st.session_state.rag_ready:
        st.divider()
        with st.expander("📊 Index Stats", expanded=False):
            cache    = _disk_cache()
            used_mb  = cache.volume() / 1024**2
            emb_key  = st.session_state.embedding_model
            try:
                from langchain_chroma import Chroma
                emb = _load_embedding_model(emb_key)
                vs  = Chroma(collection_name=f"rag_{emb_key}", embedding_function=emb, persist_directory=CHROMA_DIR)
                vs_raw = Chroma(collection_name=f"rag_raw_{emb_key}", embedding_function=emb, persist_directory=CHROMA_DIR)
                clip_col = _get_clip_col(st.session_state.pdf_hash)
                st.metric("Summary vectors",    vs._collection.count())
                st.metric("Raw atomic vectors", vs_raw._collection.count())
                st.metric("CLIP image vectors", clip_col.count())
            except Exception:
                pass
            st.metric("Cache size",  f"{used_mb:.1f} MB")
            st.metric("Text chunks", len(st.session_state.texts))
            st.metric("Tables",      len(st.session_state.tables))
            st.metric("Images",      len(st.session_state.images))
            st.caption(f"Hash: `{st.session_state.pdf_hash}`")
            st.caption(f"Pages: {st.session_state.pdf_pages}")

            if st.button("🗑️ Clear cache for this PDF", use_container_width=True):
                ph = st.session_state.pdf_hash
                for k in list(cache):
                    if ph in str(k): del cache[k]
                st.success("Cache cleared. Upload the PDF again to re-process.")

        st.divider()
        if st.button("🔄 Upload New PDF", use_container_width=True):
            for k in ["rag_ready","pdf_path","pdf_hash","pdf_name","texts","tables","images",
                      "formulas","forms","text_summaries","table_summaries","image_summaries",
                      "text_ids","table_ids","image_ids","docstore_backing","chat_history"]:
                st.session_state[k] = _SS_DEFAULTS[k]
            st.rerun()

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        if st.session_state.chat_history:
            html_export = build_html_export(st.session_state.chat_history)
            st.download_button(
                "⬇️ Download Conversation",
                data=html_export,
                file_name=f"rag_conversation_{st.session_state.pdf_name or 'export'}.html",
                mime="text/html",
                use_container_width=True,
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ── MAIN AREA ────────────────────────────────────────────────────────────────
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown('<div class="main-hdr">🔬 Context-Aware Multimodal RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-hdr">'
    '<span class="pipeline-pill" style="background:#007acc">A · LLM Summary</span>'
    '<span class="pipeline-pill" style="background:#2e7d32">B · Raw Atomic</span>'
    '<span class="pipeline-pill" style="background:#6a1b9a">C · CLIP Visual</span>'
    '&nbsp;&nbsp;·&nbsp;&nbsp;Streaming answers · Inline citations · Multi-turn chat</div>',
    unsafe_allow_html=True,
)

# ── Gate: API keys required ───────────────────────────────────────────────────
if not st.session_state.api_keys_valid:
    st.info("👈 Enter your API keys in the sidebar to get started.")
    col1, col2, col3 = st.columns(3)
    col1.markdown("**Google Gemini API**\nImage summarization + answers\n[→ Get key](https://aistudio.google.com/apikey)")
    col2.markdown("**Groq API**\nFast text/table summarization\n[→ Get key](https://console.groq.com/keys)")
    col3.markdown("**HuggingFace Token**\nEmbedding model download\n[→ Get token](https://huggingface.co/settings/tokens)")
    st.stop()


# ── PDF Upload Screen ─────────────────────────────────────────────────────────
if not st.session_state.rag_ready:
    st.markdown("### 📄 Upload a PDF Document")
    col_up, col_info = st.columns([3, 2])

    with col_up:
        uploaded = st.file_uploader(
            "Drag and drop a PDF (research paper, report, technical document)",
            type="pdf",
            help="Max recommended size: ~30 MB. First processing takes 2-5 minutes. Subsequent runs are instant (cached).",
        )

    with col_info:
        st.markdown("""
        **Processing pipeline:**
        1. 📖 Docling extraction — text, tables, figures
        2. ✍️ Groq LLaMA 3.3 — text & table summaries
        3. 👁️ Gemini Vision — image descriptions
        4. 🎨 CLIP ViT-B-32 — visual embeddings
        5. 📊 ChromaDB — 3 index collections
        6. 🔍 pdfplumber — supplemental tables

        *Everything is cached to disk — re-uploads are instant.*
        """)

    if uploaded:
        pdf_dest     = CONTENT_DIR / uploaded.name
        pdf_dest.write_bytes(uploaded.getvalue())
        pdf_hash_val = compute_pdf_hash(str(pdf_dest))

        st.session_state.pdf_path = str(pdf_dest)
        st.session_state.pdf_hash = pdf_hash_val
        st.session_state.pdf_name = uploaded.name

        size_kb = pdf_dest.stat().st_size // 1024
        st.success(f"✅ Uploaded: **{uploaded.name}** ({size_kb} KB)  |  Hash: `{pdf_hash_val}`")

        # ── Check if already fully indexed ──────────────────────────────────
        cache        = _disk_cache()
        docstore_key = _ck("docstore_v1", pdf_hash_val, st.session_state.embedding_model)
        docling_key  = _ck("docling_v1",  pdf_hash_val)

        if docstore_key in cache and docling_key in cache:
            st.info("⚡ Document already processed — loading from cache…")
            ext = cache[docling_key]
            all_t = ext["texts"] + [
                {"text": f["text"], "page": f["page"], "label": "formula", "heading": False}
                for f in ext["formulas"]
            ] + [
                {"text": f["text"], "page": f["page"], "label": f["label"], "heading": False}
                for f in ext["forms"]
            ]
            st.session_state.texts          = all_t
            st.session_state.tables         = ext["tables"]
            st.session_state.images         = ext["images"]
            st.session_state.formulas       = ext["formulas"]
            st.session_state.forms          = ext["forms"]
            st.session_state.pdf_meta       = ext["metadata"]
            st.session_state.pdf_pages      = ext["metadata"].get("pages", 0)
            st.session_state.docstore_backing = dict(cache.get(docstore_key, {}))
            st.session_state.rag_ready      = True
            st.rerun()
        else:
            if st.button("🚀 Start Processing", type="primary", use_container_width=True):
                status_box = st.container()
                success    = run_full_pipeline(str(pdf_dest), pdf_hash_val, status_box)
                if success:
                    st.session_state.rag_ready = True
                    time.sleep(1)
                    st.rerun()
    st.stop()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# READY STATE — Tabs: Chat | Document Explorer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pdf_name = st.session_state.pdf_name or "Document"
st.markdown(f"**Document:** {pdf_name} &nbsp;|&nbsp; **Pages:** {st.session_state.pdf_pages} &nbsp;|&nbsp; **Model:** `{st.session_state.answer_model}`")
st.divider()

tab_chat, tab_explorer = st.tabs(["💬 Chat", "🔍 Document Explorer"])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1: CHAT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_chat:
    # ── Display chat history ───────────────────────────────────────────────
    for msg in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(msg["question"])

        with st.chat_message("assistant"):
            st.markdown(msg["answer"])
            # Compact pipeline stats
            s = msg.get("stats", {})
            if s:
                st.caption(
                    f"Pipeline A: {s.get('A_texts',0)}t/{s.get('A_tables',0)}tbl/{s.get('A_images',0)}img  "
                    f"| B: {s.get('B_tables',0)}tbl  "
                    f"| C: {s.get('C_images',0)}img  "
                    f"| Model: `{msg.get('model_used','?')}`"
                    + ("  | HyDE ✓" if s.get("hyde_used") else "")
                )
            # Sources collapsible
            srcs = msg.get("sources", {})
            total_srcs = len(srcs.get("texts",[])) + len(srcs.get("tables",[])) + len(srcs.get("images",[]))
            if total_srcs > 0:
                with st.expander(f"📌 {total_srcs} Source(s) used", expanded=False):
                    render_sources(srcs)

    # ── Chat input ─────────────────────────────────────────────────────────
    question = st.chat_input("Ask anything about the document…  (Ctrl+Enter to send)")

    if question:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            # Build history for context
            history = st.session_state.chat_history if st.session_state.use_history else None

            with st.spinner("Retrieving from 3 pipelines…"):
                answer_stream, model_used, sources, stats = query_rag(
                    question       = question,
                    images         = st.session_state.images,
                    pdf_hash_val   = st.session_state.pdf_hash,
                    docstore_backing = st.session_state.docstore_backing,
                    embedding_key  = st.session_state.embedding_model,
                    google_key     = st.session_state.google_api_key,
                    answer_model   = st.session_state.answer_model,
                    k              = st.session_state.k_summary,
                    k_raw          = st.session_state.k_raw,
                    k_clip         = st.session_state.k_clip,
                    use_raw        = st.session_state.use_raw,
                    use_clip       = st.session_state.use_clip,
                    history        = history,
                    use_hyde       = st.session_state.use_hyde,
                )

            # ── Stream the answer ──────────────────────────────────────────
            st.caption(f"Generating with `{model_used}`…")
            answer = st.write_stream(answer_stream)

            # ── Pipeline contribution stats ────────────────────────────────
            render_pipeline_stats(stats)

            # ── Sources ────────────────────────────────────────────────────
            total_srcs = len(sources.get("texts",[])) + len(sources.get("tables",[])) + len(sources.get("images",[]))
            with st.expander(f"📌 {total_srcs} Source(s) used", expanded=False):
                render_sources(sources)

        # Save to chat history
        st.session_state.chat_history.append({
            "question":   question,
            "answer":     answer,
            "sources":    sources,
            "model_used": model_used,
            "stats":      stats,
        })


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2: DOCUMENT EXPLORER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_explorer:
    exp_tab_text, exp_tab_tables, exp_tab_images, exp_tab_meta = st.tabs(
        [f"📝 Text ({len(st.session_state.texts)})",
         f"📊 Tables ({len(st.session_state.tables)})",
         f"🖼️ Images ({len(st.session_state.images)})",
         "📋 Metadata"]
    )

    with exp_tab_text:
        if st.session_state.texts:
            col_filter, col_pg = st.columns([3, 1])
            with col_filter:
                search_txt = st.text_input("🔍 Filter text chunks", placeholder="Type to filter…", key="txt_search")
            with col_pg:
                pg_filter = st.number_input("Page filter (0=all)", min_value=0, value=0, key="txt_pg")

            filtered_texts = [
                t for t in st.session_state.texts
                if (not search_txt or search_txt.lower() in t.get("text","").lower())
                and (not pg_filter or t.get("page") == pg_filter)
            ]
            st.caption(f"Showing {len(filtered_texts)} of {len(st.session_state.texts)} chunks")

            for i, t in enumerate(filtered_texts[:100]):
                page  = t.get("page","?")
                label = t.get("label","text")
                text  = t.get("text","")
                color = LABEL_COLORS.get(label,"#555")
                with st.expander(f"{label.upper()} · Page {page} | {text[:80]}…", expanded=False):
                    st.markdown(f'<span style="background:{color};color:white;padding:2px 8px;border-radius:10px;font-size:11px">{label.upper()}</span> **Page {page}**', unsafe_allow_html=True)
                    st.text(text)
                    if st.session_state.text_summaries and i < len(st.session_state.text_summaries):
                        st.caption("**LLM Summary:**")
                        st.info(st.session_state.text_summaries[i])

            if len(filtered_texts) > 100:
                st.caption(f"…and {len(filtered_texts)-100} more (apply filter to narrow results)")
        else:
            st.info("No text chunks extracted.")

    with exp_tab_tables:
        if st.session_state.tables:
            for i, tbl in enumerate(st.session_state.tables):
                page    = tbl.get("page","?")
                caption = tbl.get("caption","")
                label   = f"Table {i+1} · Page {page}" + (f" — {caption[:60]}" if caption else "")
                with st.expander(label, expanded=i == 0):
                    if caption:
                        st.caption(caption)
                    md = tbl.get("markdown","")
                    if md:
                        try:
                            st.markdown(md)
                        except Exception:
                            st.code(md)
                    else:
                        html = tbl.get("html","")
                        if html:
                            st.markdown(html, unsafe_allow_html=True)
                    if st.session_state.table_summaries and i < len(st.session_state.table_summaries):
                        st.caption("**LLM Summary:**")
                        st.info(st.session_state.table_summaries[i])
        else:
            st.info("No tables extracted.")

    with exp_tab_images:
        if st.session_state.images:
            cols_per_row = 2
            img_list = st.session_state.images
            for row_start in range(0, len(img_list), cols_per_row):
                cols = st.columns(cols_per_row)
                for col, img in zip(cols, img_list[row_start:row_start + cols_per_row]):
                    with col:
                        page    = img.get("page","?")
                        label   = img.get("label","picture")
                        caption = img.get("caption","")
                        b64     = img.get("b64","")
                        color   = LABEL_COLORS.get(label,"#555")
                        st.markdown(f'<span style="background:{color};color:white;padding:2px 8px;border-radius:10px;font-size:11px">{label.upper()}</span> **Page {page}**', unsafe_allow_html=True)
                        if b64:
                            st.image(base64.b64decode(b64), use_container_width=True)
                        if caption:
                            st.caption(caption)
                        # Show image summary if available
                        img_idx = img_list.index(img)
                        if st.session_state.image_summaries and img_idx < len(st.session_state.image_summaries):
                            with st.expander("Vision description", expanded=False):
                                st.write(st.session_state.image_summaries[img_idx])
        else:
            st.info("No images extracted.")

    with exp_tab_meta:
        meta = st.session_state.pdf_meta
        if meta:
            st.markdown("**Document Metadata**")
            col1, col2 = st.columns(2)
            col1.metric("Title",   meta.get("title","—"))
            col1.metric("Author",  meta.get("author","—") or "—")
            col2.metric("Pages",   meta.get("pages","—"))
            col2.metric("Source",  Path(meta.get("source","")).name if meta.get("source") else "—")

            toc = meta.get("toc", [])
            if toc:
                st.markdown("**Table of Contents**")
                for entry in toc:
                    level, title, page = entry
                    indent = "&nbsp;" * (4 * (level - 1))
                    st.markdown(f"{indent}{'#'*level} {title} *(p.{page})*", unsafe_allow_html=True)
        else:
            st.info("No metadata available.")

        # ── Cache info ──────────────────────────────────────────────────────
        st.divider()
        st.markdown("**Cache Contents**")
        cache    = _disk_cache()
        ph       = st.session_state.pdf_hash or ""
        pdf_keys = [str(k) for k in cache if ph and ph in str(k)]
        if pdf_keys:
            for k in sorted(pdf_keys)[:20]:
                size = sys.getsizeof(cache.get(k, 0))
                st.caption(f"`{k}` — ~{size//1024} KB")
        else:
            st.caption("No cache entries for this PDF yet.")
