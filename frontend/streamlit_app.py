import os
import sys
import asyncio
from typing import Optional, List
import httpx
import streamlit as st

# Ensure root directory is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import get_settings
from app.services.rag_service import RAGService
from app.models.query import QueryResponse, Citation

# Set Page Config (Centered, No Sidebar)
st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Clean, Modern CSS Styling (Hides sidebar and optimizes single-window look)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide the Streamlit sidebar completely */
    [data-testid="stSidebar"] {
        display: none !important;
    }
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    
    /* Clean container cards */
    .app-header {
        text-align: center;
        padding: 20px 0 10px 0;
        margin-bottom: 20px;
    }
    .app-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 6px;
    }
    .app-subtitle {
        font-size: 1rem;
        color: #94a3b8;
    }
    
    /* Badges */
    .badge-grounded {
        background-color: #064e3b;
        color: #34d399;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        border: 1px solid #059669;
        margin-bottom: 12px;
    }
    .badge-refusal {
        background-color: #451a03;
        color: #fbbf24;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        border: 1px solid #d97706;
        margin-bottom: 12px;
    }
    
    /* Citation Excerpt Card */
    .citation-box {
        background-color: #1e293b;
        border-left: 4px solid #14b8a6;
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 10px;
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_rag_engine() -> RAGService:
    """Returns singleton RAGService."""
    return RAGService()


def execute_query(query: str, rag_engine: RAGService) -> QueryResponse:
    """Executes query through FastAPI backend or local fallback."""
    api_url = "http://localhost:8000/api/v1/query"
    try:
        with httpx.Client(timeout=45.0) as client:
            resp = client.post(
                api_url,
                json={"query": query, "top_k": 5, "score_threshold": 0.35}
            )
            if resp.status_code == 200:
                data = resp.json()
                citations = [Citation(**c) for c in data.get("citations", [])]
                return QueryResponse(
                    query=data["query"],
                    answer=data["answer"],
                    citations=citations,
                    grounded=data.get("grounded", True),
                    is_refusal=data.get("is_refusal", False),
                    confidence_score=data.get("confidence_score", 0.0),
                    retrieved_chunks_count=data.get("retrieved_chunks_count", 0),
                    model_used=data.get("model_used", "gpt-4o-mini")
                )
    except Exception:
        pass

    # Direct fallback
    return asyncio.run(rag_engine.answer_query(query=query, top_k=5, score_threshold=0.35))


def execute_upload(file_bytes: bytes, filename: str, rag_engine: RAGService):
    """Uploads file through FastAPI or local fallback."""
    api_url = "http://localhost:8000/api/v1/documents/upload"
    try:
        with httpx.Client(timeout=60.0) as client:
            files = {"file": (filename, file_bytes, "application/octet-stream")}
            resp = client.post(api_url, files=files)
            if resp.status_code == 201:
                return resp.json()
    except Exception:
        pass

    res = asyncio.run(rag_engine.ingest_file(file_bytes, filename))
    return {"chunks_created": res.chunks_created, "document_name": filename}


def main():
    rag_engine = get_rag_engine()

    # Session State
    if "qa_query" not in st.session_state:
        st.session_state.qa_query = ""
    if "qa_response" not in st.session_state:
        st.session_state.qa_response = None

    # Header
    st.markdown("""
    <div class="app-header">
        <div class="app-title">🧠 RAG Knowledge Assistant</div>
        <div class="app-subtitle">Upload documents, ask questions, and get answers strictly grounded with citations.</div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # STEP 1: UPLOAD DOCUMENTS
    # -------------------------------------------------------------------------
    st.markdown("### 1. Upload Documents")
    uploaded_file = st.file_uploader(
        "Upload PDF, TXT, or Markdown file",
        type=["pdf", "txt", "md"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        if st.button("📥 Index This Document", type="primary"):
            with st.spinner(f"Indexing '{uploaded_file.name}'..."):
                file_bytes = uploaded_file.read()
                res = execute_upload(file_bytes, uploaded_file.name, rag_engine)
                st.success(f"✅ Indexed `{uploaded_file.name}` ({res.get('chunks_created', 0)} chunks created)!")

    # Current Documents Summary
    docs = rag_engine.list_documents()
    if docs:
        doc_names = ", ".join([f"`{d.document_name}` ({d.chunk_count} chunks)" for d in docs])
        st.caption(f"📚 **Currently indexed:** {doc_names}")
    else:
        st.caption("ℹ️ No documents uploaded yet. Upload a document above or use existing documents.")

    st.divider()

    # -------------------------------------------------------------------------
    # STEP 2: ASK QUESTIONS
    # -------------------------------------------------------------------------
    st.markdown("### 2. Ask a Question")

    # Quick sample buttons
    sample_queries = [
        "What is the mandatory log retention period for Tier 1 High Risk AI systems?",
        "What are the replica count, memory allocation, and port numbers for the Qdrant cluster?",
        "What is the maximum initial response SLA for Severity 1 security incidents?",
        "What is the annual compensation salary band for Senior AI Engineers?"
    ]

    st.caption("Or click a sample question:")
    sample_cols = st.columns(len(sample_queries))
    preset_clicked = None
    for i, q in enumerate(sample_queries):
        short_label = f"Sample {i+1}"
        if sample_cols[i].button(short_label, help=q, key=f"sample_q_{i}"):
            preset_clicked = q

    # Question Input Form (Enter key submits automatically)
    with st.form(key="question_form", clear_on_submit=False):
        user_input = st.text_input(
            "Type your question:",
            value=preset_clicked if preset_clicked else st.session_state.qa_query,
            placeholder="e.g. How often must production service API keys be rotated?",
            label_visibility="collapsed"
        )
        submit_btn = st.form_submit_button("🔍 Get Answer", type="primary", use_container_width=True)

    query_to_run = None
    if preset_clicked:
        query_to_run = preset_clicked
        st.session_state.qa_query = preset_clicked
    elif submit_btn and user_input.strip():
        query_to_run = user_input.strip()
        st.session_state.qa_query = query_to_run

    if query_to_run:
        with st.spinner("Searching documents and generating grounded answer..."):
            st.session_state.qa_response = execute_query(query_to_run, rag_engine)

    # -------------------------------------------------------------------------
    # STEP 3: ANSWER & CITATIONS
    # -------------------------------------------------------------------------
    if st.session_state.qa_response is not None:
        resp = st.session_state.qa_response

        st.markdown("### 3. Answer & Evidence")

        # Grounding / Refusal Badge
        if resp.is_refusal:
            st.markdown("""
            <div class="badge-refusal">⚠️ INFORMATION UNAVAILABLE &bull; HONEST REFUSAL</div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="badge-grounded">✅ STRICTLY GROUNDED &bull; CONFIDENCE: {resp.confidence_score * 100:.1f}%</div>
            """, unsafe_allow_html=True)

        # Answer Box
        st.info(resp.answer)

        # Citations
        if resp.citations:
            st.markdown("#### 📚 Source Citations & Excerpts")
            for idx, cit in enumerate(resp.citations, start=1):
                page_str = f"Page {cit.page_number}" if cit.page_number else "N/A"
                with st.expander(f"📍 Citation [{idx}]: {cit.source} ({page_str}) &bull; Relevance: {cit.score:.3f}", expanded=(idx == 1)):
                    st.markdown(f"**Document:** `{cit.source}` &nbsp;|&nbsp; **{page_str}** &nbsp;|&nbsp; **Score:** `{cit.score:.4f}`")
                    st.markdown(f"> *\"{cit.excerpt}\"*")


if __name__ == "__main__":
    main()
