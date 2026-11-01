"""Streamlit User Interface for College Document Q&A Assistant.

Features:
- Professional college dashboard with system health indicators
- Multi-PDF document uploader and batch ingestion
- Real-time document management (list, inspect, delete individual documents, reset)
- Natural language question answering powered by local Ollama RAG
- Grounded citations (document name, page number) and expandable context inspection
- Interactive chat history with quick prompt suggestions and chat clear option
"""

from pathlib import Path
import time
import logging
import streamlit as st

from document_loader import load_pdf, load_directory, DocumentLoaderError
from chunker import split_text_into_chunks
from embeddings import OllamaEmbeddingService, OllamaEmbeddingError
from vector_store import ChromaVectorStore, VectorStoreError
from retriever import DocumentRetriever
from llm_service import OllamaLLMService, OllamaServiceError
from rag_service import RAGPipeline, FALLBACK_RESPONSE

# Configure root logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Cached Component Factories
# ---------------------------------------------------------------------------
@st.cache_resource
def get_embedding_service() -> OllamaEmbeddingService:
    """Instantiate and cache local embedding service."""
    return OllamaEmbeddingService(model_name="nomic-embed-text")


@st.cache_resource
def get_llm_service() -> OllamaLLMService:
    """Instantiate and cache local LLM service."""
    return OllamaLLMService(model_name="llama3.2:3b")


@st.cache_resource
def get_vector_store() -> ChromaVectorStore:
    """Instantiate and cache ChromaDB vector store."""
    embedding_svc = get_embedding_service()
    return ChromaVectorStore(
        persist_dir=CHROMA_DIR,
        collection_name="college_documents",
        embedding_service=embedding_svc,
    )


def get_rag_pipeline(top_k: int = 3) -> RAGPipeline:
    """Create RAG pipeline orchestrator with configured top_k."""
    store = get_vector_store()
    emb_svc = get_embedding_service()
    llm_svc = get_llm_service()
    retriever = DocumentRetriever(
        vector_store=store,
        embedding_service=emb_svc,
        top_k=top_k,
    )
    return RAGPipeline(retriever=retriever, llm_service=llm_svc)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def ingest_pdf_file(file_path: Path, store: ChromaVectorStore) -> int:
    """Ingest a single PDF file into ChromaDB."""
    pages = load_pdf(file_path, skip_empty=True)
    if not pages:
        return 0
    chunks = split_text_into_chunks(pages, chunk_size=1500, chunk_overlap=200)
    if not chunks:
        return 0
    return store.add_chunks(chunks, overwrite=True)


def load_preexisting_sample_documents() -> int:
    """Index sample college documents existing in data/ folder."""
    store = get_vector_store()
    sample_files = list(DATA_DIR.glob("*.pdf"))
    if not sample_files:
        return 0

    total_chunks = 0
    for pdf_path in sample_files:
        chunks_added = ingest_pdf_file(pdf_path, store)
        total_chunks += chunks_added
    return total_chunks


# ---------------------------------------------------------------------------
# Main Streamlit Application
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="College Document Q&A Assistant",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for polished college dashboard design
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.1rem;
            font-weight: 700;
            color: #1E3A8A;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.0rem;
            color: #4B5563;
            margin-bottom: 1.5rem;
        }
        .source-pill {
            display: inline-block;
            background-color: #EFF6FF;
            color: #1D4ED8;
            border: 1px solid #BFDBFE;
            border-radius: 14px;
            padding: 3px 12px;
            font-size: 0.82rem;
            font-weight: 600;
            margin-right: 6px;
            margin-bottom: 6px;
        }
        .metric-card {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 12px 16px;
            text-align: center;
        }
        .metric-number {
            font-size: 1.6rem;
            font-weight: 700;
            color: #1E3A8A;
        }
        .metric-label {
            font-size: 0.85rem;
            color: #64748B;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Initialize Session State
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""

    store = get_vector_store()
    emb_svc = get_embedding_service()
    llm_svc = get_llm_service()

    # -----------------------------------------------------------------------
    # Sidebar: Document Management & System Health
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.title("🎓 Knowledge Hub")
        st.caption("College Document Management")

        st.markdown("---")

        # System Health & Runtime Check
        st.subheader("🖥️ Local Runtime Status")
        ollama_online = llm_svc.is_service_available()
        if ollama_online:
            st.success("🟢 Ollama Daemon: Online", icon="✅")
            installed_models = llm_svc.list_installed_models()
            chat_ready = any("llama3.2:3b" in m.lower() for m in installed_models)
            emb_ready = any("nomic-embed-text" in m.lower() for m in installed_models)

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.caption(f"Chat: `{'Ready' if chat_ready else 'Pulling'}`")
            with col_s2:
                st.caption(f"Embed: `{'Ready' if emb_ready else 'Pulling'}`")
        else:
            st.warning("🟠 Ollama Daemon: Offline (Local Mode)")
            with st.expander("ℹ️ How to connect Ollama"):
                st.markdown(
                    """
                    1. Ensure Ollama is installed from [ollama.com](https://ollama.com).
                    2. Run in terminal:
                    ```powershell
                    ollama serve
                    ollama pull llama3.2:3b
                    ollama pull nomic-embed-text
                    ```
                    """
                )

        st.markdown("---")

        # Document Upload Section
        st.subheader("📤 Upload Documents")
        uploaded_files = st.file_uploader(
            "Select college PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload circulars, syllabi, fee structures, or campus regulations.",
        )

        if uploaded_files:
            if st.button("Process & Index Uploads", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                total_files = len(uploaded_files)

                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing ({idx+1}/{total_files}): {uploaded_file.name}")
                    target_path = DATA_DIR / uploaded_file.name
                    target_path.write_bytes(uploaded_file.getbuffer())

                    try:
                        chunks_added = ingest_pdf_file(target_path, store)
                        st.toast(f"✅ Indexed {uploaded_file.name} ({chunks_added} chunks)", icon="📄")
                    except Exception as exc:
                        st.error(f"Error processing {uploaded_file.name}: {exc}")

                    progress_bar.progress((idx + 1) / total_files)

                status_text.text("Ingestion completed successfully!")
                time.sleep(0.5)
                st.rerun()

        st.markdown("---")

        # Current Documents List & Management
        st.subheader("📚 Indexed Documents")
        indexed_docs = store.get_indexed_documents()

        if indexed_docs:
            for doc_name in indexed_docs:
                col_doc, col_del = st.columns([0.8, 0.2])
                with col_doc:
                    st.write(f"📄 `{doc_name}`")
                with col_del:
                    if st.button("🗑️", key=f"del_{doc_name}", help=f"Delete {doc_name}"):
                        deleted = store.delete_document(doc_name)
                        st.toast(f"Deleted {doc_name} ({deleted} chunks removed)", icon="🗑️")
                        st.rerun()
        else:
            st.info("No documents indexed yet.")
            if st.button("📥 Load Sample College Docs", use_container_width=True):
                with st.spinner("Indexing sample college documents..."):
                    loaded_count = load_preexisting_sample_documents()
                    st.success(f"Indexed {loaded_count} chunks from sample PDFs!")
                    st.rerun()

        st.markdown("---")

        # Configuration & Reset Controls
        st.subheader("⚙️ Settings")
        top_k = st.slider("Context chunks retrieved (Top-K)", min_value=1, max_value=5, value=3)

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col_c2:
            if st.button("Reset KB", use_container_width=True, help="Purge all indexed vectors"):
                store.reset()
                st.toast("Vector database reset.", icon="⚠️")
                st.rerun()

    # -----------------------------------------------------------------------
    # Main Panel: Dashboard & Q&A Workspace
    # -----------------------------------------------------------------------
    st.markdown('<div class="main-header">🎓 College Document Q&A Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        'Ask questions grounded strictly in official college regulations, syllabus documents, and guidelines. '
        'Powered by local RAG runtime.</div>',
        unsafe_allow_html=True,
    )

    # Dashboard Metrics Row
    stats = store.get_collection_stats()
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-number">{stats["total_documents"]}</div>'
            f'<div class="metric-label">Indexed Documents</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-number">{stats["total_chunks"]}</div>'
            f'<div class="metric-label">Knowledge Chunks</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            '<div class="metric-card"><div class="metric-number">Llama 3.2</div>'
            '<div class="metric-label">Local Chat Model (3B)</div></div>',
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            '<div class="metric-card"><div class="metric-number">Nomic</div>'
            '<div class="metric-label">Embeddings (Local)</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick Sample Query Chips (if knowledge base is populated)
    if stats["total_chunks"] > 0 and len(st.session_state.messages) == 0:
        st.markdown("##### 💡 Suggested Questions:")
        chip_col1, chip_col2, chip_col3 = st.columns(3)
        with chip_col1:
            if st.button("📋 What is the minimum 10+2 eligibility for admission?", use_container_width=True):
                st.session_state.last_query = "What is the minimum 10+2 eligibility for admission?"
        with chip_col2:
            if st.button("⏰ What are the hostel curfew hours on weekdays?", use_container_width=True):
                st.session_state.last_query = "What are the hostel curfew hours on weekdays?"
        with chip_col3:
            if st.button("📊 What is the minimum attendance requirement?", use_container_width=True):
                st.session_state.last_query = "What is the minimum attendance requirement?"

    # Empty State Notice
    if stats["total_chunks"] == 0:
        st.info(
            "📌 **Your knowledge base is empty!** "
            "Upload college PDF documents in the left sidebar or click **'Load Sample College Docs'** to get started.",
            icon="ℹ️",
        )

    # -----------------------------------------------------------------------
    # Render Conversation History
    # -----------------------------------------------------------------------
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Render citation sources if available
            sources = message.get("sources", [])
            if sources:
                st.markdown("**Citations:**")
                source_pills = "".join(
                    f'<span class="source-pill">📄 {s["source_filename"]} (p. {s["page_number"]})</span>'
                    for s in sources
                )
                st.markdown(source_pills, unsafe_allow_html=True)

            # Render expandable context chunks inspection
            chunks = message.get("context_chunks", [])
            if chunks:
                with st.expander("🔍 View Retrieved Document Passages (Evidence)"):
                    for idx, c in enumerate(chunks, start=1):
                        sim = c.get("similarity", 0.0)
                        st.markdown(
                            f"**Passage {idx}** — *{c.get('source_filename')} (Page {c.get('page_number')})* "
                            f"| *Similarity: {sim:.2%}*"
                        )
                        st.code(c.get("text", ""), language="text")

    # -----------------------------------------------------------------------
    # Question Input & Answering
    # -----------------------------------------------------------------------
    # Check if a suggested chip was clicked
    pending_prompt = st.session_state.get("last_query", "")
    user_input = st.chat_input("Ask a question about the college documents...") or pending_prompt

    # Reset last_query state
    if pending_prompt:
        st.session_state.last_query = ""

    if user_input:
        # 1. Display user query
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # 2. Execute RAG pipeline
        pipeline = get_rag_pipeline(top_k=top_k)

        with st.chat_message("assistant"):
            with st.spinner("Searching document knowledge base and generating grounded response..."):
                try:
                    result = pipeline.answer_query(user_input)
                    answer_text = result.get("answer", FALLBACK_RESPONSE)
                    sources = result.get("sources", [])
                    chunks = result.get("context_chunks", [])
                    found = result.get("found_in_documents", False)
                except Exception as exc:
                    answer_text = f"An error occurred while answering your question: {exc}"
                    sources = []
                    chunks = []
                    found = False

                st.markdown(answer_text)

                # Display Citations
                if sources:
                    st.markdown("**Citations:**")
                    source_pills = "".join(
                        f'<span class="source-pill">📄 {s["source_filename"]} (p. {s["page_number"]})</span>'
                        for s in sources
                    )
                    st.markdown(source_pills, unsafe_allow_html=True)

                # Display Evidence Passages
                if chunks:
                    with st.expander("🔍 View Retrieved Document Passages (Evidence)"):
                        for idx, c in enumerate(chunks, start=1):
                            sim = c.get("similarity", 0.0)
                            st.markdown(
                                f"**Passage {idx}** — *{c.get('source_filename')} (Page {c.get('page_number')})* "
                                f"| *Similarity: {sim:.2%}*"
                            )
                            st.code(c.get("text", ""), language="text")

                # Save response to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "sources": sources,
                    "context_chunks": chunks,
                    "found": found,
                })


if __name__ == "__main__":
    main()
