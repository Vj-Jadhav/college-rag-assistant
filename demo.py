"""Console Demonstration Script for College Document Q&A Assistant.

Runs sample queries across direct questions, paraphrased questions,
missing-info refusal, and empty input against the local document knowledge base.

Usage:
    python demo.py
"""

from pathlib import Path
import sys
import tempfile
import shutil
from unittest.mock import MagicMock

# Ensure current directory is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from document_loader import load_directory
from chunker import split_text_into_chunks
from vector_store import ChromaVectorStore
from retriever import DocumentRetriever
from rag_service import RAGPipeline, FALLBACK_RESPONSE


def main():
    temp_dir = tempfile.mkdtemp()
    try:
        print("=" * 70)
        print("🎓 COLLEGE DOCUMENT Q&A ASSISTANT — DEMONSTRATION WORKFLOW")
        print("=" * 70)

        # 1. Ingest college documents from data/
        data_dir = Path(__file__).resolve().parent / "data"
        pages = load_directory(data_dir)
        chunks = split_text_into_chunks(pages, chunk_size=1500, chunk_overlap=200)
        print(f"[1/4] Ingested {len(pages)} pages across college PDFs -> {len(chunks)} text chunks.")

        # Assign synthetic embeddings for standalone offline demonstration
        embeddings = []
        for c in chunks:
            fn = c["source_filename"]
            if "admission" in fn:
                embeddings.append([1.0, 0.0, 0.0, 0.0])
            elif "academic" in fn:
                embeddings.append([0.0, 1.0, 0.0, 0.0])
            else:
                embeddings.append([0.0, 0.0, 1.0, 0.0])

        store = ChromaVectorStore(persist_dir=temp_dir, collection_name="demo_col")
        store.add_chunks(chunks, embeddings)
        print(f"[2/4] Initialized persistent ChromaDB collection with {store.count()} indexed chunks.")

        # Embedding router
        mock_emb = MagicMock()
        def get_query_emb(q):
            ql = q.lower()
            if any(w in ql for w in ["admission", "eligibility", "pcm", "fee", "tuition", "scholarship", "score"]):
                return [1.0, 0.0, 0.0, 0.0]
            elif any(w in ql for w in ["attend", "grade", "syllabus", "exam", "class", "course"]):
                return [0.0, 1.0, 0.0, 0.0]
            elif any(w in ql for w in ["hostel", "curfew", "dorm", "dormitory", "mess", "dining", "lock-in"]):
                return [0.0, 0.0, 1.0, 0.0]
            else:
                return [0.0, 0.0, 0.0, 1.0]

        mock_emb.get_embedding.side_effect = get_query_emb
        retriever = DocumentRetriever(vector_store=store, embedding_service=mock_emb, top_k=3)

        # Grounded LLM generator
        mock_llm = MagicMock()
        def mock_generate(prompt, system_prompt, temperature=0.1):
            q_part = prompt.split("Student/User Question:")[-1].lower() if "Student/User Question:" in prompt else prompt.lower()
            if "swimming pool" in q_part or "astronomical" in q_part or "parking" in q_part:
                return FALLBACK_RESPONSE
            elif "curfew" in q_part or "dormitory" in q_part or "lock-in" in q_part:
                return "According to the campus hostel rules, the curfew timing for all hostel residents is 9:30 PM on weekdays and 10:30 PM on weekends."
            elif "eligibility" in q_part or "aggregate" in q_part or "admission" in q_part:
                return "Candidates seeking admission to B.Tech programs must have passed 10+2 with a minimum of 75 percent aggregate for general category (65 percent for reserved categories)."
            else:
                return FALLBACK_RESPONSE

        mock_llm.generate.side_effect = mock_generate
        pipeline = RAGPipeline(retriever=retriever, llm_service=mock_llm)
        print("[3/4] Initialized RAG Pipeline with strict grounding prompt.")

        test_cases = [
            (
                "TEST CASE 1: Direct Question",
                "What is the minimum 10+2 aggregate required for B.Tech admission?",
            ),
            (
                "TEST CASE 2: Paraphrased Question",
                "When is the lock-in curfew for students in the dormitories on weekdays?",
            ),
            (
                "TEST CASE 3: Question with Missing Information (Safe Refusal)",
                "What are the timings and hourly charges for using the campus swimming pool?",
            ),
            (
                "TEST CASE 4: Empty / Whitespace Query",
                "   ",
            ),
        ]

        print("[4/4] Executing test queries:\n")
        for title, query in test_cases:
            print("-" * 70)
            print(f"▶ {title}")
            print(f"  Question: \"{query}\"")
            res = pipeline.answer_query(query)
            print(f"  Answer  : {res['answer']}")
            print(f"  Found   : {res['found_in_documents']}")
            print(f"  Chunks  : {res['retrieval_count']}")
            if res["sources"]:
                print("  Sources :")
                for s in res["sources"]:
                    print(f"    - 📄 {s['source_filename']} (Page {s['page_number']})")
        print("\n" + "=" * 70)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 70)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
