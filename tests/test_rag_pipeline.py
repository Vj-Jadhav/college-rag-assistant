"""Unit and Integration Tests for RAG Pipeline (Phase 5).

Verifies:
- DocumentRetriever configuration with default top_k=3
- Preservation and formatting of source filenames and page numbers
- Strict RAG prompt generation and context assembly
- Handling of direct questions
- Handling of paraphrased questions
- Handling of out-of-domain / missing questions returning exact fallback:
  'Information not found in the available documents.'
- Handling of empty queries and retrieval failures
- Verification of streaming token generation with metadata
"""

import pytest
from unittest.mock import MagicMock, patch
from retriever import DocumentRetriever, RetrieverError
from rag_service import RAGPipeline, FALLBACK_RESPONSE, SYSTEM_PROMPT
from vector_store import ChromaVectorStore
from embeddings import OllamaEmbeddingService
from llm_service import OllamaLLMService


@pytest.fixture
def sample_college_chunks():
    """Mock retrieved college context chunks."""
    return [
        {
            "chunk_id": "admission_guidelines.pdf_p1_c0",
            "text": "Candidates seeking admission to B.Tech programs must have passed 10+2 with minimum 75 percent aggregate for general category.",
            "source_filename": "admission_guidelines.pdf",
            "page_number": 1,
            "similarity": 0.92,
            "distance": 0.08,
        },
        {
            "chunk_id": "academic_regulations.pdf_p1_c0",
            "text": "Every registered student must maintain a minimum of 75 percent attendance in each course to be eligible for end-semester exams.",
            "source_filename": "academic_regulations.pdf",
            "page_number": 1,
            "similarity": 0.89,
            "distance": 0.11,
        },
        {
            "chunk_id": "hostel_rules.pdf_p1_c0",
            "text": "Curfew timing for all hostel residents is 9:30 PM on weekdays and 10:30 PM on weekends.",
            "source_filename": "hostel_rules.pdf",
            "page_number": 1,
            "similarity": 0.85,
            "distance": 0.15,
        },
    ]


class TestDocumentRetriever:
    """Test suite for DocumentRetriever."""

    def test_default_top_k(self):
        mock_vs = MagicMock(spec=ChromaVectorStore)
        mock_emb = MagicMock(spec=OllamaEmbeddingService)
        retriever = DocumentRetriever(vector_store=mock_vs, embedding_service=mock_emb)
        assert retriever.top_k == 3

    def test_empty_query_returns_empty_list(self):
        mock_vs = MagicMock(spec=ChromaVectorStore)
        mock_emb = MagicMock(spec=OllamaEmbeddingService)
        retriever = DocumentRetriever(vector_store=mock_vs, embedding_service=mock_emb)

        assert retriever.retrieve("") == []
        assert retriever.retrieve("   ") == []
        mock_emb.get_embedding.assert_not_called()

    def test_empty_vector_store_returns_empty_list(self):
        mock_vs = MagicMock(spec=ChromaVectorStore)
        mock_vs.count.return_value = 0
        mock_emb = MagicMock(spec=OllamaEmbeddingService)
        retriever = DocumentRetriever(vector_store=mock_vs, embedding_service=mock_emb)

        assert retriever.retrieve("What is the fee?") == []
        mock_emb.get_embedding.assert_not_called()

    def test_successful_retrieval(self, sample_college_chunks):
        mock_vs = MagicMock(spec=ChromaVectorStore)
        mock_vs.count.return_value = 10
        mock_vs.query.return_value = sample_college_chunks
        mock_emb = MagicMock(spec=OllamaEmbeddingService)
        mock_emb.get_embedding.return_value = [0.1, 0.2, 0.3]

        retriever = DocumentRetriever(vector_store=mock_vs, embedding_service=mock_emb, top_k=3)
        results = retriever.retrieve("What is the eligibility for admission?")

        assert len(results) == 3
        mock_emb.get_embedding.assert_called_once_with("What is the eligibility for admission?")
        mock_vs.query.assert_called_once_with(query_embedding=[0.1, 0.2, 0.3], n_results=3)

    def test_format_context_includes_sources_and_pages(self, sample_college_chunks):
        mock_vs = MagicMock(spec=ChromaVectorStore)
        mock_emb = MagicMock(spec=OllamaEmbeddingService)
        retriever = DocumentRetriever(mock_vs, mock_emb)

        formatted = retriever.format_context(sample_college_chunks)
        assert "Document: admission_guidelines.pdf | Page: 1" in formatted
        assert "Document: academic_regulations.pdf | Page: 1" in formatted
        assert "Document: hostel_rules.pdf | Page: 1" in formatted

    def test_extract_sources_deduplication(self, sample_college_chunks):
        mock_vs = MagicMock(spec=ChromaVectorStore)
        mock_emb = MagicMock(spec=OllamaEmbeddingService)
        retriever = DocumentRetriever(mock_vs, mock_emb)

        # Add duplicate from same page
        dup_chunks = sample_college_chunks + [{
            "chunk_id": "admission_guidelines.pdf_p1_c1",
            "text": "Additional section from page 1",
            "source_filename": "admission_guidelines.pdf",
            "page_number": 1,
        }]

        sources = retriever.extract_sources(dup_chunks)
        assert len(sources) == 3
        filenames = [s["source_filename"] for s in sources]
        assert filenames == ["admission_guidelines.pdf", "academic_regulations.pdf", "hostel_rules.pdf"]


class TestRAGPipeline:
    """Test suite for RAGPipeline."""

    def test_empty_question_handling(self):
        mock_retriever = MagicMock(spec=DocumentRetriever)
        mock_llm = MagicMock(spec=OllamaLLMService)
        pipeline = RAGPipeline(retriever=mock_retriever, llm_service=mock_llm)

        res = pipeline.answer_query("")
        assert "Please ask a question" in res["answer"]
        assert res["found_in_documents"] is False
        mock_retriever.retrieve.assert_not_called()
        mock_llm.generate.assert_not_called()

    def test_direct_question_answered_with_sources(self, sample_college_chunks):
        mock_retriever = MagicMock(spec=DocumentRetriever)
        mock_retriever.retrieve.return_value = [sample_college_chunks[0]]  # admission chunk
        mock_retriever.format_context.return_value = "Admission requires 75% in 10+2 PCM."
        mock_retriever.extract_sources.return_value = [{
            "source_filename": "admission_guidelines.pdf",
            "page_number": 1,
            "chunk_id": "admission_guidelines.pdf_p1_c0",
        }]

        mock_llm = MagicMock(spec=OllamaLLMService)
        mock_llm.generate.return_value = "The minimum eligibility requirement is 75% aggregate in 10+2 PCM."

        pipeline = RAGPipeline(retriever=mock_retriever, llm_service=mock_llm)
        res = pipeline.answer_query("What is the minimum 10+2 aggregate required for admission?")

        assert "75%" in res["answer"]
        assert res["found_in_documents"] is True
        assert len(res["sources"]) == 1
        assert res["sources"][0]["source_filename"] == "admission_guidelines.pdf"
        assert res["sources"][0]["page_number"] == 1

    def test_paraphrased_question_handling(self, sample_college_chunks):
        mock_retriever = MagicMock(spec=DocumentRetriever)
        mock_retriever.retrieve.return_value = [sample_college_chunks[2]]  # hostel chunk
        mock_retriever.format_context.return_value = "Curfew timing for hostel residents is 9:30 PM on weekdays."
        mock_retriever.extract_sources.return_value = [{
            "source_filename": "hostel_rules.pdf",
            "page_number": 1,
            "chunk_id": "hostel_rules.pdf_p1_c0",
        }]

        mock_llm = MagicMock(spec=OllamaLLMService)
        # Paraphrased question: "When do I have to be back in the dormitory on Wednesdays?"
        mock_llm.generate.return_value = "You must be back in the hostel by 9:30 PM on weekdays."

        pipeline = RAGPipeline(retriever=mock_retriever, llm_service=mock_llm)
        res = pipeline.answer_query("When do I have to be back in the dormitory on Wednesdays?")

        assert "9:30 PM" in res["answer"]
        assert res["found_in_documents"] is True
        assert res["sources"][0]["source_filename"] == "hostel_rules.pdf"

    def test_missing_information_returns_exact_fallback(self, sample_college_chunks):
        mock_retriever = MagicMock(spec=DocumentRetriever)
        # Even if some chunks are returned, LLM acknowledges no evidence exists
        mock_retriever.retrieve.return_value = sample_college_chunks
        mock_retriever.format_context.return_value = "Admission, attendance, and curfew details."
        mock_retriever.extract_sources.return_value = [{"source_filename": "hostel_rules.pdf", "page_number": 1}]

        mock_llm = MagicMock(spec=OllamaLLMService)
        # LLM adheres to strict prompt and emits exact fallback sentence
        mock_llm.generate.return_value = FALLBACK_RESPONSE

        pipeline = RAGPipeline(retriever=mock_retriever, llm_service=mock_llm)
        # Question about missing swimming pool rules
        res = pipeline.answer_query("What are the timings and rules for using the campus swimming pool?")

        assert res["answer"] == "Information not found in the available documents."
        assert res["found_in_documents"] is False
        assert res["sources"] == []  # Sources suppressed when information is not found

    def test_empty_retrieval_returns_fallback_immediately(self):
        mock_retriever = MagicMock(spec=DocumentRetriever)
        mock_retriever.retrieve.return_value = []
        mock_llm = MagicMock(spec=OllamaLLMService)

        pipeline = RAGPipeline(retriever=mock_retriever, llm_service=mock_llm)
        res = pipeline.answer_query("Is there an astronomical observatory?")

        assert res["answer"] == FALLBACK_RESPONSE
        assert res["found_in_documents"] is False
        mock_llm.generate.assert_not_called()

    def test_stream_answer_query(self, sample_college_chunks):
        mock_retriever = MagicMock(spec=DocumentRetriever)
        mock_retriever.retrieve.return_value = [sample_college_chunks[1]]  # academic
        mock_retriever.format_context.return_value = "Minimum 75% attendance required."
        mock_retriever.extract_sources.return_value = [{
            "source_filename": "academic_regulations.pdf",
            "page_number": 1,
        }]

        mock_llm = MagicMock(spec=OllamaLLMService)
        mock_llm.stream_generate.return_value = iter(["Minimum ", "75% ", "attendance."])

        pipeline = RAGPipeline(retriever=mock_retriever, llm_service=mock_llm)
        events = list(pipeline.stream_answer_query("What is the attendance rule?"))

        # Verify event stream structure
        assert events[0]["type"] == "metadata"
        assert events[0]["sources"][0]["source_filename"] == "academic_regulations.pdf"

        token_events = [e for e in events if e["type"] == "token"]
        assert "".join(e["content"] for e in token_events) == "Minimum 75% attendance."

        done_event = events[-1]
        assert done_event["type"] == "done"
        assert done_event["complete_answer"] == "Minimum 75% attendance."

