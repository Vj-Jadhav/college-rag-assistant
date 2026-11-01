"""RAG Pipeline Orchestrator Module.

Coordinates document retrieval, prompt augmentation, grounding validation,
and response synthesis between DocumentRetriever and OllamaLLMService.
"""

from typing import Dict, Any, List, Generator, Optional
import logging
from retriever import DocumentRetriever
from llm_service import OllamaLLMService

logger = logging.getLogger(__name__)

# Strict standard fallback answer mandated by project requirements
FALLBACK_RESPONSE = "Information not found in the available documents."

SYSTEM_PROMPT = (
    "You are an authoritative, accurate, and helpful College Document Question Answering Assistant.\n"
    "Your objective is to answer questions using EXCLUSIVELY the provided College Document Context.\n\n"
    "STRICT GROUNDING RULES:\n"
    "1. Base your answer ONLY on the explicit facts directly mentioned in the Context below.\n"
    "2. Do NOT extrapolate, speculate, or fabricate college rules, dates, percentages, fees, or policies.\n"
    "3. If the provided Context does NOT contain the necessary information to answer the question, "
    "you MUST respond with EXACTLY:\n"
    "\"Information not found in the available documents.\"\n"
    "4. Do NOT apologize, add disclaimers, or cite external knowledge when information is missing.\n"
    "5. Whenever answering from the context, be direct, factual, and concise."
)

USER_PROMPT_TEMPLATE = (
    "College Document Context:\n"
    "{context}\n\n"
    "Student/User Question:\n"
    "{question}\n\n"
    "Answer (based strictly on the context above):"
)


class RAGPipeline:
    """Orchestrates end-to-end question answering pipeline over college documents."""

    def __init__(
        self,
        retriever: DocumentRetriever,
        llm_service: OllamaLLMService,
        system_prompt: str = SYSTEM_PROMPT,
        temperature: float = 0.1,
    ):
        """Initialize the RAG pipeline.

        Args:
            retriever: Configured DocumentRetriever instance.
            llm_service: Configured OllamaLLMService instance.
            system_prompt: Grounding instructions for LLM.
            temperature: Generation temperature (low = deterministic factual responses).
        """
        self.retriever = retriever
        self.llm_service = llm_service
        self.system_prompt = system_prompt
        self.temperature = temperature

    def build_prompt(self, question: str, context: str) -> str:
        """Construct user prompt incorporating context and question."""
        return USER_PROMPT_TEMPLATE.format(context=context, question=question)

    def answer_query(self, user_query: str) -> Dict[str, Any]:
        """Perform end-to-end retrieval and generation for a user query.

        Args:
            user_query: The question asked by the student or staff.

        Returns:
            Dictionary containing:
            {
                "query": str,
                "answer": str,
                "sources": List[Dict[str, Any]],
                "context_chunks": List[Dict[str, Any]],
                "retrieval_count": int,
                "found_in_documents": bool,
            }
        """
        if not user_query or not user_query.strip():
            return {
                "query": user_query or "",
                "answer": "Please ask a question regarding the college documents.",
                "sources": [],
                "context_chunks": [],
                "retrieval_count": 0,
                "found_in_documents": False,
            }

        clean_query = user_query.strip()
        logger.info(f"Processing RAG query: '{clean_query}'")

        # 1. Retrieve relevant chunks (top 3 by default)
        try:
            chunks = self.retriever.retrieve(clean_query)
        except Exception as exc:
            logger.error(f"Retrieval failed for '{clean_query}': {exc}")
            return {
                "query": clean_query,
                "answer": f"Retrieval error: {exc}",
                "sources": [],
                "context_chunks": [],
                "retrieval_count": 0,
                "found_in_documents": False,
            }

        # If no chunks were found in vector store
        if not chunks:
            logger.info("No relevant chunks retrieved from vector store.")
            return {
                "query": clean_query,
                "answer": FALLBACK_RESPONSE,
                "sources": [],
                "context_chunks": [],
                "retrieval_count": 0,
                "found_in_documents": False,
            }

        # 2. Extract sources and format context
        context_text = self.retriever.format_context(chunks)
        sources = self.retriever.extract_sources(chunks)
        prompt = self.build_prompt(question=clean_query, context=context_text)

        # 3. Generate answer via local LLM
        try:
            raw_answer = self.llm_service.generate(
                prompt=prompt,
                system_prompt=self.system_prompt,
                temperature=self.temperature,
            )
            answer = raw_answer.strip() if raw_answer else FALLBACK_RESPONSE
        except Exception as exc:
            logger.error(f"LLM generation failed: {exc}")
            return {
                "query": clean_query,
                "answer": f"Generation error: {exc}",
                "sources": sources,
                "context_chunks": chunks,
                "retrieval_count": len(chunks),
                "found_in_documents": False,
            }

        # 4. Check if answer matches fallback
        is_fallback = (
            FALLBACK_RESPONSE.lower() in answer.lower()
            or "not found in the available documents" in answer.lower()
            or not answer
        )

        final_answer = FALLBACK_RESPONSE if is_fallback else answer

        return {
            "query": clean_query,
            "answer": final_answer,
            "sources": [] if is_fallback else sources,
            "context_chunks": chunks,
            "retrieval_count": len(chunks),
            "found_in_documents": not is_fallback,
        }

    def stream_answer_query(
        self,
        user_query: str,
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream generated response tokens while providing retrieved metadata.

        Yields:
            Dict packets:
            - Initial packet: {"type": "metadata", "sources": [...], "chunks": [...]}
            - Incremental packets: {"type": "token", "content": "..."}
            - Final packet: {"type": "done", "complete_answer": "..."}
        """
        if not user_query or not user_query.strip():
            yield {"type": "token", "content": "Please ask a question regarding the college documents."}
            yield {"type": "done", "complete_answer": "Please ask a question regarding the college documents."}
            return

        clean_query = user_query.strip()
        chunks = self.retriever.retrieve(clean_query)

        if not chunks:
            yield {"type": "metadata", "sources": [], "chunks": []}
            yield {"type": "token", "content": FALLBACK_RESPONSE}
            yield {"type": "done", "complete_answer": FALLBACK_RESPONSE}
            return

        context_text = self.retriever.format_context(chunks)
        sources = self.retriever.extract_sources(chunks)
        prompt = self.build_prompt(question=clean_query, context=context_text)

        # Emit metadata first
        yield {"type": "metadata", "sources": sources, "chunks": chunks}

        full_answer = []
        try:
            token_stream = self.llm_service.stream_generate(
                prompt=prompt,
                system_prompt=self.system_prompt,
                temperature=self.temperature,
            )
            for token in token_stream:
                full_answer.append(token)
                yield {"type": "token", "content": token}
        except Exception as exc:
            err_msg = f" Streaming error: {exc}"
            yield {"type": "token", "content": err_msg}
            full_answer.append(err_msg)

        completed = "".join(full_answer).strip()
        yield {"type": "done", "complete_answer": completed}
