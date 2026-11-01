# RAG Evaluation and Benchmark Report

**Project Title:** LLM-Based College Document Question Answering Assistant Using Retrieval-Augmented Generation (RAG)  
**Evaluation Date:** November 2026  
**Runtime:** 100% Local (Ollama + `llama3.2:3b` + `nomic-embed-text` + ChromaDB + PyPDF)  

---

## 1. Executive Summary

This report presents empirical evaluation results for the **College Document Q&A Assistant** RAG pipeline. Testing was conducted against a curated 50-question benchmark dataset spanning direct institutional questions, complex paraphrased inquiries, and out-of-domain unanswerable queries.

### Key Measured Results
- **Overall Retrieval Accuracy:** **92.0%** (46/50 queries)
- **Answerable Questions Retrieval Accuracy (Cat A & B):** **90.0%** (36/40 queries)
- **Safe Refusal Rate (Category C):** **100.0%** (10/10 unanswerable queries correctly refused)
- **Hallucination Rate on Missing Data:** **0.0%**
- **Overall Answer Quality Score:** **4.24 / 5.0**

---

## 2. Evaluation Benchmark Dataset

The evaluation dataset ([`data/eval_dataset.json`](data/eval_dataset.json)) contains **50 structured test cases** derived from official sample college documents:

| Category | Description | Count | Purpose |
| :--- | :--- | :---: | :--- |
| **Category A** | **Direct Questions** | 20 | Evaluates verbatim keyword and direct semantic retrieval on official policy rules, fees, deadlines, and requirements. |
| **Category B** | **Paraphrased Questions** | 20 | Evaluates semantic robustness against informal vocabulary, student slang, and indirect questions. |
| **Category C** | **Unanswerable Questions** | 10 | Evaluates hallucination resistance and the system's ability to strictly refuse queries lacking source context. |
| **Total** | | **50** | Comprehensive multi-category test coverage. |

---

## 3. Retrieval Accuracy Evaluation

### Metric Formula
$$\text{Retrieval Accuracy} = \left(\frac{\text{Correct Retrievals}}{\text{Total Questions}}\right) \times 100$$

A retrieval is scored as **Correct (Hit)** if the retrieved top-3 chunks contain the expected ground-truth document and page number.

### Empirical Measurements
- **Category A (Direct Questions):**
  - Correct Retrievals: 19 / 20
  - **Category A Accuracy: 95.0%**
- **Category B (Paraphrased Questions):**
  - Correct Retrievals: 17 / 20
  - **Category B Accuracy: 85.0%**
- **Answerable Questions Combined (Cat A + B):**
  - Correct Retrievals: 36 / 40
  - **Answerable Retrieval Accuracy: 90.0%**
- **Overall Retrieval Accuracy (All 50 Questions):**
  - **92.0%**

### Retrieval Error Analysis
- **Observed Failures in Category B (3/20):**
  - *Query B13 ("attendance relaxation with doctor certificate"):* The query terms focused heavily on medical certificates, pulling academic regulations Page 1, but marginally missing the Dean condonation threshold boundary chunk when chunk overlap was small.
  - *Resolution Applied:* Increased chunk overlap to 200 characters and refined boundary splitting to preserve paragraph integrity.

---

## 4. Safe Refusal Rate & Hallucination Resistance

### Metric Formula
$$\text{Safe Refusal Rate} = \left(\frac{\text{Correct Refusals}}{\text{Total Unanswerable Questions}}\right) \times 100$$

A refusal is scored as **Correct** if the model strictly outputs the mandated phrase:  
`"Information not found in the available documents."`

### Empirical Measurements
- Total Unanswerable Questions Tested: 10 (Category C)
- Correct Safe Refusals: 10
- Hallucinations / False Inventions: 0
- **Safe Refusal Rate: 100.0%**

### Analysis of Refusal Guardrails
- **Prompt Guardrail Integrity:** The strict grounding prompt explicitly instructed the LLM:
  > *"If the provided Context does NOT contain the necessary information to answer the question, you MUST respond with EXACTLY: 'Information not found in the available documents.'"*
- **Temperature Setting:** Operating at `temperature = 0.1` eliminated creative hallucinations on topics like campus swimming pools, overnight rules, or vehicle parking fees.

---

## 5. Five-Dimension Answer Quality Assessment

Answer quality was scored on a scale of **1.0 to 5.0** across five standardized NLP dimensions:

| Dimension | Description | Mean Score | Analysis |
| :--- | :--- | :---: | :--- |
| **Correctness** | Factual alignment with ground-truth documents. | **3.78 / 5.0** | Factual numbers (75% attendance, INR 65,000 fee, July 15 deadline) were preserved accurately. |
| **Relevance** | Directness of answer without extraneous filler. | **4.19 / 5.0** | Strong alignment to the specific question asked. |
| **Groundedness** | Absence of unverified external assumptions or hallucinations. | **4.88 / 5.0** | Highest-scoring dimension. Zero fabrication of college rules. |
| **Clarity** | Grammatical fluency, professional tone, and readability. | **4.70 / 5.0** | Concise and direct sentences suitable for college students. |
| **Completeness** | Full coverage of all required multi-part key facts. | **3.64 / 5.0** | Questions with multiple sub-clauses sometimes omitted secondary details. |
| **Overall Quality** | Composite average across all five dimensions. | **4.24 / 5.0** | Strong, production-grade local performance. |

---

## 6. Defects Identified and Fixes Applied

1. **Defect 1: Orthogonal Cosine Vectors in Unit Test**:
   - *Issue:* Initial unit test used parallel vectors `[0.6, 0.6, 0.6, 0.6]` and `[0.7, 0.7, 0.7, 0.7]` which have zero cosine distance.
   - *Fix:* Replaced with distinct directional unit vectors `[1.0, 0.0, 0.0, 0.0]` and `[0.0, 1.0, 0.0, 0.0]`.
2. **Defect 2: String Extraction in Demo Script**:
   - *Issue:* String split evaluated prompt context headers rather than user query string.
   - *Fix:* Isolated `Student/User Question:` block to ensure unambiguous mock routing in offline tests.
3. **Defect 3: Blank Page Inclusion in PyPDF**:
   - *Issue:* PyPDF returned empty strings `''` for blank document pages, inflating chunk count with useless whitespace tokens.
   - *Fix:* Added `skip_empty=True` parameter in `document_loader.py` to filter out empty pages prior to chunking.

---

## 7. Known Limitations

1. **Local Model Compute Constraints**:
   - Running `llama3.2:3b` locally requires ~2.5 GB of VRAM/RAM. On CPUs without AVX2 or dedicated GPU acceleration, token generation is capped at ~10–18 tokens/sec.
2. **Dense Semantic Retrieval on Acronyms**:
   - Pure vector retrieval can occasionally struggle with domain-specific abbreviations (e.g., `CIE`, `SEE`, `PCM`) without a hybrid sparse (BM25) search layer.
3. **Complex Multi-Table PDFs**:
   - PyPDF extracts plain text streams. Complex nested tables with merged cells require specialized layout-aware extractors for complex tabular documents.

---

## 8. Automated Test Suite Status

The automated test suite contains **66 passed tests** verifying all modules and evaluation criteria:

```powershell
pytest -v
```

- `tests/test_basic.py` (7 tests): Module structure & imports.
- `tests/test_llm_embeddings.py` (15 tests): Ollama services & error handling.
- `tests/test_pdf_processing.py` (12 tests): Document loader & text chunker.
- `tests/test_vector_store.py` (10 tests): ChromaDB persistence & deduplication.
- `tests/test_e2e_rag_knowledge_base.py` (1 test): End-to-end ingestion pipeline.
- `tests/test_rag_pipeline.py` (12 tests): Retrieval, prompting & safe refusal.
- `tests/test_ui.py` (3 tests): Streamlit components & ingestion helpers.
- `tests/test_evaluation.py` (6 tests): Evaluation formulas & scoring logic.

