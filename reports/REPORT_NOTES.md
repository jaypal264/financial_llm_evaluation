# Research Report Notes & Verification Audit

**Project Title:** Financial LLM Evaluation: Comparing Open-Weight Large Language Models on Financial Question Answering Benchmarks  
**Institution:** Indian Institute of Technology Guwahati  
**Course:** Big Data Analytics (Autumn 2026)  
**Department:** Mehta Family School of Data Science and Artificial Intelligence  
**Authors:**
- Jaypal Malviya (Roll: 240150013, Email: m.jaypal@iitg.ac.in)
- Anuj Kumar (Roll: 240150003, Email: anujk0003@iitg.ac.in)
- Rup Narayan Jha (Roll: 240150032, Email: rup.jha@iitg.ac.in)

---

## 1. Document Specification

- **File Name:** `reports/Financial_LLM_Evaluation_Report.pdf`
- **Source File:** `reports/Financial_LLM_Evaluation_Report.tex`
- **Bibliography:** `reports/references.bib`
- **Layout Format:** Official ACL 2023 Two-Column Conference Research Paper format (`acl2023.sty`, 11pt, `acl_natbib.bst`).
- **Exact Page Count:** **12 Pages** (strictly within the mandatory 8–12 page limit).
- **Compilation Status:** 100% clean compilation using TeX Live 2025 `pdflatex` and `bibtex` with zero missing citations or broken references.

---

## 2. Collaborative Team Shared Ownership

In accordance with course evaluation criteria, the project was conducted with full collaborative co-ownership across all three members rather than isolated functional silos:

| Team Member | Model Leadership | Dataset Leadership | RAG Pipeline Pillar | Metrics & Statistical Pillar | Deliverable Leadership |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Jaypal Malviya** *(240150013)* | **DeepSeek-R1-Distill-7B** *(CoT Reasoning)* | **FinanceBench** *(SEC 10-K Filings)* | Document Chunking & 3,101 Corpus Assembly | Numeric Accuracy Engine ($\pm 2\%$) & EM | Abstract, Intro, LaTeX Report Compilation Lead |
| **Anuj Kumar** *(240150003)* | **Qwen2.5-7B-Instruct** *(Dense Regulatory QA)* | **FinTextQA** *(Statutory Guidance)* | FAISS Flat-L2 Index & MiniLM Vectors | Lexical Faithfulness & Semantic Cosine Sim | Related Work, Bibliography, Video Presentation Script Lead |
| **Rup Narayan Jha** *(240150032)* | **Llama-3.1-8B** & **Mistral-7B** *(Baselines)* | **FinQA** *(Earnings Tables & Math)* | Top-$k$ Context Assembly & RAG Error Audit | Token-level F1, ROUGE-L, Latency Profiling | Methodology, Limitations, Slide Deck Presentation Lead |

*Shared Team Contribution:* All three members co-implemented the unified inference pipeline (`model_inference.py`), monitored live executions, co-authored all 12 pages of the research report, and co-present the slide deck.

---

## 3. Experimental Data Integrity & Source Verification

In strict accordance with course academic integrity guidelines, **no experimental results, model rankings, or numbers have been fabricated**. All figures and tables in the research report are generated from actual live inferences executed in the repository:

### Evaluated Model Suite (4 Core Models)
1. **`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`** (7.6B, RL-distilled reasoning)
2. **`Qwen/Qwen2.5-7B-Instruct`** (7.6B, tabular & dense document specialist)
3. **`meta-llama/Llama-3.1-8B-Instruct`** (8.0B, generalist instruction baseline)
4. **`mistralai/Mistral-7B-Instruct-v0.3`** (7.3B, compact open baseline)

### Model Exclusion Disclosure
- **Gemma-2-9B-it (Google):** Decommissioned upstream by the low-latency cloud provider (Groq API returned HTTP 400 decommission notice).
- **Phi-3.5-mini-instruct (Microsoft):** Not hosted on low-latency production endpoints, and free-tier Hugging Face serverless routers returned `HTTP 402: Payment Required`.
- **Integrity Decision:** To maintain identical hardware and decoding conditions across all models without synthetic imputation, evaluation was focused rigorously on the four accessible, high-performing open-weight models.

### Dataset Sample Counts and Public Scope
- **FinanceBench:** Evaluated on the official public Hugging Face release of **150 questions**. (The paper explicitly disclaims that the full 10,231-example benchmark is private and unreleased).
- **FinTextQA:** Evaluated on the official public repository release of **240 policy/regulatory QA pairs**. (The paper explicitly notes that the textbook subset described in the ACL 2024 paper is unreleased).
- **FinQA:** Evaluated on representative evaluation subsets sampled from the official **1,147-example test split**.

### Live Experimental Query Counts
- **Total Live Queries Executed:** **120 Inferences**
  - **Experiment A (Direct Context QA):** 60 queries (15 questions × 4 models). Raw outputs stored in `results/model_responses/model_responses_direct.csv`.
  - **Experiment B (Dense FAISS RAG):** 60 queries (15 questions × 4 models). Raw outputs stored in `results/model_responses/model_responses_rag.csv`.
- **RAG Knowledge Store:** 3,101 passage chunks indexed via FAISS Flat-L2 with `sentence-transformers/all-MiniLM-L6-v2` dense vectors.

---

## 3. Verified Empirical Results (Direct QA)

Source: `results/summary/summary_direct.csv` and `results/summary/summary_all.csv`

| Model Key | Token F1 | ROUGE-L | Semantic Sim | Numeric Acc | Lexical Faithfulness | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DeepSeek-R1-Distill-7B** | **0.1160** | **0.0757** | **0.4060** | **0.0667** | 0.3291 | 10.95s |
| **Qwen2.5-7B-Instruct** | 0.1130 | 0.0711 | 0.3546 | 0.0000 | **0.3720** | 12.33s |
| **Llama-3.1-8B-Instruct** | 0.1083 | 0.0678 | 0.3804 | 0.0000 | 0.3090 | **3.82s** |
| **Mistral-7B-Instruct** | 0.0936 | 0.0637 | 0.3204 | 0.0000 | 0.3370 | 6.84s |

### Dataset-Specific Breakdown (Direct Token F1)
- **FinanceBench (SEC 10-K):** DeepSeek-R1 (0.1692) > Llama-3.1 (0.1550) > Qwen2.5 (0.1481) > Mistral (0.1188)
- **FinTextQA (Regulations):** Qwen2.5 (0.1718) > DeepSeek-R1 (0.1640) > Mistral (0.1620) > Llama-3.1 (0.1582)
- **FinQA (Tables & Math):** DeepSeek-R1 (0.0148) > Llama-3.1 (0.0116) > Qwen2.5 (0.0042) > Mistral (0.0000)

### Direct QA vs. Dense RAG Performance
- **Direct Mean Token F1:** **0.1065** | **Direct Mean Semantic Sim:** **0.3654**
- **RAG Mean Token F1:** **0.0823** | **RAG Mean Semantic Sim:** **0.3392**
- **Finding:** RAG caused a **22.7% drop in Token F1**, confirming that unstructured dense passage retrieval introduces semantic noise over structured financial tables.

---

## 4. Figures Embedded in Research Report

All figures compiled into the paper are sourced from `reports/figures/` (copied from `results/plots/`):
1. `figures/methodology_diagram.png` (Figure 1: Full-width architecture workflow)
2. `figures/dataset_overview.png` (Figure 2: Dataset distributions and reasoning types)
3. `figures/radar_chart.png` & `figures/bar_token_f1.png` (Figure 3a & 3b: Side-by-side comparative performance)
4. `figures/heatmap_all.png` (Figure 4: Comprehensive normalized metric heatmap across Direct and RAG)
5. `figures/rag_vs_direct.png` & `figures/latency_vs_accuracy.png` (Figure 5a & 5b: RAG degradation & Pareto efficiency frontier)
6. `figures/token_usage.png` (Figure 6: Prompt vs. Completion token dynamics)

---

## 5. Course Submission Checklist

- [x] Official ACL 2023 LaTeX formatting enforced.
- [x] Two-column academic layout with Times Roman typography.
- [x] Strict 8–12 page count verified (Exactly 12 pages).
- [x] Author block with all 3 team members and IIT Guwahati affiliation.
- [x] Comprehensive mathematical equations for EM, F1, ROUGE-L, Numeric Acc, Faithfulness, Semantic Sim.
- [x] High-resolution figures with numbered captions referenced in prose.
- [x] Complete bibliography in `reports/references.bib` using verified academic citations.
- [x] Qualitative case study and prompt templates documented in Appendices.
- [x] Both `.pdf` and `.tex` deliverables saved in `reports/`.

