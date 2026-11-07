# PROJECT NOTES — Financial LLM Evaluation
# IIT Guwahati — Big Data Analytics Course

---

## Status: INITIALIZED & IN PROGRESS

---

## Environment

- OS: Windows 11 (10.0.26200)
- Python: 3.14 (Path: C:\Users\jaypa\AppData\Local\Programs\Python\Python314\python.exe)
  * Note: System PATH defaults `python` to MSYS2 3.12 without site-packages. Use Python 3.14 explicitly for all execution.
- CPU cores: 20
- Installed packages: PyTorch 2.11.0, transformers 5.22.2, datasets 5.1.0, sentence-transformers 6.2.0, faiss-cpu 1.15.1, openai 3.46.0, pandas 2.3.3, numpy 2.4.3, rouge-score, bert-score, rank_bm25, tiktoken, python-dotenv, tenacity, jupyter. All verified and importing cleanly.

---

## Dataset Notes

### FinanceBench
- HF ID: PatronusAI/financebench
- Public sample: 150 questions
- Full benchmark: 10,231 questions (not public)
- Fields available: question, answer, justification, evidence, company, doc_name, question_type

### FinTextQA
- HF ID: GPS-Lab/FinTextQA
- Public portion: policy/regulation subset (~240 QA pairs)
- Textbook portion: NOT publicly available
- Fields: question, answer, context, source

### FinQA
- HF ID: ibm-research/finqa
- Splits: train, validation, test
- Fields: question, answer, gold_inds, table, pre_text, post_text, program
- Test split used for final evaluation

---

## Model Notes

### Candidate Models (all via HF Inference Providers)
1. Qwen/Qwen2.5-7B-Instruct       — open, no gate
2. meta-llama/Llama-3.1-8B-Instruct — GATED (requires HF token + access request)
3. mistralai/Mistral-7B-Instruct-v0.3 — open
4. google/gemma-2-9b-it            — GATED (requires HF token + access request)
5. microsoft/Phi-3.5-mini-instruct  — open
6. deepseek-ai/DeepSeek-R1-Distill-Qwen-7B — open

### Inference Strategy
- Primary: HF Inference Providers (Serverless) via OpenAI-compatible API
- Fallback: Together AI or Groq
- Local: Transformers (requires 16GB+ RAM) or Ollama

---

## Completed Tasks

- [x] Project directory structure created
- [x] .env.example created and copied to .env
- [x] .gitignore created
- [x] README.md created
- [x] requirements.txt created
- [x] PROJECT_NOTES.md created
- [x] All Python dependencies installed in Python 3.14 environment
- [x] Core modules implemented and verified in `src/`:
  - `data_loader.py` (enhanced with GitHub fallbacks for FinTextQA & FinQA)
  - `preprocessing.py`
  - `model_inference.py`
  - `rag_pipeline.py`
  - `evaluation.py`
  - `visualization.py`
- [x] All 3 datasets downloaded and cached locally in `data/raw/`:
  - FinanceBench (150 questions)
  - FinTextQA (240 questions)
  - FinQA test split (1,147 questions)
- [x] Preprocessing and statistics verified across all 3 datasets
- [x] Corpus construction (3,101 chunks) and BM25 retrieval verified
- [x] Dense embedding model (`sentence-transformers/all-MiniLM-L6-v2`) downloaded and cached locally
- [x] Complete test suite passed in `scratch/test_pipeline.py`
- [x] Complete Jupyter Notebook `Financial_LLM_Evaluation.ipynb` generated (41 cells)
- [x] Presentation video script (`presentation/video_script.txt`) created
- [x] Group contribution document (`reports/group_contribution.txt`) created
- [x] Research report draft (`reports/research_report_draft.txt`) created
- [x] Notebook Section 2 (Methodology diagram) executed successfully and diagram saved (`results/plots/methodology_diagram.png`)

## In Progress

- [ ] Execute Notebook cells sequentially with user approval
- [ ] Connect HF_TOKEN in `.env` for model inference

## Pending

- [ ] Run pilot evaluation (5 questions × 6 models × 3 datasets)
- [ ] Run full evaluation (upon user approval)
- [ ] Generate final evaluation metrics and plots
- [ ] Fill in actual experimental outputs into report and presentation

---

## Key Decisions

1. **Inference provider:** HF Inference Providers (serverless) as default.
   Rationale: Same API endpoint, consistent latency comparison, no local GPU needed.

2. **Pilot mode:** Default MAX_QUESTIONS_PER_DATASET = 5.
   Rationale: Verifies pipeline end-to-end before costly full evaluation.

3. **RAG embedding model:** sentence-transformers/all-MiniLM-L6-v2.
   Rationale: Small, fast, good quality, no gating required.

4. **RAG vector store:** FAISS (faiss-cpu).
   Rationale: No server required, integrates with NumPy arrays.

5. **Numerical tolerance:** ±0.5% for FinQA numerical answers.

---

## Errors Encountered

(Log errors here as they occur)

---

## Important File Locations

- Main notebook: notebooks/Financial_LLM_Evaluation.ipynb
- Config: .env
- Results: results/
- Checkpoints: results/model_responses/checkpoints/

---

## Next Session Notes

(Add notes before closing to resume efficiently)

