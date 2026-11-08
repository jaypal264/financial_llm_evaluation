# PROJECT NOTES — Financial LLM Evaluation
# IIT Guwahati — Big Data Analytics Course

---

## Status: LIVE EVALUATION EXECUTED & COMPLETED

---

## Live Evaluation Results Summary (November 8, 2026)

- Total model inferences executed: **120 live queries** (60 Direct Answering + 60 RAG Augmentation)
- Evaluated models: **Qwen2.5-7B**, **Llama-3.1-8B**, **Mistral-7B**, **DeepSeek-R1-7B**
- Across 3 financial benchmarks: **FinanceBench**, **FinTextQA**, **FinQA**
- Key empirical findings:
  * **DeepSeek-R1-Distill-7B** ranked **#1 overall** with highest Token F1 (0.1160) and highest semantic similarity (0.4060). Only model to succeed on strict numerical calculation on FinQA.
  * **Qwen2.5-7B** ranked **#1 in policy regulatory QA** on FinTextQA (Token F1: 0.1718) and highest lexical faithfulness (0.3720).
  * **Llama-3.1-8B** achieved consistent strong performance across FinanceBench (0.1550) and FinTextQA (0.1582).
  * **FinQA** is the hardest benchmark (F1 ~0.005–0.015) due to multi-step balance sheet arithmetic.
  * **Direct vs. RAG:** Direct context answering achieved higher F1 (0.1065) than RAG (0.0823); retrieval over 3,101 chunks can introduce distractor passages for isolated table QA.
- All real response logs saved to: `results/model_responses/`
- All evaluation metric CSVs saved to: `results/metrics/` and `results/summary/`
- Real publication plots generated in: `results/plots/`
- Presentation updated in: `presentation/Financial_LLM_Evaluation_Live_Results.pptx`

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

### Evaluated Models (4 Core Models)
1. Qwen/Qwen2.5-7B-Instruct            — open, high lexical faithfulness
2. meta-llama/Llama-3.1-8B-Instruct      — solid baseline across regulatory and SEC QA
3. mistralai/Mistral-7B-Instruct-v0.3   — open baseline
4. deepseek-ai/DeepSeek-R1-Distill-Qwen-7B — reasoning-specialized, highest Token F1 & numerical accuracy

### Inference Strategy
- Provider: Groq API (`GROQ_API_KEY`) and HF Inference Providers via OpenAI-compatible client
- 4 models selected for complete evaluation availability and zero rate-limiting/decommission issues

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

- [x] Run pilot live evaluation (5 questions × 4 models × 3 datasets = 60 Direct + 60 RAG queries)
- [x] Generate real evaluation metrics (EM, Token F1, ROUGE-L, Semantic Sim, Numeric Acc, Latency)
- [x] Generate empirical benchmark plots in `results/plots/`
- [x] Integrate real experimental outputs into presentation and report drafts

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

