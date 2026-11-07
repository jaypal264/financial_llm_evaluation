# Financial LLM Evaluation: Comparing Open-Weight LLMs on Financial QA Benchmarks

> **IIT Guwahati — B.Tech Data Science & AI — Big Data Analytics Course Project**

## Overview

This project evaluates and compares **six open-weight Large Language Models (LLMs)** on three financial question-answering benchmarks:

| Dataset | Questions (Public) | Focus |
|---|---|---|
| **FinanceBench** | 150 (public sample) | Financial document QA, SEC filings |
| **FinTextQA** | 240 (policy/regulation subset) | Regulation & policy understanding |
| **FinQA** | ~1,147 (test split) | Numerical reasoning over earnings tables |

### Six Models Evaluated

| # | Model | Family | Params |
|---|---|---|---|
| 1 | Qwen2.5-7B-Instruct | Qwen | 7B |
| 2 | Llama-3.1-8B-Instruct | Llama | 8B |
| 3 | Mistral-7B-Instruct-v0.3 | Mistral | 7B |
| 4 | Gemma-2-9B-it | Gemma | 9B |
| 5 | Phi-3.5-mini-instruct | Phi | 3.8B |
| 6 | DeepSeek-R1-Distill-Qwen-7B | DeepSeek | 7B |

---

## Project Structure

```
Financial-LLM-Evaluation/
├── data/
│   ├── raw/              # Downloaded datasets
│   ├── processed/        # Cleaned unified format
│   └── documents/        # Financial documents
├── notebooks/
│   └── Financial_LLM_Evaluation.ipynb   # Main notebook
├── src/
│   ├── data_loader.py    # Dataset downloading & loading
│   ├── preprocessing.py  # Cleaning & unified format
│   ├── model_inference.py # LLM inference interface
│   ├── rag_pipeline.py   # RAG with FAISS
│   ├── evaluation.py     # Metrics (EM, F1, ROUGE, BERTScore)
│   └── visualization.py  # Charts and plots
├── results/
│   ├── model_responses/  # Raw LLM outputs (JSON)
│   ├── metrics/          # Computed metric CSVs
│   ├── plots/            # Saved figures
│   └── summary/          # Aggregate tables
├── reports/              # Research report (ACL format)
├── presentation/         # PPT slides
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── PROJECT_NOTES.md
```

---

## Quick Start

### 1. Clone and set up environment

```bash
git clone <your-repo-url>
cd Financial-LLM-Evaluation
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
copy .env.example .env
# Edit .env and add your HF_TOKEN (and optionally Together/Groq keys)
```

**Get your Hugging Face token:** https://huggingface.co/settings/tokens

**Accept model access** (required for gated models):
- Llama-3.1: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- Gemma-2: https://huggingface.co/google/gemma-2-9b-it

### 3. Run the notebook

```bash
cd notebooks
jupyter notebook Financial_LLM_Evaluation.ipynb
```

Run cells **sequentially**. The notebook is organized into 18 sections.

**Default mode:** `PILOT_MODE = True` — evaluates 5 questions per dataset to verify the pipeline before a full run.

---

## Datasets

### FinanceBench
- **Source:** https://huggingface.co/datasets/PatronusAI/financebench
- **Public size:** 150 questions (not the full 10,231)
- **Content:** Questions on SEC 10-K/10-Q filings from public companies

### FinTextQA
- **Source:** https://huggingface.co/datasets/GPS-Lab/FinTextQA
- **Public size:** ~240 policy/regulation QA pairs
- **Note:** Textbook portion is NOT publicly released

### FinQA
- **Source:** https://huggingface.co/datasets/ibm-research/finqa
- **Content:** Financial numerical reasoning with table + text context
- **Splits:** Train / Dev / Test

---

## Inference Providers

This project uses the **Hugging Face Inference API** (Serverless) as the primary provider for model-consistent evaluation. All six models are queried via the same OpenAI-compatible endpoint.

**Fallback options:**
- Together AI (https://together.ai) — set `INFERENCE_PROVIDER=together`
- Groq (https://groq.com) — set `INFERENCE_PROVIDER=groq`
- Local Transformers — set `INFERENCE_PROVIDER=local_transformers` (requires 16+ GB RAM)
- Ollama — set `INFERENCE_PROVIDER=ollama` (requires local GPU/CPU)

---

## Experiments

| Experiment | Description |
|---|---|
| **A: Direct LLM** | Question + provided context → LLM answer |
| **B: RAG** | Question → FAISS retrieval → retrieved context → LLM answer |

---

## Evaluation Metrics

- Exact Match (EM)
- Token-level F1
- ROUGE-L
- Numeric Accuracy (with tolerance)
- BERTScore (F1)
- Semantic Similarity
- Faithfulness / Hallucination Rate
- Response Latency & Token Usage

---

## Academic Integrity Notice

> This project was developed as part of a Big Data Analytics course at IIT Guwahati. All results are from actual model evaluations. The research report and presentation were drafted with AI assistance and reviewed and rewritten by the student team to reflect our actual work. Fabricated results are not present anywhere in this project.

---

## Citation

If you use this work, please cite the original benchmark papers:

```bibtex
@inproceedings{islam-etal-2023-financebench,
  title     = {FinanceBench: A New Benchmark for Financial Question Answering},
  author    = {Pranab Islam and Anand Kannappan and Douwe Kiela and Rebecca Qian and Nino Scherrer and Bertie Vidgen},
  year      = {2023},
  url       = {https://arxiv.org/abs/2311.11944}
}

@inproceedings{chen-etal-2024-fintextqa,
  title     = {FinTextQA: A Dataset for Long-form Financial Question Answering},
  author    = {Jian Chen and Peilin Zhou and Yining Hua and Yunze Song and Hang Li and Brunno Goldemberg and Ravi Murthy and Sravan Bodapati and Kezhi Li},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year      = {2024},
  url       = {https://aclanthology.org/2024.acl-long.328}
}

@inproceedings{chen-etal-2021-finqa,
  title     = {FinQA: A Dataset of Numerical Reasoning over Financial Data},
  author    = {Zhiyu Chen and Wenhu Chen and Charese Shen and Oliver Fang and Xifeng Yan},
  booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  year      = {2021},
  url       = {https://arxiv.org/abs/2109.00122}
}
```

---

## Team

| Member | Role |
|---|---|
| [Member 1 — Name, Roll No.] | Dataset acquisition & preprocessing, Report |
| [Member 2 — Name, Roll No.] | Model integration, Inference pipeline, RAG |
| [Member 3 — Name, Roll No.] | Evaluation metrics, Visualizations, Presentation |

