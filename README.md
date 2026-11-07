# Financial LLM Evaluation: Comparing Open-Weight Large Language Models on Financial Question Answering Benchmarks

> **IIT Guwahati — B.Tech Data Science & Artificial Intelligence — Big Data Analytics Course Project**

---

## 1. Overview & Problem Statement

Financial question answering requires specialized capabilities that general-domain models frequently struggle with:
1. **Precise numerical reasoning** over tabular financial reports, balance sheets, and earnings calls.
2. **Document understanding & faithful evidence retrieval** from dense SEC 10-K/10-Q filings and regulatory disclosures.
3. **Domain-specific terminology comprehension** spanning accounting standards, banking regulations (e.g., Basel III), and monetary policy.
4. **Hallucination resistance**, which is critical given the severe compliance, financial, and legal implications of incorrect outputs in financial decision-making.

This project investigates and systematically compares **six open-weight Large Language Models (LLMs)** across three premier financial QA benchmarks under identical, controlled inference conditions. In addition, we design, implement, and evaluate a **Retrieval-Augmented Generation (RAG)** pipeline to examine whether external dense retrieval improves answering accuracy over direct LLM inference.

---

## 2. Research Objectives

| Research Question | Objective |
|---|---|
| **RQ1 (Model Accuracy)** | Which open-weight LLM achieves the highest answer correctness on financial question answering benchmarks? |
| **RQ2 (RAG Impact)** | Does Retrieval-Augmented Generation (FAISS + BM25) improve performance over direct LLM answering, or does context injection dilute precision? |
| **RQ3 (Numerical vs. Conceptual)** | Which model architectures excel at multi-step numerical calculation versus regulatory/conceptual text understanding? |
| **RQ4 (Efficiency & Trade-offs)** | What are the empirical latency, token usage, and cost-efficiency trade-offs across different model sizes (3.8B to 9B parameters)? |

---

## 3. Evaluated Datasets

All datasets are downloaded and cached via automated scripts with verified fallback sources:

| Dataset | Available Samples | Domain / Question Types | Primary Focus | Official Source |
|---|---|---|---|---|
| **FinanceBench** | 150 questions (public sample) | 10-K, 10-Q SEC filings, factoids, multi-hop reasoning | Financial document understanding & extraction | [PatronusAI/financebench](https://huggingface.co/datasets/PatronusAI/financebench) |
| **FinTextQA** | 240 questions (policy subset) | Regulatory guidelines, banking policies, long-form QA | Financial regulation understanding | [AlexJJJChen/FinTextQA](https://github.com/AlexJJJChen/FinTextQA) |
| **FinQA** | 1,147 questions (official test split) | Financial tables, pre/post-text, arithmetic programs | Numerical reasoning & multi-step calculations | [czyssrs/FinQA](https://github.com/czyssrs/FinQA) |

> **Documented Scope Limitations:**
> - **FinanceBench:** Evaluated on the 150 publicly released questions (the full 10,231-question benchmark is proprietary/unreleased).
> - **FinTextQA:** Evaluated on the 240 publicly released regulatory/policy QA pairs. The textbook portion described in the ACL 2024 paper is not publicly available due to institutional restrictions.
> - **FinQA:** Evaluated on the complete official test split (1,147 questions).

---

## 4. Six Evaluated Open-Weight Models

All models are accessed via an OpenAI-compatible interface using the Hugging Face Inference Providers API (with support for local execution via Transformers or Ollama):

| # | Model | Family | Parameters | Architecture Highlights | License / Access |
|---|---|---|---|---|---|
| 1 | **Qwen2.5-7B-Instruct** | Qwen | 7.6B | Dense transformer, strong multilingual & math abilities | Open (Apache 2.0) |
| 2 | **Llama-3.1-8B-Instruct** | Llama | 8.0B | Grouped-Query Attention (GQA), 128k context support | Gated (Meta Llama 3.1 Community) |
| 3 | **Mistral-7B-Instruct-v0.3** | Mistral | 7.3B | Sliding window attention, extended tokenizer vocabulary | Open (Apache 2.0) |
| 4 | **Gemma-2-9B-it** | Gemma | 9.2B | Interleaved local/global attention, knowledge distillation | Gated (Gemma Terms of Use) |
| 5 | **Phi-3.5-mini-instruct** | Phi | 3.8B | High-quality synthetic & curated pretraining data | Open (MIT) |
| 6 | **DeepSeek-R1-Distill-Qwen-7B** | DeepSeek | 7.6B | Distilled from DeepSeek-R1 with reasoning chain-of-thought | Open (MIT) |

---

## 5. Experimental Methodology

### Controlled Inference Conditions
To guarantee a fair comparison across all six models:
- **Zero-Shot Evaluation:** Uniform system prompt defining the persona of an expert financial analyst.
- **Deterministic Decoding:** `temperature = 0.0` (greedy decoding), `do_sample = False`, `max_new_tokens = 512`.
- **Identical Infrastructure:** Hugging Face Inference Providers API with automatic retry and exponential backoff.
- **Incremental Checkpointing:** MD5-hashed checkpoint database (`results/model_responses/checkpoints/`) prevents redundant API calls and enables interrupted runs to resume seamlessly.

### Experimental Settings
- **Experiment A (Direct LLM Evaluation):** The model receives the financial question and the ground-truth document context provided by the benchmark.
- **Experiment B (RAG-Augmented Evaluation):** The model receives context retrieved from a custom FAISS vector index built from all document passages.

---

## 6. RAG Pipeline Architecture

```
Financial Documents & Table Contexts
                │
                ▼
     Overlapping Text Chunking (400 words, 50-word overlap)
                │
                ▼
     Dense Embedding Generation (sentence-transformers/all-MiniLM-L6-v2)
                │
                ├───────────────────────────────────────┐
                ▼                                       ▼
     FAISS Index (IndexFlatIP)                 BM25 Sparse Index
       Cosine Inner Product                      Token Frequency Baseline
                │                                       │
                └───────────────────┬───────────────────┘
                                    │
                         Question Query Retrieval
                                    │
                                    ▼
                         Top-5 Relevant Passages
                                    │
                                    ▼
                         Context Injection into Prompt
                                    │
                                    ▼
                         LLM Answer Generation
```

*Note on Data Integrity:* Benchmark reference answers and evaluation labels are strictly excluded from the retrieval corpus. Only raw document contexts are indexed.

---

## 7. Evaluation Metrics

1. **Answer Correctness:**
   - **Exact Match (EM):** Case-insensitive string match after punctuation and whitespace stripping.
   - **Token F1:** Harmonic mean of precision and recall over normalized tokens (SQuAD standard).
   - **ROUGE-L:** Longest Common Subsequence (LCS) F1 score.
2. **Numerical Reasoning Accuracy:**
   - Custom extractor for currency symbols (`$`, `€`, `£`), commas, negative accounting parentheses `(15.4) -> -15.4`, and percentage conversions.
   - **Relative Tolerance:** $\pm 0.5\%$ relative tolerance on extracted primary numeric values.
3. **Semantic Similarity & Quality:**
   - **Semantic Similarity:** Cosine similarity over `all-MiniLM-L6-v2` embeddings.
   - **BERTScore F1:** Contextual token embedding alignment using `distilbert-base-uncased`.
4. **Faithfulness & Hallucination Analysis:**
   - **Lexical Faithfulness Score:** Ratio of non-stopword answer tokens supported by the provided context.
   - **Abstention Rate:** Detection of safe refusal patterns when evidence is absent.
5. **Efficiency & Cost:**
   - Latency per query (seconds), prompt tokens, completion tokens, and error rates.

---

## 8. Project Structure

```
big_data_project/
├── .env.example              # Template for API credentials and runtime settings
├── .gitignore                # Production gitignore excluding data, cache, and secrets
├── PROJECT_NOTES.md          # Chronological engineering log, task tracker & decisions
├── README.md                 # Complete project documentation
├── requirements.txt          # Python dependencies
│
├── data/
│   ├── raw/                  # Downloaded raw benchmarks (cached locally, ignored by git)
│   ├── processed/            # Unified schema files & FAISS indices (ignored by git)
│   └── documents/            # Financial PDF/text documents (ignored by git)
│
├── notebooks/
│   └── Financial_LLM_Evaluation.ipynb  # Primary master Jupyter notebook (18 sections)
│
├── src/
│   ├── __init__.py           # Package marker
│   ├── build_notebook.py     # Programmatic generator for the master notebook
│   ├── data_loader.py        # Robust multi-source dataset downloaders & parsers
│   ├── preprocessing.py      # Cleaning, unified schema formatting & statistics
│   ├── model_inference.py    # Common API client, checkpointing & retry mechanism
│   ├── rag_pipeline.py       # FAISS dense retriever & BM25 baseline implementation
│   ├── evaluation.py         # Multi-metric evaluation engine & numeric validator
│   └── visualization.py      # Publication-quality charts, heatmaps, and radar plots
│
├── results/
│   ├── metrics/              # Exported CSV evaluation tables
│   ├── model_responses/      # Stored model JSON responses & checkpoints
│   ├── plots/                # High-resolution saved visualization figures
│   └── summary/              # Aggregated model comparison matrices
│
├── reports/
│   ├── research_report_draft.txt   # ACL 2023 format 8-12 page research paper draft
│   └── group_contribution.txt      # Formatted 3-member contribution breakdown
│
└── presentation/
    └── video_script.txt      # 15-20 minute 3-member presentation script
```

---

## 9. Installation & Setup

### 1. Prerequisites
- Python 3.12+ (tested on Python 3.12 and Python 3.14 on Windows 11)
- Git

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/jaypal264/big_data_project.git
cd big_data_project

# Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. API Configuration
```bash
# Create local .env file from template
copy .env.example .env   # On Windows
# cp .env.example .env   # On Linux/macOS
```

Edit `.env` and provide your Hugging Face user access token:
```ini
HF_TOKEN=hf_your_actual_token_here
INFERENCE_PROVIDER=huggingface
PILOT_MODE=True
MAX_QUESTIONS_PER_DATASET=5
RANDOM_SEED=42
```

> **Gated Model Access:**
> To query gated models (`meta-llama/Llama-3.1-8B-Instruct` and `google/gemma-2-9b-it`), accept the terms of use on their Hugging Face model cards using the account corresponding to your `HF_TOKEN`.

---

## 10. Execution Instructions

Launch the Jupyter interface:
```bash
cd notebooks
jupyter notebook Financial_LLM_Evaluation.ipynb
```

The notebook is divided into 18 sequential sections:
1. **Introduction & Objectives**
2. **Research Questions & Methodology** (generates methodology architecture diagram)
3. **Environment Setup & Verification**
4. **Imports, Configuration & Token Validation**
5. **Model Registry & Access Status**
6. **Dataset Acquisition** (FinanceBench, FinTextQA, FinQA)
7. **Preprocessing & Unified Schema Conversion**
8. **Exploratory Data Analysis** (distributions, table statistics)
9. **Common Inference Pipeline Setup**
10. **Experiment A:** Direct LLM Evaluation
11. **RAG Pipeline Implementation** (FAISS dense indexing & BM25)
12. **Experiment B:** RAG-Augmented Evaluation
13. **Evaluation Metrics Computation**
14. **Publication-Quality Visualizations** (bar charts, heatmaps, radar, latency scatter)
15. **Statistical Analysis & Model Rankings**
16. **Extension:** Question Complexity Analysis
17. **Discussion & Limitations**
18. **Conclusions & Reproducibility Audit**

---

## 11. Limitations

1. **Benchmark Sample Sizes:** Only public subsets are evaluated. FinanceBench's proprietary 10,231 questions and FinTextQA's unreleased textbook corpus could not be included.
2. **Metric Boundaries:** Exact Match strictly penalizes correct answers formulated with different phrasing. Semantic similarity indicates linguistic proximity rather than financial validity. Lexical faithfulness acts as a proxy and does not guarantee logical entailment.
3. **Zero-Shot Protocol:** All evaluations are zero-shot. Domain fine-tuning or few-shot demonstration would likely elevate numeric execution performance significantly.
4. **Inference Latency:** Serverless API response times incorporate network transit and provider queue latency rather than purely isolated chip execution time.

---

## 12. Team Contributions

| Member | Focus Areas |
|---|---|
| **Member 1** [Name / Roll No.] | Dataset acquisition, preprocessing pipeline, EDA, research report drafting |
| **Member 2** [Name / Roll No.] | Model integration, API client design, RAG pipeline & FAISS vector indexing |
| **Member 3** [Name / Roll No.] | Multi-metric evaluation engine, statistical analysis, visualizations, video script |

*A complete breakdown of responsibilities and time allocation is provided in [`reports/group_contribution.txt`](reports/group_contribution.txt).*

---

## 13. Citation

If referencing this evaluation framework or the underlying benchmarks:

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
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024)},
  year      = {2024}
}

@inproceedings{chen-etal-2021-finqa,
  title     = {FinQA: A Dataset of Numerical Reasoning over Financial Data},
  author    = {Zhiyu Chen and Wenhu Chen and Charese Shen and Oliver Fang and Xifeng Yan},
  booktitle = {Proceedings of EMNLP 2021},
  year      = {2021}
}
```
