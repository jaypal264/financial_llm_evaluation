"""
build_notebook.py
=================
Generates the complete Financial_LLM_Evaluation.ipynb notebook.
Run this script to create/recreate the notebook:
    python src/build_notebook.py
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "Financial_LLM_Evaluation.ipynb"
NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)


def md(source):
    if isinstance(source, str):
        source = [source]
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source, execution_count=None):
    if isinstance(source, str):
        source = [source]
    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


cells = []

# =============================================================================
# SECTION 1: PROJECT INTRODUCTION
# =============================================================================
cells.append(md("""\
# Financial LLM Evaluation
## Comparing Open-Weight Large Language Models on Financial QA Benchmarks

**Institution:** IIT Guwahati | B.Tech Data Science & Artificial Intelligence  
**Course:** Big Data Analytics  
**Team:** Group Project — 3 Members  

---

## Project Overview

This notebook presents a **systematic empirical evaluation** of six open-weight Large Language Models (LLMs) on three established financial question-answering benchmarks.

### Datasets

| Dataset | Public Questions | Focus |
|---|---|---|
| **FinanceBench** | 150 (public sample of 10,231) | SEC filing QA, evidence extraction |
| **FinTextQA** | ~240 (policy/regulation subset) | Regulatory financial text, long-form answers |
| **FinQA** | ~1,147 (official test split) | Numerical reasoning over earnings tables |

> **Note:** We evaluate only the publicly available portions. Full FinanceBench (10,231 questions) and FinTextQA textbook portions are not publicly available. This is documented accurately throughout.

### Six Open-Weight Models

| # | Model | HF ID | Family | Params | Access |
|---|---|---|---|---|---|
| 1 | Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | Qwen | 7B | Open |
| 2 | Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` | Llama | 8B | 🔒 Gated |
| 3 | Mistral-7B-Instruct-v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` | Mistral | 7B | Open |
| 4 | Gemma-2-9B-it | `google/gemma-2-9b-it` | Gemma | 9B | 🔒 Gated |
| 5 | Phi-3.5-mini-instruct | `microsoft/Phi-3.5-mini-instruct` | Phi | 3.8B | Open |
| 6 | DeepSeek-R1-Distill-Qwen-7B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | DeepSeek | 7B | Open |

### Key Contributions

1. **Controlled comparison** — identical prompts, generation parameters, and inference infrastructure for all six models
2. **RAG pipeline** — FAISS-based dense retrieval + BM25 sparse baseline, compared with direct LLM answering
3. **Multi-metric evaluation** — EM, Token F1, ROUGE-L, Numeric Accuracy, BERTScore, Semantic Similarity, Faithfulness
4. **Question complexity analysis** — performance breakdown by question category
5. **Full reproducibility** — all code, checkpoints, and results saved incrementally
"""))

# =============================================================================
# SECTION 2: RESEARCH QUESTIONS & METHODOLOGY
# =============================================================================
cells.append(md("""\
---
## Section 2: Research Questions & Methodology

### Research Questions

| ID | Research Question |
|---|---|
| **RQ1** | Which open-weight LLM achieves the highest accuracy on financial QA benchmarks? |
| **RQ2** | Does RAG improve LLM performance compared to direct answering? |
| **RQ3** | Which model performs best on numerical reasoning vs. conceptual questions? |
| **RQ4** | What is the latency and cost trade-off across model families? |

### Experimental Design

**Experiment A — Direct LLM Evaluation:**  
Each model receives the question + benchmark-provided context. Measures ability to use already-retrieved evidence.

**Experiment B — RAG-Augmented Evaluation:**  
A FAISS retrieval index built from document contexts retrieves the top-5 relevant chunks per question. These replace the original context. Measures whether retrieval improves performance.

### Pipeline Architecture

```
Datasets (FinanceBench / FinTextQA / FinQA)
        │
        ▼
  Preprocessing & Unified Schema
        │
        ├──── Experiment A ─────────────────────────────────────────┐
        │      Question + Original Context → LLM → Answer           │
        │                                                            │
        └──── Experiment B ─────────────────────────────────────────┤
               Question → FAISS Retrieval → Top-5 Chunks → LLM → Answer
                                                                     │
                                        Evaluation Metrics ──────────┘
                                        (EM, F1, ROUGE-L, NumAcc, BERTScore, Faithfulness)
                                                │
                                        Results & Visualizations
```
"""))

cells.append(code("""\
# Methodology diagram using matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis('off')
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('#f8f9fa')

def box(ax, x, y, w, h, label, color='#1565C0', fontsize=9):
    rect = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.1',
                           facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.9)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=fontsize, color='white', fontweight='bold', wrap=True)

def arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))

# Datasets
box(ax, 0.3, 4.5, 2.5, 1.5, 'Datasets\\n(FB / FTQA / FinQA)', '#7B1FA2')
# Preprocessing
box(ax, 3.5, 4.5, 2.5, 1.5, 'Preprocessing\\n& Unified Schema', '#1565C0')
arrow(ax, 2.8, 5.25, 3.5, 5.25)

# Exp A
box(ax, 6.5, 5.5, 3.5, 1.2, 'Exp A: Direct LLM\\nQ + Context → Answer', '#2E7D32')
arrow(ax, 6.0, 5.6, 6.5, 5.9)

# Exp B  
box(ax, 6.5, 3.8, 3.5, 1.2, 'Exp B: RAG Pipeline\\nFAISS → Top-5 → Answer', '#E65100')
arrow(ax, 6.0, 4.9, 6.5, 4.4)

# FAISS box
box(ax, 6.5, 2.0, 3.5, 1.2, 'FAISS Index\\n(all-MiniLM-L6-v2)', '#00838F', fontsize=8)
arrow(ax, 8.25, 3.8, 8.25, 3.2)

# Metrics
box(ax, 10.5, 4.5, 3.0, 1.5, 'Evaluation\\nMetrics\\n(EM, F1, ROUGE-L,\\nNumAcc, Faithfulness)',
    '#B71C1C', fontsize=8)
arrow(ax, 10.0, 6.1, 10.5, 5.5)
arrow(ax, 10.0, 4.4, 10.5, 4.9)

ax.set_title('Financial LLM Evaluation — Pipeline Architecture', 
             fontsize=14, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig('../results/plots/methodology_diagram.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Methodology diagram saved to results/plots/methodology_diagram.png")
"""))

# =============================================================================
# SECTION 3: ENVIRONMENT SETUP
# =============================================================================
cells.append(md("""\
---
## Section 3: Environment Setup & Installation

This cell installs all required packages. Run once at the start of a new environment.
"""))

cells.append(code("""\
import subprocess
import sys

# Required packages (skip if already installed)
packages = [
    "transformers>=4.44.0",
    "datasets>=2.20.0",
    "huggingface_hub>=0.24.0",
    "openai>=1.40.0",
    "rouge-score>=0.1.2",
    "bert-score>=0.3.13",
    "nltk>=3.8.0",
    "rank_bm25>=0.2.2",
    "tiktoken>=0.7.0",
    "python-dotenv>=1.0.0",
    "tenacity>=8.3.0",
    "sentence-transformers>=3.0.0",
    "faiss-cpu>=1.8.0",
    "jsonlines>=4.0.0",
]

print("Installing/verifying required packages...")
for pkg in packages:
    pkg_name = pkg.split(">=")[0].split("==")[0]
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            print(f"  ✓ {pkg_name}")
        else:
            print(f"  ✗ {pkg_name}: {result.stderr[:100]}")
    except subprocess.TimeoutExpired:
        print(f"  ⏱ {pkg_name}: timeout — may already be installed")

# NLTK downloads
import nltk
for resource in ['punkt', 'stopwords', 'averaged_perceptron_tagger']:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

print("\\n✓ Environment setup complete")

# Verify critical imports
critical = ['transformers', 'datasets', 'openai', 'sentence_transformers', 'faiss']
for mod in critical:
    try:
        __import__(mod)
        print(f"  ✓ {mod} importable")
    except ImportError:
        print(f"  ✗ {mod} NOT available — restart kernel after installation")
"""))

# =============================================================================
# SECTION 4: IMPORTS, CONFIG, API SETUP
# =============================================================================
cells.append(md("""\
---
## Section 4: Imports, Configuration & API Setup

**Configuration variables** are loaded from `.env` in the project root.  
Copy `.env.example` to `.env` and set your `HF_TOKEN` before running.

```bash
copy .env.example .env
# Edit .env and set:  HF_TOKEN=hf_your_token_here
```

**Get your HF token:** https://huggingface.co/settings/tokens

**Accept gated model licenses:**
- Llama-3.1: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct  
- Gemma-2: https://huggingface.co/google/gemma-2-9b-it
"""))

cells.append(code("""\
import os
import sys
import json
import logging
import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from tqdm.auto import tqdm
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(name)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger("FinLLMEval")

# ─── Project paths ───────────────────────────────────────────────────────────
# Adjust PROJECT_ROOT if running from a different location
NOTEBOOK_DIR = Path().resolve()
if NOTEBOOK_DIR.name == "notebooks":
    PROJECT_ROOT = NOTEBOOK_DIR.parent
else:
    PROJECT_ROOT = NOTEBOOK_DIR
    
print(f"Project root: {PROJECT_ROOT}")

# Add src/ to path so we can import our modules
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ─── Load environment variables ──────────────────────────────────────────────
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✓ Loaded .env from {env_path}")
else:
    print(f"⚠️  No .env found at {env_path}")
    print(f"   Create it from .env.example and set HF_TOKEN")

# ─── Configuration ───────────────────────────────────────────────────────────
PILOT_MODE = os.environ.get("PILOT_MODE", "True").lower() == "true"
MAX_QUESTIONS_PER_DATASET = int(os.environ.get("MAX_QUESTIONS_PER_DATASET", "5"))
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
INFERENCE_PROVIDER = os.environ.get("INFERENCE_PROVIDER", "huggingface")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

np.random.seed(RANDOM_SEED)
pd.set_option("display.max_colwidth", 80)

# ─── Display config ──────────────────────────────────────────────────────────
print("\\n" + "=" * 60)
print("PROJECT CONFIGURATION")
print("=" * 60)
config = {
    "Setting": ["PILOT_MODE", "MAX_QUESTIONS_PER_DATASET", "RANDOM_SEED",
                "MAX_RETRIES", "INFERENCE_PROVIDER", "HF_TOKEN"],
    "Value": [
        str(PILOT_MODE), str(MAX_QUESTIONS_PER_DATASET),
        str(RANDOM_SEED), str(MAX_RETRIES), INFERENCE_PROVIDER,
        "✓ Set (hidden)" if HF_TOKEN else "✗ NOT SET — add to .env"
    ],
    "Note": [
        "Set False in .env for full evaluation",
        "Questions per dataset in pilot mode",
        "For reproducibility",
        "API retry attempts",
        "huggingface / together / groq / ollama",
        "https://huggingface.co/settings/tokens"
    ],
}
display(pd.DataFrame(config))

if not HF_TOKEN:
    print("\\n⚠️  WARNING: HF_TOKEN is not set.")
    print("   Model inference will be skipped.")
    print("   Set HF_TOKEN in your .env file to run evaluations.")
else:
    print("\\n✓ HF_TOKEN is set — model inference is ready")
    
print(f"\\nMode: {'🔬 PILOT (5 questions/dataset)' if PILOT_MODE else '📊 FULL EVALUATION'}")
"""))

# =============================================================================
# SECTION 5: MODEL REGISTRY & VERIFICATION
# =============================================================================
cells.append(md("""\
---
## Section 5: Model Registry & Verification

The six candidate models were verified for:
- Availability on Hugging Face
- License and gating requirements
- Support by HF Inference Providers API
- Current parameter counts and model family diversity

No models were silently substituted. Gating requirements are documented below.
"""))

cells.append(code("""\
from model_inference import MODELS, DEFAULT_GENERATION_PARAMS

# Display model registry
models_info = []
for key, info in MODELS.items():
    models_info.append({
        "Key": key,
        "Model ID (HuggingFace)": info["hf_id"],
        "Family": info["family"],
        "Params": info["params"],
        "Access": "🔒 Gated" if info["gated"] else "✓ Open",
        "Notes": info["notes"],
    })

models_df = pd.DataFrame(models_info)
print("SIX OPEN-WEIGHT MODELS FOR EVALUATION")
print("=" * 80)
display(models_df[["Key", "Model ID (HuggingFace)", "Family", "Params", "Access"]])

print("\\nAccess Notes:")
for key, info in MODELS.items():
    print(f"  {key}: {info['notes']}")

print("\\nGeneration Parameters (IDENTICAL for all models — ensures fair comparison):")
print("-" * 50)
for k, v in DEFAULT_GENERATION_PARAMS.items():
    desc = {
        "max_new_tokens": "Max response length",
        "temperature": "0 = greedy decoding (deterministic, reproducible)",
        "top_p": "Nucleus sampling (inactive at temperature=0)",
        "do_sample": "False = greedy (not sampling)",
    }.get(k, "")
    print(f"  {k:20s} = {v}   # {desc}")
    
print("\\n⚠️  Gated models require:")
print("   • HF_TOKEN set in .env")
print("   • License accepted at model page:")
print("   • Llama-3.1-8B-Instruct → https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
print("   • Gemma-2-9B-it         → https://huggingface.co/google/gemma-2-9b-it")
"""))

# =============================================================================
# SECTION 6: DATASET DOWNLOADING
# =============================================================================
cells.append(md("""\
---
## Section 6: Dataset Downloading

We download all three datasets from their official Hugging Face repositories.  
Downloaded data is cached locally in `data/raw/` to avoid repeated downloads.

### Source Links

| Dataset | HF Repository | Paper |
|---|---|---|
| FinanceBench | [PatronusAI/financebench](https://huggingface.co/datasets/PatronusAI/financebench) | [arXiv:2311.11944](https://arxiv.org/abs/2311.11944) |
| FinTextQA | [GPS-Lab/FinTextQA](https://huggingface.co/datasets/GPS-Lab/FinTextQA) | [ACL 2024](https://aclanthology.org/2024.acl-long.328/) |
| FinQA | [ibm-research/finqa](https://huggingface.co/datasets/ibm-research/finqa) | [arXiv:2109.00122](https://arxiv.org/abs/2109.00122) |

### Documented Limitations

- **FinanceBench:** Public sample = 150 questions. Full benchmark = 10,231 (not released).
- **FinTextQA:** Only policy/regulation subset (~240 QA) is public. Textbook portion NOT released.
- **FinQA:** Full dataset available; using official test split for evaluation.
"""))

cells.append(code("""\
from data_loader import load_financebench

print("=" * 60)
print("DOWNLOADING FINANCEBENCH")
print("=" * 60)
print("Source: PatronusAI/financebench")
print("Public size: 150 questions (full: 10,231 — not public)")
print()

try:
    df_fb = load_financebench(
        hf_token=HF_TOKEN if HF_TOKEN else None,
        force_download=False,  # Uses cache if available
    )
    print(f"✓ FinanceBench loaded: {len(df_fb)} questions")
    print(f"  Columns: {list(df_fb.columns)}")
    print("\\nSample record:")
    if len(df_fb) > 0:
        sample = df_fb.iloc[0]
        for col in ["question_id", "question", "reference_answer", "company", "reasoning_type"]:
            val = str(sample.get(col, "N/A"))
            display_val = val[:100] + "..." if len(val) > 100 else val
            print(f"  {col:20s}: {display_val}")
except Exception as e:
    print(f"✗ FinanceBench download failed: {e}")
    print("  Possible causes:")
    print("  1. No internet access (sandbox mode)")
    print("  2. HF_TOKEN not set for gated access")
    print("  3. Package not installed (run Section 3 first)")
    df_fb = pd.DataFrame()
    
print(f"\\nFinanceBench rows: {len(df_fb)}")
"""))

cells.append(code("""\
from data_loader import load_fintextqa

print("=" * 60)
print("DOWNLOADING FINTEXTQA")
print("=" * 60)
print("Source: GPS-Lab/FinTextQA")
print("Public size: ~240 (policy/regulation subset only)")
print("Textbook portion: NOT publicly available (documented limitation)")
print()

try:
    df_ftqa = load_fintextqa(
        hf_token=HF_TOKEN if HF_TOKEN else None,
        force_download=False,
    )
    print(f"✓ FinTextQA loaded: {len(df_ftqa)} questions")
    print(f"  Columns: {list(df_ftqa.columns)}")
    print("\\nSample record:")
    if len(df_ftqa) > 0:
        sample = df_ftqa.iloc[0]
        for col in ["question_id", "question", "reference_answer", "document_name"]:
            val = str(sample.get(col, "N/A"))
            display_val = val[:100] + "..." if len(val) > 100 else val
            print(f"  {col:20s}: {display_val}")
except Exception as e:
    print(f"✗ FinTextQA download failed: {e}")
    print(f"  Error: {e}")
    df_ftqa = pd.DataFrame()
    
print(f"\\nFinTextQA rows: {len(df_ftqa)}")
"""))

cells.append(code("""\
from data_loader import load_finqa

print("=" * 60)
print("DOWNLOADING FINQA (TEST SPLIT)")
print("=" * 60)
print("Source: ibm-research/finqa")
print("Using: Official test split for final evaluation")
print()

try:
    df_finqa = load_finqa(
        hf_token=HF_TOKEN if HF_TOKEN else None,
        force_download=False,
        split="test",
    )
    print(f"✓ FinQA (test) loaded: {len(df_finqa)} questions")
    print(f"  Columns: {list(df_finqa.columns)}")
    print("\\nSample record:")
    if len(df_finqa) > 0:
        sample = df_finqa.iloc[0]
        for col in ["question_id", "question", "reference_answer", "reference_program"]:
            val = str(sample.get(col, "N/A"))
            display_val = val[:100] + "..." if len(val) > 100 else val
            print(f"  {col:20s}: {display_val}")
except Exception as e:
    print(f"✗ FinQA download failed: {e}")
    df_finqa = pd.DataFrame()
    
print(f"\\nFinQA rows: {len(df_finqa)}")

# Summary of all downloads
print("\\n" + "=" * 60)
print("DOWNLOAD SUMMARY")
print("=" * 60)
for name, df, expected in [
    ("FinanceBench", df_fb, "150"),
    ("FinTextQA", df_ftqa, "~240"),
    ("FinQA (test)", df_finqa, "~1,147"),
]:
    status = "✓" if len(df) > 0 else "✗"
    print(f"  {status} {name:20s}: {len(df):4d} rows (expected: {expected})")
"""))

# =============================================================================
# SECTION 7: PREPROCESSING & STATISTICS
# =============================================================================
cells.append(md("""\
---
## Section 7: Dataset Preprocessing & Statistics

### Preprocessing Steps
1. Ensure all required schema columns exist
2. Drop rows with missing or too-short questions
3. Truncate very long contexts (> 2,000 estimated tokens)  
4. Strip whitespace
5. Apply pilot sampling if `PILOT_MODE=True`

### Dataset Limitations Documented
- Missing fields are filled with empty strings, not fabricated data
- Sample sizes accurately reflect publicly available data
"""))

cells.append(code("""\
from preprocessing import preprocess_dataset, sample_pilot, compute_dataset_statistics

# ─── Preprocess ──────────────────────────────────────────────────────────────
print("Preprocessing datasets...")
all_datasets = {}

raw_datasets = {
    "financebench": df_fb,
    "fintextqa": df_ftqa,
    "finqa": df_finqa,
}

for name, df in raw_datasets.items():
    if df.empty:
        print(f"  ⚠️  {name}: empty (download failed) — skipping")
        all_datasets[name] = df
        continue
    cleaned = preprocess_dataset(df, max_context_tokens=2000)
    all_datasets[name] = cleaned
    print(f"  ✓ {name}: {len(cleaned)} questions after preprocessing")

# ─── Full dataset statistics ─────────────────────────────────────────────────
print("\\nDataset Statistics (Full Available Data):")
print("=" * 80)
stats_df = compute_dataset_statistics(all_datasets)
display(stats_df.round(2))

# ─── Pilot sampling ─────────────────────────────────────────────────────────
print(f"\\n{'='*60}")
if PILOT_MODE:
    print(f"🔬 PILOT MODE: Sampling {MAX_QUESTIONS_PER_DATASET} questions per dataset")
    print(f"   (Change PILOT_MODE=False in .env for full evaluation)")
    eval_datasets = {}
    for name, df in all_datasets.items():
        if df.empty:
            eval_datasets[name] = df
        else:
            sampled = sample_pilot(df, n=MAX_QUESTIONS_PER_DATASET, seed=RANDOM_SEED)
            eval_datasets[name] = sampled
            print(f"  {name}: {len(sampled)} questions sampled")
else:
    print(f"📊 FULL MODE: Using all {sum(len(d) for d in all_datasets.values())} questions")
    eval_datasets = all_datasets

# ─── Create unified eval DataFrame ───────────────────────────────────────────
valid_dfs = [df for df in eval_datasets.values() if not df.empty]
if valid_dfs:
    unified_eval_df = pd.concat(valid_dfs, ignore_index=True)
    print(f"\\n✓ Unified evaluation set: {len(unified_eval_df)} total questions")
    print(unified_eval_df.groupby("dataset_name").size().rename("n_questions").to_string())
else:
    unified_eval_df = pd.DataFrame()
    print("\\n✗ No datasets available for evaluation")
    print("  → Check dataset download (Section 6) and internet access")
"""))

# =============================================================================
# SECTION 8: EXPLORATORY DATA ANALYSIS
# =============================================================================
cells.append(md("""\
---
## Section 8: Exploratory Data Analysis

Visualizations of dataset characteristics to understand the evaluation data before running models.
"""))

cells.append(code("""\
from preprocessing import compute_dataset_statistics, is_numerical_answer

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Financial QA Datasets — Exploratory Analysis", fontsize=14, fontweight='bold')

colors = {"financebench": "#1565C0", "fintextqa": "#2E7D32", "finqa": "#B71C1C"}
valid_names = [n for n, df in all_datasets.items() if not df.empty]

# 1. Question counts
ax = axes[0, 0]
counts = [(n, len(all_datasets[n])) for n in valid_names]
bars = ax.bar([c[0] for c in counts], [c[1] for c in counts],
              color=[colors[n] for n, _ in counts], alpha=0.85, edgecolor='white')
for bar, (_, v) in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            str(v), ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_title("Question Counts (Public Portions)")
ax.set_ylabel("Number of Questions")
ax.set_ylim(0, max([c[1] for _, c in [(n, len(all_datasets[n])) for n in valid_names]] + [10]) * 1.2)

# 2. Question length distribution
ax = axes[0, 1]
for name in valid_names:
    df = all_datasets[name]
    lengths = df["question"].str.len()
    ax.hist(lengths, bins=25, alpha=0.6, label=name, color=colors[name], edgecolor='white')
ax.set_title("Question Length Distribution (chars)")
ax.set_xlabel("Characters")
ax.set_ylabel("Count")
ax.legend(fontsize=8)

# 3. Context length distribution
ax = axes[0, 2]
for name in valid_names:
    df = all_datasets[name]
    lengths = df["context"].str.len()
    if lengths.max() > 0:
        ax.hist(lengths.clip(upper=10000), bins=25, alpha=0.6, 
                label=name, color=colors[name], edgecolor='white')
ax.set_title("Context Length Distribution (chars, clipped at 10k)")
ax.set_xlabel("Characters")
ax.set_ylabel("Count")
ax.legend(fontsize=8)

# 4. Numerical vs conceptual split
ax = axes[1, 0]
for i, name in enumerate(valid_names):
    df = all_datasets[name]
    n_numerical = df["reference_answer"].apply(is_numerical_answer).sum()
    n_total = len(df)
    ax.bar(i, n_numerical/n_total*100, color=colors[name], alpha=0.85, label=name, edgecolor='white')
    ax.bar(i, (1 - n_numerical/n_total)*100, bottom=n_numerical/n_total*100,
           color=colors[name], alpha=0.35, edgecolor='white')
ax.set_xticks(range(len(valid_names)))
ax.set_xticklabels(valid_names)
ax.set_title("Numerical vs. Conceptual Answers")
ax.set_ylabel("Percentage")
legend_patches = [mpatches.Patch(color='gray', alpha=0.85, label='Numerical'),
                  mpatches.Patch(color='gray', alpha=0.35, label='Conceptual')]
ax.legend(handles=legend_patches, fontsize=8)

# 5. Reasoning type distribution (FinanceBench)
ax = axes[1, 1]
if "financebench" in all_datasets and not all_datasets["financebench"].empty:
    df = all_datasets["financebench"]
    if "reasoning_type" in df.columns:
        type_counts = df["reasoning_type"].value_counts().head(8)
        type_counts.plot(kind="barh", ax=ax, color="#1565C0", alpha=0.8)
        ax.set_title("FinanceBench: Question Types")
        ax.set_xlabel("Count")
    else:
        ax.text(0.5, 0.5, "reasoning_type\nnot available", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("FinanceBench: Question Types")
else:
    ax.text(0.5, 0.5, "FinanceBench\nnot loaded", ha='center', va='center', transform=ax.transAxes)
    ax.set_title("FinanceBench: Question Types")

# 6. Answer length distribution
ax = axes[1, 2]
for name in valid_names:
    df = all_datasets[name]
    ans_len = df["reference_answer"].str.split().str.len()
    ax.hist(ans_len.clip(upper=100), bins=25, alpha=0.6,
            label=name, color=colors[name], edgecolor='white')
ax.set_title("Reference Answer Length (words)")
ax.set_xlabel("Words")
ax.set_ylabel("Count")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(str(PROJECT_ROOT / "results" / "plots" / "eda_overview.png"), dpi=150, bbox_inches='tight')
plt.show()
print("✓ EDA plots saved to results/plots/eda_overview.png")

# Sample questions from each dataset
print("\\nSAMPLE QUESTIONS FROM EACH DATASET")
print("=" * 60)
for name in valid_names:
    df = all_datasets[name]
    if len(df) > 0:
        row = df.sample(1, random_state=RANDOM_SEED).iloc[0]
        print(f"\\n[{name.upper()}]")
        print(f"  Q: {row['question'][:150]}")
        print(f"  A: {str(row['reference_answer'])[:100]}")
"""))

# =============================================================================
# SECTION 9: COMMON INFERENCE PIPELINE
# =============================================================================
cells.append(md("""\
---
## Section 9: Common Inference Pipeline

All six models are called through **identical infrastructure** to ensure fair comparison:
- Same API endpoint (Hugging Face Inference Providers)
- Same system prompt and user message format  
- Same generation parameters (temperature=0, max_tokens=512)
- Same retry logic (exponential backoff, 3 attempts)
- Same checkpoint/resume mechanism

### Prompt Template
"""))

cells.append(code("""\
from model_inference import (
    MODELS, DEFAULT_GENERATION_PARAMS, build_prompt, infer_single,
    run_evaluation, responses_to_dataframe, ModelResponse, load_checkpoint
)

# Show example prompt structure
sample_q = "What was Apple's total net sales for fiscal year 2022?"
sample_ctx = (
    "Apple Inc. (AAPL) Annual Report — Fiscal Year 2022\\n"
    "Net Sales by Product Category:\\n"
    "iPhone: $205,489M | Mac: $40,177M | iPad: $29,292M | "
    "Wearables, Home and Accessories: $41,241M | Services: $78,129M\\n"
    "Total Net Sales: $394,328M\\n"
    "Net income: $99,803M"
)

messages = build_prompt(sample_q, sample_ctx)

print("PROMPT TEMPLATE DEMONSTRATION")
print("=" * 70)
for msg in messages:
    print(f"\\n[{msg['role'].upper()}]")
    print("─" * 40)
    print(msg['content'])

print("\\n" + "=" * 70)
print("INFERENCE CONFIGURATION")
print("=" * 70)
print(f"Provider:           {INFERENCE_PROVIDER}")
print(f"Models to evaluate: {len(MODELS)}")
print(f"Generation params:  {DEFAULT_GENERATION_PARAMS}")
print(f"Checkpoint/resume:  Enabled (results/model_responses/checkpoints/)")
print(f"Delay between calls: 1.5 seconds (rate limit protection)")

print("\\n✓ Inference pipeline configured and ready")
print("   Run Section 10 to start Experiment A (Direct LLM Evaluation)")
"""))

# =============================================================================
# SECTION 10: EXPERIMENT A — DIRECT LLM EVALUATION
# =============================================================================
cells.append(md("""\
---
## Section 10: Experiment A — Direct LLM Evaluation

Each model receives the question + benchmark-provided context directly.  
Results are saved incrementally (checkpoint/resume enabled).

> **Cost note:** This calls the HF Inference API for each question × model.  
> In pilot mode (5 questions × 6 models × 3 datasets = 90 calls max).  
> API usage depends on your HF account tier.
"""))

cells.append(code("""\
MODEL_KEYS = list(MODELS.keys())

print("=" * 70)
print("EXPERIMENT A: DIRECT LLM EVALUATION")
print("=" * 70)
print(f"Models:    {MODEL_KEYS}")
print(f"Questions: {len(unified_eval_df) if not unified_eval_df.empty else 0} total")
print(f"Provider:  {INFERENCE_PROVIDER}")
print(f"Mode:      {'🔬 PILOT' if PILOT_MODE else '📊 FULL'}")
print()

direct_responses = []

if unified_eval_df.empty:
    print("✗ No evaluation data. Load datasets in Section 6 first.")

elif not HF_TOKEN and INFERENCE_PROVIDER in ("huggingface",):
    print("⚠️  HF_TOKEN not set → skipping live inference")
    print("   To run evaluation:")
    print("   1. Set HF_TOKEN in your .env file")
    print("   2. Re-run this cell")
    print()
    # Try to load from checkpoint
    checkpoint = load_checkpoint("direct")
    if checkpoint:
        direct_responses = [ModelResponse(**v) for v in checkpoint.values()]
        print(f"   Loaded {len(direct_responses)} responses from previous checkpoint")
    else:
        print("   No checkpoint found. Evaluation results will be empty.")

else:
    print(f"Starting evaluation... (Checkpoint/resume enabled)")
    print(f"Results saved to: results/model_responses/checkpoints/checkpoint_direct.json")
    print()
    
    direct_responses = run_evaluation(
        df=unified_eval_df,
        model_keys=MODEL_KEYS,
        provider=INFERENCE_PROVIDER,
        experiment="direct",
        delay_between_calls=1.5,
        verbose=True,
    )
    print(f"\\n✓ Experiment A complete: {len(direct_responses)} responses")

# Always try to load any existing checkpoint
if not direct_responses:
    checkpoint = load_checkpoint("direct")
    if checkpoint:
        direct_responses = [ModelResponse(**v) for v in checkpoint.values()]
        print(f"Loaded {len(direct_responses)} cached responses from checkpoint")

# Convert to DataFrame
direct_df = responses_to_dataframe(direct_responses) if direct_responses else pd.DataFrame()

if not direct_df.empty:
    print("\\nResponse Summary (Experiment A):")
    summary = direct_df.groupby("model_key").agg(
        n=("question_id", "count"),
        errors=("error", lambda x: x.notna().sum()),
        avg_latency_s=("latency_seconds", "mean"),
        avg_completion_tokens=("completion_tokens", "mean"),
    ).round(2)
    summary.index = [MODELS[m]["display_name"] if m in MODELS else m for m in summary.index]
    display(summary)
    direct_df.to_json(str(PROJECT_ROOT / "results" / "model_responses" / "direct_all.json"),
                      orient="records", indent=2)
    print("\\n✓ Saved: results/model_responses/direct_all.json")
else:
    print("\\n⚠️  No responses collected. Set HF_TOKEN and re-run.")
"""))

# =============================================================================
# SECTION 11: RAG IMPLEMENTATION
# =============================================================================
cells.append(md("""\
---
## Section 11: RAG Implementation

### Architecture
1. **Corpus construction** — All document contexts chunked into 400-word segments with 50-word overlap. Reference answers are **never** included.
2. **Dense embeddings** — `sentence-transformers/all-MiniLM-L6-v2` → 384-dim vectors
3. **FAISS index** — Inner product similarity (cosine after L2 normalization)
4. **BM25 baseline** — Sparse retrieval for comparison
5. **Top-5 retrieval** — Best chunks concatenated as context for LLM
"""))

cells.append(code("""\
from rag_pipeline import build_corpus, FAISSRetriever, BM25Retriever, build_rag_contexts

print("=" * 70)
print("RAG PIPELINE — CORPUS CONSTRUCTION")
print("=" * 70)
print()
print("Building retrieval corpus from document contexts...")
print("⚠️  Reference answers are NEVER included in the retrieval corpus.")
print()

chunks, metadata = build_corpus(all_datasets, chunk_size=400, overlap=50)
print(f"✓ Corpus built:")
print(f"  Total chunks: {len(chunks)}")
if chunks:
    avg_len = sum(len(c.split()) for c in chunks) / len(chunks)
    print(f"  Avg chunk length: {avg_len:.0f} words")
    for name in valid_names if 'valid_names' in dir() else []:
        n = sum(1 for m in metadata if m['dataset'] == name)
        print(f"  {name}: {n} chunks")
    
print()

# ─── FAISS Retriever ─────────────────────────────────────────────────────────
faiss_index_path = PROJECT_ROOT / "data" / "processed" / "faiss_index"
faiss_retriever = FAISSRetriever(index_path=faiss_index_path)

if faiss_index_path.exists() and (faiss_index_path / "index.faiss").exists():
    print("Loading existing FAISS index...")
    faiss_retriever.load(faiss_index_path)
    print(f"✓ FAISS index loaded: {faiss_retriever.index.ntotal} vectors")
elif chunks:
    print("Building FAISS index (embedding all chunks — may take 1-5 min)...")
    faiss_retriever.build_index(chunks, metadata, save=True)
    print(f"✓ FAISS index built and saved: {faiss_retriever.index.ntotal} vectors")
else:
    print("⚠️  No chunks available to build FAISS index")
    print("   Ensure at least one dataset loaded successfully")
    faiss_retriever.index = None

# ─── BM25 ───────────────────────────────────────────────────────────────────
bm25_retriever = BM25Retriever()
if chunks:
    print("\\nBuilding BM25 index...")
    bm25_retriever.build_index(chunks, metadata)
    print(f"✓ BM25 index built: {len(bm25_retriever.chunks)} documents")
"""))

cells.append(code("""\
# Test retrieval with a sample query
print("RETRIEVAL QUALITY CHECK")
print("=" * 60)

test_queries = [
    "What is Apple's total revenue for fiscal year 2022?",
    "What are the capital requirements for banks under Basel III?",
    "Calculate the return on equity for the given financial data.",
]

for query in test_queries[:2]:
    print(f"\\nQuery: '{query}'")
    print()
    
    if faiss_retriever.index is not None:
        print("FAISS (Dense) top-3:")
        try:
            results = faiss_retriever.retrieve(query, top_k=3)
            for i, (chunk, score, meta) in enumerate(results, 1):
                print(f"  [{i}] Score: {score:.3f} | Dataset: {meta['dataset']}")
                print(f"      {chunk[:120]}...")
        except Exception as e:
            print(f"  FAISS retrieval error: {e}")
    
    if bm25_retriever.bm25 is not None:
        print("\\nBM25 (Sparse) top-3:")
        try:
            results = bm25_retriever.retrieve(query, top_k=3)
            for i, (chunk, score, meta) in enumerate(results, 1):
                print(f"  [{i}] Score: {score:.2f} | Dataset: {meta['dataset']}")
                print(f"      {chunk[:120]}...")
        except Exception as e:
            print(f"  BM25 retrieval error: {e}")
"""))

cells.append(code("""\
# Build RAG-augmented eval DataFrame
print("Building RAG-augmented contexts for evaluation questions...")
print()

if not unified_eval_df.empty:
    if faiss_retriever.index is not None:
        rag_eval_df = build_rag_contexts(
            unified_eval_df,
            retriever=faiss_retriever,
            top_k=5,
            question_col="question",
            fallback_to_original=True,
        )
        print(f"✓ RAG contexts added to {len(rag_eval_df)} questions")
        
        # Show example
        ex = rag_eval_df.iloc[0]
        print(f"\\nExample — Question:")
        print(f"  {ex['question'][:150]}")
        print(f"\\nExample — Retrieved context (first 300 chars):")
        ctx = ex.get('rag_context', '')
        print(f"  {ctx[:300]}{'...' if len(ctx) > 300 else ''}")
    else:
        print("⚠️  FAISS index not available. Using original context as fallback.")
        rag_eval_df = unified_eval_df.copy()
        rag_eval_df["rag_context"] = rag_eval_df["context"]
else:
    rag_eval_df = pd.DataFrame()
    print("✗ No evaluation data to build RAG contexts")
"""))

# =============================================================================
# SECTION 12: EXPERIMENT B — RAG EVALUATION
# =============================================================================
cells.append(md("""\
---
## Section 12: Experiment B — RAG-Based Evaluation

Each model now receives retrieved context (top-5 FAISS chunks) instead of the benchmark's original context.  
Results are saved to a separate checkpoint `checkpoint_rag.json`.
"""))

cells.append(code("""\
print("=" * 70)
print("EXPERIMENT B: RAG-AUGMENTED LLM EVALUATION")
print("=" * 70)
print(f"Models:    {MODEL_KEYS}")
print(f"Questions: {len(rag_eval_df) if not rag_eval_df.empty else 0}")
print(f"Context:   FAISS-retrieved (top-5 chunks)")
print()

rag_responses = []

if rag_eval_df.empty:
    print("✗ No RAG evaluation data. Run Section 11 first.")

elif not HF_TOKEN and INFERENCE_PROVIDER in ("huggingface",):
    print("⚠️  HF_TOKEN not set → skipping RAG inference")
    checkpoint = load_checkpoint("rag")
    if checkpoint:
        rag_responses = [ModelResponse(**v) for v in checkpoint.values()]
        print(f"   Loaded {len(rag_responses)} cached RAG responses from checkpoint")
    else:
        print("   No RAG checkpoint found.")

else:
    print("Starting RAG evaluation...")
    rag_responses = run_evaluation(
        df=rag_eval_df,
        model_keys=MODEL_KEYS,
        provider=INFERENCE_PROVIDER,
        experiment="rag",
        context_col="rag_context",
        delay_between_calls=1.5,
        verbose=True,
    )
    print(f"\\n✓ Experiment B complete: {len(rag_responses)} responses")

# Try checkpoint
if not rag_responses:
    checkpoint = load_checkpoint("rag")
    if checkpoint:
        rag_responses = [ModelResponse(**v) for v in checkpoint.values()]
        print(f"Loaded {len(rag_responses)} RAG responses from checkpoint")

rag_responses_df = responses_to_dataframe(rag_responses) if rag_responses else pd.DataFrame()

if not rag_responses_df.empty:
    print("\\nRAG Response Summary:")
    summary_rag = rag_responses_df.groupby("model_key").agg(
        n=("question_id", "count"),
        errors=("error", lambda x: x.notna().sum()),
        avg_latency_s=("latency_seconds", "mean"),
    ).round(2)
    display(summary_rag)
    rag_responses_df.to_json(str(PROJECT_ROOT / "results" / "model_responses" / "rag_all.json"),
                             orient="records", indent=2)
    print("\\n✓ Saved: results/model_responses/rag_all.json")
"""))

# =============================================================================
# SECTION 13: EVALUATION METRICS
# =============================================================================
cells.append(md("""\
---
## Section 13: Evaluation Metrics Computation

### Metrics Computed

| Metric | Type | Description |
|---|---|---|
| Exact Match | Correctness | 1 if normalized prediction equals reference |
| Token F1 | Correctness | SQuAD-style token overlap F1 |
| ROUGE-L | Quality | Longest common subsequence F1 |
| Numeric Accuracy | Numerical | ±0.5% relative tolerance on extracted numbers |
| Semantic Similarity | Quality | Cosine similarity of sentence-transformer embeddings |
| BERTScore F1 | Quality | Contextual embedding similarity |
| Faithfulness | Hallucination | Fraction of answer tokens appearing in context |
| Abstention Rate | Safety | Models saying "I don't know" when appropriate |

> **Note:** High semantic/BERTScore does NOT imply factual correctness.
"""))

cells.append(code("""\
from evaluation import evaluate_responses, compute_summary_metrics

print("=" * 70)
print("COMPUTING EVALUATION METRICS")
print("=" * 70)
print()

# Reference data
if not unified_eval_df.empty:
    reference_df = unified_eval_df[
        ["question_id", "reference_answer", "context", "reasoning_type"]
    ].copy()
    print(f"Reference data: {len(reference_df)} questions")
else:
    reference_df = pd.DataFrame()
    print("⚠️  No reference data available")

# ─── Evaluate Direct LLM ─────────────────────────────────────────────────────
direct_eval = pd.DataFrame()
if not direct_df.empty and not reference_df.empty:
    print("\\nEvaluating Direct LLM responses...")
    direct_eval = evaluate_responses(
        direct_df, reference_df,
        compute_bertscore=False,   # Set True for full eval (slow ~5 min)
        compute_semantic=True,
        numeric_tolerance=0.005,
    )
    print(f"✓ Direct eval: {len(direct_eval)} scored responses")
    
    direct_summary, direct_extra = compute_summary_metrics(direct_eval)
else:
    print("⚠️  No direct responses to evaluate (run Section 10 with HF_TOKEN)")

# ─── Evaluate RAG ────────────────────────────────────────────────────────────
rag_eval = pd.DataFrame()
if not rag_responses_df.empty and not reference_df.empty:
    print("\\nEvaluating RAG responses...")
    rag_eval = evaluate_responses(
        rag_responses_df, reference_df,
        compute_bertscore=False,
        compute_semantic=True,
        numeric_tolerance=0.005,
    )
    print(f"✓ RAG eval: {len(rag_eval)} scored responses")
    
    rag_summary, rag_extra = compute_summary_metrics(rag_eval)
else:
    print("⚠️  No RAG responses to evaluate (run Section 12 with HF_TOKEN)")

# ─── Save results ────────────────────────────────────────────────────────────
metrics_dir = PROJECT_ROOT / "results" / "metrics"
if not direct_eval.empty:
    direct_eval.to_csv(str(metrics_dir / "direct_eval.csv"), index=False)
    print("\\n✓ Saved: results/metrics/direct_eval.csv")
if not rag_eval.empty:
    rag_eval.to_csv(str(metrics_dir / "rag_eval.csv"), index=False)
    print("✓ Saved: results/metrics/rag_eval.csv")

print("\\n✓ Metric computation complete")
"""))

# =============================================================================
# SECTION 14: RESULTS VISUALIZATION
# =============================================================================
cells.append(md("""\
---
## Section 14: Results Visualization

Publication-quality charts comparing all six models across all metrics and datasets.  
All plots saved to `results/plots/`.

> If model evaluation has not yet been run (no HF_TOKEN), **demo visualizations** 
> are shown with randomly generated placeholder scores, clearly labeled.
"""))

cells.append(code("""\
from visualization import (
    plot_metric_comparison, plot_heatmap, plot_radar,
    plot_latency_vs_accuracy, plot_rag_comparison,
    plot_complexity_analysis, plot_token_usage, plot_dataset_overview,
    MODEL_DISPLAY,
)
from preprocessing import compute_dataset_statistics

# ─── Dataset overview ────────────────────────────────────────────────────────
stats_df = compute_dataset_statistics(all_datasets)
fig0 = plot_dataset_overview(stats_df)
plt.show()

# ─── Determine data source ───────────────────────────────────────────────────
if not direct_eval.empty:
    plot_data = direct_eval.copy()
    data_label = "Actual Evaluation Results"
    print("✓ Using actual evaluation results for all visualizations")
else:
    # Demo data for visualization structure — CLEARLY LABELED
    print("⚠️  No evaluation results yet. Showing DEMO visualization structure.")
    print("   Run model evaluation (Section 10) with HF_TOKEN to get real results.")
    np.random.seed(RANDOM_SEED)
    demo_rows = []
    model_perf_ranges = {
        "qwen2.5-7b":     (0.28, 0.65),
        "llama3.1-8b":    (0.25, 0.60),
        "mistral-7b":     (0.22, 0.58),
        "gemma2-9b":      (0.30, 0.68),
        "phi3.5-mini":    (0.20, 0.55),
        "deepseek-r1-7b": (0.35, 0.72),
    }
    for model, (low, high) in model_perf_ranges.items():
        for dataset in ["financebench", "fintextqa", "finqa"]:
            for qid in range(5):
                mid = (low + high) / 2
                demo_rows.append({
                    "model_key": model,
                    "dataset_name": dataset,
                    "question_id": f"{model}_{dataset}_{qid}",
                    "exact_match": np.random.uniform(low * 0.3, high * 0.4),
                    "token_f1": np.random.uniform(low, high),
                    "rouge_l": np.random.uniform(low * 0.9, high * 0.9),
                    "numeric_accuracy": np.random.uniform(
                        low * 0.5 if dataset == "finqa" else low * 0.2,
                        high * 0.7 if dataset == "finqa" else high * 0.5,
                    ),
                    "faithfulness": np.random.uniform(0.5, 0.9),
                    "semantic_similarity": np.random.uniform(0.4, 0.85),
                    "bertscore_f1": np.random.uniform(0.55, 0.85),
                    "latency_seconds": np.random.uniform(1.5, 7.0),
                    "reasoning_type": np.random.choice(
                        ["numerical_reasoning", "concept_understanding",
                         "information_extraction", "comparative_reasoning"]
                    ),
                })
    plot_data = pd.DataFrame(demo_rows)
    data_label = "DEMO DATA (Placeholder — Run evaluation for real results)"

print(f"\\nData source: {data_label}")
"""))

cells.append(code("""\
# ─── Grouped bar charts ──────────────────────────────────────────────────────
print("Generating comparison charts...")

fig1 = plot_metric_comparison(
    plot_data, metric="token_f1",
    title=f"Token F1 Score by Model and Dataset\\n({data_label})"
)
plt.show()

fig2 = plot_metric_comparison(
    plot_data, metric="rouge_l",
    title=f"ROUGE-L Score by Model and Dataset\\n({data_label})"
)
plt.show()

fig3 = plot_metric_comparison(
    plot_data, metric="numeric_accuracy",
    title=f"Numeric Accuracy by Model and Dataset\\n({data_label})"
)
plt.show()

print("✓ Grouped bar charts saved")
"""))

cells.append(code("""\
# ─── Heatmap ─────────────────────────────────────────────────────────────────
fig4 = plot_heatmap(
    plot_data,
    metrics=["exact_match", "token_f1", "rouge_l", "numeric_accuracy",
             "semantic_similarity", "faithfulness"],
    title=f"Model Performance Heatmap (All Datasets)\\n({data_label})"
)
plt.show()

# ─── Radar chart ─────────────────────────────────────────────────────────────
fig5 = plot_radar(
    plot_data,
    metrics=["token_f1", "rouge_l", "numeric_accuracy", "faithfulness", "semantic_similarity"],
    title=f"Model Capability Radar\\n({data_label})"
)
plt.show()

# ─── Latency vs accuracy ─────────────────────────────────────────────────────
fig6 = plot_latency_vs_accuracy(
    plot_data,
    accuracy_metric="token_f1",
    title=f"Latency vs Token F1 Trade-off\\n({data_label})"
)
plt.show()

print("✓ Heatmap, radar, and latency charts saved")
"""))

cells.append(code("""\
# ─── RAG vs Direct comparison ─────────────────────────────────────────────────
if not rag_eval.empty and not direct_eval.empty:
    # Real comparison
    fig7 = plot_rag_comparison(
        direct_eval, rag_eval,
        metric="token_f1",
        title="Direct LLM vs RAG — Token F1 (Actual Results)"
    )
else:
    # Demo: show RAG with ~8% improvement for illustration
    rag_demo_data = plot_data.copy()
    rag_demo_data["token_f1"] = (rag_demo_data["token_f1"] * 1.08).clip(0, 1.0)
    rag_demo_data["rouge_l"] = (rag_demo_data["rouge_l"] * 1.06).clip(0, 1.0)
    fig7 = plot_rag_comparison(
        plot_data, rag_demo_data,
        metric="token_f1",
        title=f"Direct LLM vs RAG — Token F1\\n(DEMO — run evaluation for real results)"
    )
plt.show()

# Token usage (only if real data available)
if not direct_df.empty:
    fig8 = plot_token_usage(direct_df, title="Average Token Usage per Model")
    plt.show()

print("✓ RAG comparison chart saved")
"""))

# =============================================================================
# SECTION 15: STATISTICAL ANALYSIS
# =============================================================================
cells.append(md("""\
---
## Section 15: Statistical Analysis & Model Comparison

Comprehensive model rankings and per-dataset breakdowns.  
Results exported to `results/summary/model_comparison.csv`.
"""))

cells.append(code("""\
from evaluation import compute_summary_metrics

print("=" * 70)
print("MODEL COMPARISON TABLES")
print("=" * 70)

metrics_to_show = [
    "exact_match", "token_f1", "rouge_l", "numeric_accuracy",
    "semantic_similarity", "faithfulness", "latency_seconds"
]
metric_display = {
    "exact_match": "Exact Match",
    "token_f1": "Token F1",
    "rouge_l": "ROUGE-L",
    "numeric_accuracy": "Numeric Acc",
    "semantic_similarity": "Sem. Sim.",
    "faithfulness": "Faithfulness",
    "latency_seconds": "Latency (s)",
}

eval_to_show = direct_eval if not direct_eval.empty else plot_data

if not eval_to_show.empty:
    avail_metrics = [m for m in metrics_to_show if m in eval_to_show.columns]
    
    # Overall comparison table
    comp = eval_to_show.groupby("model_key")[avail_metrics].mean().round(4)
    comp.index = [MODEL_DISPLAY.get(m, m) for m in comp.index]
    comp.columns = [metric_display.get(c, c) for c in comp.columns]
    
    print("\\n📊 Mean Scores Across All Datasets")
    print("-" * 60)
    display(comp)
    
    # Per-dataset breakdown
    if "dataset_name" in eval_to_show.columns and "token_f1" in eval_to_show.columns:
        per_ds = (
            eval_to_show.groupby(["model_key", "dataset_name"])["token_f1"]
            .mean().unstack(fill_value=0).round(4)
        )
        per_ds.index = [MODEL_DISPLAY.get(m, m) for m in per_ds.index]
        print("\\n📊 Token F1 by Model × Dataset")
        print("-" * 60)
        display(per_ds)
    
    # Overall ranking
    rank_cols = ["exact_match", "token_f1", "rouge_l"]
    rank_cols = [c for c in rank_cols if c in eval_to_show.columns]
    if rank_cols:
        ranked = eval_to_show.groupby("model_key")[rank_cols].mean().mean(axis=1).sort_values(ascending=False)
        print(f"\\n🏆 Model Ranking (avg of {', '.join(rank_cols)})")
        print("-" * 60)
        for i, (model, score) in enumerate(ranked.items(), 1):
            medal = ["🥇", "🥈", "🥉", " 4.", " 5.", " 6."][i-1]
            display_name = MODEL_DISPLAY.get(model, model)
            print(f"  {medal} {display_name:30s}: {score:.4f}")
    
    # Save comparison
    summary_dir = PROJECT_ROOT / "results" / "summary"
    summary_dir.mkdir(exist_ok=True)
    comp.to_csv(str(summary_dir / "model_comparison.csv"))
    print("\\n✓ Saved: results/summary/model_comparison.csv")
    
    if not rag_eval.empty and not direct_eval.empty:
        # RAG improvement summary
        direct_agg = direct_eval.groupby("model_key")["token_f1"].mean()
        rag_agg = rag_eval.groupby("model_key")["token_f1"].mean()
        improvement = (rag_agg - direct_agg).rename("RAG Improvement (Token F1)")
        print("\\n📊 RAG vs Direct — Token F1 Improvement")
        print("-" * 50)
        for model, delta in improvement.sort_values(ascending=False).items():
            arrow = "↑" if delta >= 0 else "↓"
            print(f"  {arrow} {MODEL_DISPLAY.get(model, model):30s}: {delta:+.4f}")
else:
    print("⚠️  Run model evaluation (Section 10) for real results")
    print("   Showing demo data structure above")
"""))

# =============================================================================
# SECTION 16: COMPLEXITY ANALYSIS (EXTENSION)
# =============================================================================
cells.append(md("""\
---
## Section 16: Extension — Financial Question Complexity Analysis

Questions are categorized into types to investigate which models excel in each area:

| Category | Description | Example |
|---|---|---|
| **numerical_reasoning** | Requires arithmetic over financial figures | "What was the YoY revenue growth?" |
| **concept_understanding** | Financial concept knowledge | "What is EBITDA?" |
| **information_extraction** | Locating specific facts | "What were total assets in 2022?" |
| **comparative_reasoning** | Comparing across periods/entities | "How did margins change Q3 vs Q4?" |
| **multi_step_reasoning** | Chained calculations | "Calculate debt-to-equity ratio from..." |
"""))

cells.append(code("""\
print("=" * 70)
print("EXTENSION: QUESTION COMPLEXITY ANALYSIS")
print("=" * 70)
print()

# Auto-categorize questions if reasoning_type not set
def classify_question(q):
    q = str(q).lower()
    if any(w in q for w in ["how much", "total", "revenue", "income", "ratio",
                             "percent", "growth", "calculate", "earnings", "profit",
                             "loss", "margin", "eps", "ebitda", "cash flow"]):
        return "numerical_reasoning"
    elif any(w in q for w in ["compare", "versus", "difference between",
                              "how did", "changed", "increase", "decrease"]):
        return "comparative_reasoning"
    elif any(w in q for w in ["why", "explain", "what caused", "reason", "describe"]):
        return "multi_step_reasoning"
    elif any(w in q for w in ["what is", "define", "meaning of", "which is"]):
        return "concept_understanding"
    else:
        return "information_extraction"

# Apply to unified eval df
if not unified_eval_df.empty:
    unified_eval_df = unified_eval_df.copy()
    unified_eval_df["auto_category"] = unified_eval_df["question"].apply(classify_question)
    
    cat_counts = unified_eval_df["auto_category"].value_counts()
    print("Question Category Distribution:")
    print(cat_counts.to_string())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Category distribution
    cat_counts.plot(kind="barh", ax=axes[0], color="#1565C0", alpha=0.8, edgecolor='white')
    axes[0].set_title("Question Categories (Auto-Classified)")
    axes[0].set_xlabel("Number of Questions")
    
    # Per-dataset category breakdown
    if "dataset_name" in unified_eval_df.columns:
        cat_by_ds = unified_eval_df.groupby(["dataset_name", "auto_category"]).size().unstack(fill_value=0)
        cat_by_ds.plot(kind="bar", ax=axes[1], colormap="Set2", width=0.75, edgecolor='white')
        axes[1].set_title("Question Types by Dataset")
        axes[1].set_xlabel("Dataset")
        axes[1].set_ylabel("Count")
        axes[1].legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=20, ha='right')
    
    plt.tight_layout()
    plt.savefig(str(PROJECT_ROOT / "results" / "plots" / "question_complexity.png"),
                dpi=150, bbox_inches='tight')
    plt.show()

eval_for_complexity = direct_eval if not direct_eval.empty else plot_data

if not eval_for_complexity.empty and "reasoning_type" in eval_for_complexity.columns:
    fig_comp = plot_complexity_analysis(eval_for_complexity, metric="token_f1")
    if fig_comp:
        plt.show()

print("\\n✓ Complexity analysis complete")
"""))

# =============================================================================
# SECTION 17: LIMITATIONS & DISCUSSION
# =============================================================================
cells.append(md("""\
---
## Section 17: Limitations & Discussion

### Dataset Limitations
- **FinanceBench:** Only 150 of 10,231 questions are publicly available. Our results cannot generalize to the full benchmark.
- **FinTextQA:** Only the policy/regulation subset (~240 QA pairs) is public. The textbook portion described in the ACL 2024 paper is not released.
- **Sampling:** Pilot mode evaluates 5 questions/dataset — statistically insufficient for robust conclusions. Full evaluation requires running with `PILOT_MODE=False`.

### Model Access Limitations
- Llama-3.1 and Gemma-2 require accepting license agreements on Hugging Face. Users who have not done this will get access errors.
- HF Inference Providers API may apply rate limits or queue requests during peak hours.
- Provider-side optimizations (quantization, batching) may differ from paper-reported numbers.

### Evaluation Metric Limitations
- **Exact Match** is strict — correct answers with different wording score 0. Underestimates true accuracy, especially for FinTextQA's long-form answers.
- **Faithfulness** is a lexical proxy — not NLI-based. High lexical overlap ≠ factual correctness.
- **Semantic similarity** measures surface similarity, not factual accuracy. A hallucinated answer can score high.
- **BERTScore** similarly does not distinguish correct from plausible-but-wrong answers.

### RAG Limitations
- No ground truth for retrieval quality — we cannot measure precision@k.
- Corpus is built from benchmark contexts, not from actual full financial documents (which are not all public).
- For FinQA, the benchmark already provides the relevant table, so RAG may not help.

### Compute Limitations
- Zero-shot evaluation only — fine-tuned models would likely perform substantially better.
- No ensemble or chain-of-thought prompting applied.

### Conclusion
> Despite these limitations, this study provides a **systematic, reproducible comparison** of six open-weight LLMs under identical conditions — which is rare in existing financial NLP literature. The RAG comparison adds practical value for practitioners choosing between direct LLM and retrieval-augmented approaches.
"""))

# =============================================================================
# SECTION 18: CONCLUSIONS & REPRODUCIBILITY
# =============================================================================
cells.append(md("""\
---
## Section 18: Conclusions & Reproducibility Instructions

### Summary of Findings

*(Replace with actual findings after running full evaluation)*

| Research Question | Finding |
|---|---|
| **RQ1** (Best model for financial QA?) | *Run evaluation to determine* |
| **RQ2** (Does RAG improve performance?) | *Run evaluation to determine* |
| **RQ3** (Numerical vs conceptual?) | *Run evaluation to determine* |
| **RQ4** (Latency/cost trade-off?) | *Run evaluation to determine* |

### Reproducibility Instructions

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API
copy .env.example .env
# Edit .env: set HF_TOKEN=hf_your_token_here

# 3. Accept gated model licenses (if needed):
#    https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
#    https://huggingface.co/google/gemma-2-9b-it

# 4. Run notebook cell by cell
cd notebooks
jupyter notebook Financial_LLM_Evaluation.ipynb

# 5. For full evaluation (not pilot):
#    Set PILOT_MODE=False in .env
```
"""))

cells.append(code("""\
print("=" * 70)
print("FINAL SUMMARY AND REPRODUCIBILITY INFORMATION")
print("=" * 70)

summary_info = {
    "Project": "Financial LLM Evaluation — IIT Guwahati Big Data Analytics",
    "Python": "3.12.7",
    "Datasets": "FinanceBench (150), FinTextQA (~240), FinQA (~1147 test)",
    "Models": ", ".join(list(MODELS.keys())),
    "Experiment A": "Direct LLM — question + benchmark context → answer",
    "Experiment B": "RAG — FAISS top-5 retrieval → answer",
    "Inference provider": INFERENCE_PROVIDER,
    "Pilot mode": str(PILOT_MODE),
    "Random seed": str(RANDOM_SEED),
    "Checkpoint files": "results/model_responses/checkpoints/",
    "Results CSVs": "results/metrics/",
    "Plots": "results/plots/",
}

print()
for k, v in summary_info.items():
    print(f"  {k:25s}: {v}")

# Final file manifest
print("\\n" + "=" * 70)
print("PROJECT FILE MANIFEST")
print("=" * 70)

important_files = [
    "notebooks/Financial_LLM_Evaluation.ipynb",
    "src/data_loader.py",
    "src/preprocessing.py",
    "src/model_inference.py",
    "src/rag_pipeline.py",
    "src/evaluation.py",
    "src/visualization.py",
    "requirements.txt",
    ".env.example",
    ".gitignore",
    "README.md",
    "PROJECT_NOTES.md",
    "reports/research_report_draft.txt",
    "reports/group_contribution.txt",
    "presentation/video_script.txt",
]

for filepath in important_files:
    full = PROJECT_ROOT / filepath
    exists = "✓" if full.exists() else "✗"
    size = f"({full.stat().st_size:,} bytes)" if full.exists() else "(missing)"
    print(f"  {exists} {filepath:55s} {size}")

# Update PROJECT_NOTES
notes_path = PROJECT_ROOT / "PROJECT_NOTES.md"
try:
    with open(notes_path, "a", encoding="utf-8") as f:
        f.write(f"\\n\\n---\\n## Session: {time.strftime('%Y-%m-%d %H:%M')}\\n")
        f.write(f"- Notebook cell 18 executed successfully\\n")
        f.write(f"- Pilot mode: {PILOT_MODE}\\n")
        f.write(f"- Datasets loaded: {', '.join(n for n, df in all_datasets.items() if not df.empty)}\\n")
        f.write(f"- Eval data: {len(unified_eval_df) if not unified_eval_df.empty else 0} questions\\n")
        f.write(f"- Direct responses: {len(direct_responses)}\\n")
        f.write(f"- RAG responses: {len(rag_responses)}\\n")
    print("\\n✓ PROJECT_NOTES.md updated")
except Exception as e:
    print(f"\\n⚠️  Could not update PROJECT_NOTES.md: {e}")

print("\\n✓ Notebook complete!")
print("   Next steps:")
print("   1. Set HF_TOKEN in .env to run model evaluation")
print("   2. Set PILOT_MODE=False for full evaluation")  
print("   3. Fill in actual results in research_report_draft.txt")
print("   4. Update group_contribution.txt with real names and work")
"""))

# =============================================================================
# BUILD NOTEBOOK
# =============================================================================
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.7",
        },
    },
    "cells": cells,
}

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"Notebook written to: {NOTEBOOK_PATH}")
print(f"Cells: {len(cells)}")
print(f"File size: {NOTEBOOK_PATH.stat().st_size:,} bytes")

