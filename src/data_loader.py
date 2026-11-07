"""
data_loader.py
==============
Downloads and loads the three financial QA datasets:
  - FinanceBench (PatronusAI/financebench)
  - FinTextQA    (GPS-Lab/FinTextQA)
  - FinQA        (ibm-research/finqa)

All datasets are downloaded from Hugging Face and saved locally to data/raw/.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# FinanceBench
# ---------------------------------------------------------------------------

def load_financebench(
    hf_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Downloads the public FinanceBench sample (150 questions) from Hugging Face.

    Returns a DataFrame with unified schema.

    Note: The public HF dataset is a 150-question sample.
          The full benchmark has 10,231 questions but is not publicly released.
    """
    from datasets import load_dataset

    cache_path = DATA_RAW / "financebench_raw.json"

    if cache_path.exists() and not force_download:
        logger.info("Loading FinanceBench from local cache: %s", cache_path)
        with open(cache_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        return _financebench_to_dataframe(records)

    logger.info("Downloading FinanceBench from Hugging Face...")
    try:
        ds = load_dataset(
            "PatronusAI/financebench",
            token=hf_token,
            cache_dir=cache_dir,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to download FinanceBench. Check your HF_TOKEN and internet access.\n"
            f"Original error: {e}"
        ) from e

    # The dataset may be in a single split or 'train'
    split_name = "train" if "train" in ds else list(ds.keys())[0]
    records = [dict(row) for row in ds[split_name]]

    # Cache locally
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    logger.info("FinanceBench cached to %s (%d records)", cache_path, len(records))

    return _financebench_to_dataframe(records)


def _financebench_to_dataframe(records: list) -> pd.DataFrame:
    """Convert raw FinanceBench records to unified DataFrame schema."""
    rows = []
    for i, r in enumerate(records):
        # Handle both camelCase and snake_case field names
        rows.append({
            "dataset_name": "financebench",
            "question_id": r.get("financebench_id", r.get("id", f"fb_{i:04d}")),
            "question": r.get("question", ""),
            "context": _extract_financebench_context(r),
            "reference_answer": r.get("answer", r.get("Answer", "")),
            "evidence": json.dumps(r.get("evidence", r.get("Evidence", []))) if isinstance(r.get("evidence", r.get("Evidence")), list) else str(r.get("evidence", r.get("Evidence", ""))),
            "company": r.get("company", r.get("Company", "")),
            "document_name": r.get("doc_name", r.get("document_name", "")),
            "reasoning_type": r.get("question_type", r.get("domain", "")),
            "reference_program": None,  # Not applicable for FinanceBench
        })
    df = pd.DataFrame(rows)
    logger.info("FinanceBench loaded: %d questions", len(df))
    return df


def _extract_financebench_context(r: dict) -> str:
    """Extract context/evidence text from a FinanceBench record."""
    # Try different field names used across dataset versions
    for field in ["context", "evidence_text", "passage", "text"]:
        if field in r and r[field]:
            return str(r[field])
    # Fallback: serialize evidence list
    evidence = r.get("evidence", r.get("Evidence", []))
    if isinstance(evidence, list):
        texts = []
        for ev in evidence:
            if isinstance(ev, dict):
                texts.append(ev.get("evidence_text", ev.get("text", str(ev))))
            else:
                texts.append(str(ev))
        return "\n".join(texts)
    return str(evidence) if evidence else ""


# ---------------------------------------------------------------------------
# FinTextQA
# ---------------------------------------------------------------------------

def load_fintextqa(
    hf_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Downloads FinTextQA from official repository.

    The public dataset contains the policy/regulation subset (~240 QA pairs).
    The textbook subset described in the ACL 2024 paper is NOT publicly available.

    Returns a DataFrame with unified schema.
    """
    import requests

    cache_path = DATA_RAW / "fintextqa_raw.json"

    if cache_path.exists() and not force_download:
        logger.info("Loading FinTextQA from local cache: %s", cache_path)
        with open(cache_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        return _fintextqa_to_dataframe(records)

    logger.info("Downloading FinTextQA...")
    all_records = []

    # Method 1: Hugging Face dataset if available
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "GPS-Lab/FinTextQA",
            token=hf_token,
            cache_dir=cache_dir,
        )
        for split_name, split_ds in ds.items():
            for row in split_ds:
                record = dict(row)
                record["_split"] = split_name
                all_records.append(record)
    except Exception as e:
        logger.warning("HF load_dataset failed for FinTextQA: %s. Trying official GitHub...", e)

    # Method 2: Official GitHub repository release (Excel file with 240+ policy QA pairs)
    if not all_records:
        github_excel_url = (
            "https://raw.githubusercontent.com/AlexJJJChen/FinTextQA/"
            "afd705ffc35cf513536a4659ce2058eb105314fd/manual-qa-regulation%20policy.xlsx"
        )
        excel_path = DATA_RAW / "fintextqa_manual.xlsx"
        if not excel_path.exists():
            resp = requests.get(github_excel_url, allow_redirects=True, timeout=60)
            if resp.status_code == 200:
                with open(excel_path, "wb") as f:
                    f.write(resp.content)
            else:
                raise RuntimeError(f"Failed to download FinTextQA from GitHub (HTTP {resp.status_code})")

        excel_file = pd.ExcelFile(excel_path)
        sheet = "Sheet1" if "Sheet1" in excel_file.sheet_names else excel_file.sheet_names[0]
        df_raw = pd.read_excel(excel_path, sheet_name=sheet)
        for idx, row in df_raw.iterrows():
            q = str(row.get("question", "")).strip()
            if not q or q.lower() == "nan":
                continue
            all_records.append({
                "id": f"ftqa_{idx:04d}",
                "question": q,
                "context": str(row.get("reference", row.get("document", ""))),
                "answer": str(row.get("ground truth", "")),
                "source": str(row.get("document", "")),
                "category": "policy_regulation",
                "_split": "policy_regulation",
            })

    if not all_records:
        raise RuntimeError("No records could be retrieved for FinTextQA.")

    # Cache locally
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)
    logger.info("FinTextQA cached to %s (%d records)", cache_path, len(all_records))

    return _fintextqa_to_dataframe(all_records)


def _fintextqa_to_dataframe(records: list) -> pd.DataFrame:
    """Convert raw FinTextQA records to unified DataFrame schema."""
    rows = []
    for i, r in enumerate(records):
        rows.append({
            "dataset_name": "fintextqa",
            "question_id": r.get("id", r.get("question_id", f"ftqa_{i:04d}")),
            "question": r.get("question", r.get("Question", "")),
            "context": r.get("context", r.get("Context", r.get("passage", ""))),
            "reference_answer": r.get("answer", r.get("Answer", "")),
            "evidence": r.get("source", r.get("Source", "")),
            "company": None,  # Not applicable for FinTextQA
            "document_name": r.get("source", r.get("Source", "")),
            "reasoning_type": r.get("category", r.get("type", "policy_regulation")),
            "reference_program": None,
            "_split": r.get("_split", "unknown"),
        })
    df = pd.DataFrame(rows)
    logger.info("FinTextQA loaded: %d questions", len(df))
    return df


# ---------------------------------------------------------------------------
# FinQA
# ---------------------------------------------------------------------------

def load_finqa(
    hf_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    split: str = "test",
) -> pd.DataFrame:
    """
    Downloads FinQA from Hugging Face or official GitHub (czyssrs/FinQA).

    Uses the official test split for final evaluation.
    Returns a DataFrame with unified schema including numerical reasoning programs.
    """
    import requests

    cache_path = DATA_RAW / f"finqa_{split}_raw.json"

    if cache_path.exists() and not force_download:
        logger.info("Loading FinQA (%s) from local cache: %s", split, cache_path)
        with open(cache_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        return _finqa_to_dataframe(records)

    logger.info("Downloading FinQA (%s split)...", split)
    records = []

    # Method 1: Try Hugging Face
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "ibm-research/finqa",
            token=hf_token,
            cache_dir=cache_dir,
        )
        split_map = {"train": "train", "validation": "validation", "dev": "validation", "test": "test"}
        split_key = split_map.get(split, split)
        if split_key in ds:
            records = [dict(row) for row in ds[split_key]]
    except Exception as e:
        logger.warning("HF load_dataset failed for FinQA: %s. Trying official GitHub...", e)

    # Method 2: Official GitHub repository (czyssrs/FinQA)
    if not records:
        github_split_map = {"train": "train", "validation": "dev", "dev": "dev", "test": "test"}
        gh_split = github_split_map.get(split, split)
        url = f"https://raw.githubusercontent.com/czyssrs/FinQA/master/dataset/{gh_split}.json"
        resp = requests.get(url, allow_redirects=True, timeout=60)
        if resp.status_code == 200:
            records = resp.json()
        else:
            raise RuntimeError(f"Failed to download FinQA from GitHub (HTTP {resp.status_code})")

    # Cache locally
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    logger.info("FinQA (%s) cached to %s (%d records)", split, cache_path, len(records))

    return _finqa_to_dataframe(records)


def _finqa_to_dataframe(records: list) -> pd.DataFrame:
    """Convert raw FinQA records to unified DataFrame schema."""
    rows = []
    for i, r in enumerate(records):
        # Build context from table + pre/post text
        context = _build_finqa_context(r)
        rows.append({
            "dataset_name": "finqa",
            "question_id": r.get("id", f"finqa_{i:04d}"),
            "question": r.get("question", r.get("qa", {}).get("question", "")),
            "context": context,
            "reference_answer": _extract_finqa_answer(r),
            "evidence": json.dumps(r.get("gold_inds", {})),
            "company": r.get("filename", "").split("/")[0] if r.get("filename") else None,
            "document_name": r.get("filename", ""),
            "reasoning_type": "numerical_reasoning",
            "reference_program": json.dumps(r.get("qa", {}).get("program", r.get("program", []))),
        })
    df = pd.DataFrame(rows)
    logger.info("FinQA loaded: %d questions", len(df))
    return df


def _extract_finqa_answer(r: dict) -> str:
    """Extract the final answer from a FinQA record."""
    # IBM HF version nests under 'qa'
    qa = r.get("qa", {})
    if isinstance(qa, dict):
        return str(qa.get("exe_ans", qa.get("answer", "")))
    return str(r.get("answer", r.get("exe_ans", "")))


def _build_finqa_context(r: dict) -> str:
    """Build a text context from FinQA table + pre/post text."""
    parts = []

    pre_text = r.get("pre_text", [])
    if isinstance(pre_text, list):
        parts.extend(pre_text)
    elif pre_text:
        parts.append(str(pre_text))

    table = r.get("table", r.get("table_ori", []))
    if table:
        if isinstance(table, list) and len(table) > 0:
            if isinstance(table[0], list):
                # List of rows
                table_text = "\n".join([" | ".join([str(cell) for cell in row]) for row in table])
            else:
                table_text = " | ".join([str(cell) for cell in table])
            parts.append(f"\n[TABLE]\n{table_text}\n[/TABLE]")
        elif isinstance(table, str):
            parts.append(f"\n[TABLE]\n{table}\n[/TABLE]")

    post_text = r.get("post_text", [])
    if isinstance(post_text, list):
        parts.extend(post_text)
    elif post_text:
        parts.append(str(post_text))

    return "\n".join([str(p) for p in parts if p])


# ---------------------------------------------------------------------------
# Load all datasets
# ---------------------------------------------------------------------------

def load_all_datasets(
    hf_token: Optional[str] = None,
    force_download: bool = False,
) -> dict:
    """
    Downloads and loads all three datasets.

    Returns:
        dict with keys: 'financebench', 'fintextqa', 'finqa'
        Each value is a pd.DataFrame in unified schema.
    """
    datasets = {}

    logger.info("=" * 60)
    logger.info("Loading FinanceBench...")
    try:
        datasets["financebench"] = load_financebench(
            hf_token=hf_token, force_download=force_download
        )
    except Exception as e:
        logger.error("FinanceBench load failed: %s", e)
        datasets["financebench"] = pd.DataFrame()

    logger.info("=" * 60)
    logger.info("Loading FinTextQA...")
    try:
        datasets["fintextqa"] = load_fintextqa(
            hf_token=hf_token, force_download=force_download
        )
    except Exception as e:
        logger.error("FinTextQA load failed: %s", e)
        datasets["fintextqa"] = pd.DataFrame()

    logger.info("=" * 60)
    logger.info("Loading FinQA (test split)...")
    try:
        datasets["finqa"] = load_finqa(
            hf_token=hf_token, force_download=force_download, split="test"
        )
    except Exception as e:
        logger.error("FinQA load failed: %s", e)
        datasets["finqa"] = pd.DataFrame()

    logger.info("=" * 60)
    logger.info("All datasets loaded.")
    for name, df in datasets.items():
        logger.info("  %s: %d questions", name, len(df))

    return datasets

