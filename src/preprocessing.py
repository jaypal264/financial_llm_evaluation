"""
preprocessing.py
================
Cleans and prepares datasets for evaluation.
Handles missing values, truncation, sampling, and text normalization.
"""

import re
import logging
import random
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

UNIFIED_COLUMNS = [
    "dataset_name", "question_id", "question", "context",
    "reference_answer", "evidence", "company", "document_name",
    "reasoning_type", "reference_program",
]


# ---------------------------------------------------------------------------
# Main preprocessing
# ---------------------------------------------------------------------------

def preprocess_dataset(
    df: pd.DataFrame,
    max_context_tokens: int = 2000,
    min_question_len: int = 5,
) -> pd.DataFrame:
    """
    Cleans a unified-schema DataFrame:
      - Ensures all required columns exist
      - Drops rows with missing question or reference_answer
      - Truncates very long contexts
      - Strips whitespace
    """
    if df.empty:
        return df

    # Ensure all columns exist
    for col in UNIFIED_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # Strip whitespace
    for col in ["question", "reference_answer", "context", "evidence"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    # Drop rows with empty question
    before = len(df)
    df = df[df["question"].str.len() >= min_question_len].copy()
    if len(df) < before:
        logger.warning("Dropped %d rows with missing/short questions", before - len(df))

    # Truncate context to avoid token limit issues
    # Approximate: 1 token ≈ 4 chars
    char_limit = max_context_tokens * 4
    long_contexts = df["context"].str.len() > char_limit
    if long_contexts.any():
        logger.info(
            "Truncating %d long contexts to ~%d tokens",
            long_contexts.sum(), max_context_tokens
        )
        df.loc[long_contexts, "context"] = df.loc[long_contexts, "context"].str[:char_limit] + "\n[...TRUNCATED...]"

    # Reset index
    df = df.reset_index(drop=True)
    logger.info("Preprocessed: %d questions", len(df))
    return df


def sample_pilot(
    df: pd.DataFrame,
    n: int = 5,
    seed: int = 42,
    stratify_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Returns a small pilot sample for pipeline testing.
    If stratify_col is given, samples proportionally across its values.
    """
    if df.empty or len(df) <= n:
        return df

    if stratify_col and stratify_col in df.columns:
        groups = df[stratify_col].unique()
        per_group = max(1, n // len(groups))
        samples = []
        for g in groups:
            g_df = df[df[stratify_col] == g]
            samples.append(g_df.sample(min(per_group, len(g_df)), random_state=seed))
        sampled = pd.concat(samples).sample(min(n, sum(len(s) for s in samples)), random_state=seed)
    else:
        sampled = df.sample(n, random_state=seed)

    return sampled.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Text normalization for evaluation
# ---------------------------------------------------------------------------

def normalize_answer(text: str) -> str:
    """
    Normalizes a string answer for comparison:
      - Lowercase
      - Remove punctuation (except decimal point)
      - Strip currency symbols and thousands separators
      - Normalize whitespace
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.lower().strip()
    # Remove currency symbols
    text = re.sub(r"[\$€£¥₹]", "", text)
    # Remove thousands separators (comma between digits)
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    # Remove percent sign but keep value
    text = text.replace("%", " percent")
    # Remove parentheses (often used for negative numbers in accounting)
    text = re.sub(r"\(([0-9.,]+)\)", r"-\1", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove trailing punctuation
    text = re.sub(r"[^\w\s\.\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_number(text: str) -> Optional[float]:
    """
    Attempts to extract the primary numeric value from a string.
    Returns None if no number found.
    """
    if not isinstance(text, str):
        text = str(text)
    # Remove currency and commas
    text = re.sub(r"[\$€£¥₹,]", "", text)
    # Handle parentheses as negative
    text = re.sub(r"\(([0-9.]+)\)", r"-\1", text)
    # Find first number (including decimals and negatives)
    match = re.search(r"-?\d+\.?\d*", text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def is_numerical_answer(text: str) -> bool:
    """Returns True if the answer is primarily numerical."""
    return extract_number(normalize_answer(text)) is not None


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

def compute_dataset_statistics(datasets: dict) -> pd.DataFrame:
    """
    Computes summary statistics for each dataset.

    Returns a DataFrame with one row per dataset.
    """
    rows = []
    for name, df in datasets.items():
        if df.empty:
            rows.append({
                "dataset": name, "n_questions": 0,
                "note": "Failed to load"
            })
            continue

        numerical_mask = df["reference_answer"].apply(is_numerical_answer)

        row = {
            "dataset": name,
            "n_questions": len(df),
            "avg_question_len_chars": df["question"].str.len().mean(),
            "avg_context_len_chars": df["context"].str.len().mean(),
            "pct_numerical_answers": 100 * numerical_mask.mean(),
            "pct_missing_context": 100 * (df["context"].str.len() == 0).mean(),
            "pct_missing_answer": 100 * (df["reference_answer"].str.len() == 0).mean(),
            "unique_companies": df["company"].nunique() if "company" in df.columns else "N/A",
            "reasoning_types": str(df["reasoning_type"].value_counts().to_dict()) if "reasoning_type" in df.columns else "N/A",
        }
        rows.append(row)

    return pd.DataFrame(rows)

