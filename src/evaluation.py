"""
evaluation.py
=============
Evaluation metrics for financial QA:

  - Exact Match (EM)
  - Token-level F1
  - ROUGE-L
  - Numeric Accuracy (with tolerance)
  - BERTScore (optional, requires bert-score)
  - Semantic Similarity (sentence-transformers)
  - Hallucination/faithfulness proxy

All functions accept string inputs and return float scores.
"""

import re
import string
import logging
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text normalization helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase, remove punctuation, strip whitespace."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _tokenize(text: str) -> list[str]:
    return _normalize(text).split()


# ---------------------------------------------------------------------------
# Exact Match
# ---------------------------------------------------------------------------

def exact_match(prediction: str, reference: str) -> float:
    """
    Returns 1.0 if normalized prediction == normalized reference, else 0.0.
    """
    return float(_normalize(prediction) == _normalize(reference))


# ---------------------------------------------------------------------------
# Token-level F1
# ---------------------------------------------------------------------------

def token_f1(prediction: str, reference: str) -> float:
    """
    Computes token-level F1 score (as used in SQuAD evaluation).
    """
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------

def rouge_l(prediction: str, reference: str) -> float:
    """
    Computes ROUGE-L F1 score using the longest common subsequence.
    """
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs_len = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs_len / len(pred_tokens)
    recall = lcs_len / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list, b: list) -> int:
    """DP-based LCS length."""
    m, n = len(a), len(b)
    # Space-optimized DP
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def rouge_l_batch(predictions: list[str], references: list[str]) -> list[float]:
    """
    Computes ROUGE-L for a list of (prediction, reference) pairs.
    Falls back to pure Python implementation; uses rouge-score library if available.
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        scores = []
        for pred, ref in zip(predictions, references):
            try:
                score = scorer.score(ref, pred)
                scores.append(score["rougeL"].fmeasure)
            except Exception:
                scores.append(rouge_l(pred, ref))
        return scores
    except ImportError:
        return [rouge_l(p, r) for p, r in zip(predictions, references)]


# ---------------------------------------------------------------------------
# Numeric accuracy
# ---------------------------------------------------------------------------

def _extract_number(text: str) -> Optional[float]:
    """
    Extracts the primary numeric value from text.
    Handles: currency symbols, commas, percent signs, parentheses (negative).
    """
    if not isinstance(text, str):
        text = str(text)
    # Remove currency symbols
    text = re.sub(r"[\$€£¥₹]", "", text)
    # Remove thousands separators
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    # Parentheses → negative
    text = re.sub(r"\(([0-9.]+)\)", r"-\1", text)

    # Find first number
    match = re.search(r"-?\d+\.?\d*", text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def _is_percent(text: str) -> bool:
    return "%" in text or "percent" in text.lower()


def numeric_accuracy(
    prediction: str,
    reference: str,
    tolerance: float = 0.005,
) -> float:
    """
    Computes numeric accuracy with relative tolerance.

    - Extracts numbers from both prediction and reference.
    - Returns 1.0 if |pred - ref| / (|ref| + 1e-9) <= tolerance.
    - Handles percentages (checks unit consistency).
    - Returns 0.0 if either value cannot be parsed.

    Default tolerance: 0.5% relative error.
    """
    pred_num = _extract_number(prediction)
    ref_num = _extract_number(reference)

    if pred_num is None or ref_num is None:
        return 0.0

    # Unit consistency check for percentages
    pred_pct = _is_percent(prediction)
    ref_pct = _is_percent(reference)

    if pred_pct and not ref_pct:
        # prediction in %, reference as decimal (e.g., pred=5.2%, ref=0.052)
        if abs(ref_num) < 1.0:
            pred_num = pred_num / 100.0
    elif ref_pct and not pred_pct:
        if abs(pred_num) < 1.0:
            pred_num = pred_num * 100.0

    relative_error = abs(pred_num - ref_num) / (abs(ref_num) + 1e-9)
    return float(relative_error <= tolerance)


# ---------------------------------------------------------------------------
# BERTScore
# ---------------------------------------------------------------------------

def bertscore_batch(
    predictions: list[str],
    references: list[str],
    model_type: str = "distilbert-base-uncased",
    batch_size: int = 32,
    device: Optional[str] = None,
) -> dict:
    """
    Computes BERTScore F1 for a batch of predictions.

    Returns dict with 'precision', 'recall', 'f1' keys (lists of floats).
    Falls back to zeros if bert-score is not installed.
    """
    try:
        from bert_score import score as bert_score_fn
        P, R, F = bert_score_fn(
            predictions,
            references,
            model_type=model_type,
            lang="en",
            batch_size=batch_size,
            device=device,
            verbose=False,
        )
        return {
            "precision": P.tolist(),
            "recall": R.tolist(),
            "f1": F.tolist(),
        }
    except ImportError:
        logger.warning("bert-score not installed. BERTScore will be 0.0.")
        zeros = [0.0] * len(predictions)
        return {"precision": zeros, "recall": zeros, "f1": zeros}
    except Exception as e:
        logger.error("BERTScore computation failed: %s", e)
        zeros = [0.0] * len(predictions)
        return {"precision": zeros, "recall": zeros, "f1": zeros}


# ---------------------------------------------------------------------------
# Semantic similarity
# ---------------------------------------------------------------------------

_sem_model = None

def semantic_similarity_batch(
    predictions: list[str],
    references: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> list[float]:
    """
    Computes cosine similarity between sentence embeddings of prediction and reference.
    Uses sentence-transformers.
    """
    global _sem_model
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity

        if _sem_model is None:
            _sem_model = SentenceTransformer(model_name)

        pred_embs = _sem_model.encode(predictions, convert_to_numpy=True, show_progress_bar=False)
        ref_embs = _sem_model.encode(references, convert_to_numpy=True, show_progress_bar=False)

        sims = cosine_similarity(pred_embs, ref_embs)
        return [float(sims[i, i]) for i in range(len(predictions))]
    except ImportError:
        logger.warning("sentence-transformers not installed. Semantic similarity will be 0.0.")
        return [0.0] * len(predictions)
    except Exception as e:
        logger.error("Semantic similarity failed: %s", e)
        return [0.0] * len(predictions)


# ---------------------------------------------------------------------------
# Faithfulness / hallucination proxy
# ---------------------------------------------------------------------------

def faithfulness_score(answer: str, context: str) -> float:
    """
    Approximate faithfulness: fraction of answer tokens found in the context.
    
    This is a simple lexical proxy, NOT a full NLI-based faithfulness check.
    A high score means the model's answer tokens come from the context.
    A low score may indicate hallucination (but not definitively).
    """
    if not answer or not context:
        return 0.0

    answer_tokens = set(_tokenize(answer))
    context_tokens = set(_tokenize(context))

    if not answer_tokens:
        return 0.0

    # Remove stopwords for a more meaningful signal
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                 "being", "have", "has", "had", "do", "does", "did", "will",
                 "would", "could", "should", "may", "might", "shall", "can",
                 "of", "in", "on", "at", "to", "for", "by", "from", "with",
                 "and", "or", "but", "not", "this", "that", "it", "its",
                 "i", "you", "he", "she", "we", "they", "what", "which",
                 "how", "when", "where", "who"}

    content_tokens = answer_tokens - stopwords
    if not content_tokens:
        return 1.0  # Only stopwords — can't determine

    supported = content_tokens & context_tokens
    return len(supported) / len(content_tokens)


# ---------------------------------------------------------------------------
# Abstention detection
# ---------------------------------------------------------------------------

ABSTENTION_PHRASES = [
    "cannot determine", "cannot answer", "not enough information",
    "insufficient information", "not provided", "i don't know",
    "i do not know", "unable to answer", "not mentioned",
    "no information", "cannot be determined", "not available",
    "insufficient context", "based on the context provided, there is no",
]

def is_abstention(answer: str) -> bool:
    """Returns True if the model appears to have abstained from answering."""
    if not answer:
        return True
    lower = answer.lower()
    return any(phrase in lower for phrase in ABSTENTION_PHRASES)


# ---------------------------------------------------------------------------
# Comprehensive evaluation
# ---------------------------------------------------------------------------

def evaluate_responses(
    responses_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    compute_bertscore: bool = False,
    compute_semantic: bool = True,
    numeric_tolerance: float = 0.005,
) -> pd.DataFrame:
    """
    Evaluates all model responses in responses_df against references in reference_df.

    Args:
        responses_df: DataFrame with columns [model_key, question_id, dataset_name,
                                              generated_answer, context_used, latency_seconds, error]
        reference_df: DataFrame with columns [question_id, reference_answer, context]
        compute_bertscore: Whether to compute BERTScore (slow)
        compute_semantic: Whether to compute semantic similarity
        numeric_tolerance: Relative tolerance for numeric accuracy

    Returns:
        DataFrame with one row per (model, question) and all metric columns.
    """
    # Merge on question_id
    merged = responses_df.merge(
        reference_df[["question_id", "reference_answer", "context", "reasoning_type"]],
        on="question_id",
        how="left",
    )

    results = []
    for _, row in merged.iterrows():
        pred = str(row.get("generated_answer", ""))
        ref = str(row.get("reference_answer", ""))
        ctx = str(row.get("context", row.get("context_used", "")))

        em = exact_match(pred, ref)
        f1 = token_f1(pred, ref)
        rl = rouge_l(pred, ref)
        num_acc = numeric_accuracy(pred, ref, tolerance=numeric_tolerance)
        faith = faithfulness_score(pred, ctx)
        abstain = is_abstention(pred)
        is_error = bool(row.get("error"))

        results.append({
            "model_key": row.get("model_key"),
            "question_id": row.get("question_id"),
            "dataset_name": row.get("dataset_name", row.get("dataset_name_x", "")),
            "reasoning_type": row.get("reasoning_type"),
            "exact_match": em,
            "token_f1": f1,
            "rouge_l": rl,
            "numeric_accuracy": num_acc,
            "faithfulness": faith,
            "is_abstention": abstain,
            "is_error": is_error,
            "latency_seconds": row.get("latency_seconds", 0.0),
            "prompt_tokens": row.get("prompt_tokens", 0),
            "completion_tokens": row.get("completion_tokens", 0),
            "generated_answer": pred,
            "reference_answer": ref,
        })

    eval_df = pd.DataFrame(results)

    # Optional: BERTScore
    if compute_bertscore and not eval_df.empty:
        logger.info("Computing BERTScore...")
        bs = bertscore_batch(
            eval_df["generated_answer"].tolist(),
            eval_df["reference_answer"].tolist(),
        )
        eval_df["bertscore_f1"] = bs["f1"]
    else:
        eval_df["bertscore_f1"] = 0.0

    # Optional: Semantic similarity
    if compute_semantic and not eval_df.empty:
        logger.info("Computing semantic similarity...")
        eval_df["semantic_similarity"] = semantic_similarity_batch(
            eval_df["generated_answer"].tolist(),
            eval_df["reference_answer"].tolist(),
        )
    else:
        eval_df["semantic_similarity"] = 0.0

    return eval_df


def compute_summary_metrics(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates evaluation results by (model_key, dataset_name).

    Returns a summary DataFrame with mean metrics.
    """
    agg_cols = [
        "exact_match", "token_f1", "rouge_l", "numeric_accuracy",
        "faithfulness", "bertscore_f1", "semantic_similarity",
        "latency_seconds", "prompt_tokens", "completion_tokens",
    ]
    agg_cols = [c for c in agg_cols if c in eval_df.columns]

    summary = (
        eval_df.groupby(["model_key", "dataset_name"])[agg_cols]
        .agg(["mean", "std", "count"])
        .round(4)
    )

    # Add error rate and abstention rate
    extra = eval_df.groupby(["model_key", "dataset_name"]).agg(
        error_rate=("is_error", "mean"),
        abstention_rate=("is_abstention", "mean"),
        n_questions=("question_id", "count"),
    ).round(4)

    return summary, extra

