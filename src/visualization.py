"""
visualization.py
================
Publication-quality charts for the Financial LLM Evaluation project.

All functions save figures to results/plots/ and return the matplotlib Figure.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Style settings
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

MODEL_COLORS = {
    "qwen2.5-7b": "#2196F3",
    "llama3.1-8b": "#FF5722",
    "mistral-7b": "#9C27B0",
    "gemma2-9b": "#4CAF50",
    "phi3.5-mini": "#FF9800",
    "deepseek-r1-7b": "#00BCD4",
}

DATASET_COLORS = {
    "financebench": "#1565C0",
    "fintextqa": "#2E7D32",
    "finqa": "#B71C1C",
}

MODEL_DISPLAY = {
    "qwen2.5-7b": "Qwen2.5-7B",
    "llama3.1-8b": "Llama-3.1-8B",
    "mistral-7b": "Mistral-7B",
    "gemma2-9b": "Gemma-2-9B",
    "phi3.5-mini": "Phi-3.5-mini",
    "deepseek-r1-7b": "DeepSeek-R1-7B",
}


def _save(fig: plt.Figure, filename: str) -> Path:
    path = PLOTS_DIR / filename
    fig.savefig(path, bbox_inches="tight", dpi=150)
    logger.info("Saved plot: %s", path)
    return path


# ---------------------------------------------------------------------------
# 1. Grouped bar chart: metric per model per dataset
# ---------------------------------------------------------------------------

def plot_metric_comparison(
    summary_df: pd.DataFrame,
    metric: str = "token_f1",
    title: Optional[str] = None,
    filename: Optional[str] = None,
) -> plt.Figure:
    """
    Grouped bar chart comparing models across datasets for a given metric.
    summary_df must have columns: model_key, dataset_name, <metric>
    """
    pivot = summary_df.pivot_table(
        index="model_key", columns="dataset_name", values=metric, aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(pivot.index))
    width = 0.25
    n_datasets = len(pivot.columns)
    offsets = np.linspace(-width, width, n_datasets)

    for i, dataset in enumerate(pivot.columns):
        values = pivot[dataset].fillna(0).values
        bars = ax.bar(
            x + offsets[i], values, width,
            label=dataset.capitalize(),
            color=DATASET_COLORS.get(dataset, f"C{i}"),
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=7, rotation=45
                )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_DISPLAY.get(m, m) for m in pivot.index],
        rotation=25, ha="right"
    )
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title or f"{metric.replace('_', ' ').title()} by Model and Dataset")
    ax.legend(loc="upper right")
    ax.set_ylim(0, min(1.1, ax.get_ylim()[1] * 1.15))
    plt.tight_layout()

    fname = filename or f"bar_{metric}.png"
    _save(fig, fname)
    return fig


# ---------------------------------------------------------------------------
# 2. Heatmap: models × metrics
# ---------------------------------------------------------------------------

def plot_heatmap(
    summary_df: pd.DataFrame,
    metrics: Optional[list] = None,
    dataset: Optional[str] = None,
    title: Optional[str] = None,
    filename: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap of model × metric scores.
    If dataset is specified, filters to that dataset.
    """
    if metrics is None:
        metrics = ["exact_match", "token_f1", "rouge_l", "numeric_accuracy",
                   "semantic_similarity", "faithfulness"]

    df = summary_df.copy()
    if dataset:
        df = df[df["dataset_name"] == dataset]

    pivot = df.groupby("model_key")[metrics].mean()
    pivot.index = [MODEL_DISPLAY.get(m, m) for m in pivot.index]
    pivot.columns = [c.replace("_", " ").title() for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(len(metrics) * 1.4, len(pivot) * 0.9 + 1))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    suffix = f" ({dataset})" if dataset else " (All Datasets)"
    ax.set_title(title or f"Model Performance Heatmap{suffix}")
    plt.tight_layout()

    dname = dataset or "all"
    fname = filename or f"heatmap_{dname}.png"
    _save(fig, fname)
    return fig


# ---------------------------------------------------------------------------
# 3. Radar chart
# ---------------------------------------------------------------------------

def plot_radar(
    summary_df: pd.DataFrame,
    metrics: Optional[list] = None,
    model_keys: Optional[list] = None,
    title: str = "Model Capability Radar",
    filename: str = "radar_chart.png",
) -> plt.Figure:
    """
    Radar (spider) chart comparing models across multiple metrics.
    """
    if metrics is None:
        metrics = ["exact_match", "token_f1", "rouge_l",
                   "faithfulness", "semantic_similarity"]
    if model_keys is None:
        model_keys = summary_df["model_key"].unique().tolist()

    # Aggregate across all datasets
    agg = summary_df.groupby("model_key")[metrics].mean()

    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    labels = [m.replace("_", " ").title() for m in metrics]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for mkey in model_keys:
        if mkey not in agg.index:
            continue
        values = agg.loc[mkey, metrics].fillna(0).tolist()
        values += values[:1]
        color = MODEL_COLORS.get(mkey, "gray")
        ax.plot(angles, values, "o-", linewidth=2, color=color,
                label=MODEL_DISPLAY.get(mkey, mkey))
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], size=8)
    ax.set_title(title, size=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    plt.tight_layout()

    _save(fig, filename)
    return fig


# ---------------------------------------------------------------------------
# 4. Latency vs accuracy scatter
# ---------------------------------------------------------------------------

def plot_latency_vs_accuracy(
    summary_df: pd.DataFrame,
    accuracy_metric: str = "token_f1",
    title: str = "Latency vs Accuracy Trade-off",
    filename: str = "latency_vs_accuracy.png",
) -> plt.Figure:
    """
    Scatter plot: average latency (x) vs accuracy metric (y) per model.
    Bubble size = number of questions.
    """
    agg = summary_df.groupby("model_key").agg(
        latency=("latency_seconds", "mean"),
        accuracy=(accuracy_metric, "mean"),
        count=("question_id", "count") if "question_id" in summary_df.columns else (accuracy_metric, "count"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(9, 6))

    for _, row in agg.iterrows():
        mkey = row["model_key"]
        color = MODEL_COLORS.get(mkey, "gray")
        size = max(100, row.get("count", 100) * 5)
        ax.scatter(row["latency"], row["accuracy"],
                   s=size, color=color, alpha=0.75, edgecolors="white", linewidth=1.5)
        ax.annotate(
            MODEL_DISPLAY.get(mkey, mkey),
            (row["latency"], row["accuracy"]),
            textcoords="offset points", xytext=(8, 4), fontsize=9,
        )

    ax.set_xlabel("Average Latency (seconds)")
    ax.set_ylabel(accuracy_metric.replace("_", " ").title())
    ax.set_title(title)
    plt.tight_layout()

    _save(fig, filename)
    return fig


# ---------------------------------------------------------------------------
# 5. Direct vs RAG comparison
# ---------------------------------------------------------------------------

def plot_rag_comparison(
    direct_df: pd.DataFrame,
    rag_df: pd.DataFrame,
    metric: str = "token_f1",
    title: Optional[str] = None,
    filename: str = "rag_vs_direct.png",
) -> plt.Figure:
    """
    Bar chart comparing Direct LLM vs RAG for each model on a given metric.
    """
    direct_agg = direct_df.groupby("model_key")[metric].mean().rename("direct")
    rag_agg = rag_df.groupby("model_key")[metric].mean().rename("rag")
    combined = pd.concat([direct_agg, rag_agg], axis=1).fillna(0)

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(combined))
    width = 0.35

    bars1 = ax.bar(x - width / 2, combined["direct"], width,
                   label="Direct LLM", color="#1565C0", alpha=0.85)
    bars2 = ax.bar(x + width / 2, combined["rag"], width,
                   label="RAG", color="#E65100", alpha=0.85)

    # Improvement arrows
    for i, (d, r) in enumerate(zip(combined["direct"], combined["rag"])):
        diff = r - d
        color = "#2E7D32" if diff >= 0 else "#C62828"
        ax.annotate(
            f"{diff:+.3f}",
            xy=(x[i] + width / 2, r),
            xytext=(0, 5), textcoords="offset points",
            ha="center", va="bottom", fontsize=8, color=color,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_DISPLAY.get(m, m) for m in combined.index],
        rotation=25, ha="right"
    )
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title or f"Direct LLM vs RAG — {metric.replace('_', ' ').title()}")
    ax.legend()
    ax.set_ylim(0, min(1.1, combined.max().max() * 1.25))
    plt.tight_layout()

    _save(fig, filename)
    return fig


# ---------------------------------------------------------------------------
# 6. Question complexity analysis
# ---------------------------------------------------------------------------

def plot_complexity_analysis(
    eval_df: pd.DataFrame,
    metric: str = "token_f1",
    title: str = "Performance by Question Category",
    filename: str = "complexity_analysis.png",
) -> plt.Figure:
    """
    Grouped bar chart of model performance by question reasoning_type.
    """
    if "reasoning_type" not in eval_df.columns:
        logger.warning("No 'reasoning_type' column found. Skipping complexity analysis.")
        return None

    pivot = eval_df.groupby(["reasoning_type", "model_key"])[metric].mean().unstack(fill_value=0)
    pivot.columns = [MODEL_DISPLAY.get(c, c) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", ax=ax, colormap="tab10", width=0.75, edgecolor="white")

    ax.set_xlabel("Question Category")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    plt.tight_layout()

    _save(fig, filename)
    return fig


# ---------------------------------------------------------------------------
# 7. Token usage & cost
# ---------------------------------------------------------------------------

def plot_token_usage(
    responses_df: pd.DataFrame,
    title: str = "Average Token Usage per Model",
    filename: str = "token_usage.png",
) -> plt.Figure:
    """
    Stacked bar chart of prompt tokens + completion tokens per model.
    """
    agg = responses_df.groupby("model_key")[["prompt_tokens", "completion_tokens"]].mean()
    agg.index = [MODEL_DISPLAY.get(m, m) for m in agg.index]

    fig, ax = plt.subplots(figsize=(10, 5))
    agg.plot(
        kind="bar", stacked=True, ax=ax,
        color=["#90CAF9", "#1565C0"],
        edgecolor="white",
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Average Tokens")
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    ax.legend(["Prompt Tokens", "Completion Tokens"])
    plt.tight_layout()

    _save(fig, filename)
    return fig


# ---------------------------------------------------------------------------
# 8. Dataset statistics
# ---------------------------------------------------------------------------

def plot_dataset_overview(
    stats_df: pd.DataFrame,
    filename: str = "dataset_overview.png",
) -> plt.Figure:
    """
    Horizontal bar chart showing question counts per dataset.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Question counts
    ax = axes[0]
    colors = [DATASET_COLORS.get(d, "gray") for d in stats_df["dataset"]]
    bars = ax.barh(stats_df["dataset"], stats_df["n_questions"], color=colors, alpha=0.85)
    for bar, val in zip(bars, stats_df["n_questions"]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                str(int(val)), va="center", fontsize=10)
    ax.set_xlabel("Number of Questions")
    ax.set_title("Dataset Sizes (Public Portions)")

    # Numerical vs conceptual (if available)
    ax2 = axes[1]
    if "pct_numerical_answers" in stats_df.columns:
        x = np.arange(len(stats_df))
        ax2.bar(x, stats_df["pct_numerical_answers"], color=colors, alpha=0.85)
        ax2.set_xticks(x)
        ax2.set_xticklabels(stats_df["dataset"])
        ax2.set_ylabel("% Numerical Answers")
        ax2.set_title("Fraction of Numerical Answers")
        ax2.set_ylim(0, 100)

    plt.tight_layout()
    _save(fig, filename)
    return fig

