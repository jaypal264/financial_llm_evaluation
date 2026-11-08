"""
generate_4model_comparison_plots.py
===================================
Generates crystal-clear, high-resolution comparative figures
specifically contrasting all 4 evaluated open-weight models:
  1. DeepSeek-R1-Distill-Qwen-7B
  2. Qwen2.5-7B-Instruct
  3. Llama-3.1-8B-Instruct
  4. Mistral-7B-Instruct-v0.3
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Matplotlib configuration
os.environ["MPLCONFIGDIR"] = str(Path(__file__).resolve().parent / ".cache" / "matplotlib")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

PROJECT_ROOT = Path(__file__).resolve().parent
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# 4 Core Models Definition & Theme Colors
MODELS = [
    ("deepseek-r1-7b", "DeepSeek-R1-Distill-7B", "#6A1B9A", "Reasoning-Distilled"),
    ("qwen2.5-7b",    "Qwen2.5-7B-Instruct",     "#1565C0", "Tabular & Dense Doc"),
    ("llama3.1-8b",   "Llama-3.1-8B-Instruct",   "#2E7D32", "High-Speed Generalist"),
    ("mistral-7b",    "Mistral-7B-Instruct",     "#E65100", "Open Foundation Base"),
]

# Load actual summary data
direct_df = pd.read_csv(PROJECT_ROOT / "results" / "summary" / "summary_direct.csv")
rag_df = pd.read_csv(PROJECT_ROOT / "results" / "summary" / "summary_rag.csv")
all_df = pd.read_csv(PROJECT_ROOT / "results" / "summary" / "summary_all.csv")


# =============================================================================
# Plot 1: Overall 4-Model Multi-Metric Comparison
# =============================================================================
def plot_overall_4models():
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=300)
    fig.patch.set_facecolor('#0D1B2A')
    ax.set_facecolor('#132238')

    metrics = [
        ("token_f1", "Token F1 (Overall)"),
        ("semantic_similarity", "Semantic Similarity"),
        ("faithfulness", "Lexical Faithfulness"),
    ]
    
    n_metrics = len(metrics)
    n_models = len(MODELS)
    x = np.arange(n_metrics)
    width = 0.18

    # Aggregated direct means per model
    agg_direct = direct_df.groupby("model_key")[["token_f1", "semantic_similarity", "faithfulness"]].mean()

    for idx, (mkey, mname, mcolor, mdesc) in enumerate(MODELS):
        offsets = x + (idx - 1.5) * width
        vals = [agg_direct.loc[mkey, mcol] for mcol, _ in metrics]
        bars = ax.bar(offsets, vals, width, label=f"{mname} ({mdesc})",
                      color=mcolor, edgecolor='#E0E6ED', linewidth=0.8, alpha=0.92)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{val:.3f}", ha='center', va='bottom', fontsize=8, color='#FFFFFF', weight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics], fontsize=11, weight='bold', color='#FFFFFF')
    ax.set_ylabel("Evaluation Score (0.0 to 1.0)", fontsize=10, weight='bold', color='#00B4D8')
    ax.set_title("Overall Performance Across All 4 Models (Direct Context Evaluation)",
                 fontsize=13, weight='bold', color='#00B4D8', pad=14)
    ax.set_ylim(0, 0.48)
    ax.tick_params(colors='#CCD6E0')
    ax.grid(axis='y', color='#234160', linestyle='--', alpha=0.6)
    for spine in ax.spines.values():
        spine.set_color('#234160')

    leg = ax.legend(loc='upper right', facecolor='#1A2E44', edgecolor='#00B4D8', fontsize=8.5)
    for text in leg.get_texts():
        text.set_color('#FFFFFF')

    plt.tight_layout()
    out_path = PLOTS_DIR / "bar_token_f1.png"
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"✓ Saved updated 4-model comparison to: {out_path}")


# =============================================================================
# Plot 2: Benchmark-by-Benchmark 4-Model Breakdown
# =============================================================================
def plot_benchmark_4models():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), dpi=300, sharey=True)
    fig.patch.set_facecolor('#0D1B2A')

    benchmarks = [
        ("financebench", "FinanceBench (SEC Filings)", "Multi-hop & Earnings Reports"),
        ("fintextqa",    "FinTextQA (Regulations)",   "Long-form Statutory Guidance"),
        ("finqa",        "FinQA (Tables & Math)",     "Multi-step Arithmetic on Tables"),
    ]

    for ax_idx, (bkey, btitle, bsubtitle) in enumerate(benchmarks):
        ax = axes[ax_idx]
        ax.set_facecolor('#132238')

        b_data = direct_df[direct_df["dataset_name"] == bkey].set_index("model_key")["token_f1"]
        mkeys = [m[0] for m in MODELS]
        mnames = [m[1].replace("-Instruct", "").replace("-Distill", "") for m in MODELS]
        mcolors = [m[2] for m in MODELS]
        vals = [b_data.loc[mk] if mk in b_data else 0.0 for mk in mkeys]

        bars = ax.bar(range(len(MODELS)), vals, color=mcolors, edgecolor='#E0E6ED', width=0.55, alpha=0.92)

        for bar, val in zip(bars, vals):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.005,
                    f"{val:.3f}", ha='center', va='bottom', fontsize=8.5, color='#FFFFFF', weight='bold')

        ax.set_title(f"{btitle}\n{bsubtitle}", fontsize=9.5, weight='bold', color='#00B4D8', pad=10)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(mnames, rotation=30, ha='right', fontsize=8.5, color='#FFFFFF')
        ax.tick_params(colors='#CCD6E0')
        ax.grid(axis='y', color='#234160', linestyle='--', alpha=0.6)
        for spine in ax.spines.values():
            spine.set_color('#234160')

    axes[0].set_ylabel("Token F1 Score", fontsize=10, weight='bold', color='#00B4D8')
    axes[0].set_ylim(0, 0.22)
    fig.suptitle("Benchmark-Specific Token F1 Across All 4 Models", fontsize=13, weight='bold', color='#FFFFFF', y=1.02)

    plt.tight_layout()
    out_path = PLOTS_DIR / "heatmap_all.png"
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved 4-model benchmark breakdown to: {out_path}")


# =============================================================================
# Plot 3: Direct Context vs. RAG for All 4 Models
# =============================================================================
def plot_rag_4models():
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=300)
    fig.patch.set_facecolor('#0D1B2A')
    ax.set_facecolor('#132238')

    direct_means = direct_df.groupby("model_key")["token_f1"].mean()
    rag_means = rag_df.groupby("model_key")["token_f1"].mean()

    mkeys = [m[0] for m in MODELS]
    mnames = [m[1] for m in MODELS]

    x = np.arange(len(MODELS))
    width = 0.32

    d_vals = [direct_means.loc[mk] for mk in mkeys]
    r_vals = [rag_means.loc[mk] for mk in mkeys]

    bars1 = ax.bar(x - width/2, d_vals, width, label='Direct Context Answering (Exp A)',
                   color='#00B4D8', edgecolor='#FFFFFF', alpha=0.92)
    bars2 = ax.bar(x + width/2, r_vals, width, label='Dense RAG Retrieval (Exp B - FAISS)',
                   color='#FF9F1C', edgecolor='#FFFFFF', alpha=0.92)

    for bar, val in zip(bars1, d_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{val:.3f}", ha='center', va='bottom', fontsize=8.5, color='#00B4D8', weight='bold')

    for bar, val in zip(bars2, r_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{val:.3f}", ha='center', va='bottom', fontsize=8.5, color='#FF9F1C', weight='bold')

    # Percentage change labels
    for i, (dv, rv) in enumerate(zip(d_vals, r_vals)):
        pct = ((rv - dv) / dv) * 100
        ax.annotate(f"{pct:+.1f}%",
                    xy=(x[i], max(dv, rv) + 0.018),
                    ha='center', fontsize=8, color='#FF6B6B' if pct < 0 else '#2EC4B6',
                    weight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='#1A2E44', edgecolor='#234160'))

    ax.set_xticks(x)
    ax.set_xticklabels(mnames, fontsize=9.5, weight='bold', color='#FFFFFF')
    ax.set_ylabel("Token F1 Score", fontsize=10, weight='bold', color='#00B4D8')
    ax.set_title("Direct Context vs. Dense FAISS RAG Across All 4 Models",
                 fontsize=13, weight='bold', color='#00B4D8', pad=14)
    ax.set_ylim(0, 0.16)
    ax.tick_params(colors='#CCD6E0')
    ax.grid(axis='y', color='#234160', linestyle='--', alpha=0.6)
    for spine in ax.spines.values():
        spine.set_color('#234160')

    leg = ax.legend(loc='upper right', facecolor='#1A2E44', edgecolor='#00B4D8', fontsize=9)
    for text in leg.get_texts():
        text.set_color('#FFFFFF')

    plt.tight_layout()
    out_path = PLOTS_DIR / "rag_vs_direct.png"
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"✓ Saved 4-model RAG comparison to: {out_path}")


# =============================================================================
# Plot 4: Latency & Throughput Pareto Curve for All 4 Models
# =============================================================================
def plot_latency_4models():
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=300)
    fig.patch.set_facecolor('#0D1B2A')
    ax.set_facecolor('#132238')

    latencies = direct_df.groupby("model_key")["latency_seconds"].mean()
    accuracies = direct_df.groupby("model_key")["token_f1"].mean()

    for mkey, mname, mcolor, mdesc in MODELS:
        lat = latencies.loc[mkey]
        acc = accuracies.loc[mkey]

        ax.scatter(lat, acc, s=400, color=mcolor, edgecolors='#FFFFFF', linewidth=2, alpha=0.9, zorder=5)
        ax.annotate(f"{mname}\n({lat:.1f}s | F1: {acc:.3f})",
                    (lat, acc), textcoords="offset points", xytext=(0, 14),
                    ha='center', fontsize=8.5, weight='bold', color='#FFFFFF',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#1A2E44', edgecolor=mcolor, alpha=0.9))

    ax.set_xlabel("Average Query Latency (Seconds) — [Lower is Faster]", fontsize=10, weight='bold', color='#00B4D8')
    ax.set_ylabel("Token F1 Accuracy — [Higher is Better]", fontsize=10, weight='bold', color='#00B4D8')
    ax.set_title("Accuracy vs. Latency Efficiency Frontier Across All 4 Models",
                 fontsize=13, weight='bold', color='#00B4D8', pad=14)
    ax.set_xlim(2.0, 14.5)
    ax.set_ylim(0.08, 0.135)
    ax.tick_params(colors='#CCD6E0')
    ax.grid(color='#234160', linestyle='--', alpha=0.6)
    for spine in ax.spines.values():
        spine.set_color('#234160')

    # Background quadrant annotations
    ax.text(2.5, 0.130, "[FAST & ACCURATE]\n(Ideal Production Frontier)", fontsize=8, color='#2EC4B6', weight='bold', alpha=0.7)
    ax.text(10.2, 0.130, "[REASONING-INTENSIVE]\n(Deep Chain-of-Thought)", fontsize=8, color='#00B4D8', weight='bold', alpha=0.7)

    plt.tight_layout()
    out_path = PLOTS_DIR / "radar_chart.png"
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"✓ Saved 4-model latency-accuracy frontier to: {out_path}")


if __name__ == "__main__":
    plot_overall_4models()
    plot_benchmark_4models()
    plot_rag_4models()
    plot_latency_4models()
    print("\nAll 4-model comparison figures successfully generated!")
