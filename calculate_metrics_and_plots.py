"""
calculate_metrics_and_plots.py
==============================
Computes evaluation metrics from real model responses,
generates real summaries and plots, and prints the live evaluation results.
"""

import os
import sys
from pathlib import Path

# Fix matplotlib cache dir on Windows
os.environ["MPLCONFIGDIR"] = str(Path(__file__).resolve().parent / ".cache" / "matplotlib")
(Path(__file__).resolve().parent / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)

import pandas as pd
import numpy as np

import src.data_loader as dl
import src.preprocessing as pp
import src.evaluation as ev
import src.visualization as viz

PROJECT_ROOT = Path(__file__).resolve().parent
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
SUMMARY_DIR = PROJECT_ROOT / "results" / "summary"
RESPONSES_DIR = PROJECT_ROOT / "results" / "model_responses"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"

for d in [METRICS_DIR, SUMMARY_DIR, RESPONSES_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("COMPUTING EVALUATION METRICS FROM REAL MODEL RESPONSES")
    print("=" * 70)

    # 1. Load evaluation questions (references)
    fb = pp.preprocess_dataset(dl.load_financebench())
    ft = pp.preprocess_dataset(dl.load_fintextqa())
    fq = pp.preprocess_dataset(dl.load_finqa())

    p_fb = pp.sample_pilot(fb, n=5, seed=42)
    p_ft = pp.sample_pilot(ft, n=5, seed=42)
    p_fq = pp.sample_pilot(fq, n=5, seed=42)
    eval_df = pd.concat([p_fb, p_ft, p_fq], ignore_index=True)

    # 2. Load model responses
    df_resp_direct = pd.read_csv(RESPONSES_DIR / "model_responses_direct.csv")
    df_resp_rag = pd.read_csv(RESPONSES_DIR / "model_responses_rag.csv")
    print(f"Loaded {len(df_resp_direct)} direct responses and {len(df_resp_rag)} RAG responses.")

    # 3. Evaluate responses
    print("\nEvaluating Direct responses...")
    eval_direct = ev.evaluate_responses(
        responses_df=df_resp_direct,
        reference_df=eval_df,
        compute_bertscore=False,
        compute_semantic=True,
    )
    eval_direct.to_csv(METRICS_DIR / "metrics_direct.csv", index=False)
    print(f"  Saved metrics to {METRICS_DIR / 'metrics_direct.csv'}")

    print("\nEvaluating RAG responses...")
    eval_rag = ev.evaluate_responses(
        responses_df=df_resp_rag,
        reference_df=eval_df,
        compute_bertscore=False,
        compute_semantic=True,
    )
    eval_rag.to_csv(METRICS_DIR / "metrics_rag.csv", index=False)
    print(f"  Saved metrics to {METRICS_DIR / 'metrics_rag.csv'}")

    # 4. Compute summaries
    metric_cols = [
        "exact_match", "token_f1", "rouge_l", "numeric_accuracy",
        "faithfulness", "semantic_similarity", "latency_seconds",
        "prompt_tokens", "completion_tokens",
    ]

    sum_direct = eval_direct.groupby(["model_key", "dataset_name"])[metric_cols].mean().reset_index()
    sum_direct.to_csv(SUMMARY_DIR / "summary_direct.csv", index=False)

    sum_rag = eval_rag.groupby(["model_key", "dataset_name"])[metric_cols].mean().reset_index()
    sum_rag.to_csv(SUMMARY_DIR / "summary_rag.csv", index=False)

    eval_direct["experiment"] = "direct"
    eval_rag["experiment"] = "rag"
    combined_eval = pd.concat([eval_direct, eval_rag], ignore_index=True)
    combined_summary = combined_eval.groupby(["model_key", "dataset_name", "experiment"])[metric_cols].mean().reset_index()
    combined_summary.to_csv(SUMMARY_DIR / "summary_all.csv", index=False)

    print("\n" + "=" * 70)
    print("LIVE EVALUATION RESULTS SUMMARY (Means by Model & Experiment)")
    print("=" * 70)
    summary_table = combined_eval.groupby(["model_key", "experiment"])[
        ["token_f1", "exact_match", "numeric_accuracy", "faithfulness", "semantic_similarity", "latency_seconds"]
    ].mean()
    print(summary_table.round(4).to_string())

    # By dataset breakdown
    print("\n" + "=" * 70)
    print("LIVE EVALUATION RESULTS BY BENCHMARK (Token F1)")
    print("=" * 70)
    bench_table = combined_eval.groupby(["model_key", "dataset_name", "experiment"])["token_f1"].mean().unstack(level=1)
    print(bench_table.round(4).to_string())

    # 5. Generate Real Plots
    print("\nGenerating updated charts from real model evaluation...")
    viz.plot_metric_comparison(sum_direct, metric="token_f1", filename="bar_token_f1.png")
    viz.plot_heatmap(sum_direct, filename="heatmap_all.png")
    viz.plot_radar(sum_direct, filename="radar_chart.png")
    viz.plot_latency_vs_accuracy(sum_direct, accuracy_metric="token_f1", filename="latency_vs_accuracy.png")
    viz.plot_rag_comparison(sum_direct, sum_rag, metric="token_f1", filename="rag_vs_direct.png")
    viz.plot_token_usage(df_resp_direct, filename="token_usage.png")

    print("\nAll live evaluation charts and summaries successfully generated!")


if __name__ == "__main__":
    main()

