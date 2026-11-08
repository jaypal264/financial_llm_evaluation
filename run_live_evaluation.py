"""
run_live_evaluation.py
======================
Executes live inference across open-weight models on FinanceBench, FinTextQA, and FinQA.
Evaluates answers with 8+ metrics, generates real summary tables and plots,
and updates the academic PowerPoint presentation.
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np

# Load environment
from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import src.data_loader as dl
import src.preprocessing as pp
import src.model_inference as mi
import src.rag_pipeline as rag
import src.evaluation as ev
import src.visualization as viz

METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
SUMMARY_DIR = PROJECT_ROOT / "results" / "summary"
RESPONSES_DIR = PROJECT_ROOT / "results" / "model_responses"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"

for d in [METRICS_DIR, SUMMARY_DIR, RESPONSES_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("STARTING LIVE MODEL INFERENCE & BENCHMARK EVALUATION")
    print("=" * 70)

    # 1. Load datasets
    print("\n[Step 1/6] Loading benchmark datasets...")
    fb = pp.preprocess_dataset(dl.load_financebench())
    ft = pp.preprocess_dataset(dl.load_fintextqa())
    fq = pp.preprocess_dataset(dl.load_finqa())

    print(f"  FinanceBench: {len(fb)} records")
    print(f"  FinTextQA   : {len(ft)} records")
    print(f"  FinQA       : {len(fq)} records")

    # 2. Sample balanced pilot batch: 5 questions from each dataset
    print("\n[Step 2/6] Sampling balanced evaluation subset (5 questions/dataset)...")
    p_fb = pp.sample_pilot(fb, n=5, seed=42)
    p_ft = pp.sample_pilot(ft, n=5, seed=42)
    p_fq = pp.sample_pilot(fq, n=5, seed=42)

    eval_df = pd.concat([p_fb, p_ft, p_fq], ignore_index=True)
    print(f"  Total evaluation set: {len(eval_df)} questions across 3 benchmarks")

    # 3. Define target models to evaluate
    # Using the 4 active open-weight models available on the provider
    model_keys = ["qwen2.5-7b", "llama3.1-8b", "mistral-7b", "deepseek-r1-7b"]
    provider = "groq" if os.environ.get("GROQ_API_KEY") else "huggingface"
    print(f"\n[Step 3/6] Target Models ({len(model_keys)}):")
    for mk in model_keys:
        info = mi.MODELS[mk]
        actual_id = info.get("groq_id") if provider == "groq" else info["hf_id"]
        print(f"  • {mk:16s} -> {actual_id} ({info['display_name']})")

    # 4. Experiment A: Direct LLM Inference
    print("\n[Step 4/6] Running Experiment A (Direct Answering)...")
    direct_responses = mi.run_evaluation(
        df=eval_df,
        model_keys=model_keys,
        provider=provider,
        experiment="direct",
        context_col="context",
        delay_between_calls=0.5,
        verbose=True,
    )
    df_resp_direct = pd.DataFrame([r.to_dict() for r in direct_responses])
    df_resp_direct.to_csv(RESPONSES_DIR / "model_responses_direct.csv", index=False)
    print(f"  Saved {len(df_resp_direct)} direct responses to model_responses_direct.csv")

    # 5. Experiment B: RAG Pipeline Inference
    print("\n[Step 5/6] Building RAG index and running Experiment B (RAG Augmentation)...")
    chunks, meta = rag.build_corpus({"financebench": fb, "fintextqa": ft, "finqa": fq})
    print(f"  Built RAG corpus: {len(chunks)} chunks")

    retriever = rag.FAISSRetriever()
    retriever.build_index(chunks, meta, save=True)

    # Augment evaluation questions with retrieved context
    rag_eval_rows = []
    for _, row in eval_df.iterrows():
        q = str(row["question"])
        augmented_context = retriever.retrieve_context(q, top_k=5)[:2000]
        row_dict = row.to_dict()
        row_dict["rag_context"] = augmented_context
        rag_eval_rows.append(row_dict)

    df_rag_eval = pd.DataFrame(rag_eval_rows)

    rag_responses = mi.run_evaluation(
        df=df_rag_eval,
        model_keys=model_keys,
        provider=provider,
        experiment="rag",
        context_col="rag_context",
        delay_between_calls=0.5,
        verbose=True,
    )
    df_resp_rag = pd.DataFrame([r.to_dict() for r in rag_responses])
    df_resp_rag.to_csv(RESPONSES_DIR / "model_responses_rag.csv", index=False)
    print(f"  Saved {len(df_resp_rag)} RAG responses to model_responses_rag.csv")

    # 6. Comprehensive Metric Evaluation
    print("\n[Step 6/6] Computing multi-dimensional evaluation metrics...")
    eval_direct = ev.evaluate_responses(
        responses_df=df_resp_direct,
        reference_df=eval_df,
        compute_bertscore=False, # fast evaluation
        compute_semantic=True,
    )
    eval_direct.to_csv(METRICS_DIR / "metrics_direct.csv", index=False)

    eval_rag = ev.evaluate_responses(
        responses_df=df_resp_rag,
        reference_df=eval_df,
        compute_bertscore=False,
        compute_semantic=True,
    )
    eval_rag.to_csv(METRICS_DIR / "metrics_rag.csv", index=False)

    # Compute Summaries
    sum_direct = ev.compute_summary_metrics(eval_direct)
    sum_direct.to_csv(SUMMARY_DIR / "summary_direct.csv", index=False)

    sum_rag = ev.compute_summary_metrics(eval_rag)
    sum_rag.to_csv(SUMMARY_DIR / "summary_rag.csv", index=False)

    # Combined summary
    eval_direct["experiment"] = "direct"
    eval_rag["experiment"] = "rag"
    combined_eval = pd.concat([eval_direct, eval_rag], ignore_index=True)
    combined_summary = combined_eval.groupby(["model_key", "dataset_name", "experiment"])[
        ["exact_match", "token_f1", "rouge_l", "numeric_accuracy", "faithfulness", "semantic_similarity", "latency_seconds"]
    ].mean().reset_index()
    combined_summary.to_csv(SUMMARY_DIR / "summary_all.csv", index=False)

    print("\n" + "=" * 70)
    print("LIVE EVALUATION RESULTS SUMMARY (Token F1 & Exact Match)")
    print("=" * 70)
    overview = combined_eval.groupby(["model_key", "experiment"])[
        ["token_f1", "numeric_accuracy", "faithfulness", "latency_seconds"]
    ].mean()
    print(overview.round(4))

    # 7. Generate Real Plots
    print("\nGenerating updated charts from real model evaluation...")
    viz.plot_bar_metric_by_model(sum_direct, metric="token_f1", output_dir=PLOTS_DIR)
    viz.plot_metric_heatmap(sum_direct, metric="token_f1", output_dir=PLOTS_DIR)
    viz.plot_radar_metrics(sum_direct, output_dir=PLOTS_DIR)
    viz.plot_latency_vs_accuracy(sum_direct, output_dir=PLOTS_DIR)
    viz.plot_rag_vs_direct(sum_direct, sum_rag, metric="token_f1", output_dir=PLOTS_DIR)
    viz.plot_token_usage_breakdown(sum_direct, output_dir=PLOTS_DIR)

    print("\nAll live evaluation artifacts successfully generated!")


if __name__ == "__main__":
    main()

