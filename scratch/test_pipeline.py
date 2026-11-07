import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ["MPLCONFIGDIR"] = os.path.join(PROJECT_ROOT, ".cache", "matplotlib")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import pandas as pd
import numpy as np

class TestFullPipeline(unittest.TestCase):
    def test_evaluation_metrics(self):
        from evaluation import (
            exact_match, token_f1, rouge_l, numeric_accuracy,
            faithfulness_score, is_abstention, evaluate_responses,
            compute_summary_metrics
        )
        # Test string normalization and exact match
        self.assertEqual(exact_match("Apple Inc.", "apple inc"), 1.0)
        self.assertEqual(exact_match("hello", "world"), 0.0)

        # Test token F1
        f1 = token_f1("revenue was $394.3 billion", "revenue was 394.3 billion")
        self.assertGreater(f1, 0.9)

        # Test ROUGE-L
        rl = rouge_l("revenue grew by 5%", "revenue grew by 5%")
        self.assertEqual(rl, 1.0)

        # Test numeric accuracy with tolerance and formatting
        self.assertEqual(numeric_accuracy("$394.3 billion", "394.3"), 1.0)
        self.assertEqual(numeric_accuracy("5.2%", "0.052"), 1.0)
        self.assertEqual(numeric_accuracy("-15.4", "(15.4)"), 1.0)
        self.assertEqual(numeric_accuracy("100", "200"), 0.0)
        self.assertEqual(numeric_accuracy("No numbers here", "50"), 0.0)

        # Test faithfulness
        faith = faithfulness_score("Apple revenue grew", "Apple reported that revenue grew significantly")
        self.assertGreater(faith, 0.8)

        # Test abstention
        self.assertTrue(is_abstention("I cannot determine the answer from the context provided."))
        self.assertFalse(is_abstention("The net profit was $50 million."))

        # Test evaluate_responses function
        responses_df = pd.DataFrame([{
            "model_key": "qwen2.5-7b",
            "question_id": "q1",
            "dataset_name": "finqa",
            "generated_answer": "Revenue was $394.3 billion",
            "context_used": "Apple net sales were $394.3 billion in 2022.",
            "latency_seconds": 1.2,
            "error": None,
            "prompt_tokens": 50,
            "completion_tokens": 10
        }])
        ref_df = pd.DataFrame([{
            "question_id": "q1",
            "reference_answer": "394.3 billion",
            "context": "Apple net sales were $394.3 billion in 2022.",
            "reasoning_type": "numerical_reasoning"
        }])

        eval_df = evaluate_responses(responses_df, ref_df, compute_bertscore=False, compute_semantic=False)
        self.assertEqual(len(eval_df), 1)
        self.assertGreater(eval_df["token_f1"].iloc[0], 0.5)
        self.assertEqual(eval_df["numeric_accuracy"].iloc[0], 1.0)

        summary, extra = compute_summary_metrics(eval_df)
        self.assertFalse(summary.empty)
        print("✓ evaluation.py tested successfully!")

    def test_preprocessing(self):
        from preprocessing import (
            preprocess_dataset, sample_pilot, normalize_answer,
            extract_number, is_numerical_answer, compute_dataset_statistics
        )
        self.assertEqual(extract_number("$1,234.56"), 1234.56)
        self.assertEqual(extract_number("(500)"), -500.0)
        self.assertTrue(is_numerical_answer("$45.2M"))
        self.assertFalse(is_numerical_answer("Federal Reserve regulation"))

        raw_df = pd.DataFrame([
            {"question": "What is revenue?", "reference_answer": "$100", "context": "Revenue was $100."},
            {"question": "Too short", "reference_answer": "N/A", "context": ""},
            {"question": "   Valid question with whitespace   ", "reference_answer": "Valid", "context": "ctx"}
        ])
        cleaned = preprocess_dataset(raw_df)
        self.assertEqual(len(cleaned), 3)

        sampled = sample_pilot(cleaned, n=2, seed=42)
        self.assertEqual(len(sampled), 2)

        stats = compute_dataset_statistics({"test_ds": cleaned})
        self.assertEqual(len(stats), 1)
        print("✓ preprocessing.py tested successfully!")

    def test_rag_pipeline(self):
        from rag_pipeline import chunk_text, build_corpus, BM25Retriever
        text = "word " * 600
        chunks = chunk_text(text, chunk_size=400, overlap=50)
        self.assertGreater(len(chunks), 1)

        mock_datasets = {
            "test_ds": pd.DataFrame([{
                "question_id": "q1",
                "question": "What is operating income?",
                "context": "Operating income for the period was $50 million, up 10 percent year over year.",
                "company": "TestCorp",
                "document_name": "10-K"
            }])
        }
        all_chunks, metadata = build_corpus(mock_datasets, chunk_size=400, overlap=50)
        self.assertEqual(len(all_chunks), 1)

        bm25 = BM25Retriever()
        bm25.build_index(all_chunks, metadata)
        res = bm25.retrieve("operating income", top_k=1)
        self.assertEqual(len(res), 1)
        self.assertIn("Operating income", res[0][0])
        print("✓ rag_pipeline.py tested successfully!")

    def test_model_inference(self):
        from model_inference import MODELS, DEFAULT_GENERATION_PARAMS, build_prompt
        self.assertEqual(len(MODELS), 6)
        self.assertIn("qwen2.5-7b", MODELS)
        self.assertIn("llama3.1-8b", MODELS)
        self.assertIn("mistral-7b", MODELS)
        self.assertIn("gemma2-9b", MODELS)
        self.assertIn("phi3.5-mini", MODELS)
        self.assertIn("deepseek-r1-7b", MODELS)

        prompt = build_prompt("What is Apple's revenue?", "Revenue was $394B.")
        self.assertEqual(len(prompt), 2)
        self.assertEqual(prompt[0]["role"], "system")
        self.assertEqual(prompt[1]["role"], "user")
        print("✓ model_inference.py tested successfully!")

    def test_visualizations(self):
        import matplotlib
        matplotlib.use("Agg")
        from visualization import (
            plot_metric_comparison, plot_heatmap, plot_radar,
            plot_latency_vs_accuracy, plot_rag_comparison,
            plot_token_usage, plot_dataset_overview
        )
        dummy_df = pd.DataFrame([
            {"model_key": "qwen2.5-7b", "dataset_name": "financebench", "token_f1": 0.65, "rouge_l": 0.55,
             "exact_match": 0.3, "numeric_accuracy": 0.5, "faithfulness": 0.8, "semantic_similarity": 0.7,
             "latency_seconds": 2.1, "prompt_tokens": 100, "completion_tokens": 50, "question_id": "q1"},
            {"model_key": "llama3.1-8b", "dataset_name": "financebench", "token_f1": 0.60, "rouge_l": 0.50,
             "exact_match": 0.25, "numeric_accuracy": 0.45, "faithfulness": 0.75, "semantic_similarity": 0.68,
             "latency_seconds": 2.5, "prompt_tokens": 100, "completion_tokens": 55, "question_id": "q2"},
        ])

        fig1 = plot_metric_comparison(dummy_df, metric="token_f1")
        self.assertIsNotNone(fig1)

        fig2 = plot_heatmap(dummy_df, metrics=["token_f1", "rouge_l"])
        self.assertIsNotNone(fig2)

        fig3 = plot_radar(dummy_df, metrics=["token_f1", "rouge_l"])
        self.assertIsNotNone(fig3)

        fig4 = plot_latency_vs_accuracy(dummy_df, accuracy_metric="token_f1")
        self.assertIsNotNone(fig4)

        fig5 = plot_rag_comparison(dummy_df, dummy_df, metric="token_f1")
        self.assertIsNotNone(fig5)

        fig6 = plot_token_usage(dummy_df)
        self.assertIsNotNone(fig6)

        stats_df = pd.DataFrame([
            {"dataset": "financebench", "n_questions": 150, "pct_numerical_answers": 60.0}
        ])
        fig7 = plot_dataset_overview(stats_df)
        self.assertIsNotNone(fig7)
        print("✓ visualization.py tested successfully!")

if __name__ == "__main__":
    unittest.main()

